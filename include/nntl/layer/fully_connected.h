/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of NNTL nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <nntl/layer/_activation_wrapper.h>

//This is a basic building block of almost any feedforward neural network - fully connected layer of neurons.

namespace nntl {

	//fprop only base class
	template<typename FinalPolymorphChild, typename InterfacesT, typename ActivFunc, bool bFPropOnlyInDerived>
	class _LFC_FProp
		: public _impl::conditional_layer_stops_bprop<bFPropOnlyInDerived>
		, public _impl::_act_wrap<FinalPolymorphChild, InterfacesT, ActivFunc>
	{
	private:
		typedef _impl::_act_wrap<FinalPolymorphChild, InterfacesT, ActivFunc> _base_class_t;

	public:
		static_assert(bActivationForHidden, "ActivFunc template parameter should be derived from activations::_i_activation");

		static constexpr const char _defName[] = "fclFP";
		static constexpr bool bAssumeFPropOnly = bFPropOnlyInDerived;

	protected:

		// layer weight matrix: <m_neurons_cnt rows> x <m_incoming_neurons_cnt +1(bias)>,
		// i.e. weights for individual neuron are stored row-wise (that's necessary to make fast cut-off of bias-related weights
		// during backpropagation  - and that's the reason, why is it deformable)
		// Note that bBatchInRow() property must be set to false (the default value)
		realmtxdef_t m_weights;

		//this flag controls the weights matrix initialization and prevents reinitialization on next nnet.train() calls
		bool m_bWeightsInitialized{ false };

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void save(Archive & ar, const unsigned int version) const {
			NNTL_UNREF(version);
			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
			if (utils::binary_option<true>(ar, serialization::serialize_weights)) ar & NNTL_SERIALIZATION_NVP(m_weights);
		}

		template<class Archive>
		void load(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			if (utils::binary_option<true>(ar, serialization::serialize_weights)) {
				realmtx_t M;
				ar & serialization::make_nvp("m_weights", M);
				if (ar.success()) {
					if (!get_self().set_weights(::std::move(M))) {
						STDCOUTL("*** Failed to absorb read weights for layer " << get_self().get_layer_name_str());
						ar.mark_invalid_var();
					}
				} else {
					STDCOUTL("*** Failed to read weights for layer " << get_self().get_layer_name_str()
						<< ", " << ar.get_last_error_str());
				}
			}
			//#todo other vars!
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER();

	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 < inc_neurons_cnt);
			_base_class_t::_preinit_layer(ili, inc_neurons_cnt);
			NNTL_ASSERT(get_layer_idx() > 0);
		}

	public:
		~_LFC_FProp() noexcept {};

		_LFC_FProp(const char* pCustomName, const neurons_count_t _neurons_cnt)noexcept
			: _base_class_t(_neurons_cnt, pCustomName), m_weights(), m_bWeightsInitialized(false)
		{
			NNTL_ASSERT(_neurons_cnt > 0);//LFC should have valid neuron count from the very start
			//m_activations.emulate_biases(!bActivationForOutput);//redundant, base class do this
			NNTL_ASSERT(!bActivationForOutput == m_activations.emulatesBiases());
		};

		_LFC_FProp(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr)noexcept
			: _LFC_FProp(pCustomName, _neurons_cnt) 
		{};

		//////////////////////////////////////////////////////////////////////////

		const realmtx_t& get_weights()const noexcept { NNTL_ASSERT(m_bWeightsInitialized); return m_weights; }
		realmtx_t& get_weights() noexcept { NNTL_ASSERT(m_bWeightsInitialized); return m_weights; }

		bool isWeightsSuitable(const realmtx_t& W)const noexcept {
			if (W.empty() || W.bBatchInRow() || W.emulatesBiases()
				|| (W.cols() != get_incoming_neurons_cnt() + 1) || W.rows() != get_neurons_cnt())
			{
				NNTL_ASSERT(!"Wrong weight matrix passed!");
				return false;
			}
			NNTL_ASSERT(W.test_noNaNs());
			return true;
		}

		//note: it should be called after assembling layers into a layer_pack, b/c it requires _incoming_neurons_cnt
		bool set_weights(realmtx_t&& W)noexcept {
			if (!get_self().isWeightsSuitable(W)) return false;
			get_self().drop_weights();

			m_weights = ::std::move(W);
			m_bWeightsInitialized = true;
			return true;
		}
		bool set_weights(const realmtx_t& W)noexcept {
			if (!get_self().isWeightsSuitable(W)) return false;
			get_self().drop_weights();

			if (!m_weights.cloneFrom(W)) return false;
			m_bWeightsInitialized = true;
			return true;
		}

		bool reinit_weights()noexcept {
			NNTL_ASSERT(m_bWeightsInitialized || !"reinit_weights() can only be called after layer_init()!");
			NNTL_ASSERT(get_self().isWeightsSuitable(m_weights) || !"WTF?! Wrong state of weight matrix");
			return _activation_init_weights(m_weights);
		}

		void drop_weights()noexcept {
			m_weights.clear();
			m_bWeightsInitialized = false;
		}

		//////////////////////////////////////////////////////////////////////////

		//nothing to do here, base class has implementation
// 		void layer_deinit() noexcept {
// 			_base_class_t::layer_deinit();
// 		}

		ErrorCode layer_init(_layer_init_data_t& lid, real_t*const pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(!bActivationForOutput || !pNewActivationStorage);//output layer always uses own activation storage 

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().layer_deinit();
			});

			auto ec = _base_class_t::layer_init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			const auto neurons_cnt = get_neurons_cnt(), incoming_neurons_cnt = get_incoming_neurons_cnt();

			NNTL_ASSERT(!m_weights.emulatesBiases());
			if (m_bWeightsInitialized) {
				//just double check everything is fine
				NNTL_ASSERT(!m_weights.empty() && m_weights.bBatchInColumn());
				//prior weight initialization implies you don't want them to be reinitialized silently
				//if (mtx_size_t(neurons_cnt, incoming_neurons_cnt + 1) != m_weights.size()) {
				if (!get_self().isWeightsSuitable(m_weights)) {
					NNTL_ASSERT(!"WTF? Wrong weight matrix!");
					STDCOUTL("WTF? Wrong weight matrix @layer " << get_self().get_layer_idx() << " " << get_self().get_layer_name_str());
					abort();
				}
			} else {
				NNTL_ASSERT(!m_bWeightsInitialized);//if this assert has fired, then you've tried to use incorrectly sized
				//weight matrix. It'll be handled here, so you may safely skip the assert, but you have to know, it was a bad idea.
				m_weights.clear();

				// initializing
				if (!m_weights.resize(neurons_cnt, incoming_neurons_cnt + 1)) return ErrorCode::CantAllocateMemoryForWeights;

				m_bWeightsInitialized = true;//MUST be set prior call to reinit_weights()
				if (!get_self().reinit_weights()) {
					m_weights.clear();
					m_bWeightsInitialized = false;
					return ErrorCode::CantInitializeWeights;
				}
			}

			// there's no use of iMath _istor_alloc/_istor_free() functions during operation of this class's functions
			// (as well as inside of iMath calls we do here), so there's no need to call iMath.preinit() at all
// 			get_iMath().preinit(::std::max({
// 				m_weights.numel()
// 				, realmtx_t::sNumel(get_common_data().max_fprop_batch_size(), incoming_neurons_cnt + 1)//for prev activations.
// 				//, _activation_tmp_mem_reqs() //base class is doing it!
// 			}));

			bSuccessfullyInitialized = true;
			return ec;
		}

		//no need to check anything here
// 		vec_len_t on_batch_size_change(const vec_len_t incBatchSize, real_t*const pNewActivationStorage = nullptr)noexcept {
// 			const auto outgBs = _base_class_t::on_batch_size_change(incBatchSize, pNewActivationStorage);
// 			NNTL_ASSERT(outgBs > 0 && outgBs <= (get_common_data().is_training_mode() ? m_outgBS.maxTrainBS : m_outgBS.maxBS));
// 			NNTL_ASSERT(!m_activations.empty() && m_activations.batch_size() == outgBs);
// 			return outgBs;
// 		}

		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			get_self()._lfc_fprop(lowerLayer.get_activations());
		}

	protected:
		//separate function to isolate fprop() functionality from a previous layer type
		// #supportsBatchInRow for prevAct as well as m_activations
		void _lfc_fprop(const realmtx_t& prevAct, const bool bWillProcessActivationsLater = false)noexcept {
			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT_MTX_NO_NANS(prevAct);
			NNTL_ASSERT(get_incoming_neurons_cnt() == prevAct.sample_size());

			//just don't even check biases if the flag forbids it
			NNTL_ASSERT(is_activations_shared() || bActivationForOutput != m_activations.emulatesBiases());
			NNTL_ASSERT(bActivationForOutput || is_activations_shared() || m_activations.test_biases_strict());

			const auto bTrainingMode = get_common_data().is_training_mode();

			NNTL_ASSERT(prevAct.batch_size() <= m_incBS.max_bs4mode(bTrainingMode));
			//max() here b/c we don't know how the function is used in a derived class (which BS structure to apply)
			NNTL_ASSERT(m_activations.batch_size() <= ::std::max(m_incBS.max_bs4mode(bTrainingMode), m_outgBS.max_bs4mode(bTrainingMode)));
			NNTL_ASSERT(prevAct.batch_size() == m_activations.batch_size());

			auto& _iI = get_iInspect();
			_iI.fprop_begin(get_layer_idx(), prevAct, bTrainingMode);

			//might be necessary for Nesterov momentum application
		#pragma warning(push)
		#pragma warning(disable : 4127) //C4127: conditional expression is constant
			if (!bAssumeFPropOnly && bTrainingMode) get_self()._on_fprop_in_training_mode();
		#pragma warning(pop)

			_iI.fprop_makePreActivations(m_weights, prevAct);

			auto& iM = get_iMath();
			//iM.mMulABt_Cnb(prevAct, m_weights, m_activations);
			//note, mMul_prevAct_weights_2_act() supports bBatchInRow() for prevAct and doesn't support it for m_activations
			iM.mMul_prevAct_weights_2_act(prevAct, m_weights, m_activations);

			_iI.fprop_preactivations(m_activations);

			_activation_fprop(iM);
			_iI.fprop_activations(m_activations);

			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT(bActivationForOutput || is_activations_shared() || m_activations.test_biases_strict());
			
			if(!bWillProcessActivationsLater) _iI.fprop_end(m_activations);
			
			m_bActivationsValid = true;
		}

		static void _on_fprop_in_training_mode() noexcept {}

	public:
		template <typename LowerLayer>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(always_false<LowerLayer>::value, "_LFC_FProp::bprop() should never be called. Redefine in derived class if necessary");
			return 0;
		}
	};
	

	//For dropout combine with LDo
	// _LFC supports #supportsBatchInRow only for previous layer activations.
	// To implement the feature for this layer requires either stripping of a bias row in column-major mode
	// or reimplementing _activation_bprop() so it'd work columnwise on all but the last (bias) row and
	// output result into different matrix. Too much work for now.
	template<typename FinalPolymorphChild, typename ActivFunc, typename GradWorks>
	class _LFC 
		: public m_layer_learnable
		, public _LFC_FProp<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc, false>
	{
	private:
		typedef _LFC_FProp<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc, false> _base_class_t;

	public:
		typedef GradWorks grad_works_t;
		static_assert(::std::is_base_of<_impl::_i_grad_works<real_t>, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");

		static constexpr const char _defName[] = "fcl";

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		grad_works_t m_gradientWorks;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void save(Archive & ar, const unsigned int version) const {
			NNTL_UNREF(version);
			ar & ::boost::serialization::base_object<_base_class_t>(*this);

			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.
			if (utils::binary_option<true>(ar, serialization::serialize_grad_works)) ar & m_gradientWorks;//dont use nvp or struct here for simplicity
		}

		template<class Archive>
		void load(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			ar & ::boost::serialization::base_object<_base_class_t>(*this);
			//#todo other vars!
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER();

		//////////////////////////////////////////////////////////////////////////
		// functions
	public:
		~_LFC() noexcept {};
		_LFC(const char* pCustomName, const neurons_count_t _neurons_cnt, const real_t learningRate = real_t(.01))noexcept
			: _base_class_t(_neurons_cnt, pCustomName), m_gradientWorks(learningRate)
		{
			NNTL_ASSERT(m_activations.emulatesBiases());
		};

		_LFC(const neurons_count_t _neurons_cnt, const real_t learningRate = real_t(.01), const char* pCustomName=nullptr)noexcept
			: _LFC(pCustomName, _neurons_cnt, learningRate)
		{};
		
		grad_works_t& get_gradWorks()noexcept { return m_gradientWorks; }
		const grad_works_t& get_gradWorks()const noexcept { return m_gradientWorks; }

		ErrorCode layer_init(_layer_init_data_t& lid, real_t*const pNewActivationStorage = nullptr)noexcept {
			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().layer_deinit();
			});

			auto ec = _base_class_t::layer_init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;
			
			const numel_cnt_t prmsNumel = get_self().bUpdateWeights() ? m_weights.numel() : 0;
			lid.nParamsToLearn = prmsNumel;
			
// 			get_iMath().preinit(::std::max({
// 				realmtx_t::sNumel(get_common_data().biggest_batch_size(), get_self().get_incoming_neurons_cnt() + 1)//for prev activations.
// 			}));

			if (get_common_data().is_training_possible()) {
				//it'll be training session, therefore must allocate necessary supplementary matrices and form temporary memory reqs.
				// because we're free to reuse the dLdA space as it's no longer need, we'll also be using dLdA matrix to compute dLdW into it
				NNTL_ASSERT(lid.outgBS.maxTrainBS > 0);
				lid.max_dLdA_numel = ::std::max(realmtx_t::sNumel(lid.outgBS.maxTrainBS, get_neurons_cnt()), prmsNumel);
			}

			if (!get_self().get_gradWorks().gw_init(get_common_data(), m_weights))return ErrorCode::CantInitializeGradWorks;

			//#BUGBUG current implementation of GW::hasLossAddendum could return false because LAs are currently disabled,
			//however they could be enabled later. Seems like not a major bug, so I'll leave it to fix later.
			//#SeeAlso layer_output
			lid.bLossAddendumDependsOnWeights = get_self().get_gradWorks().hasLossAddendum();

			bSuccessfullyInitialized = true;
			return ec;
		}

		void layer_deinit() noexcept {
			get_gradWorks().gw_deinit();
			_base_class_t::layer_deinit();
		}

		void _on_fprop_in_training_mode()noexcept {
			get_self().get_gradWorks().pre_training_fprop(m_weights);
		}

	protected:

		void _cust_inspect(const realmtx_t& )const noexcept{}

		unsigned _lfc_bprop(realmtxdef_t& dLdA, const realmtx_t& prevAct, const bool bPrevLayerWBprop, realmtx_t& dLdAPrev)noexcept {
			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

			NNTL_ASSERT(get_common_data().is_training_mode());
			NNTL_ASSERT(prevAct.batch_size() <= m_incBS.maxTrainBS);
			NNTL_ASSERT(m_activations.batch_size() <= m_outgBS.maxTrainBS);
			NNTL_ASSERT(prevAct.batch_size() == m_activations.batch_size());

			NNTL_ASSERT_MTX_NO_NANS(dLdA);
			NNTL_ASSERT_MTX_NO_NANS(prevAct);
			NNTL_ASSERT_MTX_NO_NANS(m_activations);

			dLdA.assert_storage_does_not_intersect(dLdAPrev);
			NNTL_ASSERT(prevAct.emulatesBiases() && (is_activations_shared() || bActivationForOutput != m_activations.emulatesBiases()));
			NNTL_ASSERT(m_activations.size_no_bias() == dLdA.size());
			NNTL_ASSERT(get_incoming_neurons_cnt() == prevAct.sample_size());
			NNTL_ASSERT(!bPrevLayerWBprop || dLdAPrev.size() == prevAct.size_no_bias());//in vanilla simple BP we shouldn't calculate dLdAPrev for the first layer			
			NNTL_ASSERT(!bPrevLayerWBprop || dLdAPrev.bBatchInRow() == prevAct.bBatchInRow());

			auto& _iI = get_iInspect();
			_iI.bprop_begin(get_layer_idx(), dLdA);

			_iI.bprop_finaldLdA(dLdA);

			_iI.bprop_predAdZ(m_activations);
			
			//dLdZ aliases m_activations (biases skipped)
			// That is the point where the issue with bBatchInRow() arise. We can't easily strip bias row from
			// each m_activations column in this mode. Therefore leaving the old requirement untouched for now:
			// bBatchInRow() MUST be false here.
			// Possible future workaround: decrease rows count but leave proper ldim() trick? Need a workaround for 
			//     _activation_bprop() then, also for softmax, and finally everything else below here, but it would probably work
			//     in a fastest way possible... Anyway, it's not the main issue now and just having bBatchInRow() for previous
			//     activations only is fine.
			NNTL_ASSERT(m_activations.bBatchInColumn() && dLdA.bBatchInColumn());
			realmtx_t dLdZ(m_activations.data(), m_activations, realmtx_t::tag_noBias());

			auto& iM = get_iMath();

			//compile-time shortcut for linear/identity activation to turn matrix filling and ew-multiplication
			//into just copying dLdA into dLdZ
			if (activation::is_activation_identity<Activation_t>::value) {
				const auto b = dLdA.copy_to(dLdZ);
				NNTL_ASSERT(b);
			} else {
				//computing dA/dZ using m_activations (aliased to dLdZ variable, which eventually will be a dL/dZ
				_activation_bprop(dLdZ, iM);

				_iI.bprop_dAdZ(dLdZ);
				//compute dL/dZ=dL/dA.*dA/dZ into dA/dZ
				iM.evMul_ip(dLdZ, dLdA);
			}
			
			//since that moment, we no longer need the data in dLdA, therefore we're free to reuse that space for any
			//temporary computations we need provided that we keep the size of dLdA on function exit untouched.
			// We'll be using dLdA later to compute and apply dL/dW. Note, that we've stated during layer_init() that dLdA
			// must have enough size to fit dL/dW in it.
			_iI.bprop_dLdZ(dLdZ);

			//NB: if we're going to use some kind of regularization of the activation values, we should make sure, that excluded
			//activations (for example, by a dropout or by a gating layer) aren't influence the regularizer. For the dropout
			// there's a m_dropoutMask available (it has zeros for excluded and 1/m_dropoutPercentActive for included activations).
			// By convention, gating layer and other external 'things' would pass dLdA with zeroed elements, that corresponds to
			// excluded activations, because by the definition dL/dA_i == 0 means that i-th activation value should be left intact.

			get_self()._cust_inspect(dLdZ);

			//computing dL/dAPrev
			if (bPrevLayerWBprop) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				// m_weights.hide_last_col();
				// iM.mMulAB_C(dLdZ, m_weights, dLdAPrev);
				// m_weights.restore_last_col();//restore weights back
				iM.mMul_dLdZ_weights_2_dLdAPrev(dLdZ, m_weights, dLdAPrev);
			}

			if (get_self().bUpdateWeights()) {
				//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
				// BTW: even if some of neurons of this layer could have been "disabled" by a dropout (therefore their
				// corresponding dL/dZ element is set to zero), because we're working with batches, but not a single samples,
				// due to averaging the dL/dW over the whole batch 
				// (dLdW(i's neuron,j's lower layer neuron) = Sum_over_batch( dLdZ(i)*Aprev(j) ) ), it's almost impossible
				// to get some element of dLdW equals to zero, because it'll require that dLdZ entries for some neuron over the
				// whole batch were set to zero.

				realmtxdef_t& dLdW = dLdA;//we'll be using dLdA to compute dLdW and now creating a corresponding alias to it for readability
				dLdW.deform_like(m_weights);
				//ok to do that, because the storage (dLdA) was allocated using layer_init_data::max_dLdA_numel member variable,
				// that we've set to fit m_weights during layer_init()
				
				// #TODO support bBatchInRow for dLdZ!!!
				iM.mMulScaled_dLdZ_prevAct_2_dLdW(
					//(inspector::is_gradcheck_inspector<iInspect_t>::value ? real_t(1) : m_nTiledTimes) / real_t(m_activations.batch_size())
					real_t(1) / real_t(m_activations.batch_size())
					, dLdZ, prevAct, dLdW);

				_iI.bprop_dLdW(dLdZ, prevAct, dLdW);

				//now we can apply gradient to the weights
				get_gradWorks().apply_grad(m_weights, dLdW);
				//and restoring dLdA==dLdW size back (probably we can even skip this step, however, it's cheap and it's better to make it)
				dLdW.deform_like_no_bias(m_activations);
			}

			NNTL_ASSERT(prevAct.test_biases_strict());//just to make sure we didn't spoil anything

			_iI.bprop_end(dLdAPrev);
			return 1;
		}

	public:

		template <typename LowerLayer>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			//NNTL_ASSERT(get_self().bDoBProp());
			return get_self()._lfc_bprop(dLdA, lowerLayer.get_activations(), is_layer_with_bprop<LowerLayer>::value, dLdAPrev);
		}

		//////////////////////////////////////////////////////////////////////////

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept { return get_gradWorks().lossAddendum(m_weights); }
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return get_gradWorks().hasLossAddendum(); }
	};

	
	template<typename InterfacesT, typename ActivFunc>
	class LFC_FProp final : public _LFC_FProp<LFC_FProp<InterfacesT, ActivFunc>, InterfacesT, ActivFunc, true> {
		typedef _LFC_FProp<LFC_FProp<InterfacesT, ActivFunc>, InterfacesT, ActivFunc, true> _base_class_t;
	public:
		template<typename...ArgsT>
		LFC_FProp(ArgsT&&... ar)noexcept : _base_class_t(::std::forward<ArgsT>(ar)...) {}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _LFC
	// If you need to derive a new class, derive it from _LFC (to make static polymorphism work)
	template <typename ActivFunc = activation::sigm<d_interfaces::real_t>
		, typename GradWorks = grad_works<d_interfaces>
	> class LFC final : public _LFC<LFC<ActivFunc, GradWorks>, ActivFunc, GradWorks>
	{
		typedef _LFC<LFC<ActivFunc, GradWorks>, ActivFunc, GradWorks> _base_class_t;
	public:
		template<typename...ArgsT>
		LFC(ArgsT&&... ar)noexcept : _base_class_t(::std::forward<ArgsT>(ar)...) {}
	};

	template <typename ActivFunc = activation::sigm<d_interfaces::real_t>,
		typename GradWorks = grad_works<d_interfaces>
	> using layer_fully_connected = typename LFC<ActivFunc, GradWorks>;
}

