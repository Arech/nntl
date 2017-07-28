/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2016, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "_activation_wrapper.h"

//This is a basic building block of almost any feedforward neural network - fully connected layer of neurons.

namespace nntl {

	//#todo: add AddendumsTupleT template parameter, as done in LPH

	template<typename FinalPolymorphChild, typename ActivFunc, typename GradWorks, typename DropoutT>
	class _LFC 
		: public m_layer_learnable
		, public _impl::_act_wrap<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc, DropoutT>
	{
	private:
		typedef _impl::_act_wrap<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc, DropoutT> _base_class_t;

	public:
		static_assert(bActivationForHidden, "ActivFunc template parameter should be derived from activation::_i_activation");

		typedef GradWorks grad_works_t;
		static_assert(::std::is_base_of<_impl::_i_grad_works<real_t>, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");

		static constexpr const char _defName[] = "fcl";

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// layer weight matrix: <m_neurons_cnt rows> x <m_incoming_neurons_cnt +1(bias)>,
		// i.e. weights for individual neuron are stored row-wise (that's necessary to make fast cut-off of bias-related weights
		// during backpropagation  - and that's the reason, why is it deformable)
		realmtxdef_t m_weights;
		
		realmtxdef_t m_dLdW;//doesn't guarantee to retain it's value between usage in different code flows;
		// may share memory with some other data structure. Must be deformable for grad_works_t

		real_t m_nTiledTimes;

	public:
		grad_works_t m_gradientWorks; //don't use directly, use getter		
		grad_works_t& get_gradWorks()noexcept { return m_gradientWorks; }

	protected:

		//this flag controls the weights matrix initialization and prevents reinitialization on next nnet.train() calls
		bool m_bWeightsInitialized;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.
			
			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
			
			if (utils::binary_option<true>(ar, serialization::serialize_weights)) ar & NNTL_SERIALIZATION_NVP(m_weights);

			if (utils::binary_option<true>(ar, serialization::serialize_grad_works)) ar & m_gradientWorks;//dont use nvp or struct here for simplicity

			_dropout_serialize(ar, version);
		}


		//////////////////////////////////////////////////////////////////////////
		// functions
	public:
		~_LFC() noexcept {};
		_LFC(const char* pCustomName
			, const neurons_count_t _neurons_cnt
			, const real_t learningRate = real_t(.01)
			, const real_t dpa = real_t(1.0) //"dropout_percent_alive"
		)noexcept
			: _base_class_t(_neurons_cnt, pCustomName), m_weights()
			, m_bWeightsInitialized(false), m_gradientWorks(learningRate)
			, m_nTiledTimes(0.)
		{
			dropoutPercentActive(dpa);
			m_activations.will_emulate_biases();
		};
		
		//#TODO: move all generic fullyconnected stuff into a special base class!

		const realmtx_t& get_weights()const noexcept { NNTL_ASSERT(m_bWeightsInitialized); return m_weights; }
		realmtx_t& get_weights() noexcept { NNTL_ASSERT(m_bWeightsInitialized); return m_weights; }

		bool set_weights(realmtx_t&& W)noexcept {
			if (W.empty() || W.emulatesBiases()
				|| (W.cols() != get_self().get_incoming_neurons_cnt() + 1)
				|| W.rows() != get_self().get_neurons_cnt())
			{
				NNTL_ASSERT(!"Wrong weight matrix passed!");
				return false;
			}

			//m_weights = ::std::move(W);
			m_weights = ::std::forward<realmtx_t>(W);
			m_bWeightsInitialized = true;
			return true;
		}

		bool reinit_weights()noexcept {
			return _activation_init_weights(m_weights);
		}

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			auto ec = _base_class_t::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;
			
			m_nTiledTimes = real_t(lid.nTiledTimes);

			const auto neurons_cnt = get_self().get_neurons_cnt();

			NNTL_ASSERT(!m_weights.emulatesBiases());
			if (m_bWeightsInitialized) {
				//just double check everything is fine
				NNTL_ASSERT(neurons_cnt == m_weights.rows());
				NNTL_ASSERT(get_self().get_incoming_neurons_cnt() + 1 == m_weights.cols());
				NNTL_ASSERT(!m_weights.empty());
			} else {
				//TODO: initialize weights from storage for nn eval only

				// initializing
				if (!m_weights.resize(neurons_cnt, get_incoming_neurons_cnt() + 1)) return ErrorCode::CantAllocateMemoryForWeights;

				if (!reinit_weights()) return ErrorCode::CantInitializeWeights;

				m_bWeightsInitialized = true;
			}

			lid.nParamsToLearn = m_weights.numel();

			const auto training_batch_size = get_self().get_common_data().training_batch_size();

			//Math interface may have to operate on the following matrices:
			// m_weights, m_dLdW - (m_neurons_cnt, get_incoming_neurons_cnt() + 1)
			// m_activations - (biggestBatchSize, m_neurons_cnt+1) and unbiased matrices derived from m_activations - such as m_dAdZ
			// prevActivations - size (m_training_batch_size, get_incoming_neurons_cnt() + 1)
			get_self().get_iMath().preinit(::std::max({
				m_weights.numel()
				, _activation_tmp_mem_reqs()
				,realmtx_t::sNumel(training_batch_size, get_incoming_neurons_cnt() + 1)
			}));

			if (!_dropout_init(get_self().get_common_data().is_training_possible()
				, Dropout_t::bDropoutWorksAtEvaluationToo ? get_self().get_common_data().biggest_batch_size() : training_batch_size
				, neurons_cnt) )
			{
				return ErrorCode::DropoutInitFailed;
			}

			if (get_self().get_common_data().is_training_possible()) {
				//it'll be training session, therefore must allocate necessary supplementary matrices and form temporary memory reqs.

				lid.max_dLdA_numel = realmtx_t::sNumel(training_batch_size, neurons_cnt);
				// we'll need 1 temporarily matrix for bprop(): it is a dL/dW [m_neurons_cnt x get_incoming_neurons_cnt()+1]
				lid.maxMemTrainingRequire = m_weights.numel();
			}

			if (!m_gradientWorks.init(get_self().get_common_data(), m_weights.size()))return ErrorCode::CantInitializeGradWorks;

			lid.bHasLossAddendum = hasLossAddendum();

			lid.bOutputDifferentDuringTraining = layer_has_dropout<self_t>::value && get_self().get_common_data().is_training_possible();

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			m_gradientWorks.deinit();
			m_dLdW.clear();
			m_nTiledTimes = real_t(0.);
			_dropout_deinit();
			_base_class_t::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			if (get_self().get_common_data().is_training_possible()) {
				NNTL_ASSERT(ptr && cnt >= m_weights.numel());
				m_dLdW.useExternalStorage(ptr, m_weights);
				NNTL_ASSERT(!m_dLdW.emulatesBiases());
			}
		}
		
		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			_base_class_t::on_batch_size_change(pNewActivationStorage);

			const auto& CD = get_self().get_common_data();
			_dropout_on_batch_size_change(CD.is_training_mode(), CD.get_cur_batch_size());
		}

	protected:
		//help compiler to isolate fprop functionality from the specific of previous layer
		void _fprop(const realmtx_t& prevActivations)noexcept {
			const auto bTrainingMode = get_self().get_common_data().is_training_mode();
			auto& _iI = get_self().get_iInspect();
			_iI.fprop_begin(get_self().get_layer_idx(), prevActivations, bTrainingMode);

			//restoring biases, should they were altered in drop_samples()
			if (m_activations.isHoleyBiases() && !get_self().is_activations_shared()) {
				m_activations.set_biases();
			}

			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());
			NNTL_ASSERT(prevActivations.test_biases_ok());
			NNTL_ASSERT(m_activations.rows() == prevActivations.rows());
			NNTL_ASSERT(prevActivations.cols() == m_weights.cols());

			//might be necessary for Nesterov momentum application
			if (bTrainingMode) m_gradientWorks.pre_training_fprop(m_weights);

			auto& iM = get_self().get_iMath();

			_iI.fprop_makePreActivations(m_weights, prevActivations);
			iM.mMulABt_Cnb(prevActivations, m_weights, m_activations);
			_iI.fprop_preactivations(m_activations);

			NNTL_ASSERT(get_self().is_activations_shared() || m_activations.test_biases_ok());

			_activation_fprop(iM);
			_iI.fprop_activations(m_activations);

			NNTL_ASSERT(get_self().is_activations_shared() || m_activations.test_biases_ok());

			if (bDropout()) {
				_dropout_apply(m_activations, bTrainingMode, iM, get_self().get_iRng(), _iI);
				NNTL_ASSERT(get_self().is_activations_shared() || m_activations.test_biases_ok());
			}

			NNTL_ASSERT(prevActivations.test_biases_ok());
			_iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		void _cust_inspect(const realmtx_t& M)const noexcept{}

		void _bprop(realmtx_t& dLdA, const realmtx_t& prevActivations, const bool bPrevLayerIsInput, realmtx_t& dLdAPrev)noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

			auto& _iI = get_self().get_iInspect();
			_iI.bprop_begin(get_self().get_layer_idx(), dLdA);
						

			dLdA.assert_storage_does_not_intersect(dLdAPrev);
			dLdA.assert_storage_does_not_intersect(m_dLdW);
			dLdAPrev.assert_storage_does_not_intersect(m_dLdW);
			NNTL_ASSERT(get_self().get_common_data().is_training_mode());
			NNTL_ASSERT(prevActivations.test_biases_ok());

			NNTL_ASSERT(m_activations.emulatesBiases() && !m_dLdW.emulatesBiases());

			NNTL_ASSERT(m_activations.size_no_bias() == dLdA.size());
			//NNTL_ASSERT(m_dAdZ_dLdZ.size() == m_activations.size_no_bias());
			NNTL_ASSERT(m_dLdW.size() == m_weights.size());

			NNTL_ASSERT(bPrevLayerIsInput || prevActivations.emulatesBiases());//input layer in batch mode may have biases included, but no emulatesBiases() set
			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());
			NNTL_ASSERT(mtx_size_t(get_self().get_common_data().get_cur_batch_size(), get_incoming_neurons_cnt() + 1) == prevActivations.size());
			NNTL_ASSERT(bPrevLayerIsInput || dLdAPrev.size() == prevActivations.size_no_bias());//in vanilla simple BP we shouldn't calculate dLdAPrev for the first layer			

			auto& iM = get_self().get_iMath();

			if (bDropout()) {
				//we must cancel activations that was dropped out by the mask (should they've been restored by activation_f_t::df())
				//and restore the scale of dL/dA according to 1/p
				//because the true scaled_dL/dA = 1/p * computed_dL/dA
				//we must undo the scaling step from inverted dropout in order to obtain correct activation values
				//It must be done as a basis to obtain correct dA/dZ
				_dropout_restoreScaling(dLdA, m_activations, iM, _iI);

				//we must undo the scaling step from inverted dropout in order to obtain correct activation values
				//That is must be done as a basis to obtain correct dA/dZ
				//_dropout_cancelScaling(m_activations, iM, _iI);
			}
			
			_iI.bprop_finaldLdA(dLdA);

			_iI.bprop_predAdZ(m_activations);
			
			realmtx_t dLdZ;
			dLdZ.useExternalStorage_no_bias(m_activations);

			//computing dA/dZ using m_activations (aliased to dLdZ variable, which eventually will be a dL/dZ
			_activation_bprop(dLdZ, iM);

			_iI.bprop_dAdZ(dLdZ);
			//compute dL/dZ=dL/dA.*dA/dZ into dA/dZ
			iM.evMul_ip(dLdZ, dLdA);
			_iI.bprop_dLdZ(dLdZ);

			//NB: if we're going to use some kind of regularization of the activation values, we should make sure, that excluded
			//activations (for example, by a dropout or by a gating layer) aren't influence the regularizer. For the dropout
			// there's a m_dropoutMask available (it has zeros for excluded and 1/m_dropoutPercentActive for included activations).
			// By convention, gating layer and other external 'things' would pass dLdA with zeroed elements, that corresponds to
			// excluded activations, because by the definition dL/dA_i == 0 means that i-th activation value should be left intact.

			get_self()._cust_inspect(dLdZ);

			//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
			// BTW: even if some of neurons of this layer could have been "disabled" by a dropout (therefore their
			// corresponding dL/dZ element is set to zero), because we're working with batches, but not a single samples,
			// due to averaging the dL/dW over the whole batch 
			// (dLdW(i's neuron,j's lower layer neuron) = Sum_over_batch( dLdZ(i)*Aprev(j) ) ), it's almost impossible
			// to get some element of dLdW equals to zero, because it'll require that dLdZ entries for some neuron over the
			// whole batch were set to zero.
			NNTL_ASSERT(m_nTiledTimes > 0);
			iM.mScaledMulAtB_C(m_nTiledTimes / real_t(dLdZ.rows()), dLdZ, prevActivations, m_dLdW);
			_iI.bprop_dLdW(dLdZ, prevActivations, m_dLdW);

			if (!bPrevLayerIsInput) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//finally compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				m_weights.hide_last_col();
				iM.mMulAB_C(dLdZ, m_weights, dLdAPrev);
				m_weights.restore_last_col();//restore weights back
			}

			//now we can apply gradient to the weights
			m_gradientWorks.apply_grad(m_weights, m_dLdW);

			NNTL_ASSERT(prevActivations.test_biases_ok());

			_iI.bprop_end(dLdAPrev);
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._fprop(lowerLayer.get_activations());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtx_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._bprop(dLdA, lowerLayer.get_activations(), ::std::is_base_of<m_layer_input, LowerLayer>::value, dLdAPrev);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			return 1;
		}

		static constexpr bool is_trivial_drop_samples()noexcept { return true; }

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT(get_self().is_drop_samples_mbc());
			NNTL_ASSERT(!get_self().is_activations_shared() || !bBiasesToo);
			NNTL_ASSERT(!mask.emulatesBiases() && 1 == mask.cols() && m_activations.rows() == mask.rows() && mask.isBinary());
			NNTL_ASSERT(m_activations.emulatesBiases());

			m_activations.hide_last_col();
			get_self().get_iMath().mrwMulByVec(m_activations, mask.data());
			m_activations.restore_last_col();

			if (bBiasesToo) {
				m_activations.copy_biases_from(mask.data());
			}
		}

		//////////////////////////////////////////////////////////////////////////

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept { return m_gradientWorks.lossAddendum(m_weights); }
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return m_gradientWorks.hasLossAddendum(); }

		

	protected:

		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 < inc_neurons_cnt);
			_base_class_t::_preinit_layer(ili, inc_neurons_cnt);
			NNTL_ASSERT(get_self().get_layer_idx() > 0);
		}
	};

	template<typename FinalPolymorphChild, typename ActivFunc, typename GradWorks, typename DropoutT>
	using _layer_fully_connected = _LFC<FinalPolymorphChild, ActivFunc, GradWorks, DropoutT>;

	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _LFC
	// If you need to derive a new class, derive it from _LFC (to make static polymorphism work)
	template <
		typename ActivFunc = activation::sigm<d_interfaces::real_t>
		, typename GradWorks = grad_works<d_interfaces>
		, typename DropoutT = default_dropout_for<ActivFunc>
	> class LFC final 
		: public _LFC<LFC<ActivFunc, GradWorks, DropoutT>, ActivFunc, GradWorks, DropoutT>
	{
	public:
		~LFC() noexcept {};
		LFC(const neurons_count_t _neurons_cnt, const real_t learningRate = real_t(.01)
			, const real_t dropoutFrac = real_t(0.0), const char* pCustomName = nullptr
		)noexcept
			: _LFC<LFC<ActivFunc, GradWorks, DropoutT>, ActivFunc, GradWorks, DropoutT>
			(pCustomName, _neurons_cnt, learningRate, dropoutFrac) {};
		LFC(const char* pCustomName, const neurons_count_t _neurons_cnt, const real_t learningRate = real_t(.01)
			, const real_t dropoutFrac = real_t(0.0)
		)noexcept
			: _LFC<LFC<ActivFunc, GradWorks, DropoutT>, ActivFunc, GradWorks, DropoutT>
			(pCustomName, _neurons_cnt, learningRate, dropoutFrac) {};
	};

	template <typename ActivFunc = activation::sigm<d_interfaces::real_t>,
		typename GradWorks = grad_works<d_interfaces>
	> using layer_fully_connected = typename LFC<ActivFunc, GradWorks>;
}

