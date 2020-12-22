/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

namespace nntl {

	template<typename FinalPolymorphChild, typename ActivFunc, typename GradWorks>
	class _layer_output 
		: public m_layer_output
		, public m_layer_learnable
		, public _impl::_act_wrap<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc>
	{
	private:
		typedef _impl::_act_wrap<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc> _base_class_t;

	public:
		static_assert(bActivationForOutput && bActivationForHidden, "ActivFunc template parameter must be derived from "
			"activation::_i_activation_loss<> and activation::_i_activation<>");

		typedef GradWorks grad_works_t;
		static_assert(::std::is_base_of<_impl::_i_grad_works<real_t>, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");
		
		static constexpr const char _defName[] = "outp";

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// layer weight matrix: <m_neurons_cnt rows> x <m_incoming_neurons_cnt +1(bias)>
		//need it to be deformable to get rid of unnecessary bias column when doing dLdAprev
		realmtxdef_t m_weights;

		realmtxdef_t m_dLdW;//doesn't guarantee to retain it's value between usage in different code flows;
		//may share memory with some other data structure. Must be deformable for grad_works_t
		
		//real_t m_nTiledTimes{ 0 };

		real_t m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd;
		bool m_bRestrictdLdZ;//restriction flag should be permanent for init/deinit calls and changed only by explicit calls to respective functions

	    //this flag controls the weights matrix initialization and prevents reinitialization on next nnet.train() calls
		bool m_bWeightsInitialized;

	public:
		grad_works_t m_gradientWorks;
		grad_works_t& get_gradWorks()noexcept { return m_gradientWorks; }
		
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void save(Archive & ar, const unsigned int version) const {
			NNTL_UNREF(version);
			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.

			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);

			if (utils::binary_option<true>(ar, serialization::serialize_weights)) ar & NNTL_SERIALIZATION_NVP(m_weights);
			
			if (utils::binary_option<true>(ar, serialization::serialize_grad_works)) ar & m_gradientWorks;//dont use nvp or struct here for simplicity

			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_bRestrictdLdZ);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictLowerBnd);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictUpperBnd);
			}
		}
		template<class Archive>
		void load(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			if (utils::binary_option<true>(ar, serialization::serialize_weights)) {
				realmtx_t M;
				ar & serialization::make_nvp("m_weights", M);
				if (ar.success()) {
					if (!set_weights(::std::move(M))) {
						STDCOUTL("*** Failed to absorb read weights for layer " << get_layer_name_str());
						ar.mark_invalid_var();
					}
				} else {
					STDCOUTL("*** Failed to read weights for layer " << get_layer_name_str()
						<< ", " << ar.get_last_error_str());
				}
			}
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_bRestrictdLdZ);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictLowerBnd);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictUpperBnd);
			}
			//#todo other vars!
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER()

		//////////////////////////////////////////////////////////////////////////
		//methods
	public:
		~_layer_output() noexcept {};

		_layer_output(const char* pCustomName, const neurons_count_t _neurons_cnt, real_t learningRate = real_t(.01)) noexcept
			: _base_class_t(_neurons_cnt, pCustomName), m_weights(), m_dLdW(), m_bWeightsInitialized(false)
			, m_gradientWorks(learningRate)
			, m_bRestrictdLdZ(false), m_dLdZRestrictLowerBnd(.0), m_dLdZRestrictUpperBnd(.0)
		{
			m_activations.dont_emulate_biases();
		};

		//#TODO: move all generic fullyconnected stuff into a special base class!

		const realmtx_t& get_weights()const noexcept { NNTL_ASSERT(m_bWeightsInitialized); return m_weights; }
		realmtx_t& get_weights() noexcept { NNTL_ASSERT(m_bWeightsInitialized); return m_weights; }

		//should be called after assembling layers into layer_pack, - it initializes _incoming_neurons_cnt
		bool set_weights(realmtx_t&& W)noexcept {
			drop_weights();

			if (W.empty() || W.emulatesBiases() || (W.cols() != get_incoming_neurons_cnt() + 1)
				|| W.rows() != get_neurons_cnt())
			{
				NNTL_ASSERT(!"Wrong weight matrix passed!");
				return false;
			}
			NNTL_ASSERT(W.test_noNaNs());

			m_weights = ::std::move(W);
			m_bWeightsInitialized = true;
			return true;
		}

		bool reinit_weights()noexcept {
			NNTL_ASSERT(m_bWeightsInitialized || !"reinit_weights() can only be called after init()!");
			NNTL_ASSERT(!m_weights.empty() && mtx_size_t(get_neurons_cnt(), get_incoming_neurons_cnt() + 1) == m_weights.size()
				|| !"WTF?! Wrong state of weight matrix");
			return _activation_init_weights(m_weights);
		}

		void drop_weights()noexcept {
			m_weights.clear();
			m_bWeightsInitialized = false;
		}

		ErrorCode init(_layer_init_data_t& lid)noexcept {
			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			auto ec = _base_class_t::init(lid);
			if (ErrorCode::Success != ec) return ec;

			//we need this to be able to include layer_output into LPT (though, it won't work now, at least because LPT doesn't
			// propagate m_output_layer marker)
			NNTL_ASSERT(lid.nTiledTimes == 1 || !"If you've updated LPT to incapsulate this layer, uncomment every m_nTiledTimes usage if it is still necessary");
			//and note, that that nTiledTimes machinery is actually a hack to make LPT pass a grad check. So it's better to make grad checker
			//routine aware of layer differences.
			//m_nTiledTimes = real_t(lid.nTiledTimes);
			
			NNTL_ASSERT(!m_weights.emulatesBiases());
			if (m_bWeightsInitialized) {
				//just double check everything is fine
				NNTL_ASSERT(!m_weights.empty());
				//prior weight initialization implies you don't want them to be reinitialized silently
				if (mtx_size_t(get_neurons_cnt(), get_incoming_neurons_cnt() + 1) != m_weights.size()) {
					NNTL_ASSERT(!"WTF? Wrong weight matrix!");
					STDCOUTL("WTF? Wrong weight matrix for layer " << get_layer_idx() << " " << get_layer_name_str());
					abort();
				}
				
			} else {
				NNTL_ASSERT(!m_bWeightsInitialized);//if this assert has fired, then you've tried to use incorrectly sized
				//weight matrix. It'll be handled here, so you may safely skip the assert, but you have to know, it was a bad idea.
				m_weights.clear();

				//TODO: initialize weights from storage for nn eval only

				// initializing
				if (!m_weights.resize(get_neurons_cnt(), get_incoming_neurons_cnt() + 1)) return ErrorCode::CantAllocateMemoryForWeights;

				m_bWeightsInitialized = true;//MUST be set prior call to reinit_weights()
				if (!reinit_weights()) {
					m_weights.clear();
					m_bWeightsInitialized = false;
					return ErrorCode::CantInitializeWeights;
				}
			}

			const numel_cnt_t prmsNumel = get_self().bUpdateWeights()/* && get_self().bDoBProp()*/ ? m_weights.numel() : 0;
			lid.nParamsToLearn = prmsNumel;
			
			//Math interface may have to operate on the following matrices:
			// m_weights, m_dLdW - (m_neurons_cnt, get_incoming_neurons_cnt() + 1)
			// m_activations - (m_max_fprop_batch_size, m_neurons_cnt) and unbiased matrices derived from m_activations - such as m_dLdZ
			// prevActivations - size (m_training_batch_size, get_incoming_neurons_cnt() + 1)
			get_iMath().preinit(::std::max({
				m_weights.numel()//weights are used during fprop(), so conditioned prmsNumel is irrelevant here
				//, _activation_tmp_mem_reqs()//base class is doing it!
				, realmtx_t::sNumel(get_common_data().training_batch_size(), get_incoming_neurons_cnt() + 1)
			}));

			if (get_common_data().is_training_possible() && get_self().bUpdateWeights() /*&& get_self().bDoBProp()*/) {
				//There's no dLdA coming into the output layer, therefore leave max_dLdA_numel it zeroed
				//lid.max_dLdA_numel = 0;
				
				// we'll need 1 temporarily matrix for bprop(): dL/dW [m_neurons_cnt x get_incoming_neurons_cnt()+1]
				lid.maxMemTrainingRequire = prmsNumel;
			}

			if (!m_gradientWorks.init(get_common_data(), m_weights))return ErrorCode::CantInitializeGradWorks;

			//#BUGBUG current implementation of GW::hasLossAddendum could return false because LAs are currently disabled,
			//however they could be enabled later. Seems like not a major bug, so I'll leave it to fix later.
			//#SeeAlso LFC
			lid.bLossAddendumDependsOnWeights = m_gradientWorks.hasLossAddendum();

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit()noexcept {
			m_gradientWorks.deinit();
			m_dLdW.clear();
			//m_nTiledTimes = real_t(0);
			_base_class_t::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			NNTL_UNREF(cnt);
			if (get_common_data().is_training_possible() && get_self().bUpdateWeights() /*&& get_self().bDoBProp()*/) {
				NNTL_ASSERT(ptr && cnt >= m_weights.numel());
				m_dLdW.useExternalStorage(ptr, m_weights);
				NNTL_ASSERT(!m_dLdW.emulatesBiases());
			} else NNTL_ASSERT(m_dLdW.empty());
		}

	protected:
		void _outp_fprop(const realmtx_t& prevActivations)noexcept {
#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
			NNTL_ASSERT(prevActivations.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK
			NNTL_ASSERT(prevActivations.test_biases_strict());
			NNTL_ASSERT(!m_activations.emulatesBiases());

			auto& _iI = get_iInspect();
			_iI.fprop_begin(get_layer_idx(), prevActivations, get_common_data().is_training_mode());

			NNTL_ASSERT(m_activations.rows() == get_common_data().get_cur_batch_size());
			NNTL_ASSERT(m_activations.rows() == prevActivations.rows());
			NNTL_ASSERT(prevActivations.cols() == m_weights.cols());

			//might be necessary for Nesterov momentum application
			if (get_common_data().is_training_mode()) m_gradientWorks.pre_training_fprop(m_weights);

			auto& iM = get_iMath();
			_iI.fprop_makePreActivations(m_weights, prevActivations);
			iM.mMulABt_Cnb(prevActivations, m_weights, m_activations);

			_iI.fprop_preactivations(m_activations);
			_activation_fprop(iM);
			_iI.fprop_activations(m_activations);

			_iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		void _cust_inspect(const realmtx_t&)const noexcept { }

		template<typename YT>
		unsigned _outp_bprop(const math::smatrix<YT>& data_y, const realmtx_t& prevActivations, const bool bPrevLayerIsInput, realmtx_t& dLdAPrev)noexcept {
			NNTL_ASSERT(prevActivations.test_biases_strict());
			//NNTL_ASSERT(real_t(1.) == m_gradientWorks._learning_rate_scale());//output layer mustn't have lr scaled.
			// Don't implement lrdecay using the _learning_rate_scale(), it's for internal use only!
			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
			NNTL_ASSERT(prevActivations.test_noNaNs());
			NNTL_ASSERT(data_y.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK

			auto& _iI = get_iInspect();
			_iI.bprop_begin(get_layer_idx(), data_y);
			//_iI.bprop_finaldLdA(dLdA); //--doesn't apply here actually

			data_y.assert_storage_does_not_intersect(dLdAPrev);
			NNTL_ASSERT(get_common_data().is_training_mode());
			//NNTL_ASSERT(m_activations.size() == data_y.size()); //too strong. We need special loss functions that may require different columns count
			NNTL_ASSERT(m_activations.rows() == data_y.rows());
			NNTL_ASSERT(m_activations.rows() == get_common_data().get_cur_batch_size());
			NNTL_ASSERT(mtx_size_t(get_common_data().get_cur_batch_size(), get_incoming_neurons_cnt() + 1) == prevActivations.size());
			NNTL_ASSERT(bPrevLayerIsInput || dLdAPrev.size() == prevActivations.size_no_bias());

			auto& iM = get_iMath();

			_iI.bprop_predLdZOut(m_activations, data_y);
			//compute dL/dZ
			_activation_bprop_output(data_y, iM);

			//now dLdZ is calculated into m_activations
			realmtx_t & dLdZ = m_activations;
			_iI.bprop_dLdZ(dLdZ);

			if (m_bRestrictdLdZ) {
				iM.evClamp(dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);
				_iI.bprop_postClampdLdZ(dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);
			}

			get_self()._cust_inspect(dLdZ);

			if (!bPrevLayerIsInput) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				m_weights.hide_last_col();
				iM.mMulAB_C(dLdZ, m_weights, dLdAPrev);
				m_weights.restore_last_col();//restore weights back
			}

			if (get_self().bUpdateWeights()) {
				dLdAPrev.assert_storage_does_not_intersect(m_dLdW);
				NNTL_ASSERT(!m_dLdW.emulatesBiases());
				NNTL_ASSERT(m_dLdW.size() == m_weights.size());

				//NNTL_ASSERT(m_nTiledTimes > 0);
				//#hack to make gradient check of LPT happy, we would ignore the tiling count here entirely (because numerical error is
				// implicitly normalized to batch size, however, analytical - doesn't and it's normalized to the current batch size. Inner
				// layers of LPT have different batch sizes, so we just going to ignore that here. Probably we should:
				//#todo update grad checking algo to take tiling count into account.
				//However, the tiled layer in a real life would have a m_nTiledTimes bigger batch size, than the modeled layer should have,
				// therefore we'll upscale the gradient m_nTiledTimes times.
				//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
				//iM.mScaledMulAtB_C((inspector::is_gradcheck_inspector<iInspect_t>::value ? real_t(1) : m_nTiledTimes) / real_t(dLdZ.rows())
				iM.mScaledMulAtB_C(real_t(1.) / real_t(dLdZ.rows())
					, dLdZ, prevActivations, m_dLdW);
				_iI.bprop_dLdW(dLdZ, prevActivations, m_dLdW);

				//now we can apply gradient to the weights
				m_gradientWorks.apply_grad(m_weights, m_dLdW);
			}			

			_iI.bprop_end(dLdAPrev);
			return 1;
		}
	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer");
			get_self()._outp_fprop(lowerLayer.get_activations());
		}		
		template <typename YT, typename LowerLayer>
		unsigned bprop(const math::smatrix<YT>& data_y, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			//NNTL_ASSERT(get_self().bDoBProp());
			return get_self()._outp_bprop(data_y, lowerLayer.get_activations(), is_layer_input<LowerLayer>::value, dLdAPrev);
		}
		
		//////////////////////////////////////////////////////////////////////////

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept { return m_gradientWorks.lossAddendum(m_weights); }
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return m_gradientWorks.hasLossAddendum(); }

		//////////////////////////////////////////////////////////////////////////

		//use this function to put a restriction on dL/dZ value - this may help in training large networks
		//(see Alex Graves's "Generating Sequences With Recurrent Neural Networks(2013)" )
		self_ref_t restrict_dL_dZ(real_t lowerBnd, real_t upperBnd)noexcept {
			if (upperBnd == 0 || lowerBnd==0) {
				NNTL_ASSERT(upperBnd == 0 && lowerBnd == 0);
				m_bRestrictdLdZ = false;
			} else {
				NNTL_ASSERT(lowerBnd < 0 && upperBnd>0);
				m_bRestrictdLdZ = true;
			}
			m_dLdZRestrictLowerBnd = lowerBnd;
			m_dLdZRestrictUpperBnd = upperBnd;
			return get_self();
		}
		self_ref_t drop_dL_dZ_restriction()noexcept { 
			m_bRestrictdLdZ = false; 
			return get_self();
		}

	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 < inc_neurons_cnt);
			_base_class_t::_preinit_layer(ili, inc_neurons_cnt);
			NNTL_ASSERT(get_self().get_layer_idx() > 0);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_output
	// If you need to derive a new class, derive it from _layer_output (to make static polymorphism work)
	template <typename ActivFunc = activation::sigm_quad_loss<d_interfaces::real_t>,
		typename GradWorks = grad_works<d_interfaces>
	> class layer_output final 
		: public _layer_output<layer_output<ActivFunc, GradWorks>, ActivFunc, GradWorks>
	{
	public:
		~layer_output() noexcept {};
		layer_output(const neurons_count_t _neurons_cnt, const real_t learningRate=real_t(0.01)
			, const char* pCustomName = nullptr) noexcept 
			: _layer_output<layer_output<ActivFunc, GradWorks>, ActivFunc, GradWorks>(pCustomName, _neurons_cnt, learningRate)
		{};

		layer_output(const char* pCustomName, const neurons_count_t _neurons_cnt, const real_t learningRate = real_t(0.01)) noexcept
			: _layer_output<layer_output<ActivFunc, GradWorks>, ActivFunc, GradWorks>(pCustomName, _neurons_cnt, learningRate)
		{};
	};
}

