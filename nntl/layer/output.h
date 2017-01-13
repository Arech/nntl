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

#include "_layer_base.h"

namespace nntl {

	template<typename ActivFunc, typename GradWorks, typename FinalPolymorphChild>
	class _layer_output : public m_layer_output, public _layer_base<typename GradWorks::interfaces_t, FinalPolymorphChild> {
	private:
		typedef _layer_base<typename GradWorks::interfaces_t, FinalPolymorphChild> _base_class;

	public:
		typedef ActivFunc activation_f_t;
		//output layer uses only _i_function::f() (during forward propagation). During backward propagation _i_activation_loss::dLdZ()
		//is used instead of _i_activation::df()
		static_assert(std::is_base_of<activation::_i_function<real_t>, activation_f_t>::value, "ActivFunc template parameter should be derived from activation::_i_function");
		static_assert(std::is_base_of<activation::_i_activation_loss<real_t>, activation_f_t>::value, "ActivFunc template parameter should be derived from activation::_i_activation_loss");

		typedef GradWorks grad_works_t;
		static_assert(std::is_base_of<_impl::_i_grad_works<real_t>, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// matrix of layer neurons activations: <batch_size rows> x <m_neurons_cnt+1(bias) cols> for fully connected layer
		realmtxdef_t m_activations;
		// layer weight matrix: <m_neurons_cnt rows> x <m_incoming_neurons_cnt +1(bias)>
		//need it to be deformable to get rid of unnecessary bias column when doing dLdAprev
		realmtxdef_t m_weights;

		//realmtx_t m_dLdZ;//doesn't guarantee to retain it's value between usage in different code flows; may share memory with some other data structure

		realmtxdef_t m_dLdW;//doesn't guarantee to retain it's value between usage in different code flows;
		//may share memory with some other data structure. Must be deformable for grad_works_t
		
		real_t m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd;
		bool m_bRestrictdLdZ;//restriction flag should be permanent for init/deinit calls and changed only by explicit calls to respective functions
		
	    //this flag controls the weights matrix initialization and prevents reinitialization on next nnet.train() calls
		bool m_bWeightsInitialized;

	public:
		grad_works_t m_gradientWorks;
		
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.

			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);

			if (utils::binary_option<true>(ar, serialization::serialize_weights)) ar & NNTL_SERIALIZATION_NVP(m_weights);
			
			if (utils::binary_option<true>(ar, serialization::serialize_grad_works)) ar & m_gradientWorks;//dont use nvp or struct here for simplicity

			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_bRestrictdLdZ);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictLowerBnd);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictUpperBnd);

				//ar & NNTL_SERIALIZATION_NVP(m_max_fprop_batch_size);
				//ar & NNTL_SERIALIZATION_NVP(m_training_batch_size);
			}
		}


		//////////////////////////////////////////////////////////////////////////
		//methods
	public:
		~_layer_output() noexcept {};

		_layer_output(const neurons_count_t _neurons_cnt, real_t learningRate = 0.01, const char* pCustomName = nullptr) noexcept
			: _base_class(_neurons_cnt, pCustomName), m_activations(), m_weights(), m_dLdW(), m_bWeightsInitialized(false)
			, m_gradientWorks(learningRate)
			, m_bRestrictdLdZ(false), m_dLdZRestrictLowerBnd(.0), m_dLdZRestrictUpperBnd(.0)
		{
			//dont need biases in last layer!  --- it is OFF by default
			//m_activations.dont_emulate_biases();
		};
		static constexpr const char _defName[] = "outp";

		//constexpr const bool is_output_layer()const noexcept { return true; }
		const realmtxdef_t& get_activations()const noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			return m_activations;
		}
		const mtx_size_t get_activations_size()const noexcept { return m_activations.size(); }

		//#TODO: move all generic fullyconnected stuff into a special base class!

		const realmtx_t& get_weights()const noexcept { 
			NNTL_ASSERT(m_bWeightsInitialized);
			return m_weights; 
		}
		//should be called after assembling layers into layer_pack, - it initializes _incoming_neurons_cnt
		bool set_weights(realmtx_t&& W)noexcept {
			if (W.empty() || W.emulatesBiases() || (W.cols() != get_incoming_neurons_cnt() + 1)
				|| W.rows() != get_self().get_neurons_cnt())
			{
				NNTL_ASSERT(!"Wrong weight matrix passed!");
				return false;
			}
			//m_weights = std::move(W);
			m_weights = std::forward<realmtx_t>(W);
			m_bWeightsInitialized = true;
			return true;
		}

		ErrorCode init(_layer_init_data_t& lid)noexcept {
			auto ec = _base_class::init(lid);
			if (ErrorCode::Success != ec) return ec;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			NNTL_ASSERT(!m_weights.emulatesBiases());
			if (m_bWeightsInitialized) {
				//just double check everything is fine
				NNTL_ASSERT(get_self().get_neurons_cnt() == m_weights.rows());
				NNTL_ASSERT(get_self().get_incoming_neurons_cnt() + 1 == m_weights.cols());
				NNTL_ASSERT(!m_weights.empty());
			} else {
				//TODO: initialize weights from storage for nn eval only

				// initializing
				if (!m_weights.resize(get_self().get_neurons_cnt(), get_incoming_neurons_cnt() + 1)) return ErrorCode::CantAllocateMemoryForWeights;

				if (!activation_f_t::weights_scheme::init(m_weights, get_self().get_iRng(), get_self().get_iMath()))return ErrorCode::CantInitializeWeights;

				m_bWeightsInitialized = true;
			}

			lid.nParamsToLearn = m_weights.numel();

			NNTL_ASSERT(!m_activations.emulatesBiases());
			if (!m_activations.resize(get_self().get_max_fprop_batch_size(), get_self().get_neurons_cnt())) return ErrorCode::CantAllocateMemoryForActivations;

			//Math interface may have to operate on the following matrices:
			// m_weights, m_dLdW - (m_neurons_cnt, get_incoming_neurons_cnt() + 1)
			// m_activations - (m_max_fprop_batch_size, m_neurons_cnt) and unbiased matrices derived from m_activations - such as m_dLdZ
			// prevActivations - size (m_training_batch_size, get_incoming_neurons_cnt() + 1)
			get_self().get_iMath().preinit(std::max({
				m_weights.numel()
				, activation_f_t::needTempMem(m_activations, get_self().get_iMath())
				, realmtx_t::sNumel(get_self().get_training_batch_size(), get_incoming_neurons_cnt() + 1)
			}));

			if (get_self().get_training_batch_size() > 0) {
				//There's no dLdA coming into the output layer, therefore leave max_dLdA_numel it zeroed
				//lid.max_dLdA_numel = 0; ///realmtx_t::sNumel(get_self().get_training_batch_size(), get_self().get_neurons_cnt());
				
				// we'll need 1 temporarily matrix for bprop(): for dL/dW [m_neurons_cnt x get_incoming_neurons_cnt()+1]
				lid.maxMemBPropRequire = m_weights.numel();
// 				lid.maxMemBPropRequire = realmtx_t::sNumel(get_self().get_training_batch_size(), get_self().get_neurons_cnt())
// 					+ m_weights.numel();
			}

			if (!m_gradientWorks.init(get_self().get_common_data(), m_weights.size()))return ErrorCode::CantInitializeGradWorks;

			lid.bHasLossAddendum = hasLossAddendum();

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit()noexcept {
			m_gradientWorks.deinit();
			m_activations.clear();
			//m_dLdZ.clear();
			m_dLdW.clear();
			_base_class::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			if (get_self().get_training_batch_size() > 0) {
// 				m_dLdZ.useExternalStorage(ptr, get_self().get_training_batch_size(), m_activations.cols_no_bias());
// 				m_dLdW.useExternalStorage(ptr + m_dLdZ.numel(), m_weights);
// 				NNTL_ASSERT(!m_dLdZ.emulatesBiases() && !m_dLdW.emulatesBiases());
// 				NNTL_ASSERT(cnt >= m_dLdZ.numel() + m_dLdW.numel());

				NNTL_ASSERT(ptr && cnt >= m_weights.numel());
				m_dLdW.useExternalStorage(ptr, m_weights);
				NNTL_ASSERT(!m_dLdW.emulatesBiases());
			}
		}

		void set_mode(vec_len_t batchSize)noexcept {
			m_bActivationsValid = false;
			m_bTraining = batchSize == 0;
			NNTL_ASSERT(!m_activations.emulatesBiases());
			if (m_bTraining) {
				m_activations.deform_rows(get_self().get_training_batch_size());
			} else {
				NNTL_ASSERT(batchSize <= get_self().get_max_fprop_batch_size());
				m_activations.deform_rows(batchSize);
			}
		}
	protected:
		void _fprop(const realmtx_t& prevActivations)noexcept {
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), prevActivations, m_bTraining);

			NNTL_ASSERT(m_activations.rows() == prevActivations.rows());
			NNTL_ASSERT(prevActivations.cols() == m_weights.cols());

			//might be necessary for Nesterov momentum application
			if (m_bTraining) m_gradientWorks.pre_training_fprop(m_weights);

			auto& _Math = get_self().get_iMath();
			iI.fprop_makePreActivations(m_weights, prevActivations);
			_Math.mMulABt_Cnb(prevActivations, m_weights, m_activations);
			iI.fprop_preactivations(m_activations);
			activation_f_t::f(m_activations, _Math);
			iI.fprop_activations(m_activations);
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		void _cust_inspect(const realmtx_t& M)const noexcept {}

		void _bprop(const realmtx_t& data_y, const realmtx_t& prevActivations, const bool bPrevLayerIsInput, realmtx_t& dLdAPrev)noexcept {
			m_bActivationsValid = false;
			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), data_y);

			data_y.assert_storage_does_not_intersect(dLdAPrev);
			dLdAPrev.assert_storage_does_not_intersect(m_dLdW);
			NNTL_ASSERT(m_bTraining);
			NNTL_ASSERT(!m_dLdW.emulatesBiases());
			NNTL_ASSERT(m_activations.size() == data_y.size());
			//NNTL_ASSERT(m_dLdZ.size() == m_activations.size());
			NNTL_ASSERT(m_dLdW.size() == m_weights.size());
			NNTL_ASSERT(bPrevLayerIsInput || prevActivations.emulatesBiases());//input layer in batch mode may have biases included, but no emulatesBiases() set
			NNTL_ASSERT(mtx_size_t(get_self().get_training_batch_size(), get_self().get_incoming_neurons_cnt() + 1) == prevActivations.size());
			NNTL_ASSERT(bPrevLayerIsInput || dLdAPrev.size() == prevActivations.size_no_bias());

			auto& _Math = get_self().get_iMath();

			//compute dL/dZ
			iI.bprop_predLdZOut(m_activations, data_y);
			
			activation_f_t::dLdZ(data_y, m_activations, _Math);
			//now dLdZ is calculated into m_activations
			realmtx_t & dLdZ = m_activations;
			iI.bprop_dLdZ(dLdZ);

			if (m_bRestrictdLdZ) {
				_Math.evClamp(dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);
				iI.bprop_postClampdLdZ(dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);
			}

			get_self()._cust_inspect(dLdZ);

			//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
			_Math.mScaledMulAtB_C(real_t(1.0) / real_t(dLdZ.rows()), dLdZ, prevActivations, m_dLdW);

			if (!bPrevLayerIsInput) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//finally compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				m_weights.hide_last_col();
				_Math.mMulAB_C(dLdZ, m_weights, dLdAPrev);
				m_weights.restore_last_col();//restore weights back
			}

			//now we can apply gradient to the weights
			m_gradientWorks.apply_grad(m_weights, m_dLdW);

			iI.bprop_end(dLdAPrev);
		}
	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer");
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._fprop(lowerLayer.get_activations());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
		}		
		template <typename LowerLayer>
		const unsigned bprop(const realmtx_t& data_y, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._bprop(data_y, lowerLayer.get_activations(), std::is_base_of<m_layer_input, LowerLayer>::value, dLdAPrev);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			return 1;
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
			_base_class::_preinit_layer(ili, inc_neurons_cnt);
			NNTL_ASSERT(get_self().get_layer_idx() > 0);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_output
	// If you need to derive a new class, derive it from _layer_output (to make static polymorphism work)
	template <typename ActivFunc = activation::sigm_quad_loss<d_interfaces::real_t>,
		typename GradWorks = grad_works<d_interfaces>
	> class layer_output final 
		: public _layer_output<ActivFunc, GradWorks, layer_output<ActivFunc, GradWorks>>
	{
	public:
		~layer_output() noexcept {};
		layer_output(const neurons_count_t _neurons_cnt, const real_t learningRate=real_t(0.01)
			, const char* pCustomName = nullptr) noexcept 
			: _layer_output<ActivFunc, GradWorks, layer_output<ActivFunc, GradWorks>>(_neurons_cnt, learningRate, pCustomName)
		{};
	};
}

