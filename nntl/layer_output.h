/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015, Arech (aradvert@gmail.com; https://github.com/Arech)
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

	template<typename ActivFunc, typename Interfaces, typename GradWorks, typename FinalPolymorphChild>
	class _layer_output : public m_layer_output, public _layer_base<FinalPolymorphChild> {
	private:
		typedef _layer_base<FinalPolymorphChild> _base_class;

	public:
		typedef math_types::floatmtxdef_ty floatmtxdef_t;
		static_assert(std::is_base_of<floatmtx_t, floatmtxdef_t>::value, "math_types::floatmtxdef_ty must be derived from math_types::floatmtx_ty!");

		typedef typename Interfaces::iMath_t iMath_t;
		static_assert(std::is_base_of<math::_i_math, iMath_t>::value, "Interfaces::iMath type should be derived from _i_math");

		typedef typename Interfaces::iRng_t iRng_t;
		static_assert(std::is_base_of<rng::_i_rng, iRng_t>::value, "Interfaces::iRng type should be derived from _i_rng");

		typedef ActivFunc activation_f_t;
		static_assert(std::is_base_of<activation::_i_activation, activation_f_t>::value, "ActivFunc template parameter should be derived from activation::_i_activation");
		static_assert(std::is_base_of<activation::_i_activation_loss, activation_f_t>::value, "ActivFunc template parameter should be derived from activation::_i_activation_loss");

		typedef GradWorks grad_works_t;
		static_assert(std::is_base_of<_i_grad_works, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// matrix of layer neurons activations: <batch_size rows> x <m_neurons_cnt+1(bias) cols> for fully connected layer
		floatmtxdef_t m_activations;
		// layer weight matrix: <m_neurons_cnt rows> x <m_incoming_neurons_cnt +1(bias)>
		//need it to be deformable to get rid of unnecessary bias column when doing dLdAprev
		floatmtxdef_t m_weights;

		floatmtx_t m_dLdZ;//doesn't guarantee to retain it's value between usage in different code flows; may share memory with some other data structure
		floatmtx_t m_dLdW;//doesn't guarantee to retain it's value between usage in different code flows; may share memory with some other data structure

		iMath_t* m_pMath;
		iRng_t* m_pRng;

		float_t_ m_learningRate;

		bool m_bRestrictdLdZ;//restriction flag should be permanent for init/deinit calls and changed only by explicit calls to respective functions
		float_t_ m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd;

	public:
		grad_works_t m_gradientWorks;

	protected:

		vec_len_t m_max_fprop_batch_size, m_training_batch_size;

		//this flag controls the weights matrix initialization and prevents reinitialization on next nnet.train() calls
		bool m_bWeightsInitialized;

		//////////////////////////////////////////////////////////////////////////
		//methods
	public:
		~_layer_output() noexcept {};

		_layer_output(const neurons_count_t _neurons_cnt, float_t_ learningRate = 0.01) noexcept 
			: _base_class(_neurons_cnt), m_activations(), m_weights(), m_dLdZ(), m_dLdW(), m_bWeightsInitialized(false)
			, m_learningRate(learningRate), m_max_fprop_batch_size(0), m_training_batch_size(0)
			, m_pMath(nullptr), m_pRng(nullptr), m_bRestrictdLdZ(false)
		{
			NNTL_ASSERT(learningRate > 0);
			//dont need biases in last layer!  --- it is OFF by default
			//m_activations.dont_emulate_biases();
		};

		constexpr bool is_output_layer()const noexcept { return true; }
		const floatmtx_t& get_activations()const noexcept { return m_activations; }

		template<typename _layer_init_data_t>
		ErrorCode init(_layer_init_data_t& lid)noexcept {
			static_assert(std::is_same<iMath_t, _layer_init_data_t::i_math_t>::value, "_layer_init_data_t::i_math_t type must be the same as given by class template parameter Interfaces::iMath");
			static_assert(std::is_same<iRng_t, _layer_init_data_t::i_rng_t>::value, "_layer_init_data_t::i_rng_t must be the same as given by class template parameter Interfaces::iRng");

			bool bSuccessfullyInitialized = false;
			utils::scope_exit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) {
					deinit();
				}
			});

			m_pMath = &lid.iMath;
			m_pRng = &lid.iRng;

			m_max_fprop_batch_size = lid.max_fprop_batch_size;
			m_training_batch_size = lid.training_batch_size;

			NNTL_ASSERT(!m_weights.emulatesBiases());
			if (m_bWeightsInitialized) {
				//just double check everything is fine
				NNTL_ASSERT(m_neurons_cnt == m_weights.rows());
				NNTL_ASSERT(get_incoming_neurons_cnt() + 1 == m_weights.cols());
				NNTL_ASSERT(!m_weights.empty());
			} else {
				//TODO: initialize weights from storage for nn eval only

				// initializing
				if (!m_weights.resize(m_neurons_cnt, get_incoming_neurons_cnt() + 1)) return ErrorCode::CantAllocateMemoryForWeights;

				activation_f_t::init_weights(m_weights, *m_pRng);
				m_bWeightsInitialized = true;
			}

			lid.nParamsToLearn = m_weights.numel();

			NNTL_ASSERT(!m_activations.emulatesBiases());
			if (! m_activations.resize(m_max_fprop_batch_size, m_neurons_cnt)) return ErrorCode::CantAllocateMemoryForActivations;

			//Math interface may have to operate on the following matrices:
			// m_weights, m_dLdW - (m_neurons_cnt, get_incoming_neurons_cnt() + 1)
			// m_activations - (m_max_fprop_batch_size, m_neurons_cnt) and unbiased matrices derived from m_activations - such as m_dLdZ
			// prevActivations - size (m_training_batch_size, get_incoming_neurons_cnt() + 1)
			m_pMath->preinit( std::max(std::max(m_weights.numel(), m_activations.numel()),
				floatmtx_t::sNumel(m_training_batch_size, get_incoming_neurons_cnt() + 1)));

			if (m_training_batch_size > 0) {
				//we need 2 temporarily matrices for training: one for dA/dZ -> dL/dZ [batchSize x m_neurons_cnt]
				// and one for dL/dW [m_neurons_cnt x get_incoming_neurons_cnt()+1]
				lid.max_dLdA_numel = floatmtx_t::sNumel(m_training_batch_size, m_activations.cols_no_bias());
				lid.maxMemBPropRequire = lid.max_dLdA_numel + m_weights.numel();
			}

			if (!m_gradientWorks.init(grad_works_t::init_struct_t(m_pMath, m_weights.size())))return ErrorCode::CantInitializeGradWorks;

			bSuccessfullyInitialized = true;
			return ErrorCode::Success;
		}

		void deinit()noexcept {
			m_gradientWorks.deinit();
			m_activations.clear();
			m_dLdZ.clear();
			m_dLdW.clear();
			m_pMath = nullptr;
			m_pRng = nullptr;
		}

		void initMem(float_t_* ptr, numel_cnt_t cnt)noexcept {
			if (m_training_batch_size > 0) {
				m_dLdZ.useExternalStorage(ptr, m_training_batch_size, m_activations.cols_no_bias());
				m_dLdW.useExternalStorage(&ptr[m_dLdZ.numel()], m_weights);
				NNTL_ASSERT(!m_dLdZ.emulatesBiases() && !m_dLdW.emulatesBiases());
				NNTL_ASSERT(cnt >= m_dLdZ.numel() + m_dLdW.numel());
				//NNTL_ASSERT(m_dAdZ.size() == m_activations.size() && m_dLdW.size() == m_weights.size());
			}
		}
		
		void set_mode(vec_len_t batchSize)noexcept {
			m_bTraining = batchSize == 0;
			NNTL_ASSERT(!m_activations.emulatesBiases());
			if (m_bTraining) {
				m_activations.deform_rows(m_training_batch_size);
			} else {
				NNTL_ASSERT(batchSize <= m_max_fprop_batch_size);
				m_activations.deform_rows(batchSize);
			}
		}

		//template <typename i_math_t = nnet_def_interfaces::Math, typename i_rng_t = nnet_def_interfaces::Rng>
		//void fprop(const floatmtx_t& prevActivations, i_math_t& iMath, i_rng_t& iRng, const bool bInTraining)noexcept {
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer");
			const auto& prevActivations = lowerLayer.get_activations();

			NNTL_ASSERT(m_activations.rows() == prevActivations.rows());
			NNTL_ASSERT(prevActivations.cols() == m_weights.cols());

			//might be necessary for Nesterov momentum application
			if (m_bTraining) m_gradientWorks.pre_training_fprop(m_weights);

			m_pMath->mMulABt_Cnb(prevActivations, m_weights, m_activations);
			activation_f_t::f(m_activations, *m_pMath);
		}

		//template <typename i_math_t = nnet_def_interfaces::Math>
		//void bprop(const floatmtx_t& data_y, const floatmtx_t& prevActivations, floatmtx_t& dLdAPrev, i_math_t& iMath, const bool bPrevLayerIsInput)noexcept{
		template <typename LowerLayer>
		void bprop(const floatmtx_t& data_y, const LowerLayer& lowerLayer, floatmtx_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer");
			const auto& prevActivations = lowerLayer.get_activations();

			data_y.assert_storage_does_not_intersect(dLdAPrev);
			NNTL_ASSERT(!m_dLdZ.emulatesBiases() && !m_dLdW.emulatesBiases());
			NNTL_ASSERT(m_activations.size() == data_y.size());
			NNTL_ASSERT(m_dLdZ.size() == m_activations.size());
			NNTL_ASSERT(m_dLdW.size() == m_weights.size());
			NNTL_ASSERT(lowerLayer.is_input_layer() || prevActivations.emulatesBiases());//input layer in batch mode may have biases included, but no emulatesBiases() set
			NNTL_ASSERT(mtx_size_t(m_training_batch_size, get_incoming_neurons_cnt() + 1) == prevActivations.size());
			NNTL_ASSERT(dLdAPrev.size() == prevActivations.size_no_bias());
			

			//compute dL/dZ
			activation_f_t::dLdZ(m_activations, data_y, m_dLdZ, *m_pMath);

			if (m_bRestrictdLdZ) m_pMath->evClamp(m_dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);

			//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
			m_pMath->mScaledMulAtB_C(float_t_(1.0) / float_t_(m_dLdZ.rows()), m_dLdZ, prevActivations, m_dLdW);
			//now we can apply gradient to the weights
			m_gradientWorks.apply_grad(m_weights, m_dLdW, m_learningRate);

			if (!lowerLayer.is_input_layer()) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//finally compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				m_weights.hide_last_col();
				m_pMath->mMulAB_C(m_dLdZ, m_weights, dLdAPrev);
				m_weights.restore_last_col();//restore weights back
			}
		}

		//////////////////////////////////////////////////////////////////////////

		//use this function to put a restriction on dL/dZ value - this may help in training large networks
		//(see Alex Graves's “Generating Sequences With Recurrent Neural Networks”(2013) )
		self_ref_t restrict_dL_dZ(float_t_ lowerBnd, float_t_ upperBnd)noexcept {
			NNTL_ASSERT(lowerBnd < upperBnd);
			m_bRestrictdLdZ = true;
			m_dLdZRestrictLowerBnd = lowerBnd;
			m_dLdZRestrictUpperBnd = upperBnd;
			return get_self();
		}
		self_ref_t drop_dL_dZ_restriction()noexcept { 
			m_bRestrictdLdZ = false; 
			return get_self();
		}
		const float_t_ learning_rate()const noexcept { return m_learningRate; }
		self_ref_t learning_rate(float_t_ lr)noexcept {
			NNTL_ASSERT(lr > 0);
			m_learningRate = lr;
			return get_self();
		}

	protected:
		friend class _preinit_layers;
		void _preinit_layer(const layer_index_t idx, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 < idx);
			NNTL_ASSERT(0 < inc_neurons_cnt);
			_base_class::_preinit_layer(idx, inc_neurons_cnt);

			//m_activations.resize(m_neurons_cnt);//there is no need to initialize allocated memory
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final input layer with all default functionality of _layer_output
	// If you need to derive a new class, derive from _layer_output
	// 
	template <typename ActivFunc = activation::sigm_quad_loss<>,
		typename Interfaces = nnet_def_interfaces,
		typename GradWorks = grad_works<typename Interfaces::iMath_t>
	> class layer_output final 
		: public _layer_output<ActivFunc, Interfaces, GradWorks, layer_output<ActivFunc, Interfaces, GradWorks>>
	{
	public:
		~layer_output() noexcept {};
		layer_output(const neurons_count_t _neurons_cnt, const float_t_ learningRate=0.01) noexcept :
			_layer_output<ActivFunc, Interfaces, GradWorks, layer_output<ActivFunc, Interfaces, GradWorks>>(_neurons_cnt, learningRate) {};
	};

}