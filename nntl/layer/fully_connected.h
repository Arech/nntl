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

//This is a basic building block of almost any feedforward neural network - fully connected layer of neurons.

namespace nntl {

	template<typename ActivFunc, typename Interfaces, typename GradWorks, typename FinalPolymorphChild>
	class _layer_fully_connected : public _layer_base<Interfaces, FinalPolymorphChild>
		//, public _i_layer_gate<typename Interfaces::iMath_t::real_t>
	{
	private:
		typedef _layer_base<Interfaces, FinalPolymorphChild> _base_class;

	public:
		typedef ActivFunc activation_f_t;
		static_assert(std::is_base_of<activation::_i_activation, activation_f_t>::value, "ActivFunc template parameter should be derived from activation::_i_function");

		typedef GradWorks grad_works_t;
		static_assert(std::is_base_of<_i_grad_works, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// matrix of layer neurons activations: <batch_size rows> x <m_neurons_cnt+1(bias) cols> for fully connected layer
		realmtxdef_t m_activations;

		// layer weight matrix: <m_neurons_cnt rows> x <m_incoming_neurons_cnt +1(bias)>,
		// i.e. weights for individual neuron are stored row-wise (that's necessary to make fast cut-off of bias-related weights
		// during backpropagation  - and that's the reason, why is it deformable)
		realmtxdef_t m_weights;
		
		//matrix of dropped out neuron activations, used when m_dropoutFraction>0
		realmtx_t m_dropoutMask;//<batch_size rows> x <m_neurons_cnt cols> (must not have a bias column)
		real_t m_dropoutFraction;

		realmtx_t m_dAdZ_dLdZ;//doesn't guarantee to retain it's value between usage in different code flows; may share memory with some other data structure

		realmtxdef_t m_dLdW;//doesn't guarantee to retain it's value between usage in different code flows;
		// may share memory with some other data structure. Must be deformable for grad_works_t

	public:
		grad_works_t m_gradientWorks;

	protected:
		//this flag controls the weights matrix initialization and prevents reinitialization on next nnet.train() calls
		bool m_bWeightsInitialized;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.

			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_dropoutFraction);
				//ar & NNTL_SERIALIZATION_NVP(m_max_fprop_batch_size);
				//ar & NNTL_SERIALIZATION_NVP(m_training_batch_size);
			}
			
			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
			
			if (utils::binary_option<true>(ar, serialization::serialize_weights)) ar & NNTL_SERIALIZATION_NVP(m_weights);

			if (utils::binary_option<true>(ar, serialization::serialize_grad_works)) ar & m_gradientWorks;//dont use nvp or struct here for simplicity

			if (bDropout() && utils::binary_option<true>(ar, serialization::serialize_dropout_mask)) ar & NNTL_SERIALIZATION_NVP(m_dropoutMask);
		}


		//////////////////////////////////////////////////////////////////////////
		// functions
	public:
		~_layer_fully_connected() noexcept {};
		_layer_fully_connected(const neurons_count_t _neurons_cnt, real_t learningRate = .01, real_t dropoutFrac=0.0)noexcept
			: _base_class(_neurons_cnt), m_activations(), m_weights(), m_bWeightsInitialized(false)
				, m_gradientWorks(learningRate)
				, m_dropoutMask(), m_dropoutFraction(dropoutFrac)
		{
			NNTL_ASSERT(0 <= m_dropoutFraction && m_dropoutFraction < 1);
			m_activations.will_emulate_biases();
		};

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, "fcl%d", static_cast<unsigned>(get_self().get_layer_idx()));
		}
		
		const realmtxdef_t& get_activations()const noexcept { return m_activations; }

		//#TODO: move all generic fullyconnected stuff into a special base class!

		const realmtx_t& get_weights()const noexcept {
			NNTL_ASSERT(m_bWeightsInitialized);
			return m_weights;
		}
		bool set_weights(realmtx_t&& W)noexcept {
			if (W.empty() || W.emulatesBiases()
				|| (W.cols() != get_self().get_incoming_neurons_cnt() + 1)
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

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			auto ec = _base_class::init(lid, pNewActivationStorage);
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

				if (!activation_f_t::weights_scheme::init(m_weights, get_self().get_iRng()))return ErrorCode::CantInitializeWeights;

				m_bWeightsInitialized = true;
			}

			lid.nParamsToLearn = m_weights.numel();

			NNTL_ASSERT(m_activations.emulatesBiases());
			if (pNewActivationStorage) {
				m_activations.useExternalStorage(pNewActivationStorage
					, get_self().get_max_fprop_batch_size(), get_self().get_neurons_cnt() + 1, true);
			} else {
				if (!m_activations.resize(get_self().get_max_fprop_batch_size(), get_self().get_neurons_cnt()))
					return ErrorCode::CantAllocateMemoryForActivations;
			}

			//Math interface may have to operate on the following matrices:
			// m_weights, m_dLdW - (m_neurons_cnt, get_incoming_neurons_cnt() + 1)
			// m_activations - (m_max_fprop_batch_size, m_neurons_cnt) and unbiased matrices derived from m_activations - such as m_dAdZ
			// prevActivations - size (m_training_batch_size, get_incoming_neurons_cnt() + 1)
			get_self().get_iMath().preinit(std::max({
				m_weights.numel()
				,activation_f_t::needTempMem(m_activations,get_self().get_iMath())
				,realmtx_t::sNumel(get_self().get_training_batch_size(), get_incoming_neurons_cnt() + 1) 
			}));

			if (get_self().get_training_batch_size() > 0) {
				//it'll be training session, therefore must allocate necessary supplementary matrices and form temporary memory reqs.
// 				if (bDropout()) {
// 					NNTL_ASSERT(!m_dropoutMask.emulatesBiases());
// 					if (!m_dropoutMask.resize(m_training_batch_size, m_neurons_cnt)) return ErrorCode::CantAllocateMemoryForDropoutMask;
// 				}
 				if(!_check_init_dropout()) return ErrorCode::CantAllocateMemoryForDropoutMask;

				lid.max_dLdA_numel = realmtx_t::sNumel(get_self().get_training_batch_size(), get_self().get_neurons_cnt());
				// we'll need 2 temporarily matrices for bprop(): one for dA/dZ -> dL/dZ [batchSize x m_neurons_cnt] and
				// the other for dL/dW [m_neurons_cnt x get_incoming_neurons_cnt()+1]
				lid.maxMemBPropRequire = lid.max_dLdA_numel + m_weights.numel();
			}

			if (!m_gradientWorks.init(grad_works_t::init_struct_t(& get_self().get_iMath(), m_weights.size())))return ErrorCode::CantInitializeGradWorks;

			lid.bHasLossAddendum = hasLossAddendum();

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			m_gradientWorks.deinit();
			m_activations.clear();
			m_dropoutMask.clear();
			m_dAdZ_dLdZ.clear();
			m_dLdW.clear();
			_base_class::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			if (get_self().get_training_batch_size() > 0) {
				m_dAdZ_dLdZ.useExternalStorage(ptr, get_self().get_training_batch_size(), get_self().get_neurons_cnt());
				m_dLdW.useExternalStorage( ptr + m_dAdZ_dLdZ.numel(), m_weights);
				NNTL_ASSERT(!m_dAdZ_dLdZ.emulatesBiases() && !m_dLdW.emulatesBiases());
				NNTL_ASSERT(cnt >= m_dAdZ_dLdZ.numel() + m_dLdW.numel());
				//NNTL_ASSERT(m_dAdZ.size() == m_activations.size() && m_dLdW.size() == m_weights.size());
			}
		}
		
		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(m_activations.emulatesBiases());

			m_bTraining = batchSize == 0;

			const auto _training_batch_size = get_self().get_training_batch_size();

			if (pNewActivationStorage) {
				NNTL_ASSERT(m_activations.bDontManageStorage());
				//m_neurons_cnt + 1 for biases
				m_activations.useExternalStorage(pNewActivationStorage
					, m_bTraining ? _training_batch_size : batchSize, get_self().get_neurons_cnt() + 1, true);
				//should not restore biases here, because for compound layers its a job for their fprop() implementation
			} else {
				NNTL_ASSERT(!m_activations.bDontManageStorage());
				const auto _max_fprop_batch_size = get_self().get_max_fprop_batch_size();
				bool bRestoreBiases;
				//we don't need to restore biases in one case - if new row count equals to maximum (m_max_fprop_batch_size). Then the original
				//(filled during resize()) bias column has been untouched
				if (m_bTraining) {
					m_activations.deform_rows(_training_batch_size);
					bRestoreBiases = _training_batch_size != _max_fprop_batch_size;
				} else {
					NNTL_ASSERT(batchSize <= _max_fprop_batch_size);
					m_activations.deform_rows(batchSize);
					bRestoreBiases = batchSize != _max_fprop_batch_size;
				}
				if (bRestoreBiases) m_activations.set_biases();
				NNTL_ASSERT(m_activations.test_biases_ok());
			}
		}

	protected:
		//help compiler to isolate fprop functionality from the specific of previous layer
		void _fprop(const realmtx_t& prevActivations)noexcept {
			NNTL_ASSERT(prevActivations.test_biases_ok());
			NNTL_ASSERT(m_activations.rows() == prevActivations.rows());
			NNTL_ASSERT(prevActivations.cols() == m_weights.cols());

			//might be necessary for Nesterov momentum application
			if (m_bTraining) m_gradientWorks.pre_training_fprop(m_weights);

			auto& _Math = get_self().get_iMath();

			_Math.mMulABt_Cnb(prevActivations, m_weights, m_activations);
			NNTL_ASSERT(m_activations.bDontManageStorage() || m_activations.test_biases_ok());
			activation_f_t::f(m_activations, _Math);
			NNTL_ASSERT(m_activations.bDontManageStorage() || m_activations.test_biases_ok());

			if (bDropout()) {
				NNTL_ASSERT(0 < m_dropoutFraction && m_dropoutFraction < 1);
				if (m_bTraining) {
					//must make dropoutMask and apply it
					get_self().get_iRng().gen_matrix_norm(m_dropoutMask);
					_Math.make_dropout(m_activations, m_dropoutFraction, m_dropoutMask);
				} else {
					//only applying dropoutFraction
					_Math.evMulC_ip_Anb(m_activations, real_t(1.0) - m_dropoutFraction);
				}
				NNTL_ASSERT(m_activations.bDontManageStorage() || m_activations.test_biases_ok());
			}

			NNTL_ASSERT(prevActivations.test_biases_ok());

			//TODO?: sparsity penalty here
		}

		void _bprop(realmtx_t& dLdA, const realmtx_t& prevActivations, const bool bPrevLayerIsInput, realmtx_t& dLdAPrev)noexcept {
			dLdA.assert_storage_does_not_intersect(dLdAPrev);
			dLdA.assert_storage_does_not_intersect(m_dLdW);
			dLdA.assert_storage_does_not_intersect(m_dAdZ_dLdZ);
			dLdAPrev.assert_storage_does_not_intersect(m_dLdW);
			dLdAPrev.assert_storage_does_not_intersect(m_dAdZ_dLdZ);
			NNTL_ASSERT(prevActivations.test_biases_ok());

			NNTL_ASSERT(m_activations.emulatesBiases() && !m_dAdZ_dLdZ.emulatesBiases() && !m_dLdW.emulatesBiases());

			NNTL_ASSERT(m_activations.size_no_bias() == dLdA.size());
			NNTL_ASSERT(m_dAdZ_dLdZ.size() == m_activations.size_no_bias());
			NNTL_ASSERT(m_dLdW.size() == m_weights.size());

			NNTL_ASSERT(bPrevLayerIsInput || prevActivations.emulatesBiases());//input layer in batch mode may have biases included, but no emulatesBiases() set
			NNTL_ASSERT(mtx_size_t(get_self().get_training_batch_size(), get_incoming_neurons_cnt() + 1) == prevActivations.size());
			NNTL_ASSERT(bPrevLayerIsInput || dLdAPrev.size() == prevActivations.size_no_bias());//in vanilla simple BP we shouldn't calculate dLdAPrev for the first layer

			auto& _Math = get_self().get_iMath();

			//computing dA/dZ(no_bias)
			activation_f_t::df(m_activations, m_dAdZ_dLdZ, _Math);
			//compute dL/dZ=dL/dA.*dA/dZ into dA/dZ
			_Math.evMul_ip(m_dAdZ_dLdZ, dLdA);

			if (bDropout()) {
				//we must do it even though for sigmoid it's not necessary. But other activation function may
				//have non-zero derivative at y=0. #todo Probably, could #consider a speedup here by introducing a special
				// flag in an activation function that shows whether this step is necessary.
				_Math.evMul_ip(m_dAdZ_dLdZ, m_dropoutMask);
			}

			//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
			// BTW: even if some of neurons of this layer could have been "disabled" by a dropout (therefore their
			// corresponding dL/dZ element is set to zero), because we're working with batches, but not a single samples,
			// due to averaging the dL/dW over the whole batch 
			// (dLdW(i's neuron,j's lower layer neuron) = Sum_over_batch( dLdZ(i)*Aprev(j) ) ), it's almost impossible
			// to get some element of dLdW equals to zero, because it'll require that dLdZ entries for some neuron over the
			// whole batch were set to zero.
			_Math.mScaledMulAtB_C(real_t(1.0) / real_t(m_dAdZ_dLdZ.rows()), m_dAdZ_dLdZ, prevActivations, m_dLdW);

			if (!bPrevLayerIsInput) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//finally compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				m_weights.hide_last_col();
				_Math.mMulAB_C(m_dAdZ_dLdZ, m_weights, dLdAPrev);
				m_weights.restore_last_col();//restore weights back
			}

			//now we can apply gradient to the weights
			m_gradientWorks.apply_grad(m_weights, m_dLdW);

			NNTL_ASSERT(prevActivations.test_biases_ok());
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept{
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			//STDCOUTL("In " << get_layer_name_str());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._fprop(lowerLayer.get_activations());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
		}


		
		template <typename LowerLayer>
		const unsigned bprop(realmtx_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept{
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			//STDCOUTL("bprop " << get_layer_name_str());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._bprop(dLdA, lowerLayer.get_activations(), std::is_base_of<m_layer_input, LowerLayer>::value, dLdAPrev);
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

		const real_t dropoutFraction()const noexcept { return m_dropoutFraction; }
		self_ref_t dropoutFraction(real_t dfrac)noexcept {
			NNTL_ASSERT(0 <= dfrac && dfrac < 1);
			m_dropoutFraction = dfrac;
			if (!_check_init_dropout()) {
				NNTL_ASSERT(!"Failed to init dropout, probably no memory");
				abort();
			}
			return get_self();
		}
		const bool bDropout()const noexcept { return 0 < m_dropoutFraction; }

	protected:
		const bool _check_init_dropout()noexcept {
			if (get_self().get_training_batch_size() > 0 && bDropout()) {
				NNTL_ASSERT(!m_dropoutMask.emulatesBiases());
				return m_dropoutMask.resize(get_self().get_training_batch_size(), get_self().get_neurons_cnt());
			}
			return true;
		}

		friend class _impl::_preinit_layers;
		void _preinit_layer(layer_index_t& idx, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 < idx);
			NNTL_ASSERT(0 < inc_neurons_cnt);
			_base_class::_preinit_layer(idx, inc_neurons_cnt);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_fully_connected
	// If you need to derive a new class, derive it from _layer_fully_connected (to make static polymorphism work)
	template <typename ActivFunc = activation::sigm<>,
		typename Interfaces=nnet_def_interfaces,
		typename GradWorks = grad_works<typename Interfaces::iMath_t>
	> class LFC final 
		: public _layer_fully_connected<ActivFunc, Interfaces, GradWorks, LFC<ActivFunc, Interfaces, GradWorks>>
	{
	public:
		~LFC() noexcept {};
		LFC(const neurons_count_t _neurons_cnt,
			const real_t learningRate=.01, const real_t dropoutFrac = 0.0) noexcept 
			: _layer_fully_connected<ActivFunc, Interfaces, GradWorks, LFC<ActivFunc, Interfaces, GradWorks>>
			(_neurons_cnt, learningRate, dropoutFrac) {};
	};

	template <typename ActivFunc = activation::sigm<>,
		typename Interfaces = nnet_def_interfaces,
		typename GradWorks = grad_works<typename Interfaces::iMath_t>
	> using layer_fully_connected = typename LFC<ActivFunc, Interfaces, GradWorks>;
}

