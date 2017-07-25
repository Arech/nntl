/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2017, Arech (aradvert@gmail.com; https://github.com/Arech)
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

// This file defines a wrapper around _i_activation-derived class and a corresponding dropout class
// Layer classes should derive from this class.
// Using inheritance allows to define a state-full _i_activation classes as well as zero-cost stateless.

#include "_layer_base.h"
#include "../activation.h"

namespace nntl {
	namespace _impl {

		template<typename FinalPolymorphChild, typename InterfacesT, typename ActivFuncT>
		class _activation_wrapper
			: public _layer_base<FinalPolymorphChild, InterfacesT>
			, private ActivFuncT
			, public ::std::conditional_t <
				::std::is_base_of <activation::_i_activation_loss< typename ActivFuncT::real_t >, ActivFuncT>::value
				, _impl::_No_Dropout_at_All <typename ActivFuncT::real_t>
				, typename ActivFuncT::Dropout_t
			>
		{
		private:
			typedef _layer_base<FinalPolymorphChild, InterfacesT> _base_class_t;

		public:
			using _base_class_t::real_t;
			using _base_class_t::realmtx_t;
			using _base_class_t::realmtxdef_t;
			//using _base_class_t::numel_cnt_t;

			typedef ActivFuncT Activation_t;
			typedef typename Activation_t::weights_scheme_t Weights_Init_t;
			//we'll define Dropout_t conditioning on bActivationForOutput

			static constexpr bool bActivationForOutput = ::std::is_base_of <activation::_i_activation_loss<real_t>, Activation_t>::value;
			static constexpr bool bActivationForHidden = ::std::is_base_of<activation::_i_activation<typename Activation_t::Dropout_t, Weights_Init_t>, Activation_t>::value;
			
			typedef ::std::conditional_t<bActivationForOutput, _impl::_No_Dropout_at_All<real_t>, typename Activation_t::Dropout_t> Dropout_t;

		protected:
			// matrix of layer neurons activations: <batch_size rows> x <m_neurons_cnt+1(bias) cols> for fully connected layer
			// Class assumes, that it's content on the beginning of the bprop() is the same as it was on exit from fprop().
			// Don't change it outside of the class!
			realmtxdef_t m_activations;

		protected:
			~_activation_wrapper()noexcept {}
			_activation_wrapper(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr)noexcept 
				: _base_class_t(_neurons_cnt, pCustomName)
			{}

		public:
			// Class assumes, that the content of the m_activations matrix on the beginning of the bprop() is the same as it was on exit from fprop().
			// Don't change it outside of the class!
			const realmtxdef_t& get_activations()const noexcept {
				NNTL_ASSERT(m_bActivationsValid);
				return m_activations;
			}
			mtx_size_t get_activations_size()const noexcept { return m_activations.size(); }

			bool is_activations_shared()const noexcept {
				const auto r = _base_class_t::is_activations_shared();
				NNTL_ASSERT(!r || m_activations.bDontManageStorage());//shared activations can't manage their own storage
				return r;
			}

			template<bool _b = bActivationForOutput>
			::std::enable_if_t<_b, real_t> calc_loss(const realmtx_t& data_y)const noexcept {
				return Activation_t::loss(m_activations, data_y, get_self().get_iMath());
			}

			//////////////////////////////////////////////////////////////////////////

			ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
				const auto ec = _base_class_t::init(lid, pNewActivationStorage);
				if (ErrorCode::Success != ec) return ec;
				
				//non output layers must have biases while output layers must not
				NNTL_ASSERT(m_activations.emulatesBiases() ^ is_layer_output<self_t>::value);

				const auto neurons_cnt = get_self().get_neurons_cnt();
				const auto biggestBatchSize = get_self().get_common_data().biggest_batch_size();
				if (pNewActivationStorage) {
					NNTL_ASSERT(!is_layer_output<self_t>::value || !"WTF? pNewActivationStorage can't be set for output layer");
					m_activations.useExternalStorage(pNewActivationStorage, biggestBatchSize, neurons_cnt + 1, true); //+1 for biases
				} else {
					if (!m_activations.resize(biggestBatchSize, neurons_cnt))
						return ErrorCode::CantAllocateMemoryForActivations;
				}

				return ErrorCode::Success;
			}

			void deinit() noexcept {
				Activation_t::act_deinit();
				NNTL_ASSERT(m_activations.emulatesBiases() ^ is_layer_output<self_t>::value);
				m_activations.clear();
				_base_class_t::deinit();
			}

			real_t act_scaling_coeff() const noexcept {
				return Activation_t::act_scaling_coeff();
			}

			void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
				constexpr bool bOutputLayer = is_layer_output<self_t>::value;

				NNTL_ASSERT(m_activations.emulatesBiases() ^ bOutputLayer);
				m_bActivationsValid = false;

				const auto& CD = get_self().get_common_data();
				const vec_len_t batchSize = CD.get_cur_batch_size();
				const auto _biggest_batch_size = CD.biggest_batch_size();
				NNTL_ASSERT(batchSize > 0 && batchSize <= _biggest_batch_size);

				if (!bOutputLayer && pNewActivationStorage) {
					NNTL_ASSERT(m_activations.bDontManageStorage());
					//m_neurons_cnt + 1 for biases
					m_activations.useExternalStorage(pNewActivationStorage, batchSize, get_self().get_neurons_cnt() + 1, true);
					//should not restore biases here, because for compound layers its a job for their fprop() implementation
				} else {
					NNTL_ASSERT(bOutputLayer || !pNewActivationStorage);
					NNTL_ASSERT(!m_activations.bDontManageStorage());
					//we don't need to restore biases in one case - if new row count equals to maximum (_biggest_batch_size). Then the original
					//(filled during resize()) bias column has been untouched
					m_activations.deform_rows(batchSize);
					if (!bOutputLayer && batchSize != _biggest_batch_size) m_activations.set_biases();
					NNTL_ASSERT(bOutputLayer || m_activations.test_biases_ok());
				}
			}

		protected:
			bool _activation_init_weights(realmtx_t& weights) noexcept {
				NNTL_ASSERT(!weights.emulatesBiases());
				if (!Weights_Init_t::init(weights, get_self().get_iRng(), get_self().get_iMath()))return false;
				NNTL_ASSERT(!weights.emulatesBiases());

				return Activation_t::act_init();
			}

			//this is to return how many temporary real_t elements activation function might require the iMath interface to have
			auto _activation_tmp_mem_reqs()const noexcept {
				return Activation_t::needTempMem(m_activations, get_self().get_iMath());
			}

			template<typename iMathT>
			void _activation_fprop(iMathT& iM)noexcept {
				if (!bLayerIsLinear()) {
					Activation_t::f(m_activations, iM);
				}
			}

			template<typename iMathT, bool _b = bActivationForHidden>
			::std::enable_if_t<_b>
			_activation_bprop(realmtx_t& act2dAdZ_nb,iMathT& iM)noexcept {
				NNTL_ASSERT(m_activations.emulatesBiases() && !act2dAdZ_nb.emulatesBiases());
				NNTL_ASSERT(m_activations.data() == act2dAdZ_nb.data() && m_activations.size_no_bias() == act2dAdZ_nb.size());
				if (bLayerIsLinear()) {
					Activation_t::dIdentity(act2dAdZ_nb, iM);
				} else {
					Activation_t::df(act2dAdZ_nb, iM);
				}
			}

			template<typename iMathT, bool _b = bActivationForOutput>
			::std::enable_if_t<_b> _activation_bprop_output(const realmtx_t& data_y, iMathT& iM)noexcept {
				NNTL_ASSERT(!m_activations.emulatesBiases() && !data_y.emulatesBiases());
				if (bLayerIsLinear()) {
					Activation_t::dLdZIdentity(data_y, m_activations, iM);
				} else {
					Activation_t::dLdZ(data_y, m_activations, iM);
				}
			}
		};

	}
}
