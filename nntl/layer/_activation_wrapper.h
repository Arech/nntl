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

// This file defines a wrapper around _i_activation-derived class.
// Using inheritance allows to define a state-full _i_activation classes as well as zero-cost stateless.

#include "_activation_storage.h"
#include "../activation.h"

namespace nntl {
	namespace _impl {

		template<typename FinalPolymorphChild, typename InterfacesT, typename ActivFuncT>
		class _act_wrap
			: public _act_stor<FinalPolymorphChild, InterfacesT>
			, private ActivFuncT
		{
		private:
			typedef _act_stor<FinalPolymorphChild, InterfacesT> _base_class_t;

		public:
			using _base_class_t::real_t;
			using _base_class_t::realmtx_t;
			using _base_class_t::realmtxdef_t;
			
			static_assert(::std::is_same<typename InterfacesT::real_t, typename ActivFuncT::real_t>::value, "Invalid real_t");

			typedef ActivFuncT Activation_t;
			typedef typename Activation_t::weights_scheme_t Weights_Init_t;

			static constexpr bool bActivationForOutput = ::std::is_base_of<activation::_i_activation_loss<real_t>, Activation_t>::value;
			static constexpr bool bActivationForHidden = ::std::is_base_of<activation::_i_activation<real_t, Weights_Init_t, Activation_t::bFIsZeroStable>, Activation_t>::value;

			/* static_assert(::std::is_base_of<activation::_i_function<real_t, Weights_Init_t, true>, Activation_t>::value
				|| ::std::is_base_of<activation::_i_function<real_t, Weights_Init_t, false>, Activation_t>::value
				, "ActivFuncT template parameter should be derived from activation::_i_function"); */
			
			static_assert(::std::is_base_of<activation::_i_function<real_t, Weights_Init_t, Activation_t::bFIsZeroStable>, Activation_t>::value
				, "ActivFuncT template parameter should be derived from activation::_i_function");

		protected:
			~_act_wrap()noexcept {}
			_act_wrap(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr)noexcept 
				: _base_class_t(_neurons_cnt, pCustomName)
			{}

		public:

			template<bool _b = bActivationForOutput>
			::std::enable_if_t<_b, real_t> calc_loss(const realmtx_t& data_y)const noexcept {
				return Activation_t::loss(m_activations, data_y, get_iMath());
			}

			//////////////////////////////////////////////////////////////////////////
			ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
				const auto ec = _base_class_t::init(lid, pNewActivationStorage);
				if (ErrorCode::Success != ec) return ec;

				if (!Activation_t::act_init()) return ErrorCode::CantInitializeActFunc;

				return ErrorCode::Success;
			}

			void deinit() noexcept {
				Activation_t::act_deinit();
				_base_class_t::deinit();
			}

			real_t act_scaling_coeff() const noexcept {
				return Activation_t::act_scaling_coeff();
			}

		protected:
			bool _activation_init_weights(realmtx_t& weights) noexcept {
				NNTL_ASSERT(!weights.emulatesBiases());
				return Weights_Init_t::init(weights, get_iRng(), get_iMath());
			}

			//this is to return how many temporary real_t elements activation function might require the iMath interface to have
			auto _activation_tmp_mem_reqs()const noexcept {
				return Activation_t::needTempMem(m_activations, get_iMath());
			}

			template<typename iMathT>
			void _activation_fprop(iMathT& iM)noexcept {
#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
				NNTL_ASSERT(m_activations.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK

				if (!bLayerIsLinear()) {
					Activation_t::f(m_activations, iM);

#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
					NNTL_ASSERT(m_activations.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK
				}
			}

			template<typename iMathT, bool _b = bActivationForHidden>
			::std::enable_if_t<_b> _activation_bprop(realmtx_t& act2dAdZ_nb,iMathT& iM)noexcept {
				NNTL_ASSERT(m_activations.emulatesBiases() && !act2dAdZ_nb.emulatesBiases());
				NNTL_ASSERT(m_activations.data() == act2dAdZ_nb.data() && m_activations.size_no_bias() == act2dAdZ_nb.size());
				NNTL_ASSERT(act2dAdZ_nb.test_noNaNs());
				if (bLayerIsLinear()) {
					Activation_t::dIdentity(act2dAdZ_nb, iM);
				} else {
					Activation_t::df(act2dAdZ_nb, iM);
				}
				NNTL_ASSERT(act2dAdZ_nb.test_noNaNs());
			}

			template<typename iMathT, bool _b = bActivationForOutput>
			::std::enable_if_t<_b> _activation_bprop_output(const realmtx_t& data_y, iMathT& iM)noexcept {
				NNTL_ASSERT(!m_activations.emulatesBiases() && !data_y.emulatesBiases());
				if (bLayerIsLinear()) {
					Activation_t::dLdZIdentity(data_y, m_activations, iM);
				} else {
					Activation_t::dLdZ(data_y, m_activations, iM);
				}
				NNTL_ASSERT(m_activations.test_noNaNs());
			}
		};

	}
}
