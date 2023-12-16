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

#include <nntl/activations/_i_activation.h>

namespace nntl {
namespace activation {

	//activation types should not be templated (probably besides real_t), because they are intended to be used
	//as means to recognize activation function type
	struct type_sigm {};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//sigmoid
	template<typename RealT = d_interfaces::real_t
		, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>>
	class sigm
		: public _i_activation<RealT, WeightsInitScheme, false>
		, public type_sigm
	{
	public:
		//apply f to each srcdest matrix element to compute activation values. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.sigm(srcdest);
		};
		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dsigm(f_df);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>, bool bNumericStable = false>
	class sigm_quad_loss : public sigm<RealT, WeightsInitScheme>, public _i_quadratic_loss<RealT> {
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.dSigmQuadLoss_dZ(data_y, act_dLdZ);
		}

		template <typename iMath, bool bNS = bNumericStable>
		static ::std::enable_if_t<!bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_quadratic(activations, data_y);
		}
		template <typename iMath, bool bNS = bNumericStable>
		static ::std::enable_if_t<bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_quadratic_ns(activations, data_y);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>, bool bNumericStable = false>
	class sigm_xentropy_loss : public sigm<RealT, WeightsInitScheme>, public _i_xentropy_loss<RealT> {
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			// L = -y*log(a)-(1-y)log(1-a) (dL/dZ = dL/dA * dA/dZ = (a-y)/(a*(1-a)) * dA/dZ )
			// dA/dZ = a(1-a)
			//dL/dz = dL/dA * dA/dZ = (a-y)
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			m.evSub_ip(act_dLdZ, data_y);
		}


		template <typename iMath, bool bNS = bNumericStable>
		static ::std::enable_if_t<!bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_xentropy(activations, data_y);
		}

		template <typename iMath, bool bNS = bNumericStable>
		static ::std::enable_if_t<bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_xentropy_ns(activations, data_y);
		}
	};

}
}