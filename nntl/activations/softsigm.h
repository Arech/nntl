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

#include "_i_activation.h"

namespace nntl {
namespace activation {

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// SoftSigm, y = (x/(2*(a+|x|)) +.5), dy/dx = (.5-|y-.5|)^2 * 2/a, parameter 'a' controls the slope of the curve
	template<typename RealT, unsigned int A1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class softsigm : public _i_activation<RealT> {
		softsigm() = delete;
		~softsigm() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;
		static constexpr real_t A = real_t(A1e3) / real_t(1000.0);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.softsigm(srcdest, A);
		};

		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dsoftsigm(f_df, A);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using softsigm_ua = softsigm<RealT, 1000, WeightsInitScheme>;


	//////////////////////////////////////////////////////////////////////////
	//bNumericStable selects slightly more stable loss function implementation (usually drops relative error by the factor of 2)
	// use it for numeric gradient checks. Useless for usual nnet training
	template<typename RealT, unsigned int A1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>, bool bNumericStable = false>
	class softsigm_quad_loss : public softsigm<RealT, A1e3, WeightsInitScheme>, public _i_quadratic_loss<RealT> {
		softsigm_quad_loss() = delete;
		~softsigm_quad_loss() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.dSoftSigmQuadLoss_dZ(data_y, act_dLdZ, A);
		}

		template <typename iMath, bool bNS = bNumericStable>
		static std::enable_if_t<!bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_quadratic(activations, data_y);
		}
		template <typename iMath, bool bNS = bNumericStable>
		static std::enable_if_t<bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_quadratic_ns(activations, data_y);
		}
	};

	template<typename RealT, unsigned int A1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class debug_softsigm_zeros : public softsigm<RealT, A1e3, WeightsInitScheme>, public _i_quadratic_loss<RealT> {
		debug_softsigm_zeros() = delete;
		~debug_softsigm_zeros() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			//m.dSoftSigmQuadLoss_dZ(data_y, act_dLdZ, A);
			act_dLdZ.zeros();
		}

		template <typename iMath>
		static real_t loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			//return m.loss_quadratic(activations, data_y);
			return real_t(0.);
		}
	};

	//NB: http://neuralnetworksanddeeplearning.com/chap3.html
	template<typename RealT, unsigned int A1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>, bool bNumericStable = false>
	class softsigm_xentropy_loss : public softsigm<RealT, A1e3, WeightsInitScheme>, public _i_xentropy_loss<RealT> {
		softsigm_xentropy_loss() = delete;
		~softsigm_xentropy_loss() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.dSoftSigmXEntropyLoss_dZ(data_y, act_dLdZ, A);
		}

		template <typename iMath, bool bNS = bNumericStable>
		static std::enable_if_t<!bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_xentropy(activations, data_y);
		}

		template <typename iMath, bool bNS = bNumericStable>
		static std::enable_if_t<bNS, real_t> loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_xentropy_ns(activations, data_y);
		}
	};
}
}
