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

#include "_i_activation.h"
#include "_loss_parts.h"

namespace nntl {
namespace activation {

	//activation types should not be templated (probably besides real_t), because they are intended to be used
	//as means to recognize activation function type
	struct type_identity {};
	// note that identity activation implies that dA/dZ (implemented in void df(..)) == 1 and due to that fact
	// some optimization become possible.
	// Therefore derive classes from the type_identity keeping it mind.
	
	template<typename ActT>
	using is_activation_identity = is_type_of<ActT, type_identity>;

	template<typename RealT = d_interfaces::real_t
		, typename WeightsInitScheme = weights_init::SNNInit>
		class identity
		: public _i_activation<RealT, WeightsInitScheme, true>
		, public type_identity
	{
	public:
		//apply f to each srcdest matrix element to compute activation values. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_UNREF(srcdest); NNTL_UNREF(m);
			//should do nothing.
		};
		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			dIdentity(f_df, m);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::SNNInit, bool bNumericStable = false>
	class identity_quad_loss : public identity<RealT, WeightsInitScheme>, public _i_quadratic_loss<RealT> {
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			dLdZIdentity(data_y, act_dLdZ, m);
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

	//linear_output is a linear output activation function that allows to specify custom loss function as a template parameter
	// 
	//LossT is a generic loss type that for example allows to put more weight on one kind of errors (like false positives)
	// to drag gradient harder away from it.
	// Here's how it works for false positives (Linear_Loss_quadWeighted_FP):
	//		(1) if data_y<bnd && activation>bnd, then apply more heavyweight loss function, like 2*(a-y)^2
	//		(2) else apply less heavyweight function like (a-y)^2
	// See the _loss_parts.h file for examples
	template<typename LossT /*= Linear_Loss_quadWeighted_FP<RealT>*/, typename WeightsInitScheme = weights_init::SNNInit>
	class identity_custom_loss : public identity<typename LossT::real_t, WeightsInitScheme>, public _i_activation_loss<typename LossT::real_t> {
	public:
		typedef LossT Loss_t;

		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.dLoss_dZ<LossT>(data_y, act_dLdZ);
		}
		template <typename iMath>
		static void dLdZIdentity(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m) noexcept {
			//same as dLdZ
			dLdZ(data_y, act_dLdZ, m);
		}

		template <typename iMath>
		static real_t loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.compute_loss<LossT>(activations, data_y);
		}
	};


}
}