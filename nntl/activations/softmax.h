/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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

	//activation types should not be templated (probably besides real_t), because they are intended to be used
	//as means to recognize activation function type
	struct type_softmax {};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// SoftMax (for output layer only - it's easier to get dL/dL than dA/dL for SoftMax)
	// #TODO: which weight initialization scheme is better for SoftMax?
	// #TODO: may be it's worth to implement SoftMax activation for hidden layers, i.e. make a dA/dZ implementation
	template<typename RealT, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>>
	class softmax_xentropy_loss
		: public _i_function<RealT, WeightsInitScheme, false>
		, public _i_xentropy_loss<RealT>
		, public type_softmax
	{
	public:

		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtxdef_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.softmax(srcdest);
		};
		//get requirements on temporary memory size needed to calculate f() over matrix act (need it for memory
		// preallocation algorithm of iMath).
		template <typename iMath>
		static numel_cnt_t needTempMem(const realmtx_t& act, iMath& m) noexcept {
			return m.softmax_needTempMem(act);
		}

		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			//SoftMax dL/dZ = dL/dA * dA/dZ = (a-y)
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			m.evSub_ip(act_dLdZ, data_y);
		}

		template <typename iMath>
		static real_t loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_softmax_xentropy(activations, data_y);
		}
	};

}
}