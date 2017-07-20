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
	// SELU, arxiv:1706.02515 "Self-Normalizing Neural Networks", by Günter Klambauer et al.
	// pass zeros to Alpha&Lambda to get a "standard" SELU with stable fixed point at (0,1)
	template<typename RealT, uint64_t Alpha1e9 = 0, uint64_t Lambda1e9 = 0, typename WeightsInitScheme = weights_init::SNNInit>
	class selu : public _i_activation<RealT> {
		selu() = delete;
		~selu() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;

		static constexpr ext_real_t AlphaExt = Alpha1e9 ? ext_real_t(Alpha1e9) / ext_real_t(1e9) : ext_real_t(1.6732632423543772848170429916717);
		static constexpr ext_real_t LambdaExt = Lambda1e9 ? ext_real_t(Lambda1e9) / ext_real_t(1e9) : ext_real_t(1.0507009873554804934193349852946);

		static constexpr real_t Alpha = real_t(AlphaExt);
		static constexpr real_t Lambda = real_t(LambdaExt);
		static constexpr real_t Alpha_t_Lambda = real_t(AlphaExt*LambdaExt);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.selu(srcdest, Alpha_t_Lambda, Lambda);
		};
		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dselu(f_df, Alpha_t_Lambda, Lambda);
		}
	};

}
}
