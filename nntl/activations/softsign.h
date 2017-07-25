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

	//activation types should not be templated (probably besides real_t), because they are intended to be used
	//as means to recognize activation function type
	struct type_softsign {};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// SoftSign, y = c*(x/(a+|x|)), dy/dx = (c-|y|)^2 /(c*a), parameter 'a' controls the slope of the curve, c - amplitude
	// pass 0 to C1e3 to get predefined c=1.59253... (see https://www.reddit.com/r/MachineLearning/comments/6g5tg1/r_selfnormalizing_neural_networks_improved_elu/diwq7rb/)
	template<typename RealT, unsigned int A1e6 = 1000000, unsigned int C1e6=1000000
		, typename WeightsInitScheme = weights_init::He_Zhang<>, typename DropoutT = Dropout<RealT>>
	class softsign
		: public _i_activation<DropoutT, WeightsInitScheme>
		, public type_softsign
	{
	public:
		
		static constexpr real_t A = real_t(A1e6) / real_t(1e6);
		static constexpr real_t C = C1e6 == 0 ? real_t(1.5925374197228312) : real_t(C1e6) / real_t(1e6);
		static constexpr bool bIsUnitA = (A1e6 == 1000000);
		static constexpr bool bIsUnitC = (C1e6 == 1000000);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath, bool bUnitC = bIsUnitC>
		static ::std::enable_if_t<!bUnitC> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.softsign(srcdest, A, C);
		};
		template <typename iMath, bool bUnitC = bIsUnitC>
		static ::std::enable_if_t<bUnitC> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.softsign_uc(srcdest, A);
		};


		template <typename iMath, bool bUnitAll = bIsUnitA && bIsUnitC>
		static ::std::enable_if_t<!bUnitAll> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dsoftsign(f_df, A, C);
		}
		template <typename iMath, bool bUnitAll = bIsUnitA && bIsUnitC>
		static ::std::enable_if_t<bUnitAll> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dsoftsign_ua_uc(f_df);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>, typename DoT=Dropout<RealT>>
	using softsign_uc_ua = softsign<RealT, 1000000, 1000000, WeightsInitScheme, DoT>;

}
}
