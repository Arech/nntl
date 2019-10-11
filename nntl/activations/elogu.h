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

#include "_i_activation.h"

namespace nntl {
namespace activation {

	//activation types should not be templated (probably besides real_t), because they are intended to be used
	//as means to recognize activation function type
	struct type_elogu {};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//ELogU : log(x+1)/log(b) | x>0,  alpha*(exp(x)-1) | x<0
	template<typename RealT, unsigned int Alpha1e3 = 1000, unsigned int LogBase1e3 = 2000
		, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class elogu 
		: public _i_activation<RealT, WeightsInitScheme, true>
		, public type_elogu
	{
	public:


		static constexpr real_t Alpha = real_t(Alpha1e3) / real_t(1000.0);
		static constexpr bool bIsUnitAlpha = (Alpha1e3 == 1000);

		static constexpr real_t LogBase = real_t(LogBase1e3) / real_t(1000.0);
		static constexpr bool bIsNaturalBase = (LogBase1e3 == 2718);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<!bUnitAlpha && !bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu(srcdest, Alpha, LogBase);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<bUnitAlpha && !bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu_ua(srcdest, LogBase);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<!bUnitAlpha && bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu_nb(srcdest, Alpha);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<bUnitAlpha && bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu_ua_nb(srcdest);
		};

		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<!bUnitAlpha && !bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu(f_df, Alpha, LogBase);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<bUnitAlpha && !bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu_ua(f_df, LogBase);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<!bUnitAlpha && bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu_nb(f_df, Alpha);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalBase>
		static ::std::enable_if_t<bUnitAlpha && bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu_ua_nb(f_df);
		}
	};

	template<typename RealT, unsigned int LogBase1e3 = 2000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elogu_ua = elogu<RealT, 1000, LogBase1e3, WeightsInitScheme>;
	template<typename RealT, unsigned int Alpha1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elogu_nb = elogu<RealT, Alpha1e3, 2718, WeightsInitScheme>;
	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elogu_ua_nb = elogu<RealT, 1000, 2718, WeightsInitScheme>;

}
}