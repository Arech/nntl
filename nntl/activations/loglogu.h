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

namespace nntl {
namespace activation {

	//activation types should not be templated (probably besides real_t), because they are intended to be used
	//as means to recognize activation function type
	struct type_loglogu {};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//LogLogU : -log(1-x)/log(b_neg) | x<0,   log(x+1)/log(b_pos) | x>0
	template<typename RealT, unsigned int LogBaseNeg1e3 = 2718, unsigned int LogBasePos1e3 = 2000
		, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class loglogu 
		: public _i_activation<RealT, WeightsInitScheme, true>
		, public type_loglogu
	{
	public:
		//typedef WeightsInitScheme weights_scheme;
		static constexpr real_t LogBaseNeg = real_t(LogBaseNeg1e3) / real_t(1000.0);
		static constexpr bool bIsNBN = (LogBaseNeg1e3 == 2718);

		static constexpr real_t LogBasePos = real_t(LogBasePos1e3) / real_t(1000.0);
		static constexpr bool bIsNBP = (LogBasePos1e3 == 2718);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<!bNBN && !bNBP> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.loglogu(srcdest, LogBaseNeg, LogBasePos);
		};
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<bNBN && !bNBP> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.loglogu_nbn(srcdest, LogBasePos);
		};
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<!bNBN && bNBP> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.loglogu_nbp(srcdest, LogBaseNeg);
		};
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<bNBN && bNBP> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.loglogu_nbn_nbp(srcdest);
		};

		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<!bNBN && !bNBP> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dloglogu(f_df, LogBaseNeg, LogBasePos);
		}
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<bNBN && !bNBP> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dloglogu_nbn(f_df, LogBasePos);
		}
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<!bNBN && bNBP> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dloglogu_nbp(f_df, LogBaseNeg);
		}
		template <typename iMath, bool bNBN = bIsNBN, bool bNBP = bIsNBP>
		static ::std::enable_if_t<bNBN && bNBP> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dloglogu_nbn_nbp(f_df);
		}
	};

	template<typename RealT, unsigned int LogBasePos1e3 = 2000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using loglogu_nbn = loglogu<RealT, 2718, LogBasePos1e3, WeightsInitScheme>;
	template<typename RealT, unsigned int LogBaseNeg1e3 = 2718, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using loglogu_nbp = loglogu<RealT, LogBaseNeg1e3, 2718, WeightsInitScheme>;
	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using loglogu_nbn_nbp = loglogu<RealT, 2718, 2718, WeightsInitScheme>;


}
}
