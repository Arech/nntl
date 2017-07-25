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

#include "../serialization/serialization.h"

namespace nntl {
namespace GW { //GW namespace is for grad_works mixins and other stuff, that helps to implement gradient processing voodooo things

	template<typename RealT>
	struct ILR_Dummy_props {
		constexpr ILR_Dummy_props()noexcept {}
	};

	template<typename RealT>
	struct ILR_props {
		typedef RealT real_t;
		real_t mulDecr, mulIncr, capLow, capHigh;

	public:
		//~ILR_props()noexcept {}
		//ILR_props(const ILR_props& i)noexcept { set(i); }
		constexpr ILR_props(const ILR_props& i)noexcept : mulDecr(i.mulDecr), mulIncr(i.mulIncr), capLow(i.capLow), capHigh(i.capHigh) {}

		ILR_props& operator=(const ILR_props& rhs) noexcept {
			if (this != &rhs) set(rhs);
			return *this;
		}

		constexpr ILR_props()noexcept:mulDecr(real_t(0.0)), mulIncr(real_t(0.0)), capLow(real_t(0.0)), capHigh(real_t(0.0)) {}
		constexpr ILR_props(const real_t decr, const real_t incr, const real_t cLow, const real_t cHigh, const ::std::nullptr_t _tag)noexcept 
			: mulDecr(decr), mulIncr(incr), capLow(cLow), capHigh(cHigh) {}

		ILR_props(const real_t& decr, const real_t& incr, const real_t& cLow, const real_t& cHigh)noexcept {
			set(decr, incr, cLow cHigh);
		}
		void set(const real_t& decr, const real_t& incr, const real_t& cLow, const real_t& cHigh)noexcept {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
			mulDecr = decr;
			mulIncr = incr;
			capLow = cLow;
			capHigh = cHigh;
		}
		void set(const ILR_props& i)noexcept { set(i.mulDecr, i.mulIncr, i.capLow, i.capHigh); }
		void clear()noexcept { set(real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)); }
		const bool bUseMe()const noexcept {
			NNTL_ASSERT((mulDecr > real_t(0.0) && mulIncr > real_t(0.0) && capLow > real_t(0.0) && capHigh > real_t(0.0))
				|| (mulDecr == real_t(0.0) && mulIncr == real_t(0.0) && capLow == real_t(0.0) && capHigh == real_t(0.0)));
			return mulDecr > real_t(0.0);
		}

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & NNTL_SERIALIZATION_NVP(mulDecr);
			ar & NNTL_SERIALIZATION_NVP(mulIncr);
			ar & NNTL_SERIALIZATION_NVP(capLow);
			ar & NNTL_SERIALIZATION_NVP(capHigh);
		}
	};

}
}