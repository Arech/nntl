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

#include <chrono>

#ifndef NNTL_COMP_SILENCE_RETVAL_IGNORED
#define NNTL_COMP_SILENCE_RETVAL_IGNORED __pragma(warning( push )) \
    __pragma(warning(disable : 6031))
#endif

#ifndef NNTL_COMP_POP
#define NNTL_COMP_POP __pragma(warning( pop ))
#endif

namespace nntl {
namespace utils {

	namespace chrono {
		/*template<typename per_t> struct period_3oom_bigger {
			typedef ::std::chrono::minutes type;
		};
		template<> struct period_3oom_bigger<::std::chrono::milliseconds> {
			typedef ::std::chrono::seconds type;
		};*/
		template<typename per_t> struct period_3oom_bigger {
			typedef ::std::chrono::seconds type;
		};
		template<> struct period_3oom_bigger<::std::chrono::nanoseconds> {
			typedef ::std::chrono::microseconds type;
		};
		template<> struct period_3oom_bigger<::std::chrono::microseconds> {
			typedef ::std::chrono::milliseconds type;
		};

		template<typename per_t> struct period_name {
			//static constexpr const strchar_t* name = NNTL_STRING("???");
			static constexpr const char* name = "??";
		};
		/*template<> struct period_name<::std::chrono::minutes> {
			//static constexpr const strchar_t* name = NNTL_STRING("s");
			static constexpr const char* name = "m";
		};*/
		template<> struct period_name<::std::chrono::seconds> {
			//static constexpr const strchar_t* name = NNTL_STRING("s");
			static constexpr const char* name = "s";
		};
		template<> struct period_name<::std::chrono::milliseconds> {
			//static constexpr const strchar_t* name = NNTL_STRING("ms");
			static constexpr const char* name = "ms";
		};
		template<> struct period_name<::std::chrono::microseconds> {
			//static constexpr const strchar_t* name = NNTL_STRING("mcs");
			static constexpr const char* name = "us";
		};
		template<> struct period_name<::std::chrono::nanoseconds> {
			//static constexpr const strchar_t* name = NNTL_STRING("ns");
			static constexpr const char* name = "ns";
		};
	}
	

	template <typename durType>
	static ::std::string duration_readable(durType d, uint64_t repeats=1, double* ptrSingleTime=nullptr)noexcept {
		double t = static_cast<double>(d.count())/repeats;
		if (ptrSingleTime) *ptrSingleTime = t;
		const char* name;
		if (t<1000) {
			name = chrono::period_name<durType>::name;
		}else if (t < 1000000) {
			t /= 1000;
			name = chrono::period_name<
				typename chrono::period_3oom_bigger<durType>::type
			>::name;
		} else if (t < 1000000000) {
			t /= 1000000;
			name = chrono::period_name<
				typename chrono::period_3oom_bigger<
				typename chrono::period_3oom_bigger<durType>::type
				>::type >::name;
		} else {
			t /= 1000000000;
			name = chrono::period_name<
				typename chrono::period_3oom_bigger<
				typename chrono::period_3oom_bigger<
				typename chrono::period_3oom_bigger<durType>::type
				>::type >::type >::name;
		}
		constexpr unsigned MAX_STR_SIZE = 256;
		char str[MAX_STR_SIZE];

		NNTL_COMP_SILENCE_RETVAL_IGNORED
		::std::snprintf(str, MAX_STR_SIZE, "%8.3f %-2s", t, name);
		NNTL_COMP_POP

		return ::std::string(str);
	};

}
}