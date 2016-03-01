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

#include <type_traits>
#include <bitset>

namespace nntl {
namespace utils {

	template<typename OptionsFinalT>
	struct options {
		//marker for SFINAE implementation of options getter functions
		typedef OptionsFinalT options_t;
	};

	//////////////////////////////////////////////////////////////////////////
	// thanks to http://en.cppreference.com/w/cpp/types/void_t
	// primary template handles types that have no nested ::options_t member:
	template< class, class = std::void_t<> >
	struct has_options : std::false_type { };
	// specialization recognizes types that do have a nested ::options_t member:
	template< class T >
	struct has_options<T, std::void_t<typename T::options_t>> : std::true_type {};

	//////////////////////////////////////////////////////////////////////////
	template<typename EnumT>
	struct binary_options : public options<binary_options<EnumT>> {
		typedef EnumT binary_options_enum_t;

		std::bitset<binary_options_enum_t::total_options> m_binary_options;

		void turn_on_all_options()noexcept { m_binary_options.set(); }
		void turn_off_all_options()noexcept { m_binary_options.reset(); }
	};

	// primary template handles types that have no nested ::options_t and ::binary_options_enum_t members:
	template< class, class = std::void_t<> >
	struct has_binary_options : std::false_type { };
	// specialization recognizes types that do have a nested ::options_t and binary_options_enum_t members:
	template< class T >
	struct has_binary_options<T, std::void_t<typename T::options_t, typename T::binary_options_enum_t>> : std::true_type {};


	//////////////////////////////////////////////////////////////////////////
	// helper function that can provide default binary options for classes, that don't derived from options class
	// first - defaults to false
	template<bool bDefault = false, typename ClassT>
	inline std::enable_if_t<has_binary_options<ClassT>::value, bool> binary_option(const ClassT& ar, const typename ClassT::binary_options_enum_t optId)noexcept {
		return ar.m_binary_options[optId];
	}
	template<bool bDefault = false, typename ClassT>
	inline std::enable_if_t<!has_binary_options<ClassT>::value, constexpr bool> binary_option(const ClassT& ar, const size_t optId)noexcept {
		return bDefault;
	}

}
}