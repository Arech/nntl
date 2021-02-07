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

//this an addition to generic tuple_utils.h to work over tuple of layers

#include "../utils/tuple_utils.h"

#include "_init_layers.h"

namespace nntl {
namespace tuple_utils {

	//walk over every tuple element except last in back-wise manner until is_layer_stops_bprop<current element type> returns true
	namespace _impl {
		template<int I, class Tuple, typename F> struct _down4bprop_go {
			static void for_each(Tuple&& t, F&& f) noexcept {
				f(::std::get<I>(t), ::std::get<I - 1>(t));
				_down4bprop<I - 1, Tuple, F>::for_each(::std::forward<Tuple>(t), ::std::forward<F>(f));
			}
		};
		template<class Tuple, typename F> struct _down4bprop_go<1, Tuple, F> {
			static void for_each(Tuple&& t, F&& f) noexcept {
				f(::std::get<1>(t), ::std::get<0>(t));
			}
		};

		template<int I, class Tuple, typename F> struct _down4bprop_stop {
			static void for_each(Tuple&& t, F&& f) noexcept {
				//f(::std::get<I>(t), ::std::get<I - 1>(t));
			}
		};

		template<int I, class Tuple, typename F>
		struct _down4bprop : public ::std::conditional_t<
			is_layer_stops_bprop<::std::decay_t<::std::tuple_element_t<I, ::std::decay_t<Tuple>>>>::value
			, _down4bprop_stop<I, Tuple, F>
			, _down4bprop_go<I, Tuple, F>
		> {};

		template<class Tuple, typename F> struct _down4bprop<0, Tuple, F> {
			static void for_each(Tuple&&, F&&)noexcept {}
		};
	}

	template<class Tuple, typename F>
	inline void for_each_down4bprop_no_last(Tuple&& t, F&& f)noexcept {
		constexpr auto lei = ::std::tuple_size<::std::remove_reference_t<Tuple>>::value - 1;
		static_assert(lei > 0, "Tuple must have more than 1 element");
		_impl::_down4bprop<lei - 1, Tuple, F>::for_each(::std::forward<Tuple>(t), ::std::forward<F>(f));
	}


	//////////////////////////////////////////////////////////////////////////
	//same as above, but the last element is also included
	template<class Tuple, typename F>
	inline void for_each_down4bprop(Tuple&& t, F&& f)noexcept {
		constexpr auto lei = ::std::tuple_size<::std::remove_reference_t<Tuple>>::value - 1;
		static_assert(lei > 0, "Tuple must have more than 1 element");
		_impl::_down4bprop<lei, Tuple, F>::for_each(::std::forward<Tuple>(t), ::std::forward<F>(f));
	}

}
}

