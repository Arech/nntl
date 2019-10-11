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

#include <type_traits>

namespace nntl {
namespace utils {

	//////////////////////////////////////////////////////////////////////////
	// y_combinator concept to make recursive lambdas possible
	// thanks to http://stackoverflow.com/documentation/c%2b%2b/572/lambdas/8508/recursive-lambdas#t=201703101123559331192
	template <class F>
	struct y_combinator {
		F f; // the lambda will be stored here

			 // a forwarding operator():
		template <class... Args>
		decltype(auto) operator()(Args&&... args) const {
			// we pass ourselves to f, then the arguments.
			// the lambda should take the first argument as `auto&& recurse` or similar.
			return f(*this, ::std::forward<Args>(args)...);
		}
	};
	// helper function that deduces the type of the lambda:
	template <class F>
	y_combinator<::std::decay_t<F>> make_y_combinator(F&& f) {
		return{ ::std::forward<F>(f) };
	}
	// (Be aware that in C++17 we can do better than a `make_` function)

	//example
	/*auto gcd = make_y_combinator(
		[](auto&& gcd, int a, int b) {
		return b == 0 ? a : gcd(b, a%b);
	}
	);*/

}
}