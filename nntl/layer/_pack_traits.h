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

#include <map>
#include <set>

// defines some common traits and struct for layer_pack_* layers

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	// traits recognizer
	// primary template handles types that have no nested ::phl_original_t member:
	template< class, class = ::std::void_t<> >
	struct is_PHL : ::std::false_type { };
	// specialization recognizes types that do have a nested ::phl_original_t member:
	template< class T >
	struct is_PHL<T, ::std::void_t<typename T::phl_original_t>> : ::std::true_type {};

	//////////////////////////////////////////////////////////////////////////
	// Tests if a layer applies gating capabilities to its inner layers by checking the existance of ::gating_layer_t type
	template< class, class = ::std::void_t<> >
	struct is_pack_gated : ::std::false_type { };
	template< class T >
	struct is_pack_gated<T, ::std::void_t<typename T::gating_layer_t>> : ::std::true_type {};

	//////////////////////////////////////////////////////////////////////////
	// Helper traits recognizer
	// primary template handles types that have no nested ::wrapped_layer_t member:
	template< class, class = ::std::void_t<> >
	struct is_layer_wrapper : ::std::false_type { };
	// specialization recognizes types that do have a nested ::wrapped_layer_t member:
	template< class T >
	struct is_layer_wrapper<T, ::std::void_t<typename T::wrapped_layer_t>> : ::std::true_type {};

	//////////////////////////////////////////////////////////////////////////
	// Tests if a layer applies tiling capabilities to its inner layer by checking the existance of ::tiled_layer_t type
	template< class, class = ::std::void_t<> >
	struct is_pack_tiled : ::std::false_type { };
	template< class T >
	struct is_pack_tiled<T, ::std::void_t<typename T::tiled_layer_t>> : ::std::true_type {};
}