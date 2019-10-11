/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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

namespace nntl {

	//math_types should be defined elsewhere
	//using math_types = math::_basic_types;
	// OBSOLETTE

	//typename for referring layer numbers, indexes, counts and so on. *640k* 256 layers should enough for everyone :-D
	//Damn, I've never thought, it'll not be enought so soon!)
	// must be unsigned
	typedef ::std::uint16_t layer_index_t;

	// must be unsigned
	typedef ::std::uint32_t neurons_count_t;

	//by convention layer_type_id_t can't be zero
	typedef ::std::uint64_t layer_type_id_t;

	//real_t with extended precision for some temporarily calculations
	typedef double ext_real_t;

	//thread id must be in range [0,workers_count())
	//worker threads should have par_range_t::tid>=1. tid==0 is reserved to main thread.
	// If scheduler will launch less than workers_count() threads to process task, 
	// then maximum tid must be equal to <scheduled workers count>+1 (+1 refers to a main thread, that's also
	// used in scheduling)
	typedef unsigned int thread_id_t;

	//see also NNTL_STRING macro
	// by now some code such as file-related functions in _supp{} may be bounded to char only in strchar_t
	//TODO: Don't think it is good idea to solve it now.
	// OBSOLETTE, use char instead
	typedef char strchar_t;

	namespace utils {
		//////////////////////////////////////////////////////////////////////////
		//https://bitbucket.org/martinhofernandes/wheels/src/default/include/wheels/meta/type_traits.h%2B%2B?fileviewer=file-view-default#cl-161
		//! Tests if T is a specialization of Template
		template <typename T, template <typename...> class Template>
		struct is_specialization_of : ::std::false_type {};
		template <template <typename...> class Template, typename... Args>
		struct is_specialization_of<Template<Args...>, Template> : ::std::true_type {};

		// 		template <typename T>
		// 		using is_tuple2 = is_specialization_of<T, ::std::tuple>;
	}
}