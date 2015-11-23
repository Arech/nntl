/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015, Arech (aradvert@gmail.com; https://github.com/Arech)
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

//like threadsc, but run() restored to templated version able to receive capturing lambdas

//include header after nntl
#include "threads/parallel_range.h"

namespace nntl {
namespace threads {
	
	template <typename range_type>
	class _i_threadsc2 {
	public:
		typedef range_type range_t;
		typedef parallel_range<range_t> par_range_t;
		typedef typename par_range_t::thread_id_t thread_id_t;

		typedef math_types::float_ty float_t_;

		typedef float_t_(*fnReduce_t)(const par_range_t& r, void* context);
		typedef float_t_(*fnFinalReduce_t)(const float_t_* ptr, const range_t cnt, void* context);

		// TODO: add param (range_t cacheHint=0) to be the max length of data to pass to F
		template<typename Func>
		nntl_interface void run(Func&& F, const range_t cnt) noexcept;

		//expecting r.offset()==0.
		// TODO: add param (range_t cacheHint=0) to be the max length of data to pass to F
		nntl_interface float_t_ reduce(fnReduce_t FRed, fnFinalReduce_t FRF, const range_t cnt, void* redContext=nullptr) noexcept;

		nntl_interface thread_id_t workers_count()noexcept;

		//returns a head of container with thread objects (count threadsCnt)
		auto get_worker_threads(thread_id_t& threadsCnt)noexcept;
	};

}
}