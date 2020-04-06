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

#include "../utils/denormal_floats.h"
#include "threads/parallel_range.h"
#include "../utils/call_wrappers.h"
#include "threads/_sync_primitives.h"
#include "threads/prioritize_workers.h"

namespace nntl {
namespace threads {

	template <typename RealT, typename RangeT>
	struct _i_threads {
		typedef RealT real_t;

		typedef RangeT range_t;
		typedef parallel_range<range_t> par_range_t;
		//typedef typename par_range_t::thread_id_t thread_id_t;

		// some ugly reduce() staff necessary to get rid of C++ UB that comes from pointer casting.
		// See reduce() implementation for details
		typedef ::std::uint64_t reduce_data_t;
		static_assert(sizeof(real_t) <= sizeof(reduce_data_t), "");
		static_assert(::std::alignment_of<real_t>::value <= ::std::alignment_of<reduce_data_t>::value, "");
		
		template<typename T>
		struct converter_reduce_data_t {
			typedef T type;

			static_assert(::std::is_integral<T>::value || ::std::is_floating_point<T>::value, "");
			static_assert(!::std::is_const<T>::value, "");
			static_assert(sizeof(T) <= sizeof(reduce_data_t), "");
			static_assert(::std::alignment_of<T>::value <= ::std::alignment_of<reduce_data_t>::value, "");

			static reduce_data_t to(const type v)noexcept {
				reduce_data_t r=0;
				::std::memcpy(&r, &v, sizeof(type));
				return r;
			}
			static type from(const reduce_data_t v)noexcept {
				type r;
				::std::memcpy(&r, &v, sizeof(type));
				return r;
			}
		};

		//////////////////////////////////////////////////////////////////////////

		nntl_interface static thread_id_t workers_count()noexcept;
		nntl_interface thread_id_t cur_workers_count()const noexcept; //non static faster version of workers_count()

		//returns a head of container with thread objects (count threadsCnt)
		nntl_interface auto get_worker_threads(thread_id_t& threadsCnt)noexcept;

		nntl_interface bool denormalsOnInAnyThread()noexcept;

		// useNThreads (if greater than 1 and less or equal to workers_count() specifies the number of threads to serve request.
		// if pThreadsUsed is specified, it'll contain total number of threads (including the main thread),
		// that is used to serve the request. It'll be less or equal to workers_count()
		template<typename Func>
		nntl_interface void run(Func&& F, const range_t cnt, const thread_id_t useNThreads = 0, thread_id_t* pThreadsUsed = nullptr) noexcept;

		// useNThreads (if greater than 1 and less or equal to workers_count() specifies the number of threads to serve request.
		// see workers::reduce() implementation for details
		template<typename Func, typename FinalReduceFunc>
		nntl_interface auto reduce(Func&& FRed, FinalReduceFunc&& FRF, const range_t cnt, const thread_id_t useNThreads = 0) noexcept
			-> decltype(::std::forward<FinalReduceFunc>(FRF)(static_cast<const reduce_data_t*>(nullptr), range_t(0)));

		/*moved to _threads_td
		 *nntl_interface thread_id_t workers_count()noexcept;

		//returns a head of container with thread objects (count threadsCnt)
		nntl_interface auto get_worker_threads(thread_id_t& threadsCnt)noexcept;

		nntl_interface bool denormalsOnInAnyThread()noexcept;*/

		nntl_interface static constexpr char* name="_i_threads";
	};
}
}
