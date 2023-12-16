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

//This file defines _i_asynch common interface to a provider of asynchronous worker threads

#include "../utils/denormal_floats.h"
//#include "threads/parallel_range.h"
#include "../utils/call_wrappers.h"
#include "threads/_sync_primitives.h"
#include "threads/prioritize_workers.h"

namespace nntl {
namespace threads {

	//interface to a asynchronous thread workers pool
	struct _i_bgworkers {
	private:
		typedef _i_bgworkers self_t;

	public:
		nntl_interface thread_id_t workers_count()noexcept;
		nntl_interface auto get_worker_threads(thread_id_t& threadsCnt)noexcept;

		nntl_interface self_t& expect_tasks_count(const unsigned expectedTasksCnt)noexcept;

		template<class Rep, class Period>
		nntl_interface self_t& set_task_wait_timeout(const ::std::chrono::duration<Rep, Period>& to)noexcept;

		//in general, func must have a duration greater than duration of the task
		template<typename FTask>
		nntl_interface self_t& add_task(FTask&& func, int priority = 0) noexcept;

		nntl_interface self_t& delete_tasks()noexcept;

		//never call recursively or from non-main thread
		template<typename FExec>
		nntl_interface self_t& exec(FExec&& func) noexcept;
	};


}
}