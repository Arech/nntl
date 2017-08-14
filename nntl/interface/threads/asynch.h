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

#include "../_i_asynch.h"

namespace nntl {
namespace threads {

	template <typename RangeT, typename SyncT = threads::sync_primitives, typename CallHandlerT = utils::forwarderWrapper<>>
	class Asynch : public _i_asynch<RangeT> {
		//!! copy constructor not needed
		Asynch(const Asynch& other)noexcept = delete;
		Asynch(Asynch&& other)noexcept = delete;
		//!!assignment is not needed
		Asynch& operator=(const Asynch& rhs) noexcept = delete;

	public:
		typedef CallHandlerT CallH_t;
		typedef SyncT Sync_t;

	protected:
		//typedef typename CallH_t::template call_tpl<void(const par_range_t& r)> func_run_t;
		//typedef typename CallH_t::template call_tpl<real_t(const par_range_t& r)> func_reduce_t;

		typedef typename Sync_t::mutex_comp_t mutex_comp_t;
		typedef ::std::atomic_ptrdiff_t interlocked_t;
		//typedef typename Sync_t::cond_var_t cond_var_t;
		typedef typename Sync_t::cond_var_comp_t cond_var_comp_t;

	public:
		//typedef ::std::vector<::std::thread> threads_cont_t;
		//typedef threads_cont_t::iterator ThreadObjIterator_t;

		//////////////////////////////////////////////////////////////////////////
		//Members
	protected:
// 		cond_var_comp_t m_waitingOrders;
// 		cond_var_comp_t m_orderDone;
// 		mutex_comp_t m_mutex;
		//interlocked_t m_workingCnt;
		
		//threads_cont_t m_threads;
		//bool m_bStop;

	protected:

	public:
		~Asynch()noexcept {
		}

		Asynch()noexcept {
			
		}
		

	};

}
}