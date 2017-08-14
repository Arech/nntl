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

#include "../_i_threads.h"

namespace nntl {
namespace threads {

	//TODO: error handling!!!

	template <typename RealT, typename RangeT, typename SyncT = threads::sync_primitives, typename CallHandlerT = utils::forwarderWrapper<>>
	class Workers : public _i_threads<RealT, RangeT> {
		//!! copy constructor not needed
		Workers(const Workers& other)noexcept = delete;
		Workers(Workers&& other)noexcept = delete;
		//!!assignment is not needed
		Workers& operator=(const Workers& rhs) noexcept = delete;

	public:
		typedef CallHandlerT CallH_t;
		typedef SyncT Sync_t;

	protected:
		typedef typename CallH_t::template call_tpl<void(const par_range_t& r)> func_run_t;
		typedef typename CallH_t::template call_tpl<real_t(const par_range_t& r)> func_reduce_t;

		enum class JobType { Run, Reduce };

		typedef typename Sync_t::mutex_comp_t mutex_comp_t;
		typedef ::std::atomic_ptrdiff_t interlocked_t;
		//typedef typename Sync_t::cond_var_t cond_var_t;
		typedef typename Sync_t::cond_var_comp_t cond_var_comp_t;

	public:
		typedef ::std::vector<::std::thread> threads_cont_t;
		typedef threads_cont_t::iterator ThreadObjIterator_t;
		
		//////////////////////////////////////////////////////////////////////////
		//Members
	protected:
		cond_var_comp_t m_waitingOrders;
		cond_var_comp_t m_orderDone;
		mutex_comp_t m_mutex;
		interlocked_t m_workingCnt;
		JobType m_jobType;

		//we'll be using fences to make sure load/store operations on this variables are coherent
		::std::vector<par_range_t> m_ranges;

		func_run_t m_fnRun;
		::std::vector<real_t> m_reduceCache;
		func_reduce_t m_fnReduce;

		const thread_id_t m_workersCnt;
		threads_cont_t m_threads;
		::std::atomic<bool> m_bStop;

	public:
		~Workers()noexcept {
			m_bStop = true;

			m_mutex.lock();
			m_waitingOrders.notify_all();
			m_mutex.unlock();

			for (auto& t : m_threads)  t.join();
		}

		Workers()noexcept : m_bStop(false), m_workersCnt(workers_count() - 1), m_workingCnt(0) {
			NNTL_ASSERT(m_workersCnt > 0);

			m_ranges.reserve(m_workersCnt);
			m_threads.resize(m_workersCnt);
			m_reduceCache.resize(m_workersCnt + 1);

			m_mutex.lock();
			m_workingCnt = m_workersCnt;

			for (thread_id_t i = 0; i < m_workersCnt; ++i) {
				//worker threads should have par_range_t::tid>=1. tid==0 is reserved to main thread
				m_ranges.push_back(par_range_t(0, 0, i + 1));
				m_threads[i] = ::std::thread(_s_worker, this, i);
			}
			m_ranges.shrink_to_fit();
			NNTL_ASSERT(m_ranges.size() == m_workersCnt);
			m_mutex.unlock();

			::std::unique_lock<decltype(m_mutex)> lk(m_mutex);
			m_orderDone.wait(lk, [&wc = m_workingCnt]() throw() {return wc <= 0; });
		}

		static thread_id_t workers_count()noexcept {
			return ::std::thread::hardware_concurrency();
		}
		auto get_worker_threads(thread_id_t& threadsCnt)noexcept ->ThreadObjIterator_t {
			threadsCnt = m_workersCnt;
			return m_threads.begin();
		}

		bool denormalsOnInAnyThread()noexcept {
			for (auto&e : m_reduceCache) e = real_t(-9999);

			thread_id_t thdUsed = 0;
			const auto thrCnt = m_workersCnt + 1;
			run([&rc = m_reduceCache](const par_range_t& r) {
				rc[r.tid()] = isDenormalsOn() ? real_t(1.) : real_t(0.);
			}, thrCnt, thrCnt, &thdUsed);

			if (thdUsed == thrCnt) {
				real_t s(real_t(0.));
				for (const auto& e : m_reduceCache) s += e;
				return real_t(0.) != s;
			}
			return true;
		}

		//NEVER CALL IN RECURSION!
		template<typename Func>
		void run(Func&& F, const range_t cnt, const thread_id_t useNThreads = 0, thread_id_t* pThreadsUsed = nullptr) noexcept {
			if (cnt <= 1) {
				if (pThreadsUsed) *pThreadsUsed = 1;
				::std::forward<Func>(F)(par_range_t(cnt));
			} else {
				_run(CallH_t::wrap<Func>(::std::forward<Func>(F)), cnt, useNThreads, pThreadsUsed);
			}
		}

	protected:

		template<typename Func>
		void _run(Func&& F, const range_t cnt, const thread_id_t useNThreads = 0, thread_id_t* pThreadsUsed = nullptr) noexcept {
			NNTL_ASSERT(cnt > 1);
			m_mutex.lock();

			m_fnRun = F;
			m_jobType = JobType::Run;

			const auto prevOfs = partition_count_to_workers(cnt, useNThreads);
			NNTL_ASSERT(prevOfs < cnt);
			if (pThreadsUsed) *pThreadsUsed = static_cast<thread_id_t>(m_workingCnt) + 1;

			m_waitingOrders.notify_all();
			m_mutex.unlock();

			::std::forward<Func>(F)(par_range_t(prevOfs, cnt - prevOfs, 0));

			if (m_workingCnt > 0) {
// 				::std::unique_lock<decltype(m_mutex)> lk(m_mutex);
// 				while (m_workingCnt > 0) {
// 					m_orderDone.wait(lk);
// 				}
				Sync_t::lock_wait_unlock(m_mutex, m_orderDone, [&wc = m_workingCnt]() {return wc <= 0; });
			}
		}

	public:
		template<typename Func, typename FinalReduceFunc>
		real_t reduce(Func&& FRed, FinalReduceFunc&& FRF, const range_t cnt, const thread_id_t useNThreads = 0) noexcept {
			return cnt <= 1
				? ::std::forward<Func>(FRed)(par_range_t(cnt))
				: _reduce(CallH_t::wrap<Func>(::std::forward<Func>(FRed)), ::std::forward<FinalReduceFunc>(FRF), cnt, useNThreads);
		}


	protected:

		template<typename Func, typename FinalReduceFunc>
		real_t _reduce(Func&& FRed, FinalReduceFunc&& FRF, const range_t cnt, const thread_id_t useNThreads = 0) noexcept {
			NNTL_ASSERT(cnt > 1);
			m_mutex.lock();

			m_fnReduce = FRed;
			m_jobType = JobType::Reduce;

			//TODO: need cache friendly partitioning here
			const auto prevOfs = partition_count_to_workers(cnt, useNThreads);
			NNTL_ASSERT(prevOfs < cnt);
			auto* rc = &m_reduceCache[0];

			const range_t workersOnReduce = m_workingCnt + 1;
			NNTL_ASSERT(workersOnReduce <= m_reduceCache.size());

			m_waitingOrders.notify_all();
			m_mutex.unlock();

			*rc = ::std::forward<Func>(FRed)(par_range_t(prevOfs, cnt - prevOfs, 0));

			if (m_workingCnt > 0) {
				Sync_t::lock_wait_unlock(m_mutex, m_orderDone, [&wc = m_workingCnt]() {return wc <= 0; });
			}
			return ::std::forward<FinalReduceFunc>(FRF)(rc, workersOnReduce);
		}

	protected:

		//returns an offset after last partitioned item
		range_t partition_count_to_workers(const range_t cnt, const thread_id_t _useNThreads)noexcept {
			//TODO: need cache friendly partitioning here
			const thread_id_t useNThreads = _useNThreads > 1 && _useNThreads <= m_workersCnt + 1 ? _useNThreads - 1 : m_workersCnt;
			const thread_id_t _workingCnt = cnt > useNThreads ? useNThreads : static_cast<thread_id_t>(cnt - 1);
			m_workingCnt = _workingCnt;
			const range_t totalWorkers = _workingCnt + 1;
			range_t eachCnt = cnt / totalWorkers;
			const range_t residual = cnt % totalWorkers;
			range_t prevOfs = 0;

			for (thread_id_t i = 0; i < m_workersCnt; ++i) {
				if (i >= _workingCnt) {
					m_ranges[i].cnt(0);
				} else {
					const range_t n = eachCnt + (i < residual ? 1 : 0);
					m_ranges[i].cnt(n).offset(prevOfs);
					prevOfs += n;
				}
			}
			return prevOfs;
		}

		static void _s_worker(Workers* p, const thread_id_t id)noexcept {
			global_denormalized_floats_mode();
			p->_worker(id);
		}

		void _worker(const thread_id_t id)noexcept {
			m_mutex.lock();
			m_workingCnt--;
			m_orderDone.notify_one();
			m_mutex.unlock();		

			auto& thrdRange = m_ranges[id];
			while (true) {				
				Sync_t::lockShared_wait_unlock(m_mutex, m_waitingOrders, [&bStop = m_bStop, &tr = thrdRange]()noexcept{
					return bStop || 0 != tr.cnt();
				});
				if (m_bStop) break;

				switch (m_jobType) {
				case JobType::Run:
					m_fnRun(thrdRange);
					break;
				case JobType::Reduce:
					m_reduceCache[id + 1] = m_fnReduce(thrdRange);
					break;
				default:
					NNTL_ASSERT(!"WTF???");
					abort();
				}

				//must set lock here to prevent deadlock during lk.lock();while (m_workingCnt > 0)  m_orderDone.wait(lk);...

				Sync_t::lockShared(m_mutex);
				thrdRange.cnt(0);
				m_workingCnt--;
				m_orderDone.notify_one();
				Sync_t::unlockShared(m_mutex);
			}
		}
	};

}
}

