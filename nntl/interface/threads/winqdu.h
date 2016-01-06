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

#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)
#pragma message "native SRW Locks and conditional variables not available, use threads/std.h instead"

#else

#define NNTL_THREADS_WINQDU_AVAILABLE

//basic implementation on windows core performance primitives. Non-portable, quick, dirty, ugly

#include <windows.h>
#include <functional>
#include <thread>

#include "../_i_threads.h"

namespace nntl {
namespace threads {

	//TODO: error handling!!!

	template <typename range_type>
	class WinQDU : public _i_threads<range_type>{
		//!! copy constructor not needed
		WinQDU(const WinQDU& other)noexcept = delete;
		//!!assignment is not needed
		WinQDU& operator=(const WinQDU& rhs) noexcept = delete;
		
	protected:
		typedef std::function<void(const par_range_t& r)> func_run_t;
		typedef std::function<real_t(const par_range_t& r)> func_reduce_t;

		enum class JobType{Run, Reduce};

	public:
		typedef std::vector<std::thread> threads_cont_t;
		typedef threads_cont_t::iterator ThreadObjIterator_t;

		static constexpr char* name = "WinQDU";

		//////////////////////////////////////////////////////////////////////////
		//Members
	protected:
		CONDITION_VARIABLE m_waitingOrders;
		CONDITION_VARIABLE m_orderDone;
		SRWLOCK m_srwlock;
		volatile long m_workingCnt;
		JobType m_jobType;

		std::vector<par_range_t> m_ranges;
		func_run_t m_fnRun;
		std::vector<real_t> m_reduceCache;
		func_reduce_t m_fnReduce;

		const thread_id_t m_workersCnt;
		threads_cont_t m_threads;
		bool m_bStop;

	public:
		~WinQDU()noexcept {
			m_bStop = true;
			//locking execution to make sure threads are in wait state
			AcquireSRWLockExclusive(&m_srwlock);
			WakeAllConditionVariable(&m_waitingOrders);
			ReleaseSRWLockExclusive(&m_srwlock);

			for (auto& t : m_threads)  t.join();
		}

		WinQDU()noexcept : m_srwlock(SRWLOCK_INIT), m_bStop(false), m_workersCnt(workers_count() - 1), m_workingCnt(0){
			InitializeConditionVariable(&m_waitingOrders);
			InitializeConditionVariable(&m_orderDone);

			//STDCOUTL("** Supports " << m_workersCnt << " concurent threads");
			NNTL_ASSERT(m_workersCnt > 0);

			m_ranges.reserve(m_workersCnt);
			m_threads.resize(m_workersCnt);
			m_reduceCache.resize(m_workersCnt + 1);

			AcquireSRWLockExclusive(&m_srwlock);
			m_workingCnt = m_workersCnt;

			for (thread_id_t i = 0; i < m_workersCnt; ++i) {
				//worker threads should have par_range_t::tid>=1. tid==0 is reserved to main thread
				m_ranges.push_back(par_range_t(0, 0, i + 1));
				m_threads[i] = std::thread(_s_worker, this, i);
			}
			m_ranges.shrink_to_fit();
			NNTL_ASSERT(m_ranges.size() == m_workersCnt);
			
			ReleaseSRWLockExclusive(&m_srwlock);

			AcquireSRWLockExclusive(&m_srwlock);
			while (m_workingCnt > 0) {
				SleepConditionVariableSRW(&m_orderDone, &m_srwlock, INFINITE, 0);
			}
			ReleaseSRWLockExclusive(&m_srwlock);
		}

		static thread_id_t workers_count()noexcept {
			return std::thread::hardware_concurrency();
		}
		auto get_worker_threads(thread_id_t& threadsCnt)noexcept ->ThreadObjIterator_t {
			threadsCnt = m_workersCnt;
			return m_threads.begin();
		}

		template<typename Func>
		void run(Func&& F, const range_t cnt, const thread_id_t useNThreads = 0, thread_id_t* pThreadsUsed = nullptr) noexcept {
			//TODO: decide whether it is worth to use workers here
			//DONE: well, it worth less than 9mks to parallelize execution therefore won't bother...
			if (cnt <= 1) {
				if (pThreadsUsed) *pThreadsUsed = 1;
				F(par_range_t(cnt));
			} else {
				AcquireSRWLockExclusive(&m_srwlock);
				m_fnRun = F;
				m_jobType = JobType::Run;

				const auto prevOfs = partition_count_to_workers(cnt, useNThreads);
				NNTL_ASSERT(prevOfs < cnt);
				if (pThreadsUsed) *pThreadsUsed = static_cast<thread_id_t>(m_workingCnt)+1;

				WakeAllConditionVariable(&m_waitingOrders);
				ReleaseSRWLockExclusive(&m_srwlock);

				F(par_range_t(prevOfs, cnt - prevOfs, 0));

				if (m_workingCnt > 0) {
					AcquireSRWLockExclusive(&m_srwlock);
					while (m_workingCnt > 0) {
						SleepConditionVariableSRW(&m_orderDone, &m_srwlock, INFINITE, 0);
					}
					ReleaseSRWLockExclusive(&m_srwlock);
				}
			}
		}

		template<typename Func, typename FinalReduceFunc>
		real_t reduce(Func&& FRed, FinalReduceFunc&& FRF, const range_t cnt, const thread_id_t useNThreads = 0) noexcept {
			//TODO: decide whether it is worth to use workers here
			//DONE: well, it worth less than 9mks to parallelize execution therefore won't bother...
			if (cnt <= 1) {
				return FRed( par_range_t(cnt) );
			} else {
				AcquireSRWLockExclusive(&m_srwlock);
				m_fnReduce = FRed;
				m_jobType = JobType::Reduce;

				//TODO: need cache friendly partitioning here
				const auto prevOfs = partition_count_to_workers(cnt, useNThreads);
				NNTL_ASSERT(prevOfs < cnt);
				auto* rc = &m_reduceCache[0];

				const range_t workersOnReduce = m_workingCnt + 1;
				NNTL_ASSERT(workersOnReduce <= m_reduceCache.size());

				WakeAllConditionVariable(&m_waitingOrders);
				ReleaseSRWLockExclusive(&m_srwlock);

				*rc = FRed(par_range_t(prevOfs, cnt - prevOfs, 0));

				if (m_workingCnt > 0) {
					AcquireSRWLockExclusive(&m_srwlock);
					while (m_workingCnt > 0) {
						SleepConditionVariableSRW(&m_orderDone, &m_srwlock, INFINITE, 0);
					}
					ReleaseSRWLockExclusive(&m_srwlock);
				}
				return FRF(rc, workersOnReduce);
			}
		}

	protected:

		//returns an offset after last partitioned item
		range_t partition_count_to_workers(const range_t cnt, const thread_id_t _useNThreads)noexcept {
			//TODO: need cache friendly partitioning here
			const thread_id_t useNThreads = _useNThreads > 1 && _useNThreads <= m_workersCnt + 1 ? _useNThreads - 1 : m_workersCnt;

			const thread_id_t _workingCnt = cnt > useNThreads ? useNThreads : static_cast<thread_id_t>(cnt - 1);
			m_workingCnt = static_cast<std::remove_volatile<decltype(m_workingCnt)>::type> (_workingCnt);
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

		static void _s_worker(WinQDU* p, const thread_id_t id)noexcept { p->_worker(id); }

		void _worker(const thread_id_t id)noexcept {
			InterlockedDecrement(&m_workingCnt);
			WakeConditionVariable(&m_orderDone);

			while (true) {
				AcquireSRWLockShared(&m_srwlock);

				while (!m_bStop && 0 == m_ranges[id].cnt()) {
					SleepConditionVariableSRW(&m_waitingOrders, &m_srwlock, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED);
				}

				if (m_bStop) {
					ReleaseSRWLockShared(&m_srwlock);
					break;
				}

				auto& thrdRange = m_ranges[id];
				switch (m_jobType){
				case JobType::Run:
					m_fnRun(thrdRange);
					break;
				case JobType::Reduce:
					m_reduceCache[id+1] = m_fnReduce(thrdRange);
					break;
				default:
					NNTL_ASSERT(!"WTF???");
					abort();
				}
				thrdRange.cnt(0);

				InterlockedDecrement(&m_workingCnt);
				WakeConditionVariable(&m_orderDone);
				ReleaseSRWLockShared(&m_srwlock);
			}
		}
	};
}
}


#endif

