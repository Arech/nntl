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

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "../threads.h"

namespace nntl {
namespace threads {

	//TODO: error handling!!!
	
	template <typename range_type>
	class Std : public _i_threads<range_type>{
		//!! copy constructor not needed
		Std(const Std& other)noexcept = delete;
		//!!assignment is not needed
		Std& operator=(const Std& rhs) noexcept = delete;

	protected:
		typedef std::function<void(const par_range_t& r)> func_run_t;
		typedef std::function<real_t(const par_range_t& r)> func_reduce_t;

		enum class JobType { Run, Reduce };

		typedef std::mutex lock_t;
		typedef std::unique_lock<lock_t> locker_t;
		typedef std::atomic_ptrdiff_t interlocked_t;

public:
		typedef std::vector<std::thread> threads_cont_t;
		typedef threads_cont_t::iterator ThreadObjIterator_t;

		//////////////////////////////////////////////////////////////////////////
		//Members
	protected:
		std::condition_variable m_waitingOrders;
		std::condition_variable m_orderDone;
		lock_t m_lock;
		interlocked_t m_workingCnt;
		JobType m_jobType;

		std::vector<par_range_t> m_ranges;
		func_run_t m_fnRun;
		std::vector<real_t> m_reduceCache;
		func_reduce_t m_fnReduce;
		
		const thread_id_t m_workersCnt;
		threads_cont_t m_threads;
		bool m_bStop;

	public:
		~Std()noexcept {
			m_bStop = true;
			
			locker_t lk(m_lock);
			m_waitingOrders.notify_all();
			lk.unlock();

			for (auto& t : m_threads)  t.join();
		}

		Std()noexcept : m_bStop(false), m_workersCnt(workers_count() - 1), m_workingCnt(0) {
			NNTL_ASSERT(m_workersCnt > 0);

			m_ranges.reserve(m_workersCnt);
			m_threads.resize(m_workersCnt);
			m_reduceCache.resize(m_workersCnt + 1);

			locker_t lk(m_lock);
			m_workingCnt = m_workersCnt;

			for (thread_id_t i = 0; i < m_workersCnt; ++i) {
				//worker threads should have par_range_t::tid>=1. tid==0 is reserved to main thread
				m_ranges.push_back(par_range_t(0, 0, i + 1));
				m_threads[i] = std::thread(_s_worker, this, i);
			}
			m_ranges.shrink_to_fit();
			NNTL_ASSERT(m_ranges.size() == m_workersCnt);
			lk.unlock();

			lk.lock();
			m_orderDone.wait(lk, [&] () throw() {return m_workingCnt <= 0; });
			lk.unlock();
		}

		static thread_id_t workers_count()noexcept {
			return std::thread::hardware_concurrency();
		}
		auto get_worker_threads(thread_id_t& threadsCnt)noexcept ->ThreadObjIterator_t {
			threadsCnt = m_workersCnt;
			return m_threads.begin();
		}

		template<typename Func>
		void run(Func&& F, const range_t cnt, thread_id_t* pThreadsUsed = nullptr) noexcept {
			//TODO: decide whether it is worth to use workers here
			//DONE: well, it worth about 14mks to parallelize execution therefore won't bother...
			if (cnt <= 1) {
				if (pThreadsUsed) *pThreadsUsed = 1;
				F(par_range_t(cnt));
			} else {
				locker_t lk(m_lock);

				m_fnRun = F;
				m_jobType = JobType::Run;
				
				const auto prevOfs = partition_count_to_workers(cnt);
				NNTL_ASSERT(prevOfs < cnt);
				if (pThreadsUsed) *pThreadsUsed = static_cast<thread_id_t>(m_workingCnt) + 1;

				m_waitingOrders.notify_all();
				lk.unlock();

				F(par_range_t(prevOfs, cnt - prevOfs, 0));

				if (m_workingCnt > 0) {
					lk.lock();
					//m_orderDone.wait(lk, [&] () throw() {return m_workingCnt <= 0; });
					while (m_workingCnt>0) {
						m_orderDone.wait(lk);
					}
					lk.unlock();
				}
			}
		}

		template<typename Func, typename FinalReduceFunc>
		real_t reduce(Func&& FRed, FinalReduceFunc FRF, const range_t cnt) noexcept {
			//TODO: decide whether it is worth to use workers here
			//DONE: well, it worth less than 9mks to parallelize execution therefore won't bother...
			if (cnt <= 1) {
				return FRed( par_range_t(cnt) );
			} else {
				locker_t lk(m_lock);

				m_fnReduce = FRed;
				m_jobType = JobType::Reduce;

				//TODO: need cache friendly partitioning here
				const auto prevOfs = partition_count_to_workers(cnt);
				NNTL_ASSERT(prevOfs < cnt);
				auto* rc = &m_reduceCache[0];

				const range_t workersOnReduce = m_workingCnt + 1;
				NNTL_ASSERT(workersOnReduce <= m_reduceCache.size());

				m_waitingOrders.notify_all();
				lk.unlock();

				*rc = FRed(par_range_t(prevOfs, cnt - prevOfs, 0));

				if (m_workingCnt > 0) {
					lk.lock();
					while (m_workingCnt > 0)  m_orderDone.wait(lk);
					lk.unlock();
				}
				return FRF(rc, workersOnReduce);
			}
		}

	protected:

//returns an offset after last partitioned item
		range_t partition_count_to_workers(const range_t cnt)noexcept {
			//TODO: need cache friendly partitioning here
			const range_t lastWrkr = cnt - 1;
			m_workingCnt = (cnt > m_workersCnt ? m_workersCnt : lastWrkr);
			const range_t totalWorkers = m_workersCnt + 1;
			range_t eachCnt = cnt / totalWorkers;
			const range_t residual = cnt % totalWorkers;
			range_t prevOfs = 0;

			for (range_t i = 0; i < m_workersCnt; ++i) {
				if (i >= lastWrkr) {
					m_ranges[i].cnt(0);
				} else {
					const range_t n = eachCnt + (i < residual ? 1 : 0);
					m_ranges[i].cnt(n).offset(prevOfs);
					prevOfs += n;
				}
			}
			return prevOfs;
		}

		static void _s_worker(Std* p, const thread_id_t id)noexcept { p->_worker(id); }

		void _worker(const thread_id_t id)noexcept {
			m_workingCnt--;
			m_orderDone.notify_one();

			while (true) {
				locker_t lk(m_lock);		
				while(!m_bStop && 0 == m_ranges[id].cnt()) m_waitingOrders.wait(lk);

				lk.unlock();
				if (m_bStop) break;

				auto& thrdRange = m_ranges[id];
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

				lk.lock();
				thrdRange.cnt(0);
				m_workingCnt--;
				m_orderDone.notify_one();
				lk.unlock();

				/*
				 *wrong (only one thread at a time will process data), but working
				 *
				 *if (m_bStop) {
					lk.unlock();
					break;
				}

				auto& thrdRange = m_ranges[id];
				switch (m_jobType) {
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

				m_workingCnt--;
				m_orderDone.notify_one();
				lk.unlock();*/
			}
		}
	};

}
}

