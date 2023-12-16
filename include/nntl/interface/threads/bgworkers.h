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

#include <nntl/interface/threads/_i_bgworkers.h>
#include <vector>
#include <algorithm>


namespace nntl {
namespace threads {

	namespace _impl {
		template<typename FuncT>
		struct TaskDescr {
			typedef FuncT func_task_t;

			func_task_t task;
			int priority;

			TaskDescr(const func_task_t&t, int p)noexcept : task(t), priority(p) {}
			TaskDescr(func_task_t&&t, int p)noexcept : task(::std::move(t)), priority(p) {}
		};

		struct TaskDescrComp_t {
			template<typename FuncT>
			constexpr bool operator()(const TaskDescr<FuncT>& lhs, const TaskDescr<FuncT>& rhs) const {
				return lhs.priority > rhs.priority;
			}
		};

		template<typename MtxT>
		struct disperse_locker : public ::std::unique_lock<MtxT> {
		private:
			typedef ::std::unique_lock<MtxT> _base_class_t;
		public:
			::std::atomic<bool>& bDisperser;

			disperse_locker(MtxT& m, ::std::atomic<bool>& d) : _base_class_t(m, ::std::defer_lock), bDisperser(d) {
				lock();
			}
			~disperse_locker()noexcept {
				unlock();
			}

			void lock()noexcept {
				bDisperser = true;
				_base_class_t::lock();
			}
			void unlock()noexcept {
				bDisperser = false;
				_base_class_t::unlock();
			}

		};
	}

	template <typename SyncT = threads::sync_primitives
		, typename CallHandlerT = utils::cmcforwarderWrapper<2 * sizeof(void*)>
	>
	class BgWorkers : public _i_bgworkers {
		//!! copy constructor not needed
		BgWorkers(const BgWorkers& other)noexcept = delete;
		BgWorkers(BgWorkers&& other)noexcept = delete;
		//!!assignment is not needed
		BgWorkers& operator=(const BgWorkers& rhs) noexcept = delete;

	private:
		typedef BgWorkers self_t;

	public:
		typedef CallHandlerT CallH_t;
		typedef SyncT Sync_t;

	protected:
		typedef typename CallH_t::template call_tpl<bool(const thread_id_t tId)> func_task_t;
		typedef typename CallH_t::template call_tpl<void(const thread_id_t tId)> func_exec_t;

		typedef typename Sync_t::shared_mutex_t shared_mutex_t;
		//typedef typename Sync_t::cond_var_t cond_var_t;
		typedef typename Sync_t::cond_var_any_t cond_var_any_t;

		typedef _impl::disperse_locker<shared_mutex_t> disperse_locker_t;

		typedef _impl::TaskDescr<func_task_t> TaskDescr_t;
		
		typedef ::std::vector<TaskDescr_t> TaskSet_t;

	public:
		typedef ::std::vector<::std::thread> threads_cont_t;
		typedef threads_cont_t::iterator ThreadObjIterator_t;
		
		//////////////////////////////////////////////////////////////////////////
		//Members
	protected:
		shared_mutex_t m_mutexTasks;

		cond_var_any_t m_waitingOrders;
		cond_var_any_t m_orderDone;

		::std::atomic<bool> m_bStop;
		::std::atomic<bool> m_bGo2Waiting;

		::std::chrono::milliseconds m_taskWaitTO{ 10 };

		TaskSet_t m_tasks;

		threads_cont_t m_threads;
		::std::atomic_ptrdiff_t m_workingCnt;

		::std::vector<char> m_threadHasSmth2Exec;
		func_exec_t m_execFn;


	private:
		void _ctor(const thread_id_t nThreads, const PriorityClass pc)noexcept {
			NNTL_ASSERT(nThreads);
			m_workingCnt = nThreads;
			m_bGo2Waiting = false;
			m_bStop = false;
			//m_taskWaitTO = 250;

			m_threadHasSmth2Exec.resize(nThreads, char(false));
			m_threads.reserve(nThreads);
			m_mutexTasks.lock();
			for (thread_id_t i = 0; i < nThreads; ++i) {
				m_threads.emplace_back(_s_worker, this, i);
			}
			m_mutexTasks.unlock();

			{
				::std::unique_lock<decltype(m_mutexTasks)> lk(m_mutexTasks);
				m_orderDone.wait(lk, [&wc = m_workingCnt]() throw() {return wc <= 0; });
			}

			if (PriorityClass::threads_priority_no_change != pc) {
				const auto b = Funcs::ChangeThreadsPriorities(*this, pc);
				NNTL_ASSERT(b);
			}
		}

	public:
		~BgWorkers()noexcept {
			m_bStop = true;
			m_bGo2Waiting = true;

			//m_mutexTasks.lock();
			m_waitingOrders.notify_all();
		//	m_mutexTasks.unlock();

			for (auto& t : m_threads)  t.join();
		}

		BgWorkers(const thread_id_t nThreads
			, const PriorityClass pc = PriorityClass::threads_priority_below_current)noexcept 
		{
			_ctor(nThreads, pc);
		}
		BgWorkers(const PriorityClass pc = PriorityClass::threads_priority_below_current)noexcept
		{
			_ctor(::std::thread::hardware_concurrency() - 1, pc);
		}

		thread_id_t workers_count()noexcept {
			return static_cast<thread_id_t>(m_threads.size());
		}
		auto get_worker_threads(thread_id_t& threadsCnt)noexcept ->ThreadObjIterator_t {
			threadsCnt = workers_count();
			return m_threads.begin();
		}

		self_t& expect_tasks_count(const unsigned expectedTasksCnt)noexcept {
			NNTL_ASSERT(expectedTasksCnt);
			if (expectedTasksCnt) {
				disperse_locker_t d(m_mutexTasks, m_bGo2Waiting);
				m_tasks.reserve(expectedTasksCnt);
			}
			return *this;
		}

		template<class Rep, class Period>
		self_t& set_task_wait_timeout(const ::std::chrono::duration<Rep, Period>& to)noexcept {
			NNTL_ASSERT(to >= ::std::chrono::milliseconds(1));
			disperse_locker_t d(m_mutexTasks, m_bGo2Waiting);
			m_taskWaitTO = ::std::chrono::duration_cast<decltype(m_taskWaitTO)>(to);
			if (!m_taskWaitTO.count()) m_taskWaitTO = ::std::chrono::milliseconds(1);
			return *this;
		}

		size_t tasks_count()const noexcept {
			return m_tasks.size(); //should be safe to call from the main thread without mutex
		}

		/*template<typename T> static void S() {
			static_assert(false, "you've requested type of T, see below");
		};*/

		//in general, func must have a duration greater than duration of the task
		template<typename FTask>
		self_t& add_task(FTask&& func, int priority = 0) noexcept {
			typedef decltype(CallH_t::wrap<FTask>(::std::forward<FTask>(func))) stored_f_t;
			
			//S<decltype(CallH_t::wrap<FTask>(func))>();
			//S<decltype(CallH_t::wrap<FTask>(::std::forward<FTask>(func)))>();

			static_assert(
				::std::is_lvalue_reference<stored_f_t>::value
				|| (::std::is_class<stored_f_t>::value
				&& utils::is_specialization_of<stored_f_t, ::std::reference_wrapper>::value)
				, "func representation can be rvalue only if it is a ::std::reference_wrapper"
				);

			disperse_locker_t d(m_mutexTasks, m_bGo2Waiting);
			m_tasks.emplace_back(CallH_t::wrap<FTask>(::std::forward<FTask>(func)), priority);
			::std::sort(m_tasks.begin(), m_tasks.end(), _impl::TaskDescrComp_t());
			return *this;
		}

		self_t& delete_tasks()noexcept {
			disperse_locker_t d(m_mutexTasks, m_bGo2Waiting);
			m_tasks.clear();
			return *this;
		}
		
		//never call recursively or from non-main thread
		template<typename FExec>
		self_t& exec(FExec&& func) noexcept {
			m_bGo2Waiting = true;
			m_mutexTasks.lock();

			m_execFn = CallH_t::wrap<FExec>(::std::forward<FExec>(func));
			for (auto& e : m_threadHasSmth2Exec) e = char(true);
			m_workingCnt = m_threadHasSmth2Exec.size();

			m_bGo2Waiting = false;
			m_mutexTasks.unlock();

			Sync_t::lock_wait_unlock(m_mutexTasks, m_orderDone, [&wc = m_workingCnt]() {return wc <= 0; });

			m_execFn.reset();//shouldn't harm when done here while outside of mutex
			return *this;
		}

	protected:

		static void _s_worker(BgWorkers* p, const thread_id_t tId)noexcept {
			const auto b = threads::Funcs::AllowCurrentThreadPriorityBoost(false);
			NNTL_ASSERT(b);
			global_denormalized_floats_mode();
			p->_worker(tId);
		}

		void _worker(const thread_id_t id)noexcept {
			m_mutexTasks.lock_shared();
			m_workingCnt--;
			m_orderDone.notify_one();
			m_mutexTasks.unlock_shared();

			auto& bHas2Exec = m_threadHasSmth2Exec[id];
			while (true) {
				::std::atomic_thread_fence(::std::memory_order_acquire);//for m_taskWaitTO usage before mutex acquisition.
				const auto sleepUntil = ::std::chrono::steady_clock::now() + m_taskWaitTO;
				Sync_t::lockShared_waitFor_unlock(m_mutexTasks, m_waitingOrders, m_taskWaitTO
					, [&bStop = m_bStop, &bGo2Waiting = m_bGo2Waiting
					, &tasks = m_tasks, &utime = sleepUntil, &h2e = bHas2Exec]()noexcept
				{
					return bStop || (::std::chrono::steady_clock::now() > utime && !bGo2Waiting && tasks.size() > 0) || h2e;
				});
				if (m_bStop) break;

				if (bHas2Exec) {
					m_execFn(id);

					m_mutexTasks.lock_shared();
					
					bHas2Exec = char(false);
					m_workingCnt--;
					m_orderDone.notify_one();

					m_mutexTasks.unlock_shared();
				} else {
					m_mutexTasks.lock_shared();

					auto itCur = m_tasks.cbegin();
					const auto itLast = m_tasks.cend();
					while (!m_bGo2Waiting && itCur != itLast) {
						const auto& fn = (*itCur++).task;
						while (!m_bGo2Waiting && fn(id)) {}
					}

					m_mutexTasks.unlock_shared();
				}
			}
		}
	};

}
}