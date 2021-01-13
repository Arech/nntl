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

// one could use standard primitives from STL, however they have quite restrictive API and there almost
// always exist a native OS primitive that works faster. And it may matter a lot.
// This file should help to isolate NNTL from OS thread synchronization differences
// 

#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>

#include <chrono>

#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)

#pragma message "native SRW Locks and conditional variables not available, using STL primitives instead"
#define NNTL_HAS_NATIVE_SRWLOCKS_AND_CODITIONALS 0

#else // !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)

#define NNTL_HAS_NATIVE_SRWLOCKS_AND_CODITIONALS 1
#include <windows.h>

#endif//else !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)



namespace nntl {
namespace threads {

	template<class CondVarT, class MutexT, class = ::std::void_t<>>
	struct has_wait_shared : ::std::false_type {};
	template<typename CondVarT, typename MutexT>
	struct has_wait_shared<CondVarT, MutexT, ::std::void_t<decltype(::std::declval<CondVarT>().wait_shared(::std::declval<MutexT&>()))>>
		: ::std::true_type {};

	template<class CondVarT, class MutexT, class Rep, class Period, class = ::std::void_t<>>
	struct has_wait_for_shared : ::std::false_type {};
	template<typename CondVarT, typename MutexT, class Rep, class Period>
	struct has_wait_for_shared<CondVarT, MutexT, Rep, Period, ::std::void_t<decltype(::std::declval<CondVarT>()
		.wait_for_shared(::std::declval<MutexT&>(), ::std::declval<const ::std::chrono::duration<Rep,Period>&>()))>>
		: ::std::true_type {};

	template<class MutexT, class = ::std::void_t<>>
	struct has_lock_shared : ::std::false_type {};
	template<typename MutexT>
	struct has_lock_shared<MutexT, ::std::void_t<decltype(::std::declval<MutexT>().lock_shared())>> : ::std::true_type {};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//#todo: triple check it, might need another memory ordering
	//#WARNING! It seems that it is almost the worst possible implementation of spinlock.
	//https://stackoverflow.com/questions/26583433/c11-implementation-of-spinlock-using-atomic#comment47623821_26583433
	// "Among other problems: 1) When you finally do acquire the lock, you take the mother of all mispredicted branches
	// leaving the while loop, which is the worst possible time for such a thing. 2) The lock function can starve another
	// thread running in the same virtual core on a hyper-threaded CPU."
	//and btw, why it's derived from ::std::atomic_flag ???
	//better use #include <boost/smart_ptr/detail/spinlock.hpp>
	struct spin_lock : public ::std::atomic_flag {
	protected:
		::std::atomic_flag m_lock{ ATOMIC_FLAG_INIT };

	public:
		spin_lock()noexcept {
			//m_lock = ATOMIC_FLAG_INIT;
		}
		spin_lock(const spin_lock&) = delete;
		spin_lock& operator=(const spin_lock&) = delete;

		//#consider not sure we need atomic_thread_fence() there

		void lock()noexcept {
			while (m_lock.test_and_set(::std::memory_order_acquire)) {}
			::std::atomic_thread_fence(::std::memory_order_acquire);
		}
		void lock_yield()noexcept {
			while (m_lock.test_and_set(::std::memory_order_acquire)) {
				::std::this_thread::yield();
			}
			::std::atomic_thread_fence(::std::memory_order_acquire);
		}
		void unlock()noexcept {
			::std::atomic_thread_fence(::std::memory_order_release);
			m_lock.clear(::std::memory_order_release);
		}
	};

	/*inline void spin_lock::lock()noexcept {
		while (test_and_set(::std::memory_order_acquire)) {}
	}
	inline void spin_lock::lock_yield()noexcept {
		while (test_and_set(::std::memory_order_acquire)) {
			::std::this_thread::yield();
		}
	}
	inline void spin_lock::unlock()noexcept {
		clear(::std::memory_order_release);
	}*/
	
	//taken from ::std::lock_guard
	class lock_guard_yield {	// specialization for a single mutex
	public:
		typedef spin_lock mutex_type;

		explicit lock_guard_yield(spin_lock& _Mtx)
			: _MyMutex(_Mtx)
		{	// construct and lock
			_MyMutex.lock_yield();
		}

		lock_guard_yield(spin_lock& _Mtx, ::std::adopt_lock_t)
			: _MyMutex(_Mtx)
		{	// construct but don't lock
		}

		~lock_guard_yield() noexcept
		{	// unlock
			_MyMutex.unlock();
		}

		lock_guard_yield(const lock_guard_yield&) = delete;
		lock_guard_yield& operator=(const lock_guard_yield&) = delete;
	protected:
		spin_lock& _MyMutex;
	};

	template<class _Mutex>
	using yieldable_lock = ::std::conditional_t <
		::std::is_same<_Mutex, spin_lock>::value
		, lock_guard_yield
		, ::std::lock_guard<_Mutex>
	>;

	//////////////////////////////////////////////////////////////////////////

	namespace _impl {

#if NNTL_HAS_NATIVE_SRWLOCKS_AND_CODITIONALS

		//////////////////////////////////////////////////////////////////////////
		//define a STL-style wrappers for locks and conditionals
		// #todo ::std::atomic_thread_fence() calls might be redundant here, but need proofs for it...
		struct win_srwlock {
		private:
			//!! copy constructor not needed
			win_srwlock(const win_srwlock& other)noexcept = delete;
			win_srwlock(win_srwlock&& other)noexcept = delete;
			//!!assignment is not needed
			win_srwlock& operator=(const win_srwlock& rhs) noexcept = delete;

		public:
			typedef ::SRWLOCK lock_t;
		protected:
			lock_t m_srwlock;

		public:
			~win_srwlock()noexcept {}
			win_srwlock()noexcept : m_srwlock(SRWLOCK_INIT) {}
			void lock()noexcept {
				::AcquireSRWLockExclusive(&m_srwlock);
				::std::atomic_thread_fence(::std::memory_order_acquire);
			}
			void lock_shared()noexcept {
				::AcquireSRWLockShared(&m_srwlock);
				::std::atomic_thread_fence(::std::memory_order_acquire);
			}

			void unlock()noexcept {
				::std::atomic_thread_fence(::std::memory_order_release);
				::ReleaseSRWLockExclusive(&m_srwlock);
			}
			void unlock_shared()noexcept {
				::std::atomic_thread_fence(::std::memory_order_release);
				::ReleaseSRWLockShared(&m_srwlock);
			}

			operator lock_t()noexcept { return m_srwlock; }
			operator lock_t*()noexcept { return &m_srwlock; }
		};

		struct win_condition_var {
		private:
			//!! copy constructor not needed
			win_condition_var(const win_condition_var& other)noexcept = delete;
			win_condition_var(win_condition_var&& other)noexcept = delete;
			//!!assignment is not needed
			win_condition_var& operator=(const win_condition_var& rhs) noexcept = delete;

		protected:
			::CONDITION_VARIABLE m_var;

		protected:
			template< class Rep, class Period >
			DWORD _to_ms(const ::std::chrono::duration<Rep, Period>& t)noexcept {
				NNTL_ASSERT(t >= ::std::chrono::milliseconds(1));
				return static_cast<DWORD>((::std::chrono::duration_cast<::std::chrono::milliseconds>(t)).count());
			}
			DWORD _to_ms(const ::std::chrono::milliseconds& t)noexcept {
				NNTL_ASSERT(t >= ::std::chrono::milliseconds(1));
				return static_cast<DWORD>(t.count());
			}

		public:
			~win_condition_var()noexcept {}
			win_condition_var()noexcept {
				::InitializeConditionVariable(&m_var);
			}

			//////////////////////////////////////////////////////////////////////////
			void wait(win_srwlock& l)noexcept {
				::SleepConditionVariableSRW(&m_var, l, INFINITE, 0);
				::std::atomic_thread_fence(::std::memory_order_acquire);
			}
			void wait_shared(win_srwlock& l)noexcept {
				::SleepConditionVariableSRW(&m_var, l, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED);
				::std::atomic_thread_fence(::std::memory_order_acquire);
			}

			void wait(::std::unique_lock<win_srwlock>& l)noexcept { wait(*l.mutex()); }
			void wait(::std::shared_lock<win_srwlock>& l)noexcept { wait_shared(*l.mutex()); }

			//////////////////////////////////////////////////////////////////////////
			template<typename Predicate>
			void wait(win_srwlock& l, Predicate&& pred)noexcept {
				//while (!((::std::forward<Predicate>(pred))())) {
				while (!(pred())) {
					wait(l);
				}
			}
			template<typename Predicate>
			void wait_shared(win_srwlock& l, Predicate&& pred)noexcept {
				//while (!((::std::forward<Predicate>(pred))())) {
				while (!(pred())) {
					wait_shared(l);
				}
			}

			template<typename Predicate>
			void wait(::std::unique_lock<win_srwlock>& l, Predicate&& pred)noexcept {
				wait(*l.mutex(), ::std::forward<Predicate>(pred));
			}
			template<typename Predicate>
			void wait(::std::shared_lock<win_srwlock>& l, Predicate&& pred)noexcept {
				wait_shared(*l.mutex(), ::std::forward<Predicate>(pred));
			}

			//////////////////////////////////////////////////////////////////////////
			template< class Rep, class Period >
			::std::cv_status wait_for(win_srwlock& l, const ::std::chrono::duration<Rep, Period>& rel_time) {
				auto sleepUntil = ::std::chrono::steady_clock::now() + rel_time;
				::SleepConditionVariableSRW(&m_var, l, _to_ms(rel_time), 0);
				::std::atomic_thread_fence(::std::memory_order_acquire);
				return sleepUntil > ::std::chrono::steady_clock::now() ? ::std::cv_status::timeout : ::std::cv_status::no_timeout;
			}
			template< class Rep, class Period >
			::std::cv_status wait_for_shared(win_srwlock& l, const ::std::chrono::duration<Rep, Period>& rel_time) {
				auto sleepUntil = ::std::chrono::steady_clock::now() + rel_time;
				::SleepConditionVariableSRW(&m_var, l, _to_ms(rel_time), CONDITION_VARIABLE_LOCKMODE_SHARED);
				::std::atomic_thread_fence(::std::memory_order_acquire);
				return sleepUntil > ::std::chrono::steady_clock::now() ? ::std::cv_status::timeout : ::std::cv_status::no_timeout;
			}

			template< class Rep, class Period >
			::std::cv_status wait_for(::std::unique_lock<win_srwlock>& l, const ::std::chrono::duration<Rep, Period>& rel_time) {
				return wait_for(*l.mutex(), rel_time);
			}
			template< class Rep, class Period >
			::std::cv_status wait_for(::std::shared_lock<win_srwlock>& l, const ::std::chrono::duration<Rep, Period>& rel_time) {
				return wait_for_shared(*l.mutex(), rel_time);
			}
			//////////////////////////////////////////////////////////////////////////
#pragma warning(disable:4706)
			template<class Rep, class Period, typename Predicate>
			bool wait_for(win_srwlock& l, const ::std::chrono::duration<Rep, Period>& rel_time, Predicate&& pred)noexcept {
				::std::cv_status r(::std::cv_status::no_timeout);
				bool b;
				//while (!(b = ((::std::forward<Predicate>(pred))()))) {
				while (!(b = (pred()))) {
					r = wait_for(l, rel_time);
				}
				return r == ::std::cv_status::no_timeout ? true : b;
#pragma warning(default:4706)
			}
			template<class Rep, class Period, typename Predicate>
			bool wait_for(::std::unique_lock<win_srwlock>& l, const ::std::chrono::duration<Rep, Period>& rel_time, Predicate&& pred)noexcept {
				return wait_for(*l.mutex(), rel_time, ::std::forward<Predicate>(pred));
			}
#pragma warning(disable:4706)
			template<class Rep, class Period, typename Predicate>
			bool wait_for_shared(win_srwlock& l, const ::std::chrono::duration<Rep, Period>& rel_time, Predicate&& pred)noexcept {
				::std::cv_status r(::std::cv_status::no_timeout);
				bool b;
				//while (!(b = ((::std::forward<Predicate>(pred))()))) {
				while (!(b = (pred()))) {
					r = wait_for_shared(l, rel_time);
				}
				return r == ::std::cv_status::no_timeout ? true : b;
#pragma warning(default:4706)
			}
			template<class Rep, class Period, typename Predicate>
			bool wait_for(::std::shared_lock<win_srwlock>& l, const ::std::chrono::duration<Rep, Period>& rel_time, Predicate&& pred)noexcept {
				return wait_for_shared(*l.mutex(), rel_time, ::std::forward<Predicate>(pred));
			}

			//////////////////////////////////////////////////////////////////////////

			void notify_one()noexcept {
				::WakeConditionVariable(&m_var);
			}
			void notify_all()noexcept {
				::WakeAllConditionVariable(&m_var);
			}
		};

#endif

		struct _sync_base {
			template<typename MutexT, typename CondVarT, typename Predicate>
			static void lock_wait_unlock(MutexT& l, CondVarT& cv, Predicate&& p)noexcept
			{
				::std::unique_lock<MutexT> lk(l);
				cv.wait(lk, ::std::forward<Predicate>(p));
			}
			//::std::unique_lock could be used with win_srwlock, however special handling works faster
			template<typename Predicate>
			static void lock_wait_unlock(win_srwlock& l, win_condition_var& cv, Predicate&& p)noexcept
			{
				l.lock();
				cv.wait(l, ::std::forward<Predicate>(p));
				l.unlock();
			}

			//////////////////////////////////////////////////////////////////////////

			template<typename MutexT, typename CondVarT, typename Predicate>
			//nntl_static_warning("locks: Using native shared mode")
			static ::std::enable_if_t<has_lock_shared<MutexT>::value && has_wait_shared<CondVarT, MutexT>::value>
			lockShared_wait_unlock(MutexT& l, CondVarT& cv, Predicate&& p)noexcept//runs in shared mode
			{
				l.lock_shared();
				cv.wait_shared(l, ::std::forward<Predicate>(p));
				l.unlock_shared();
			}

			template<typename MutexT, typename CondVarT, typename Predicate>
			//nntl_static_warning("locks: Using std shared mode")
			static ::std::enable_if_t<has_lock_shared<MutexT>::value && !has_wait_shared<CondVarT, MutexT>::value>
			lockShared_wait_unlock(MutexT& l, CondVarT& cv, Predicate&& p)noexcept//runs in shared mode
			{
				::std::shared_lock<MutexT> lk(l);
				cv.wait(lk, ::std::forward<Predicate>(p));
			}

			template<typename MutexT, typename CondVarT, typename Predicate>
			//nntl_static_warning("locks: Using exclusive mode instead of shared")
			static ::std::enable_if_t<!has_lock_shared<MutexT>::value>
			lockShared_wait_unlock(MutexT& l, CondVarT& cv, Predicate&& p)noexcept//runs in exclusive mode
			{
				::std::unique_lock<MutexT> lk(l);
				cv.wait(lk, ::std::forward<Predicate>(p));
			}

			//////////////////////////////////////////////////////////////////////////

			template<typename MutexT, typename CondVarT, class Rep, class Period, typename Predicate>
			//nntl_static_warning("locks: Using native shared mode")
			static ::std::enable_if_t<has_lock_shared<MutexT>::value && has_wait_for_shared<CondVarT, MutexT, Rep, Period>::value>
				lockShared_waitFor_unlock(MutexT& l, CondVarT& cv, const ::std::chrono::duration<Rep,Period>& rt, Predicate&& p)noexcept//runs in shared mode
			{
				l.lock_shared();
				cv.wait_for_shared(l, rt, ::std::forward<Predicate>(p));
				l.unlock_shared();
			}

			template<typename MutexT, typename CondVarT, class Rep, class Period, typename Predicate>
			//nntl_static_warning("locks: Using std shared mode")
			static ::std::enable_if_t<has_lock_shared<MutexT>::value && !has_wait_for_shared<CondVarT, MutexT, Rep, Period>::value>
				lockShared_waitFor_unlock(MutexT& l, CondVarT& cv, const ::std::chrono::duration<Rep, Period>& rt, Predicate&& p)noexcept//runs in shared mode
			{
				::std::shared_lock<MutexT> lk(l);
				cv.wait_for(lk, rt, ::std::forward<Predicate>(p));
			}

			template<typename MutexT, typename CondVarT, class Rep, class Period, typename Predicate>
			//nntl_static_warning("locks: Using exclusive mode instead of shared")
			static ::std::enable_if_t<!has_lock_shared<MutexT>::value>
				lockShared_waitFor_unlock(MutexT& l, CondVarT& cv, const ::std::chrono::duration<Rep, Period>& rt, Predicate&& p)noexcept//runs in exclusive mode
			{
				::std::unique_lock<MutexT> lk(l);
				cv.wait_for(lk, rt, ::std::forward<Predicate>(p));
			}

			//////////////////////////////////////////////////////////////////////////
			template<typename MutexT>
			//nntl_static_warning("locks: locking shared")
			static ::std::enable_if_t<has_lock_shared<MutexT>::value>
			lockShared(MutexT& l)noexcept { l.lock_shared(); }

			template<typename MutexT>
			//nntl_static_warning("locks: locking exclusive")
			static ::std::enable_if_t<!has_lock_shared<MutexT>::value>
			lockShared(MutexT& l)noexcept { l.lock(); }

			template<typename MutexT>
			//nntl_static_warning("locks: unlocking shared")
			static ::std::enable_if_t<has_lock_shared<MutexT>::value>
			unlockShared(MutexT& l)noexcept { l.unlock_shared(); }

			template<typename MutexT>
			//nntl_static_warning("locks: unlocking exclusive")
			static ::std::enable_if_t<!has_lock_shared<MutexT>::value>
			unlockShared(MutexT& l)noexcept { l.unlock(); }
		};

		struct _dummySync {
			typedef void mutex_t;
			typedef void shared_mutex_t;
			typedef void cond_var_t;
			typedef void cond_var_any_t;
		};
	}

	//pure STL
	struct stdSync : public _impl::_sync_base {
		typedef ::std::mutex mutex_t;
		typedef ::std::shared_mutex shared_mutex_t;
		typedef ::std::condition_variable cond_var_t;
		typedef ::std::condition_variable_any cond_var_any_t;
	};

#if NNTL_HAS_NATIVE_SRWLOCKS_AND_CODITIONALS
	struct winNativeSync : public _impl::_sync_base {
		typedef _impl::win_srwlock mutex_t;
		typedef _impl::win_srwlock shared_mutex_t;
		typedef _impl::win_condition_var cond_var_t;
		typedef _impl::win_condition_var cond_var_any_t;
	};
#else
	typedef void winNativeSync;
#endif

	struct sync_primitives : public ::std::conditional_t<::std::is_void<winNativeSync>::value, stdSync, winNativeSync> {
		typedef mutex_t mutex_comp_t;
		typedef cond_var_t cond_var_comp_t;
	};

	struct std_sync_primitives : public stdSync {
		typedef mutex_t mutex_comp_t;
		typedef cond_var_t cond_var_comp_t;
	};

	struct win_sync_primitives : public ::std::conditional_t < ::std::is_void<winNativeSync>::value, _impl::_dummySync, winNativeSync > {
		typedef mutex_t mutex_comp_t;
		typedef cond_var_t cond_var_comp_t;
	};

}
}