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

// one could use standard primitives from STL, however they have quite restrictive API and there almost
// always exist a native OS primitive that works faster. And it may matter a lot.
// This file should help to isolate NNTL from OS thread synchronization differences
// 

#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>

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

	template<class MutexT, class = ::std::void_t<>>
	struct has_lock_shared : ::std::false_type {};
	template<typename MutexT>
	struct has_lock_shared<MutexT, ::std::void_t<decltype(::std::declval<MutexT>().lock_shared())>> : ::std::true_type {};

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

		public:
			~win_condition_var()noexcept {}
			win_condition_var()noexcept {
				::InitializeConditionVariable(&m_var);
			}

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

			template<typename Predicate>
			void wait(win_srwlock& l, Predicate&& pred)noexcept {
				while (!(::std::forward<Predicate>(pred))()) {
					wait(l);
				}
			}
			template<typename Predicate>
			void wait_shared(win_srwlock& l, Predicate&& pred)noexcept {
				while (!(::std::forward<Predicate>(pred))()) {
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

}
}