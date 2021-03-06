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

#ifdef _WIN32_WINNT
#include <windows.h>
#else
#pragma message("prioritize_workers class is implemented only for Windows platform. Implement it for your OS or the dummy/empty class will be used instead.")
#endif

#include "../_i_threads.h"

namespace nntl {
namespace threads {

	enum class PriorityClass {
		Normal = 0
		, Working
		, PerfTesting
		, global_last

		, threads_priority_first
		, threads_priority_no_change
		, threads_priority_below_current
		, threads_priority_below_current2
		, threads_priority_last
	};

	namespace _impl {

		template<PriorityClass _mode, typename iThreadsT>
		class prioritize_workers_dummy {
		public:
			~prioritize_workers_dummy()noexcept {}
			prioritize_workers_dummy(iThreadsT& iT, const bool bAllThreads = true)noexcept {
				NNTL_UNREF(iT); NNTL_UNREF(bAllThreads);
				//STDCOUTL("*** prioritize_workers not implemented for this OS, using dummy class");
			}
		};
	}

#ifdef _WIN32_WINNT

	namespace _impl {

		struct Funcs {
			static bool AllowCurrentThreadPriorityBoost(bool bAllow)noexcept {
				return TRUE == ::SetThreadPriorityBoost(::GetCurrentThread(), static_cast<BOOL>(!bAllow));
			}

			template<typename ThreadObjT>
			static bool ChangeThreadsPriorities(ThreadObjT& iT, const PriorityClass pc)noexcept {
				NNTL_ASSERT(pc > PriorityClass::threads_priority_first && pc < PriorityClass::threads_priority_last);

				const auto origThreadPriority = ::GetThreadPriority(::GetCurrentThread());
				const int newPriority = origThreadPriority -
					(pc == PriorityClass::threads_priority_below_current ? 1 :
					(pc == PriorityClass::threads_priority_below_current2 ? 2 : 0));

				thread_id_t cnt, i = 0;
				ThreadObjT::ThreadObjIterator_t head = iT.get_worker_threads(cnt);
				for (; i < cnt; ++i) {
					if (!::SetThreadPriority(head->native_handle(), newPriority)) {
						STDCOUTL("***Failed to set thread priority for thread #" << i);
						head = iT.get_worker_threads(cnt);
						for (thread_id_t j = 0; j < i; ++j) {
							if (!::SetThreadPriority(head->native_handle(), origThreadPriority))
								STDCOUTL("***Failed to restore original thread priority for thread #" << j);
							head++;
						}
						break;
					}
					head++;
				}
				return i == cnt;
			}
		};

		
		template<PriorityClass _mode, typename iThreadsT>
		class prioritize_workers_win {
		public:
			typedef iThreadsT iThreads_t;

		protected:
			const DWORD m_origPriorityClass;
			const int m_origThreadPriority;
			iThreads_t& m_iT;
			bool m_bRestore;
			const bool m_bAllThreads;

			template<PriorityClass _m> struct PrCThP {};
			template<> struct PrCThP<PriorityClass::Normal> {
				static constexpr DWORD priorityClass = NORMAL_PRIORITY_CLASS;
				static constexpr int threadPriority = THREAD_PRIORITY_NORMAL;
			};
			template<> struct PrCThP<PriorityClass::Working> {
				static constexpr DWORD priorityClass = HIGH_PRIORITY_CLASS;
				static constexpr int threadPriority = THREAD_PRIORITY_TIME_CRITICAL; // THREAD_PRIORITY_ABOVE_NORMAL;
			};
			template<> struct PrCThP<PriorityClass::PerfTesting> {
				static constexpr DWORD priorityClass = REALTIME_PRIORITY_CLASS;
				static constexpr int threadPriority = THREAD_PRIORITY_TIME_CRITICAL; //THREAD_PRIORITY_HIGHEST;//THREAD_PRIORITY_NORMAL;// THREAD_PRIORITY_LOWEST;
			};

		public:
			~prioritize_workers_win()noexcept {
				if (m_bRestore) {
					if (!::SetPriorityClass(GetCurrentProcess(), m_origPriorityClass)) STDCOUTL("***Failed to restore original priority class");
					if (!::SetThreadPriority(GetCurrentThread(),m_origThreadPriority)) STDCOUTL("***Failed to restore original thread priority for main thread");

					if (m_bAllThreads) {
						thread_id_t cnt;
						iThreads_t::ThreadObjIterator_t head = m_iT.get_worker_threads(cnt);
						for (thread_id_t i = 0; i < cnt; i++) {
							if (!::SetThreadPriority(head->native_handle(), m_origThreadPriority))
								STDCOUTL("***Failed to restore original thread priority for thread #" << i);
							head++;
						}
					}
				}
			}

			prioritize_workers_win(iThreads_t& iT, const bool bAllThreads=true)noexcept : m_iT(iT),
				m_origThreadPriority(::GetThreadPriority(::GetCurrentThread())),
				m_origPriorityClass(::GetPriorityClass(::GetCurrentProcess())),
				m_bRestore(false), m_bAllThreads(bAllThreads)
			{
				NNTL_ASSERT(m_origPriorityClass);
				NNTL_ASSERT(THREAD_PRIORITY_ERROR_RETURN != m_origThreadPriority);
				_apply();
			}

		protected:
			/*template<PriorityClass m = _mode>
			::std::enable_if_t < (m > PriorityClass::global_last)> _apply()noexcept {
				NNTL_ASSERT(m_bAllThreads);
				m_bRestore = false;
				const int newPriority = m_origThreadPriority - 
					(m == PriorityClass::threads_priority_below_current ? 1 :
					(m == PriorityClass::threads_priority_below_current2 ? 2 : 0));

				iThreads_t::thread_id_t cnt, i = 0;
				iThreads_t::ThreadObjIterator_t head = m_iT.get_worker_threads(cnt);
				for (; i < cnt; ++i) {
					if (!::SetThreadPriority(head->native_handle(), newPriority)) {
						STDCOUTL("***Failed to set thread priority for thread #" << i);
						head = m_iT.get_worker_threads(cnt);
						for (iThreads_t::thread_id_t j = 0; j < i; ++j) {
							if (!::SetThreadPriority(head->native_handle(), m_origThreadPriority))
								STDCOUTL("***Failed to restore original thread priority for thread #" << j);
							head++;
						}
						break;
					}
					head++;
				}
			}*/

			template<PriorityClass m = _mode>
			::std::enable_if_t < (m < PriorityClass::global_last)> _apply()noexcept {
				if (m_origPriorityClass && THREAD_PRIORITY_ERROR_RETURN != m_origThreadPriority) {
					typedef PrCThP<_mode> PriorityData;

					if (::SetPriorityClass(::GetCurrentProcess(), PriorityData::priorityClass)) {
						bool bMainThreadPrioritySet = false;
						if (::SetThreadPriority(::GetCurrentThread(), PriorityData::threadPriority)) {
							bMainThreadPrioritySet = true;

							if (m_bAllThreads) {
								thread_id_t cnt, i = 0;
								iThreads_t::ThreadObjIterator_t head = m_iT.get_worker_threads(cnt);
								for (; i < cnt; ++i) {
									if (!::SetThreadPriority(head->native_handle(), PriorityData::threadPriority)) {
										STDCOUTL("***Failed to set thread priority for thread #" << i);
										head = m_iT.get_worker_threads(cnt);
										for (thread_id_t j = 0; j < i; ++j) {
											if (!::SetThreadPriority(head->native_handle(), m_origThreadPriority))
												STDCOUTL("***Failed to restore original thread priority for thread #" << j);
											head++;
										}
										break;
									}
									head++;
								}
								if (i == cnt) m_bRestore = true;
							} else m_bRestore = true;
						} else {
							STDCOUTL("***Failed to set main thread priority");
						}

						if (!m_bRestore) {
							if (bMainThreadPrioritySet)
								if (!::SetThreadPriority(::GetCurrentThread(), m_origThreadPriority))
									STDCOUTL("***Failed to restore original thread priority for main thread");
							if (!::SetPriorityClass(::GetCurrentProcess(), m_origPriorityClass))
								STDCOUTL("***Failed to restore original priority class");
						}
					} else {
						STDCOUTL("***Failed to set priority class");
					}
				} else {
					STDCOUTL("****** Prioritization failed - can't get original values");
				}
			}

		};
	}
	
	template<PriorityClass mode,typename iThreads_t>
	using prioritize_workers = _impl::prioritize_workers_win<mode,iThreads_t>;

	using Funcs = _impl::Funcs;
	
#else

	template<PriorityClass mode, typename iThreads_t>
	using prioritize_workers = _impl::prioritize_workers_dummy<mode, iThreads_t>;

	struct Funcs {
		static bool AllowCurrentThreadPriorityBoost(bool bAllow)noexcept {
			return false;
			// not implemented
		}

		template<typename ThreadObjT>
		static bool ChangeThreadsPriorities(ThreadObjT& iT, const PriorityClass pc)noexcept {
			return false;
		}
	};

#pragma message( __FILE__ "[" STRING(__LINE__) "]: *** prioritize_workers was not implemented for current OS, using dummy class")

#endif

}
}