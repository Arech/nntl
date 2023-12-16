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

#include "../interface/threads/_sync_primitives.h"

namespace nntl {
namespace utils {

	template<typename RealT>
	struct CircBufferRange {
	public:
		typedef RealT real_t;

		real_t* ptr1, *ptr2;
		size_t n1, n2; //length of arrays addressed by ptr1 and ptr2

		CircBufferRange()noexcept : ptr1(nullptr), ptr2(nullptr) {}

		CircBufferRange(real_t*const p, const size_t n) noexcept : ptr1(p), n1(n), ptr2(nullptr) {
			NNTL_ASSERT(p && n);
		}
		CircBufferRange(real_t*const p1, const size_t nn1, real_t*const p2, const size_t nn2) noexcept
			: ptr1(p1), n1(nn1), ptr2(p2), n2(nn2)
		{
			NNTL_ASSERT(p1 && nn1 && p2 && nn2);
		}

		bool acquired()const noexcept { return !!ptr1; }
		bool isTwoArrays()const noexcept { return !!ptr2; }
	};

	template<typename RealT, unsigned int _Scale1e6>
	class CircBufferFeeder : public CircBufferRange<RealT> {
	private:
		typedef CircBufferRange<RealT> _base_class_t;
		
	public:
		using _base_class_t::real_t;
		
		static constexpr unsigned int Scale1e6 = _Scale1e6 ? _Scale1e6 : 1000000;
		static constexpr real_t scaleCoeff = real_t(Scale1e6) / real_t(1e6);

	protected:
		mutable ::std::atomic_size_t m_n;
		const size_t maxN;

		template<unsigned int sv = Scale1e6>
		static ::std::enable_if_t<sv == 1000000, real_t> _applyScale(const real_t v)noexcept { return v; }
		template<unsigned int sv = Scale1e6>
		static ::std::enable_if_t<sv != 1000000, real_t> _applyScale(const real_t v)noexcept { return v*scaleCoeff; }

	public:
		~CircBufferFeeder()noexcept = default;
		CircBufferFeeder()noexcept: _base_class_t(), m_n(0), maxN(0) {}
		CircBufferFeeder(const _base_class_t& b) noexcept : _base_class_t(b), m_n(0), maxN((ptr1 ? n1 : 0) + (ptr2 ? n2 : 0)) {}
		CircBufferFeeder(_base_class_t&& b) noexcept : _base_class_t(::std::move(b)), m_n(0), maxN((ptr1 ? n1 : 0) + (ptr2 ? n2 : 0)) {}

		real_t next(const thread_id_t tId)const noexcept {
			NNTL_UNREF(tId);
			NNTL_ASSERT(maxN && acquired());

			const auto n = m_n++;
			NNTL_ASSERT(n < maxN);
			const real_t*const pCur = (n >= n1 ? ptr2 : ptr1);
			NNTL_ASSERT(pCur);
			if (!pCur || n >= maxN) {
				STDCOUTL(NNTL_FUNCTION << " - no more data in buffer!");
				abort();
			}
			return _applyScale(pCur[n >= n1 ? n - n1 : n]);
		}

		void shrink()noexcept {
			NNTL_ASSERT(maxN && acquired());

			const auto n = m_n.load(::std::memory_order_relaxed);
			NNTL_ASSERT(n < maxN);

			if (n>=n1) {
				NNTL_ASSERT(ptr2);
				n2 = n - n1;
			} else {
				n1 = n;
				ptr2 = nullptr;
				n2 = 0;
			}
		}
	};

	/*namespace tests {
		template<typename RealT, unsigned int _Scale1e6 = 0>
		class SeqMtxFeeder{
		public:
			typedef RealT real_t;

			static constexpr unsigned int Scale1e6 = _Scale1e6 ? _Scale1e6 : 1000000;
			static constexpr real_t scaleCoeff = real_t(Scale1e6) / real_t(1e6);

		protected:
			const real_t* pBegin;
			const size_t numEl;

			template<unsigned int sv = Scale1e6>
			static ::std::enable_if_t<sv == 1000000, real_t> _applyScale(const real_t v)noexcept { return v; }
			template<unsigned int sv = Scale1e6>
			static ::std::enable_if_t<sv != 1000000, real_t> _applyScale(const real_t v)noexcept { return v*scaleCoeff; }

		public:
			~SeqMtxFeeder()noexcept = default;
			SeqMtxFeeder(const math::smatrix<real_t>& m, const ) noexcept : pBegin(m.data()), numEl(m.numel()){}

			real_t next(const thread_id_t tId)const noexcept {
				

				return _applyScale(pCur[n >= n1 ? n - n1 : n]);
			}

		};
	}
*/

	namespace _impl {
		template<typename SyncT_or_void>
		struct _DataBufferAPI {
			static constexpr bool bUseAtomics = false;
			typedef typename SyncT_or_void::mutex_t mutex_t;
		};

		template<>
		struct _DataBufferAPI<void> {
			static constexpr bool bUseAtomics = true;
			typedef ::nntl::threads::spin_lock mutex_t;
		};
	}

	template<typename RealT, typename SyncT_or_void>
	class DataBuffer : public _impl::_DataBufferAPI<SyncT_or_void> {
	public:
		typedef RealT real_t;
		typedef CircBufferRange<RealT> BufferRange_t;

	protected:
		template<typename MtT>
		using ylock_t = threads::yieldable_lock<MtT>;

// 		template<typename MtT>
// 		using ylock_t = ::std::lock_guard<MtT>;

	protected:
		mutex_t m_bufMutex;

		real_t*const m_pBegin;
		const size_t m_total;

		size_t m_curIdx{ 0 };//index of the first initialized element. Modifiable only by a main thread.
		size_t m_afterLastValidIdx{ 0 }; //last initialized element+1, so for a full container it is either 
										 // equals to m_curIdx if m_curIdx!=0, or equals to m_total. Modifiable only by a one of bgthreads
		size_t m_afterLastClaimedIdx{ 0 };//last claimed by bgthread element+1. zero means uninitialized
		bool m_bEmpty{ true };

	public:
		~DataBuffer()noexcept {}
		DataBuffer()noexcept : m_pBegin(nullptr), m_total(0) {}
		DataBuffer(real_t*const pBuf, const size_t s) noexcept : m_pBegin(pBuf), m_total(s) {}

		//bool bFull()const noexcept { return m_curIdx == m_afterLastValidIdx && m_afterLastValidIdx; }
		//bool bEverInitialized()const noexcept { return m_afterLastValidIdx; }

		bool isInitialized()const noexcept { return !!m_pBegin; }

		bool bStateOK()const noexcept {
			const auto cur2ALV = m_curIdx <= m_afterLastValidIdx;
			const auto ALV2ALC = m_afterLastValidIdx <= m_afterLastClaimedIdx;
			const auto ALC2cur = m_afterLastClaimedIdx <= m_curIdx;
			const auto ALVok = m_afterLastValidIdx > 0;
			const auto ALCok = m_afterLastClaimedIdx > 0;
			return (
				(
					(ALVok && ALCok &&
						((cur2ALV && ALV2ALC && m_afterLastClaimedIdx <= m_total)
							|| (ALC2cur && cur2ALV && m_afterLastValidIdx <= m_total)
							|| (ALV2ALC && ALC2cur && m_curIdx < m_total))
						)
					|| (!m_curIdx && (!ALVok || !ALCok))
				) && (
					(!m_bEmpty || m_curIdx==m_afterLastValidIdx)
				)				
				);
		}

		bool TestState() noexcept {
			::std::lock_guard<decltype(m_bufMutex)> l(m_bufMutex);
			return bStateOK();
		}

		//must be called from the main thread only
		// updates nothing
		//if succeed, release_data() with same bufferrange must be called.
		BufferRange_t acquire_data(const size_t n)noexcept {
			NNTL_ASSERT(/*n && */n <= m_total);
			::std::lock_guard<decltype(m_bufMutex)> l(m_bufMutex);
			NNTL_ASSERT(bStateOK());

			if (!m_bEmpty && n) {
				NNTL_ASSERT(m_afterLastValidIdx);
				real_t* const pStart = m_pBegin + m_curIdx;
				const size_t afterLastReqIdx = m_curIdx + n;
				if (m_curIdx < m_afterLastValidIdx) {
					//the valid zone spans from [m_curIdx to m_afterLastValidIdx)
					if (afterLastReqIdx <= m_afterLastValidIdx) {
						return BufferRange_t(pStart, n);
					}
				} else {
					//valid zone spans from [m_curIdx to m_total), and from [0 to m_afterLastValidIdx)
					if (afterLastReqIdx <= m_total) {
						return BufferRange_t(pStart, n);//we're fitting in [m_curIdx to m_total)
					} else {
						//[m_curIdx to m_total) is too small for request, splitting into 2 arrays [0 to m_afterLastValidIdx)
						const size_t n1 = m_total - m_curIdx;
						const size_t n2 = n - n1;
						NNTL_ASSERT(n2);
						if (n2 <= m_afterLastValidIdx) {
							return BufferRange_t(pStart, n1, m_pBegin, n2);
						}
					}
				}
			}
			return BufferRange_t();
		}

		//Don't call if acquire_data() returned empty 
		//must be called from the main thread only
		// updates all
		void release_data(const BufferRange_t& br)noexcept {
			NNTL_ASSERT(br.acquired());

			m_bufMutex.lock();
			NNTL_ASSERT(bStateOK());

			m_curIdx = br.isTwoArrays() ? br.n2 : m_curIdx + br.n1;
			NNTL_ASSERT(m_curIdx <= m_total);
			if (m_curIdx >= m_total) {
				m_curIdx = 0;
				m_afterLastValidIdx = (m_afterLastValidIdx == m_total ? 0 : m_afterLastValidIdx);
				m_afterLastClaimedIdx = (m_afterLastClaimedIdx == m_total ? 0 : m_afterLastClaimedIdx);
			}
			m_bEmpty = (m_curIdx == m_afterLastValidIdx);

			NNTL_ASSERT(bStateOK());
			m_bufMutex.unlock();
		}

		//////////////////////////////////////////////////////////////////////////
		// the _as* functions must be called from worker threads only
		// _as_should_work(n) gets called by bgthread to request a generation of data of maximum size n.
		// If the generation should be done, returns size_t<=n. Else returns 0.
		// updates m_afterLastClaimedIdx only
		size_t _as_should_work(const size_t _n)noexcept {
			NNTL_ASSERT(_n);
			const size_t n = _n <= m_total ? _n : m_total;

			//m_bufMutex.lock();
			ylock_t<decltype(m_bufMutex)> lk(m_bufMutex);
			NNTL_ASSERT(bStateOK());

			size_t r;
			if (!m_bEmpty && ((m_afterLastClaimedIdx && m_afterLastClaimedIdx == m_curIdx) || (!m_curIdx && m_afterLastClaimedIdx == m_total))) {
				//the buffer is either already full or will been generated to its full condition soon. Nothing should be done.
				r = 0;
			} else {
				r = n;
				NNTL_ASSERT(m_curIdx != m_afterLastClaimedIdx || m_bEmpty);
				if (m_curIdx <= m_afterLastClaimedIdx) {
					if (m_afterLastClaimedIdx < m_total) {
						//zone to generate spans from [m_afterLastClaimedIdx to m_total)
						const auto res = m_total - m_afterLastClaimedIdx;
						if (res >= n) {
							//n fits to [m_afterLastClaimedIdx to m_total) entirely
							m_afterLastClaimedIdx += n;
						} else {
							//n has to be broken into 2 arrays: to be placed before m_total and between 0 and m_curIdx
							const auto remToGen = n - res;
							if (remToGen <= m_curIdx) {
								m_afterLastClaimedIdx = remToGen;
							} else {
								if (m_curIdx) {
									m_afterLastClaimedIdx = m_curIdx;
									r = res + m_curIdx;
								} else {
									m_afterLastClaimedIdx = m_total;
									r = res;
								}
							}
						}
					} else {
						//m_afterLastClaimedIdx == m_total, therefore a zone to generate spans from [0 to m_curIdx)
						if (m_curIdx) {
							//NNTL_ASSERT(m_curIdx);//m_curIdx==0 must be caught earlier in first if()
							r = m_curIdx > n ? n : m_curIdx;
							m_afterLastClaimedIdx = r;
						} else {
							//shouldn't be here??
							NNTL_ASSERT(!m_bEmpty);
							NNTL_ASSERT(!"How did I get here?");
							r = 0;
						}
					}
				} else {
					//zone to generate spans from [m_afterLastClaimedIdx to m_curIdx)
					const auto res = m_curIdx - m_afterLastClaimedIdx;
					r = res > n ? n : res;
					m_afterLastClaimedIdx += r;
				}
			}
			NNTL_ASSERT(r <= n);
			NNTL_ASSERT(m_afterLastClaimedIdx > 0);
			NNTL_ASSERT(bStateOK());
			//m_bufMutex.unlock();
			return r;
		}

		//this function is called by bgthread to merge the values of ptr[n] into the buffer
		template<bool bValidateBuffer = false>
		void _as_done(const real_t*const ptr, const size_t n)noexcept {
			NNTL_ASSERT(n && n <= m_total);

			//m_bufMutex.lock();
			ylock_t<decltype(m_bufMutex)> lk(m_bufMutex);

			NNTL_ASSERT(bStateOK());
			//NNTL_ASSERT(!m_afterLastValidIdx || m_curIdx != m_afterLastValidIdx);
			NNTL_ASSERT(m_afterLastValidIdx != m_afterLastClaimedIdx);

			//deciding where to put the data
			if (m_afterLastValidIdx < m_afterLastClaimedIdx) {
				//between [m_afterLastValidIdx, m_afterLastClaimedIdx)
				const auto bs = m_afterLastClaimedIdx - m_afterLastValidIdx;
				const auto ds = bs < n ? bs : n;
				_copy2buf<bValidateBuffer>(m_afterLastValidIdx, ptr, ds);
				m_afterLastValidIdx += ds;
			} else {
				//zone spans from [m_afterLastValidIdx to m_total) and [0, m_afterLastClaimedIdx]
				const auto bs1 = m_total - m_afterLastValidIdx;
				if (bs1 >= n) {
					//fits entirely into [m_afterLastValidIdx to m_total)
					_copy2buf<bValidateBuffer>(m_afterLastValidIdx, ptr, n);
					m_afterLastValidIdx += n;
				} else {
					//dividing in two parts
					if (bs1) {
						_copy2buf<bValidateBuffer>(m_afterLastValidIdx, ptr, bs1);
					}
					const auto ds = n - bs1;
					const auto bs2 = ds < m_afterLastClaimedIdx ? ds : m_afterLastClaimedIdx;
					m_afterLastValidIdx = bs2;
					_copy2buf<bValidateBuffer>(0, ptr + bs1, bs2);
				}
			}
			m_bEmpty = false;
			NNTL_ASSERT(m_afterLastValidIdx && m_afterLastValidIdx <= m_total);
			NNTL_ASSERT(bStateOK());

			//m_bufMutex.unlock();
		}

	protected:
		template<bool bValidateBuffer>
		::std::enable_if_t<bValidateBuffer> _copy2buf(const size_t ofs, const real_t*const ptr, const size_t n)noexcept {
			const auto b = ::std::all_of(m_pBegin + ofs, m_pBegin + ofs + n, [](const real_t e)->bool {
				return e == real_t(0.);
			});
			//you must clean the buffer content before doing release_data()
			NNTL_ASSERT(b || !"Buffer clean content check failed!");
			if (!b) {
				STDCOUTL("error #BCF!");
			}
			_copy2buf<false>(ofs, ptr, n);
		}
		template<bool bValidateBuffer>
		::std::enable_if_t<!bValidateBuffer> _copy2buf(const size_t ofs, const real_t*const ptr, const size_t n)noexcept {
			::std::memcpy(m_pBegin + ofs, ptr, n * sizeof(real_t));
		}
	};

}
}