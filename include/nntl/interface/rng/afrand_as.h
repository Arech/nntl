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

//Well, it was kinda experiment, and it proved to be unsuccessful.
//The code is probably correct (though, not 100% sure and there might be bugs), but actually it is slower, than a synchronous
// version. Either should change some scheduling parameters and buffer sizes, or more likely the root cause is in cache incoherency:
// background threads permanently working with a huge memory buffer contaminates processor's cache and cause much more cache misses
// for main algorithms. Cache misses actually slows down computations significantly more than a synchronous RNG.
// Leave the code here, thought it's kinda at max "early beta" quality.


#include <nntl/interface/rng/afrand_mt.h>
#include <nntl/interface/threads/bgworkers.h>
#include <nntl/utils/data_buffer.h>

namespace nntl {
namespace rng {

	namespace as {

		namespace _impl {
			template<class T>
			struct Call_normal_distr {
				T*const ptr;

				Call_normal_distr(T*const p)noexcept:ptr(p) {}
				bool operator()(const thread_id_t t) {
					return ptr->_as_normal_distr(t);
				}
			};

			template<class T>
			struct Call_norm {
				T*const ptr;

				Call_norm(T*const p)noexcept:ptr(p) {}
				bool operator()(const thread_id_t t) {
					return ptr->_as_norm(t);
				}
			};
		}

		template<typename RealT, typename AgnerFogRNG>
		class AsynchRng : public math::smatrix_td {
		private:
			typedef AsynchRng<RealT, AgnerFogRNG> self_t;

		public:
			typedef RealT real_t;
			typedef AgnerFogRNG base_rng_t;
			typedef ::nntl::rng::_impl::AFRAND_MT_THR<base_rng_t, real_t> Thresholds_t;

		protected:
			typedef threads::BgWorkers<> bgworkers_t;

			struct BgThreadCtx {
				base_rng_t Rng;
				::std::normal_distribution<real_t> normDistr;
				real_t* pTmpMem;

				~BgThreadCtx()noexcept {
					pTmpMem = nullptr;
				}
				BgThreadCtx(const int RgSeed, real_t*const pMem, const real_t ndMean, const real_t ndStdev)noexcept
					: Rng(RgSeed), normDistr(ndMean, ndStdev), pTmpMem(pMem)
				{}
			};

			typedef ::std::vector<BgThreadCtx> ctx_vector_t;

			enum class _TaskId : size_t {
				normal_distr
				, vector_norm

				, _totalTasks
			};
			static constexpr size_t maxTasksCount = static_cast<size_t>(_TaskId::_totalTasks);

			typedef _impl::Call_normal_distr<self_t> call_normal_distr_t;
			static constexpr real_t normal_distr_mean = real_t(0.);
			static constexpr real_t normal_distr_stdev = real_t(1.);

			typedef _impl::Call_norm<self_t> call_norm_t;

			static constexpr real_t threadmemMul = real_t(20);
			static constexpr size_t thread_mem_count4_normal_distr = size_t(threadmemMul*Thresholds_t::bnd_normal_vector);
			static constexpr size_t thread_mem_count4_norm = size_t(threadmemMul*Thresholds_t::bnd_gen_vector_norm);
			//static constexpr size_t thread_mem_count4_bernoulli = Thresholds_t::bnd_bernoulli_vector;

			static constexpr real_t bufsizeMul = real_t(3);

			//static constexpr ::std::array<size_t, maxTasksCount> a_thread_mem_count{ Thresholds_t::bnd_normal_vector, Thresholds_t::bnd_gen_vector_norm };

		public:
			typedef typename bgworkers_t::CallH_t CallH_t;
			typedef typename bgworkers_t::Sync_t Sync_t;

		protected:
			typedef utils::DataBuffer<RealT,void> DataBuffer_t;//, Sync_t> DataBuffer_t;
			typedef typename DataBuffer_t::BufferRange_t BufferRange_t;

		protected:
			ctx_vector_t mas_ThreadCtx;

// 			::std::aligned_storage_t<sizeof(DataBuffer_t)> m_storage_Normal_distr;
// 			::std::aligned_storage_t<sizeof(DataBuffer_t)> m_storage_Norm;

			::std::array<::std::aligned_storage_t<sizeof(DataBuffer_t)>, maxTasksCount> ma_Storage;

			bgworkers_t m_bgThreads;

			real_t* m_pMainBuffer{ nullptr };
			
			//size_t m_bufferSize_normal_distr{ 0 }, m_bufferSize_norm{ 0 };
			::std::array<size_t, maxTasksCount> ma_bufferSize;

			call_normal_distr_t m_call_normal_distr{ this };
			call_norm_t m_call_norm{ this };

		public:
			~AsynchRng()noexcept {
				deinit_rng();

				if (mas_ThreadCtx.size()) {
					delete[] (mas_ThreadCtx[0].pTmpMem);
					mas_ThreadCtx.clear();
				}
			}
			AsynchRng()noexcept {// : m_bgThreads(threads::PriorityClass::threads_priority_no_change) {
				::std::fill(ma_bufferSize.begin(), ma_bufferSize.end(), size_t(0));
			}

		protected:

			DataBuffer_t& _get_buffer_Normal_distr()noexcept {
				//available after init_rng was called()
				NNTL_ASSERT(m_pMainBuffer && ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)]);
				//auto& r = *reinterpret_cast<DataBuffer_t*>(&m_storage_Normal_distr);
				//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
				auto& r = *reinterpret_cast<DataBuffer_t*>(&ma_Storage[static_cast<size_t>(_TaskId::normal_distr)]);
				NNTL_ASSERT(r.isInitialized());
				return r;
			}

			DataBuffer_t& _get_buffer_Norm()noexcept {
				//available after init_rng was called()
				NNTL_ASSERT(m_pMainBuffer && ma_bufferSize[static_cast<size_t>(_TaskId::vector_norm)]);
				//auto& r = *reinterpret_cast<DataBuffer_t*>(&m_storage_Norm);
				//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
				auto& r = *reinterpret_cast<DataBuffer_t*>(&ma_Storage[static_cast<size_t>(_TaskId::vector_norm)]);
				NNTL_ASSERT(r.isInitialized());
				return r;
			}

			void _construct_rngs(int s)noexcept {
				NNTL_ASSERT(s);
				const auto wc = m_bgThreads.workers_count();
				NNTL_ASSERT(wc);
				NNTL_ASSERT(!mas_ThreadCtx.size());

				//sanity check
				if (mas_ThreadCtx.size()) {
					STDCOUTL("WTF? Second initialization for " << NNTL_FUNCTION);
					abort();
				}

				//allocating memory for thread buffers
				//constexpr auto maxSize = ::std::max({ thread_mem_count4_bernoulli , thread_mem_count4_normal_distr, thread_mem_count4_norm });
				//thread_mem_count
				const size_t maxSize = ::std::max({ thread_mem_count4_normal_distr, thread_mem_count4_norm });
				real_t* pTM = ::new(::std::nothrow) real_t[wc*maxSize];
				if (!pTM) {
					NNTL_ASSERT(!"Failed to allocate internal thread memory");
					STDCOUTL("Failed to allocate internal thread memory, " << (wc*maxSize * sizeof(real_t)) << " bytes");
					abort();
				}

				mas_ThreadCtx.reserve(wc);

				for (unsigned i = 0; i < wc; ++i) {
					mas_ThreadCtx.emplace_back(s + i, pTM, normal_distr_mean, normal_distr_stdev);
					pTM += maxSize;
				}
			}

		public:
			void seed(int s, const thread_id_t syncWc) noexcept {
				NNTL_ASSERT(mas_ThreadCtx.size());
				auto& ctx = mas_ThreadCtx;
				m_bgThreads.exec([s, &ctx, syncWc](const thread_id_t tId) {
					int sd[2];
					sd[0] = static_cast<int>(s);
					sd[1] = static_cast<int>(tId + syncWc);
					ctx[tId].Rng.RandomInitByArray(sd, 2);
				});
			}

			auto& bgThreads()noexcept {
				return m_bgThreads;
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			void preinit_additive_normal_distr(const numel_cnt_t ne)noexcept {
				//m_bufferSize_normal_distr += ne;
				ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)] += ne;
			}

			void preinit_additive_norm(const numel_cnt_t ne)noexcept {
				//m_bufferSize_norm += ne;
				ma_bufferSize[static_cast<size_t>(_TaskId::vector_norm)] += ne;
			}

			bool init_rng()noexcept {
				if (m_pMainBuffer) {
					STDCOUTL("Double initialization of " << NNTL_FUNCTION);
					abort();
				}

				for (auto& b : ma_bufferSize) b *= bufsizeMul;

				const size_t totalBufferSize = ::std::accumulate(ma_bufferSize.begin(), ma_bufferSize.end(), size_t(0));
				if (!totalBufferSize) return true;

				//#todo aggregation of buffers
				//#todo preconditions on generateable thread buffers size for _as_should_work

				m_pMainBuffer = ::new(::std::nothrow) real_t[totalBufferSize];
				if (!m_pMainBuffer) return false;

				unsigned int tc{ 0 };
				real_t* ptrBuf = m_pMainBuffer;

				for (size_t i = 0; i < maxTasksCount; ++i) {
					const auto bufSize = ma_bufferSize[i];
					if (bufSize) {
						new(&ma_Storage[i]) DataBuffer_t(ptrBuf, bufSize);
						ptrBuf += bufSize;
						++tc;
					} else {
						new(&ma_Storage[i]) DataBuffer_t();
					}
				}

				m_bgThreads.expect_tasks_count(tc);
				if (ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)]) {
					m_bgThreads.add_task(m_call_normal_distr);
				}
				if (ma_bufferSize[static_cast<size_t>(_TaskId::vector_norm)]) {
					m_bgThreads.add_task(m_call_norm);
				}
				return true;
			}
			bool isInitialized()const noexcept {
				return !!m_pMainBuffer;
			}
			void deinit_rng()noexcept {
				m_bgThreads.delete_tasks();

				if (m_pMainBuffer) {
					for (auto& e : ma_Storage) {
						//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
						reinterpret_cast<DataBuffer_t*>(&e)->~DataBuffer();
					}

					delete[] m_pMainBuffer;
					m_pMainBuffer = nullptr;
				}

				::std::fill(ma_bufferSize.begin(), ma_bufferSize.end(), size_t(0));
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
		protected:
			struct rgWrapper {
				base_rng_t & rg;
				typedef decltype(::std::declval<base_rng_t>().BRandom()) result_type;
				static_assert(::std::is_unsigned<result_type>::value, "result_type must be unsigned according to UniformRandomBitGenerator concept");

				result_type operator()()noexcept { return rg.BRandom(); }
				//returns the minimum value that is returned by the generator's operator().
				static constexpr result_type min() noexcept { return ::std::numeric_limits<result_type>::min(); }
				//returns the maximum value that is returned by the generator's operator().
				static constexpr result_type max() noexcept { return ::std::numeric_limits<result_type>::max(); }
			};

			template<_TaskId _tsk>
			bool _move_data(real_t*const ptr, const size_t n)noexcept {
				static constexpr size_t tsk = static_cast<size_t>(_tsk);

				NNTL_ASSERT(m_pMainBuffer && ma_bufferSize[tsk]);
				if (!m_pMainBuffer || !ma_bufferSize[tsk])return false;

				//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
				auto& Buf = *reinterpret_cast<DataBuffer_t*>(&ma_Storage[tsk]);
				NNTL_ASSERT(Buf.isInitialized());

				const auto br = Buf.acquire_data(n);
				if (!br.acquired()) return false;

				::std::memcpy(ptr, br.ptr1, sizeof(real_t)*br.n1);
				if (br.isTwoArrays()) {
					NNTL_ASSERT(n == br.n1 + br.n2);
					::std::memcpy(ptr + br.n1, br.ptr2, sizeof(real_t)*br.n2);
				} else NNTL_ASSERT(n == br.n1);

				Buf.release_data(br);
				return true;
			}

		public:
			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			// all functions starting from _as* are called from the background threads
			bool _as_normal_distr(const thread_id_t tId)noexcept {
				auto& Buf = _get_buffer_Normal_distr();
				const auto genSize = Buf._as_should_work(thread_mem_count4_normal_distr);
				if (!genSize) return false;

				auto& Ctx = mas_ThreadCtx[tId];
				real_t*const pTmpMem = Ctx.pTmpMem;
				auto& distr = Ctx.normDistr;
				rgWrapper Rng = { Ctx.Rng };
				
				for (size_t i = 0; i < genSize; ++i) {
					pTmpMem[i] = distr(Rng);
				}

				Buf._as_done(pTmpMem, genSize);
				return true;
			}

			bool _as_norm(const thread_id_t tId)noexcept {
				auto& Buf = _get_buffer_Norm();
				const auto genSize = Buf._as_should_work(thread_mem_count4_norm);
				if (!genSize) return false;

				auto& Ctx = mas_ThreadCtx[tId];
				real_t*const pTmpMem = Ctx.pTmpMem;
				auto& rg = Ctx.Rng;

				for (size_t i = 0; i < genSize; ++i) {
					pTmpMem[i] = static_cast<real_t>(rg.Random());
				}

				Buf._as_done(pTmpMem, genSize);
				return true;
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
		protected:
			static void _apply_scaling(real_t*const ptr, const size_t n, const real_t span, const real_t ofs
				, const real_t bestSpan = real_t(1.), const real_t bestOfs = real_t(0))noexcept
			{
				const bool bHasSpan = span != bestSpan, bHasOfs = ofs != bestOfs;
				if (bHasOfs || bHasSpan) {
					if (bHasOfs && bHasSpan) {
						for (size_t i = 0; i < n; ++i) {
							const auto v = ptr[i];
							ptr[i] = span*v + ofs;
						}
					} else {
						if (bHasOfs) {
							for (size_t i = 0; i < n; ++i) {
								ptr[i] += ofs;
							}
						} else {
							for (size_t i = 0; i < n; ++i) {
								ptr[i] *= span;
							}
						}
					}
				}
			}
		public:
			bool normal_vector(real_t*const ptr, const size_t n, const real_t m, const real_t st)noexcept {
				if (!_move_data<_TaskId::normal_distr>(ptr, n)) return false;
				_apply_scaling(ptr, n, st, m, normal_distr_stdev, normal_distr_mean);
				return true;
			}

			/*BufferRange_t acquire_standard_normal_distr(const size_t n)noexcept {
				NNTL_ASSERT(m_pMainBuffer && ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)]);
				if (!m_pMainBuffer || !ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)]) return BufferRange_t();
				return _get_buffer_Normal_distr().acquire_data(n);
			}

			void release_standard_normal_distr(const BufferRange_t& br)noexcept {
				NNTL_ASSERT(m_pMainBuffer && ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)]);
				_get_buffer_Normal_distr().release_data(br);
			}
			template<unsigned int i>
			void release_standard_normal_distr_cbf(utils::CircBufferFeeder<real_t, i>& cbf)noexcept {
				NNTL_ASSERT(m_pMainBuffer && ma_bufferSize[static_cast<size_t>(_TaskId::normal_distr)]);
				cbf.shrink();
				release_standard_normal_distr(static_cast<BufferRange_t>(cbf));
			}*/

			//////////////////////////////////////////////////////////////////////////

			bool gen_vector_norm(real_t*const ptr, const size_t n)noexcept {
				return _move_data<_TaskId::vector_norm>(ptr, n);
			}
			bool gen_vector_uni(const real_t span, const real_t ofs, real_t*const ptr, const size_t n)noexcept {
				if (!_move_data<_TaskId::vector_norm>(ptr, n)) return false;
				_apply_scaling(ptr, n, span, ofs, real_t(1.), real_t(0.));
				return true;
			}
		};

	}

	template<typename FCT, typename RealT, typename AgnerFogRNG, typename iThreadsT>
	class _AFRand_as 
		: public _AFRand_mt<FCT, RealT, AgnerFogRNG, iThreadsT>
		, public as::AsynchRng<RealT, AgnerFogRNG>	
	{
		typedef _AFRand_mt<FCT, RealT, AgnerFogRNG, iThreadsT> _base_class_t;
	public:
		using _base_class_t::real_t;
		using _base_class_t::base_rng_t;
		using _base_class_t::Thresholds_t;
		using _base_class_t::rng_vector_t;

		typedef as::AsynchRng<RealT, AgnerFogRNG> asynch_rng_t;
		typedef _base_class_t mt_rng_t;

	protected:

	public:
		~_AFRand_as()noexcept {}
		_AFRand_as()noexcept : _base_class_t() {}

		_AFRand_as(iThreads_t& t)noexcept : _base_class_t(t) {
			NNTL_ASSERT(m_pThreads);
			asynch_rng_t::_construct_rngs(m_lastSeed + m_pThreads->workers_count());
		}
		_AFRand_as(iThreads_t& t, const seed_t s)noexcept : _base_class_t(t, s){
			NNTL_ASSERT(m_pThreads);
			asynch_rng_t::_construct_rngs(m_lastSeed + m_pThreads->workers_count());
		}

		bool init_ithreads(iThreads_t& t, const seed_t s = static_cast<seed_t>(s64to32(::std::time(0))))noexcept {
			const auto b = _base_class_t::init_ithreads(t, s);
			if (b) {
				NNTL_ASSERT(m_pThreads);
				asynch_rng_t::_construct_rngs(m_lastSeed + m_pThreads->workers_count());
			}
			return b;
		}

		void seed(const seed_t s) noexcept {
			_base_class_t::seed(s);
			asynch_rng_t::seed(s, m_pThreads->workers_count());
		}

		void preinit_additive_normal_distr(const numel_cnt_t ne)noexcept {
			_base_class_t::preinit_additive_normal_distr(ne);
			asynch_rng_t::preinit_additive_normal_distr(ne);
		}
		void preinit_additive_norm(const numel_cnt_t ne)noexcept {
			_base_class_t::preinit_additive_norm(ne);
			asynch_rng_t::preinit_additive_norm(ne);
		}

		bool init_rng()noexcept { 
			const bool b = _base_class_t::init_rng();
			if (b && !asynch_rng_t::init_rng()) _base_class_t::deinit_rng();
			return b;
		}
		void deinit_rng()noexcept {
			_base_class_t::deinit_rng();
			asynch_rng_t::deinit_rng();
		}


		//////////////////////////////////////////////////////////////////////////

		//int_4_random_shuffle_t gen_i(int_4_random_shuffle_t lessThan)noexcept {		}
		//int_4_distribution_t gen_int()noexcept { return static_cast<int_4_distribution_t>(m_Rngs[0].BRandom()); }

		//////////////////////////////////////////////////////////////////////////
		//generate FP value in range [0,1]
		//real_t gen_f_norm()noexcept { return static_cast<real_t>(m_Rngs[0].Random()); }

		//////////////////////////////////////////////////////////////////////////
		// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
		void gen_vector(real_t* ptr, const size_t n, const real_t a)noexcept {
			if (!asynch_rng_t::gen_vector_uni(a*real_t(2), -a, ptr, n)) {
				STDCOUT("_GVa!");
				_base_class_t::gen_vector(ptr, n, a);
			}
		}
		void gen_vector(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
			if (!asynch_rng_t::gen_vector_uni(pos - neg, neg, ptr, n)) {
				STDCOUT("_GVnp!");
				_base_class_t::gen_vector(ptr, n, neg, pos);
			}
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,1]
		void gen_vector_norm(real_t* ptr, const size_t n)noexcept {
			if (!asynch_rng_t::gen_vector_norm(ptr, n)) {
				STDCOUT("_GVN!");
				_base_class_t::gen_vector_norm(ptr, n);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,a]
		/*template<typename BaseType>
		void gen_vector_gtz(BaseType* ptr, const size_t n, const BaseType a)noexcept {
			NNTL_ASSERT(m_pThreads);
			if (n < Thresholds_t::bnd_gen_vector_gtz) {
				get_self().gen_vector_gtz_st(ptr, n, a);
			} else get_self().gen_vector_gtz_mt(ptr, n, a);
		}
		template<typename BaseType>
		void gen_vector_gtz_st(BaseType* ptr, const size_t n, const BaseType a)noexcept {
			NNTL_ASSERT(m_pThreads && ptr);
			get_self()._igen_vector_gtz_st(ptr, a, elms_range(0, n), 0);
		}

		template<typename BaseType>
		void _igen_vector_gtz_st(BaseType* ptr, const BaseType a, const elms_range& er, const thread_id_t tId)noexcept {
			NNTL_ASSERT(ptr);
			const auto pE = ptr + er.elmEnd;
			ptr += er.elmBegin;
			auto& rg = m_Rngs[tId];
			typedef decltype(rg.Random()) rgRandom_t;
			const rgRandom_t a2 = static_cast<rgRandom_t>(a);
			while (ptr != pE) {
				*ptr++ = static_cast<BaseType>(rg.Random()*a2);
			}
		}

		template<typename BaseType>
		void gen_vector_gtz_mt(BaseType* ptr, const size_t n, const BaseType a)noexcept {
			NNTL_ASSERT(m_pThreads && ptr);
			m_pThreads->run([ptr, a, this](const par_range_t&r) {
				get_self()._igen_vector_gtz_st(ptr, a, elms_range(r), r.tid());
			}, n);
		}*/

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//it works slower than if just separately generate mtx and then binarize it.
		/*void bernoulli_vector(real_t* ptr, const size_t n, const real_t p, const real_t posVal = real_t(1.), const real_t negVal = real_t(0.))noexcept {
			if (n < Thresholds_t::bnd_bernoulli_vector) {
				get_self().bernoulli_vector_st(ptr, n, p, posVal, negVal);
			} else get_self().bernoulli_vector_mt(ptr, n, p, posVal, negVal);
		}
		void bernoulli_vector_st(real_t* ptr, const size_t n, const real_t p, const real_t posVal, const real_t negVal)noexcept {
			NNTL_ASSERT(ptr);
			NNTL_ASSERT(p > real_t(0) && p < real_t(1));
			get_self()._ibernoulli_vector_st(ptr, p, posVal, negVal, elms_range(0, n), 0);
		}
		void bernoulli_vector_mt(real_t* ptr, const size_t n, const real_t p, const real_t posVal, const real_t negVal)noexcept {
			NNTL_ASSERT(ptr);
			NNTL_ASSERT(p > real_t(0) && p < real_t(1));
			m_pThreads->run([ptr, p, posVal, negVal, this](const par_range_t& r) {
				get_self()._ibernoulli_vector_st(ptr, p, posVal, negVal, elms_range(r), r.tid());
			}, n);
		}
		void _ibernoulli_vector_st(real_t* ptr, const real_t p, const real_t posVal, const real_t negVal
			, const elms_range& er, const thread_id_t tId)noexcept
		{
			NNTL_ASSERT(ptr);
			NNTL_ASSERT(p > real_t(0) && p < real_t(1));
			auto& rg = m_Rngs[tId];
			typedef decltype(rg.Random()) rgRandom_t;
			const rgRandom_t rP = static_cast<rgRandom_t>(p);
			const auto pE = ptr + er.elmEnd;
			ptr += er.elmBegin;
			while (ptr != pE) {
				*ptr++ = rg.Random() < rP ? posVal : negVal;
			}
		}*/

		/*void _ibernoulli_vector_st(real_t*const ptr, const real_t p, const real_t posVal, const real_t negVal
		, const elms_range& er, const thread_id_t tId)noexcept
		{
		NNTL_ASSERT(ptr);
		NNTL_ASSERT(p > real_t(0) && p < real_t(1));
		auto& __restrict rg = m_Rngs[tId];
		typedef decltype(rg.Random()) rgRandom_t;
		const rgRandom_t rP = static_cast<rgRandom_t>(p);

		const bool bOdd = er.totalElements() & 1;
		if (bOdd) {
		ptr[er.elmBegin] = rg.Random() < rP ? posVal : negVal;
		}
		for (auto i = er.elmBegin + bOdd; i < er.elmEnd; i+=2) {
		/ *const auto b1 = rg.Random() < rP;
		const auto b2 = rg.Random() < rP;
		ptr[i] = b1 ? posVal : negVal;
		ptr[i+1] = b2 ? posVal : negVal;* /
		const auto v1 = rg.Random();
		const auto v2 = rg.Random();
		ptr[i] = v1 < rP ? posVal : negVal;
		ptr[i + 1] = v2 < rP ? posVal : negVal;
		}
		}*/

		///////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		void normal_vector(real_t*const ptr, const size_t n, const real_t m = real_t(0.), const real_t st = real_t(1.))noexcept {
			if (!asynch_rng_t::normal_vector(ptr, n, m, st)) {
				STDCOUT("_NV!");
				_base_class_t::normal_vector(ptr, n, m, st);
			}
		}

	};

	template<typename RealT, typename AgnerFogRNG, typename iThreadsT>
	class AFRand_as final : public _AFRand_as<AFRand_as<RealT, AgnerFogRNG, iThreadsT>, RealT, AgnerFogRNG, iThreadsT> {
		typedef _AFRand_as<AFRand_as<RealT, AgnerFogRNG, iThreadsT>, RealT, AgnerFogRNG, iThreadsT> _base_class_t;
	public:
		~AFRand_as()noexcept { }
		AFRand_as()noexcept : _base_class_t() {}
		AFRand_as(iThreads_t& t)noexcept : _base_class_t(t) {}
		AFRand_as(iThreads_t& t, seed_t s)noexcept : _base_class_t(t, s) {}
	};

	

}
}
