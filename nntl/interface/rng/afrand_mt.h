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

//CRandomMersenne from Agner Fog's randomc.zip package

#include "../../../_extern/agner.org/AF_randomc_h/random.h"

#include "../_i_rng.h"
#include "../_i_threads.h"

#include "AFRAND_MT_THR.h"

namespace nntl {
	namespace rng {

		template<typename RealT, typename AgnerFogRNG, typename iThreadsT>
		class AFRand_mt final : public rng_helper<RealT, ptrdiff_t, uint32_t, AFRand_mt<RealT, AgnerFogRNG, iThreadsT>> {
			static_assert(::std::is_base_of<threads::_i_threads<RealT, typename iThreadsT::range_t>, iThreadsT>::value, "iThreads must implement threads::_i_threads");

		public:
			typedef AgnerFogRNG base_rng_t;
			typedef iThreadsT iThreads_t;
			typedef typename iThreads_t::range_t range_t;
			typedef typename iThreads_t::par_range_t par_range_t;
			typedef typename iThreads_t::thread_id_t thread_id_t;

			typedef _impl::AFRAND_MT_THR<base_rng_t,real_t> Thresholds_t;

		protected:
			typedef ::std::vector<base_rng_t> rng_vector_t;

		protected:
			iThreads_t* m_pThreads;
			rng_vector_t m_Rngs;

			//AFog::CRandomMersenne m_rng;
			//base_rng_t m_rng;
		
		protected:
			void _construct_rngs(int s)noexcept {
				NNTL_ASSERT(m_pThreads);
				const auto wc = m_pThreads->workers_count();
				//TODO: exception handling here
				m_Rngs.reserve(wc);
				for (unsigned i = 0; i < wc; ++i) {
					m_Rngs.push_back(base_rng_t(s+i));
				}
			}

		public:
			static constexpr bool is_multithreaded = true;

			AFRand_mt()noexcept:m_pThreads(nullptr) {}
			bool set_ithreads(iThreads_t& t)noexcept {
				if (m_pThreads) return false;
				m_pThreads = &t;
				_construct_rngs(static_cast<int>(s64to32(::std::time(0))));
				return true;
			}
			bool set_ithreads(iThreads_t& t, seed_t s)noexcept {
				if (m_pThreads) return false;
				m_pThreads = &t;
				_construct_rngs(static_cast<int>(s));
				return true;
			}

			iThreads_t& ithreads()noexcept { return *m_pThreads; }

			AFRand_mt(iThreads_t& t)noexcept : m_pThreads(&t) {
				_construct_rngs(static_cast<int>(s64to32(::std::time(0))));
			}
			AFRand_mt(iThreads_t& t, seed_t s)noexcept : m_pThreads(&t) {
				_construct_rngs(static_cast<int>(s));
			}

			void seed(seed_t s) noexcept {
				NNTL_ASSERT(m_pThreads);
				auto& rngs = m_Rngs;
				m_pThreads->run([s,&rngs](const par_range_t&r) {
					int sd[2];
					sd[0] = static_cast<int>(s);
					sd[1] = static_cast<int>(r.tid());
					rngs[r.tid()].RandomInitByArray(sd, 2);
				},m_pThreads->workers_count());
			}

			// int_4_random_shuffle_t is either int on 32bits or int64 on 64bits
			int_4_random_shuffle_t gen_i(int_4_random_shuffle_t lessThan)noexcept {
				NNTL_ASSERT(m_pThreads);
				//TODO: pray we'll never need it bigger (because we'll possible do need and this may break everything)
				NNTL_ASSERT(lessThan <= INT32_MAX);
				int v = m_Rngs[0].IRandomX(0, static_cast<int>(lessThan - 1));
				NNTL_ASSERT(v != AFog::GEN_ERROR);
				return static_cast<int_4_random_shuffle_t>(v);
			}

			int_4_distribution_t gen_int()noexcept { return static_cast<int_4_distribution_t>(m_Rngs[0].BRandom()); }

			//////////////////////////////////////////////////////////////////////////
			//generate FP value in range [0,1]
			real_t gen_f_norm()noexcept { return static_cast<real_t>(m_Rngs[0].Random()); }

			//////////////////////////////////////////////////////////////////////////
			// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
			void gen_vector(real_t* ptr, const size_t n, const real_t a)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector) {
					get_self().gen_vector_st(ptr, n, a);
				}else get_self().gen_vector_mt(ptr, n, a);
			}
			void gen_vector_st(real_t* ptr, const size_t n, const real_t a)noexcept {
				NNTL_ASSERT(ptr);
				get_self()._igen_vector_st(ptr, a*real_t(2), -a, elms_range(0, n), 0);
			}
			/*void _igen_vector_st(real_t* ptr, const real_t a, const elms_range& er, const thread_id_t tId)noexcept {
				NNTL_ASSERT(ptr);
				const auto scale = real_t(2) * a;
				const auto pE = ptr + er.elmEnd;
				ptr += er.elmBegin;
				auto& rg = m_Rngs[tId];
				while (ptr != pE) {
					*ptr++ = scale * (static_cast<real_t>(rg.Random()) - real_t(.5));
				}
			}*/
			void gen_vector_mt(real_t* ptr, const size_t n, const real_t a)noexcept {
				NNTL_ASSERT(m_pThreads);
				m_pThreads->run([ptr, span = a*real_t(2), ofs = -a, this](const par_range_t&r) {
					get_self()._igen_vector_st(ptr, span, ofs, elms_range(r), r.tid());
				}, n);
			}

			// matrix/vector generation (sequence of numbers drawn from uniform distribution in [neg,pos])
			void gen_vector(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector) {
					get_self().gen_vector_st(ptr, n, neg,pos);
				} else get_self().gen_vector_mt(ptr, n, neg, pos);
			}
			void gen_vector_st(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
				NNTL_ASSERT(m_pThreads);
				get_self()._igen_vector_st(ptr, pos-neg, neg, elms_range(0, n), 0);
			}
			void _igen_vector_st(real_t* ptr, const real_t span, const real_t ofs, const elms_range& er, const thread_id_t tId)noexcept {
				NNTL_ASSERT(ptr);
				const auto pE = ptr + er.elmEnd;
				ptr += er.elmBegin;
				auto& rg = m_Rngs[tId];
				typedef decltype(rg.Random()) rgRandom_t;
				const rgRandom_t rSpan = static_cast<rgRandom_t>(span);
				while (ptr != pE) {
					*ptr++ = static_cast<real_t>(rg.Random()*rSpan) + ofs;
				}
			}
			void gen_vector_mt(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
				NNTL_ASSERT(m_pThreads && ptr);
				m_pThreads->run([ptr, span = pos - neg, ofs = neg, this](const par_range_t&r) {
					get_self()._igen_vector_st(ptr, span, ofs, elms_range(r), r.tid());
				}, n);
			}
			
			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			//generate vector with values in range [0,1]
			void gen_vector_norm(real_t* ptr, const size_t n)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector_norm) {
					get_self().gen_vector_norm_st(ptr, n);
				} else get_self().gen_vector_norm_mt(ptr, n);
			}
			void gen_vector_norm_st(real_t* ptr, const size_t n)noexcept {
				NNTL_ASSERT(m_pThreads);
				get_self()._igen_vector_norm_st(ptr, elms_range(0, n), 0);
			}
			void _igen_vector_norm_st(real_t* ptr, const elms_range& er, const thread_id_t tId)noexcept {
				NNTL_ASSERT(ptr);
				auto& rg = m_Rngs[tId];
				const auto pE = ptr + er.elmEnd;
				ptr += er.elmBegin;
				while (ptr != pE) {
					*ptr++ = static_cast<real_t>(rg.Random());
				}
			}
			void gen_vector_norm_mt(real_t* ptr, const size_t n)noexcept {
				NNTL_ASSERT(m_pThreads && ptr);
				m_pThreads->run([ptr, this](const par_range_t&r) {
					get_self()._igen_vector_norm_st(ptr, elms_range(r), r.tid());
				}, n);
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			//generate vector with values in range [0,a]
			template<typename BaseType>
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
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			//it works slower than if just separately generate mtx and then binarize it.
			void bernoulli_vector(real_t* ptr, const size_t n, const real_t p, const real_t posVal = real_t(1.), const real_t negVal = real_t(0.))noexcept {
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
			}

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

		};

	}
}
