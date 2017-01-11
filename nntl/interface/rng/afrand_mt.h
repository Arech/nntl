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
		class AFRand_mt final : public rng_helper<RealT, AFRand_mt<RealT, AgnerFogRNG, iThreadsT>> {
			static_assert(std::is_base_of<threads::_i_threads<RealT, typename iThreadsT::range_t>, iThreadsT>::value, "iThreads must implement threads::_i_threads");

		public:
			typedef AgnerFogRNG base_rng_t;
			typedef iThreadsT ithreads_t;
			typedef typename ithreads_t::range_t range_t;
			typedef typename ithreads_t::par_range_t par_range_t;
			typedef typename ithreads_t::thread_id_t thread_id_t;

			typedef _impl::AFRAND_MT_THR<base_rng_t,real_t> Thresholds_t;

		protected:
			typedef std::vector<base_rng_t> rng_vector_t;

		protected:
			ithreads_t* m_pThreads;
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
			bool set_ithreads(ithreads_t& t)noexcept {
				if (m_pThreads) return false;
				m_pThreads = &t;
				_construct_rngs(static_cast<int>(s64to32(std::time(0))));
				return true;
			}
			bool set_ithreads(ithreads_t& t, seed_t s)noexcept {
				if (m_pThreads) return false;
				m_pThreads = &t;
				_construct_rngs(static_cast<int>(s));
				return true;
			}

			AFRand_mt(ithreads_t& t)noexcept : m_pThreads(&t) {
				_construct_rngs(static_cast<int>(s64to32(std::time(0))));
			}
			AFRand_mt(ithreads_t& t, seed_t s)noexcept : m_pThreads(&t) {
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
// 			void seed_array(const seed_t s[], unsigned seedsCnt) noexcept {
// 				//m_rng.RandomInitByArray(static_cast<const int*>(s), static_cast<int>(seedsCnt));
// 				m_rng.RandomInitByArray(s, static_cast<int>(seedsCnt));//better to get here an error than silent convertion that will ruin everything
// 			}

			// generated_scalar_t is either int on 32bits or int64 on 64bits
			// gen_i() is going to be used with random_shuffle()
			generated_scalar_t gen_i(generated_scalar_t lessThan)noexcept {
				NNTL_ASSERT(m_pThreads);
				//TODO: pray we'll never need it bigger (because we'll possible do need and this may break everything)
				NNTL_ASSERT(lessThan <= INT32_MAX);
				int v = m_Rngs[0].IRandomX(0, static_cast<int>(lessThan - 1));
				NNTL_ASSERT(v != AFog::GEN_ERROR);
				return static_cast<generated_scalar_t>(v);
			}

			int gen_int()noexcept { return m_Rngs[0].IRandom(min(), max()); }

			//////////////////////////////////////////////////////////////////////////
			//generate FP value in range [0,1]
			real_t gen_f_norm()noexcept { return static_cast<real_t>(m_Rngs[0].Random()); }

			//////////////////////////////////////////////////////////////////////////
			// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
			void gen_vector(real_t* ptr, const size_t n, const real_t a)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector) {
					gen_vector_st(ptr, n, a);
				}else gen_vector_mt(ptr, n, a);
			}
			void gen_vector_st(real_t* ptr, const size_t n, const real_t a)noexcept {
				NNTL_ASSERT(m_pThreads);
				const ext_real_t scale = 2 * a;
				const auto pE = ptr + n;
				while (ptr != pE) {
					*ptr++ = static_cast<real_t>(scale*(m_Rngs[0].Random() - 0.5));
				}
			}
			void gen_vector_mt(real_t* ptr, const size_t n, const real_t a)noexcept {
				NNTL_ASSERT(m_pThreads);
				const ext_real_t scale = 2 * a;
				auto& rngs = m_Rngs;
				m_pThreads->run([ptr,scale,&rngs](const par_range_t&r) {
					auto& rg = rngs[r.tid()];
					auto p = ptr + r.offset();
					const auto pE = p + r.cnt();
					while (p != pE) {
						*p++ = static_cast<real_t>(scale*(rg.Random() - 0.5));
					}
				}, n);
			}

			// matrix/vector generation (sequence of numbers drawn from uniform distribution in [neg,pos])
			void gen_vector(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector) {
					gen_vector_st(ptr, n, neg,pos);
				} else gen_vector_mt(ptr, n, neg, pos);
			}
			void gen_vector_st(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
				NNTL_ASSERT(m_pThreads);
				const auto span = static_cast<ext_real_t>(pos - neg);
				const auto rNeg = static_cast<ext_real_t>(neg);
				const auto pE = ptr + n;
				while (ptr != pE) {
					*ptr++ = static_cast<real_t>(m_Rngs[0].Random()*span + rNeg);
				}
			}
			void gen_vector_mt(real_t* ptr, const size_t n, const real_t neg, const real_t pos)noexcept {
				NNTL_ASSERT(m_pThreads);
				const auto span = static_cast<ext_real_t>(pos - neg);
				const auto rNeg = static_cast<ext_real_t>(neg);
				auto& rngs = m_Rngs;
				m_pThreads->run([ptr, span,rNeg, &rngs](const par_range_t&r) {
					auto& rg = rngs[r.tid()];
					auto p = ptr + r.offset();
					const auto pE = p + r.cnt();
					while (p != pE) {
						*p++ = static_cast<real_t>(span*rg.Random() + rNeg);
					}
				}, n);
			}
			
			//////////////////////////////////////////////////////////////////////////
			//generate vector with values in range [0,1]
			void gen_vector_norm(real_t* ptr, const size_t n)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector_norm) {
					gen_vector_norm_st(ptr, n);
				} else gen_vector_norm_mt(ptr, n);
			}
			void gen_vector_norm_st(real_t* ptr, const size_t n)noexcept {
				NNTL_ASSERT(m_pThreads);
				const auto pE = ptr + n;
				while (ptr != pE) {
					*ptr++ = gen_f_norm();
				}
			}
			void gen_vector_norm_mt(real_t* ptr, const size_t n)noexcept {
				NNTL_ASSERT(m_pThreads);
				auto& rngs = m_Rngs;
				m_pThreads->run([ptr, &rngs](const par_range_t&r) {
					auto& rg = rngs[r.tid()];
					auto p = ptr + r.offset();
					const auto pE = p + r.cnt();
					while (p != pE) {
						*p++ = static_cast<real_t>(rg.Random());
					}
				}, n);
			}

			//////////////////////////////////////////////////////////////////////////
			//generate vector with values in range [0,a]
			template<typename BaseType>
			void gen_vector_gtz(BaseType* ptr, const size_t n, const BaseType a)noexcept {
				NNTL_ASSERT(m_pThreads);
				if (n < Thresholds_t::bnd_gen_vector_gtz) {
					gen_vector_gtz_st(ptr, n, a);
				} else gen_vector_gtz_mt(ptr, n, a);
			}
			template<typename BaseType>
			void gen_vector_gtz_st(BaseType* ptr, const size_t n, const BaseType a)noexcept {
				NNTL_ASSERT(m_pThreads);
				const auto pE = ptr + n;
				while (ptr != pE) {
					*ptr++ = static_cast<BaseType>(m_Rngs[0].Random()*a);
				}
			}
			template<typename BaseType>
			void gen_vector_gtz_mt(BaseType* ptr, const size_t n, const BaseType a)noexcept {
				NNTL_ASSERT(m_pThreads);
				auto& rngs = m_Rngs;
				m_pThreads->run([ptr, a, &rngs](const par_range_t&r) {
					auto& rg = rngs[r.tid()];
					auto p = ptr + r.offset();
					const auto pE = p + r.cnt();
					while (p != pE) {
						*p++ = static_cast<BaseType>(rg.Random()*a);
					}
				}, n);
			}
		};

	}
}
