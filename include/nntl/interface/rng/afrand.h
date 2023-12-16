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

// code obsolete, use AFRand_mt


//CRandomMersenne from Agner Fog's randomc.zip package
#include "../../../_extern/agner.org/AF_randomc_h/random.h"

#include <nntl/interface/_i_rng.h>

namespace nntl {
namespace rng {

	template<typename RealT, typename AgnerFogRNG>
	class AFRand final : public rng_helper<RealT, ptrdiff_t, uint32_t, AFRand<RealT, AgnerFogRNG>> {
	public:
		typedef AgnerFogRNG base_rng_t;

	protected:
		//AFog::CRandomMersenne m_rng;
		base_rng_t m_rng;

	public:

		AFRand()noexcept : m_rng(static_cast<int>(s64to32(::std::time(0)))) {}
		AFRand(seed_t s)noexcept : m_rng(static_cast<int>(s)) {}

		void seed(seed_t s) noexcept { m_rng.RandomInit(static_cast<int>(s)); }

		// int_4_random_shuffle_t is either int on 32bits or int64 on 64bits
		int_4_random_shuffle_t gen_i(int_4_random_shuffle_t lessThan)noexcept {
			//TODO: pray we'll never need it bigger (because we'll possible do need and this may break everything)
			NNTL_ASSERT(lessThan <= INT32_MAX);
			int v = m_rng.IRandomX(0, static_cast<int>(lessThan-1));
			NNTL_ASSERT(v != AFog::GEN_ERROR);
			return static_cast<int_4_random_shuffle_t>(v);
		}

		int_4_distribution_t gen_int()noexcept { return static_cast<int_4_distribution_t>(m_rng.BRandom()); }

		//////////////////////////////////////////////////////////////////////////
		//generate FP value in range [0,1]
		//real_t gen_f_norm()noexcept { return static_cast<real_t>(m_rng.Random()); }

		static constexpr bool AFRng_has_FRandom = AFog::has_FRandom<base_rng_t>::value;
		
		template<typename T> using type_of_gen_f_norm = decltype(::std::declval<base_rng_t>().FPRandom<T>());

		template<typename T> auto gen_f_norm()noexcept { return m_rng.FPRandom<T>(); }

		//////////////////////////////////////////////////////////////////////////
		// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
		void gen_vector(real_t* ptr, const numel_cnt_t n, const real_t a)noexcept {
			//const ext_real_t scale = 2 * a;
			const real_t scale = 2 * a;
			const auto pE = ptr + n;

			typedef type_of_gen_f_norm<real_t> gen_f_norm_t;

			while (ptr != pE) {
				//*ptr++ = static_cast<real_t>(scale*(m_rng.Random() - 0.5));
				*ptr++ = scale*static_cast<real_t>(gen_f_norm<real_t>() - gen_f_norm_t(0.5));
			}
		}

		// matrix/vector generation (sequence of numbers drawn from uniform distribution in [neg,pos])
		void gen_vector(real_t* ptr, const numel_cnt_t n, const real_t neg, const real_t pos)noexcept {
			typedef type_of_gen_f_norm<real_t> gen_f_norm_t;

			const auto span = static_cast<gen_f_norm_t>(pos - neg);
			const auto rNeg = static_cast<gen_f_norm_t>(neg);
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<real_t>(gen_f_norm<real_t>()*span + rNeg);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,1]
		void gen_vector_norm(real_t* ptr, const numel_cnt_t n)noexcept {
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<real_t>(gen_f_norm<real_t>());
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,a]
		//specialization for float, as almost any other 32+bit types require more randomness than float provides
		template<typename BaseType, typename RandomT = ::std::conditional_t<
			::std::is_same<float, BaseType>::value && AFRng_has_FRandom, float ,double
		>>
		void gen_vector_gtz(BaseType* ptr, const numel_cnt_t n, const BaseType a)noexcept {
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<BaseType>(m_rng.FPRandom<RandomT>()*a);
			}
		}
	};

}
}
