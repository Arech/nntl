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

//CRandomMersenne from Agner Fog's randomc.zip package

#include "../../../_extern/agner.org/AF_randomc_h/random.h"

#include "../_i_rng.h"

namespace nntl {
namespace rng {

	template<typename AgnerFogRNG>
	class AFRandom final : public rng_helper<AFRandom<AgnerFogRNG>> {
	public:
		typedef AgnerFogRNG base_rng_t;

		AFRandom()noexcept : m_rng(static_cast<int>(s64to32(std::time(0)))) {}
		AFRandom(seed_t s)noexcept : m_rng(static_cast<int>(s)) {}

		void seed(seed_t s) noexcept { m_rng.RandomInit(static_cast<int>(s)); }
// 		void seed_array(const seed_t s[], unsigned seedsCnt) noexcept { 
// 			//m_rng.RandomInitByArray(static_cast<const int*>(s), static_cast<int>(seedsCnt));
// 			m_rng.RandomInitByArray( s, static_cast<int>(seedsCnt));//better to get here an error than silent convertion that will ruin everything
// 		}

		// generated_scalar_t is either int on 32bits or int64 on 64bits
		// gen_i() is going to be used with random_shuffle()
		generated_scalar_t gen_i(generated_scalar_t lessThan)noexcept {
			//TODO: pray we'll never need it bigger (because we'll possible do need and this may break everything)
			NNTL_ASSERT(lessThan <= INT32_MAX);
			int v = m_rng.IRandom(0, static_cast<int>(lessThan-1));
			NNTL_ASSERT(v != Agner_Fog::GEN_ERROR);
			return static_cast<generated_scalar_t>(v);
		}

		//////////////////////////////////////////////////////////////////////////
		//generate FP value in range [0,1]
		real_t gen_f_norm()noexcept { return static_cast<real_t>(m_rng.Random()); }

		//////////////////////////////////////////////////////////////////////////
		// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
		void gen_vector(real_t* ptr, const size_t n, const real_t a)noexcept {
			const double scale = 2 * a;
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<real_t>(scale*(m_rng.Random() - 0.5));
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,1]
		void gen_vector_norm(real_t* ptr, const size_t n)noexcept {
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = gen_f_norm();
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,a]
		template<typename BaseType>
		void gen_vector_gtz(BaseType* ptr, const size_t n, const BaseType a)noexcept {
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<BaseType>(m_rng.Random()*a);
			}
		}

	protected:
		//Agner_Fog::CRandomMersenne m_rng;
		base_rng_t m_rng;
	};

}
}