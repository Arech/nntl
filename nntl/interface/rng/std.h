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

#include <cstdlib>      // std::rand, std::srand

// this file will be included by default.
// It defines an interface to a rng generators, provided by STL.
// Don't use it, it's slower than you can expect.
#include "_i_rng.h"

namespace nntl {
namespace rng {
		
	//////////////////////////////////////////////////////////////////////////
	//defining RNG functor. Functions can be non static, because a rng can have a state
	//Don't use in production unless you are on WinXP+ and use #define _CRT_RAND_S (and even then don't use it too)
	class Std final : public rng_helper<Std> {
	protected:
		typedef unsigned int real_seed_t;
		typedef uint32_t real_rand_max_t;
	public:
		
		Std()noexcept {
			seed(static_cast<seed_t>(s64to32(std::time(0))));
		}
		Std(seed_t s)noexcept {
			seed(s);
		}

		static void seed(seed_t s) noexcept { std::srand(s); }// _64to32(s)); }
// 		static void seed_array(const seed_t s[], unsigned seedsCnt) noexcept {
// 			seed_t sv=0;
// 			for (unsigned i = 0; i < seedsCnt; ++i) sv += s[i];
// 			seed(sv);
// 		}

		//////////////////////////////////////////////////////////////////////////
		// family of generator subfunctions
		// If there was an error and RNG can't be generated, by convention return 0 and assume, it's a caller responsibility
		// to make sure the gen is ok.
		// 
		// generated_scalar_t is either int on 32bits or int64 on 64bits
		// gen_i() is going to be used with random_shuffle()
		static generated_scalar_t gen_i(generated_scalar_t lessThan)noexcept {
			//TODO: pray we'll never need it bigger (because we'll possible do need and this may break everything)
			NNTL_ASSERT(lessThan <= _rand_max());
			return static_cast<generated_scalar_t>(_rand()) % lessThan;
		}
		//generated_scalar_t operator()(generated_scalar_t lessThan)noexcept { return gen_i(lessThan); }

		//////////////////////////////////////////////////////////////////////////
		//generate FP value in range [0,a]
		//static real_t gen_f(const real_t a)noexcept { return a*gen_f_norm(); }
		//generate FP value in range [0,1]
		static real_t gen_f_norm()noexcept {
			return static_cast<real_t>(_rand()) / static_cast<real_t>(_rand_max());
		}

		//////////////////////////////////////////////////////////////////////////
		// weights generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
// 		static void gen_matrix(realmtx_t& mtx, const real_t a)noexcept {
// 			NNTL_ASSERT(!mtx.emulatesBiases());
// 			gen_vector(mtx.data(), mtx.numel(), a);
// 		}
// 		static void gen_matrix_no_bias(realmtx_t& mtx, const real_t a)noexcept {
// 			NNTL_ASSERT();
// 			NNTL_ASSERT(mtx.emulatesBiases());
// 			gen_vector(mtx.data(), mtx.numel_no_bias(), a);
// 		}
		static void gen_vector(real_t* ptr, const size_t n, const real_t a)noexcept {
			const real_t scale = 2 * a;
			const real_t rm = static_cast<real_t>(_rand_max());
			const auto pE = ptr + n;
			while (ptr!=pE){
				*ptr++ = scale*(static_cast<real_t>(_rand()) / rm - static_cast<real_t>(0.5));
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,1]
		static void gen_vector_norm(real_t* ptr, const size_t n)noexcept {
			const real_t rm = static_cast<real_t>(_rand_max());
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<real_t>(_rand()) / rm;
			}
		}
// 		//generate matrix with values in range [0,1]
// 		static void gen_matrix_norm(realmtx_t& mtx)noexcept {
// 			NNTL_ASSERT(!mtx.emulatesBiases());
// 			gen_vector_norm(mtx.data(), mtx.numel());
// 		}
// 		static void gen_matrix_no_bias_norm(realmtx_t& mtx)noexcept {
// 			NNTL_ASSERT();
// 			NNTL_ASSERT(mtx.emulatesBiases());
// 			gen_vector_norm(mtx.data(), mtx.numel_no_bias());
// 		}
// 
// 		//////////////////////////////////////////////////////////////////////////
// 		//generate matrix with values in range [0,a]
// 		static void gen_matrix_gtz(realmtx_t& mtx, const real_t a)noexcept {
// 			NNTL_ASSERT(!mtx.emulatesBiases());
// 			gen_vector_gtz(mtx.data(), mtx.numel(), a);
// 		}
// 		static void gen_matrix_no_bias_gtz(realmtx_t& mtx, const real_t a)noexcept {
// 			NNTL_ASSERT();
// 			NNTL_ASSERT(mtx.emulatesBiases());
// 			gen_vector_gtz( mtx.data(), mtx.numel_no_bias(), a);
// 		}
		//generate vector with values in range [0,a]
		template<typename BaseType>
		static void gen_vector_gtz(BaseType* ptr, const size_t n, const BaseType a)noexcept {
			const real_t scale = static_cast<real_t>(a);
			const real_t rm = static_cast<real_t>(_rand_max());
			const auto pE = ptr + n;
			while (ptr != pE) {
				*ptr++ = static_cast<BaseType>(scale*(static_cast<real_t>(_rand()) / rm));
			}
		}
		
	protected:
		
#ifdef _CRT_RAND_S
		static unsigned int _rand()noexcept {
			unsigned int v;
			rand_s(&v);
			return v;
		}

		static real_rand_max_t _rand_max()noexcept { return UINT_MAX; }
#else
		static int _rand()noexcept { return std::rand(); }
		static real_rand_max_t _rand_max()noexcept { return RAND_MAX; }
#endif // _CRT_RAND_S
		

	};


}
}