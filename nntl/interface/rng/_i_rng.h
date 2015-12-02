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

#include <cstdint>
#include <ctime>        // std::time

namespace nntl {
namespace rng {

	struct _i_rng {

		//typedef uint64_t seed_t;
		typedef int seed_t;
		typedef math_types::floatmtx_ty floatmtx_t;
		typedef floatmtx_t::value_type float_t_;
		// ptrdiff_t is either int on 32bits or int64 on 64bits. Type required by random_shuffle()
		typedef ptrdiff_t generated_scalar_t;

		static constexpr bool is_multithreaded = false;
		template<typename itt>
		bool set_ithreads(itt& t)noexcept { return false; }
		template<typename itt>
		bool set_ithreads(itt& t, seed_t s)noexcept { return false; }

		//nntl_interface _i_rng()noexcept {}
		//nntl_interface _i_rng(seed_t s)noexcept {}

		nntl_interface void seed(seed_t s) noexcept;
		//nntl_interface void seed_array(const seed_t s[], unsigned seedsCnt) noexcept;

		// generated_scalar_t is either int on 32bits or int64 on 64bits
		// gen_i() is going to be used with random_shuffle()
		nntl_interface generated_scalar_t gen_i(generated_scalar_t lessThan)noexcept;
		
		//////////////////////////////////////////////////////////////////////////
		//generate FP value in range [0,1]
		nntl_interface float_t_ gen_f_norm()noexcept;

		//////////////////////////////////////////////////////////////////////////
		// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
		nntl_interface void gen_vector(float_t_* ptr, const size_t n, const float_t_ a)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,1]
		nntl_interface void gen_vector_norm(float_t_* ptr, const size_t n)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//generate vector with values in range [0,a]
		template<typename BaseType>
		nntl_interface void gen_vector_gtz(BaseType* ptr, const size_t n, const BaseType a)noexcept;



		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//following functions could be implemented in _i_rng_helper

		//to be used by random_shuffle()
		nntl_interface generated_scalar_t operator()(generated_scalar_t lessThan)noexcept;// { return gen_i(lessThan); }

		//generate FP value in range [0,a]
		nntl_interface float_t_ gen_f(const float_t_ a)noexcept; //{ return a*gen_f_norm(); }

		// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
		nntl_interface void gen_matrix(floatmtx_t& mtx, const float_t_ a)noexcept;
		nntl_interface void gen_matrix_no_bias(floatmtx_t& mtx, const float_t_ a)noexcept;

		//generate matrix with values in range [0,1]
		nntl_interface void gen_matrix_norm(floatmtx_t& mtx)noexcept;
		nntl_interface void gen_matrix_no_bias_norm(floatmtx_t& mtx)noexcept;

		//generate matrix with values in range [0,a]
		nntl_interface void gen_matrix_gtz(floatmtx_t& mtx, const float_t_ a)noexcept;
		nntl_interface void gen_matrix_no_bias_gtz(floatmtx_t& mtx, const float_t_ a)noexcept;
	};

	template<typename FinalPolymorphChild>
	struct rng_helper : public _i_rng {
	protected:
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;

		self_ref_t get_self() noexcept {
			static_assert(std::is_base_of<rng_helper<FinalPolymorphChild>, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _i_rng_helper<FinalPolymorphChild>");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<rng_helper<FinalPolymorphChild>, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _i_rng_helper<FinalPolymorphChild>");
			return static_cast<self_cref_t>(*this);
		}

	public:
		generated_scalar_t operator()(generated_scalar_t lessThan)noexcept { return get_self().gen_i(lessThan); }

		void seed64(uint64_t s) noexcept {
			get_self().seed(static_cast<seed_t>(s64to32(s)));
		}

		//generate FP value in range [0,a]
		float_t_ gen_f(const float_t_ a)noexcept { return a*get_self().gen_f_norm(); }

		// matrix/vector generation (sequence from begin to end of numbers drawn from uniform distribution in [-a,a])
		void gen_matrix(floatmtx_t& mtx, const float_t_ a)noexcept {
			NNTL_ASSERT(!mtx.emulatesBiases());
			get_self().gen_vector(mtx.dataAsVec(), mtx.numel(), a);
		}
		void gen_matrix_no_bias(floatmtx_t& mtx, const float_t_ a)noexcept {
			NNTL_ASSERT(mtx.emulatesBiases());
			get_self().gen_vector(mtx.dataAsVec(), mtx.numel_no_bias(), a);
		}

		//generate matrix with values in range [0,1]
		void gen_matrix_norm(floatmtx_t& mtx)noexcept {
			NNTL_ASSERT(!mtx.emulatesBiases());
			get_self().gen_vector_norm(mtx.dataAsVec(), mtx.numel());
		}
		void gen_matrix_no_bias_norm(floatmtx_t& mtx)noexcept {
			NNTL_ASSERT(mtx.emulatesBiases());
			get_self().gen_vector_norm(mtx.dataAsVec(), mtx.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		//generate matrix with values in range [0,a]
		void gen_matrix_gtz(floatmtx_t& mtx, const float_t_ a)noexcept {
			NNTL_ASSERT(!mtx.emulatesBiases());
			get_self().gen_vector_gtz(mtx.dataAsVec(), mtx.numel(), a);
		}
		void gen_matrix_no_bias_gtz(floatmtx_t& mtx, const float_t_ a)noexcept {
			NNTL_ASSERT(mtx.emulatesBiases());
			get_self().gen_vector_gtz(mtx.dataAsVec(), mtx.numel_no_bias(), a);
		}

		static uint32_t s64to32(uint64_t v)noexcept {
			return static_cast<uint32_t>(v & UINT32_MAX) ^ static_cast<uint32_t>((v >> 32)&UINT32_MAX);
		}
	};

}
}