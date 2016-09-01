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

#include <yepLibrary.h>
#include <yepCore.h>
#include <yepMath.h>

#pragma comment(lib,"yeppp.lib")

namespace nntl {
namespace math {

	class b_Yeppp {
		//!! copy constructor not needed
		b_Yeppp(const b_Yeppp& other)noexcept = delete;
		//!!assignment is not needed
		b_Yeppp& operator=(const b_Yeppp& rhs) noexcept = delete;

	public:
		b_Yeppp() noexcept : m_lastError(yepLibrary_Init()) {
			NNTL_ASSERT(YepStatusOk == m_lastError);
		};
		~b_Yeppp()noexcept {
			m_lastError = yepLibrary_Release();
			NNTL_ASSERT(YepStatusOk == m_lastError);
		};

		YepStatus last_error()const noexcept { return m_lastError; }
		bool succeded()const noexcept { return YepStatusOk == m_lastError; }

		//////////////////////////////////////////////////////////////////////////
		//Negation (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evNegate(const fl_t *src, fl_t *res, const sz_t n) 
		{
			const auto rv = yepCore_Negate_V64f_V64f(src, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evNegate(const fl_t *src, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Negate_V32f_V32f(src, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		//Negation (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evNegate_ip(fl_t *srcdest, const sz_t n)
		{
			const auto rv = yepCore_Negate_IV64f_IV64f(srcdest, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evNegate_ip(fl_t *srcdest, const sz_t n)
		{
			const auto rv = yepCore_Negate_IV32f_IV32f(srcdest, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//////////////////////////////////////////////////////////////////////////
		//Vector Addition (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evAdd(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Add_V64fV64f_V64f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evAdd(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Add_V32fV32f_V32f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Vector Addition (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evAdd_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Add_IV64fV64f_IV64f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evAdd_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Add_IV32fV32f_IV32f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Constant + Vector Addition (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evAddC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Add_V64fS64f_V64f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evAddC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Add_V32fS32f_V32f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Constant + Vector Addition (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evAddC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Add_IV64fS64f_IV64f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evAddC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Add_IV32fS32f_IV32f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//////////////////////////////////////////////////////////////////////////
		//vector subtraction (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evSub(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_V64fV64f_V64f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evSub(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_V32fV32f_V32f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Vector subtraction (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evSub_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Subtract_IV64fV64f_IV64f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evSub_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Subtract_IV32fV32f_IV32f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evSub_ip2(const fl_t *A, fl_t *B_res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_V64fIV64f_IV64f(A, B_res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evSub_ip2(const fl_t *A, fl_t *B_res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_V32fIV32f_IV32f(A, B_res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Vector - Constant subtraction (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evSubC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_V64fS64f_V64f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evSubC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_V32fS32f_V32f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Constant - Vector subtraction (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evCSub(const fl_t b, const fl_t *A, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_S64fV64f_V64f(b, A, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evCSub(const fl_t b, const fl_t *A, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_S32fV32f_V32f(b, A, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Vector - Constant subtraction  (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evSubC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Subtract_IV64fS64f_IV64f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evSubC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Subtract_IV32fS32f_IV32f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Constant - Vector subtraction  (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evCSub_ip(const fl_t b, fl_t *A_res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_S64fIV64f_IV64f(b, A_res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evCSub_ip(const fl_t b, fl_t *A_res, const sz_t n)
		{
			const auto rv = yepCore_Subtract_S32fIV32f_IV32f(b, A_res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//////////////////////////////////////////////////////////////////////////
		//Elementwise Vector Multiplication (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMul(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Multiply_V64fV64f_V64f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMul(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Multiply_V32fV32f_V32f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Elementwise Vector Multiplication (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMul_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Multiply_IV64fV64f_IV64f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMul_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Multiply_IV32fV32f_IV32f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Elementwise Constant + Vector Multiplication (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMulC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Multiply_V64fS64f_V64f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMulC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Multiply_V32fS32f_V32f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

		//Elementwise Constant + Vector Multiplication (in-place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMulC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Multiply_IV64fS64f_IV64f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMulC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Multiply_IV32fS32f_IV32f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		


		//////////////////////////////////////////////////////////////////////////
		// Maximum value of vector
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value, fl_t>
			vMax(const fl_t *src, const sz_t n)
		{
			fl_t res;
			const auto rv = yepCore_Max_V64f_S64f(src, &res, n);
			NNTL_ASSERT(YepStatusOk == rv);
			return res;
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value, fl_t>
			vMax(const fl_t *src, const sz_t n)
		{
			fl_t res;
			const auto rv = yepCore_Max_V32f_S32f(src, &res, n);
			NNTL_ASSERT(YepStatusOk == rv);
			return res;
		}
		// elementwise maximum values of two vectors (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMax(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Max_V64fV64f_V64f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMax(const fl_t *A, const fl_t *B, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Max_V32fV32f_V32f(A, B, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		// elementwise maximum values of two vectors (in place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMax_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Max_IV64fV64f_IV64f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMax_ip(fl_t *A_res, const fl_t *B, const sz_t n)
		{
			const auto rv = yepCore_Max_IV32fV32f_IV32f(A_res, B, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		// elementwise maximum values of vector and constant (different destination)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMaxC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Max_V64fS64f_V64f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMaxC(const fl_t *A, const fl_t b, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_Max_V32fS32f_V32f(A, b, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		// elementwise maximum values of vector and constant (in place)
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evMaxC_ip(fl_t *A_ip, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Max_IV64fS64f_IV64f(A_ip, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evMaxC_ip(fl_t *A_res, const fl_t b, const sz_t n)
		{
			const auto rv = yepCore_Max_IV32fS32f_IV32f(A_res, b, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}


		//////////////////////////////////////////////////////////////////////////
		// exponent
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			evExp(const fl_t *src, fl_t *res, const sz_t n)
		{
			const auto rv = yepMath_Exp_V64f_V64f(src, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			evExp(const fl_t *src, fl_t *res, const sz_t n)
		{
			static_assert(!"Yeppp! doesn't have exp<float> specialization. You should probably do it yourself.");
		}

		//////////////////////////////////////////////////////////////////////////
		// sum of squares
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value>
			vSumSquares(const fl_t *src, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_SumSquares_V64f_S64f(src, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}
		template<typename sz_t, typename fl_t>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value>
			vSumSquares(const fl_t *src, fl_t *res, const sz_t n)
		{
			const auto rv = yepCore_SumSquares_V32f_S32f(src, res, n);
			NNTL_ASSERT(YepStatusOk == rv);
		}

	protected:
		YepStatus m_lastError;
	};


}
}