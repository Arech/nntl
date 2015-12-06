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

#include <cblas.h>
//TODO: function definitions (like dgemm()) conflicts with similar function definitions in ACML. It builds successfully, but
//links to wrong library and access violation happens in run-time.

#pragma comment(lib,"libopenblas.dll.a")

//what else to do to use OpenBLAS:
// -in Solution's VC++ Directories property page set Library Directories to point to a folder with libopenblas.dll.a file
// -copy correct libopenblas.dll and another dlls that libopenblas.dll requires to debug/release solution folder
// -there might be a need to change cblas.h function declarations to include __cdecl keyword. Following functions have to
// be updated (take into account your basic data type - be it float or double):
// cblas_dgemm, 


//http://www.christophlassner.de/using-blas-from-c-with-row-major-data.html

// BTW: lda,ldb,ldc is a "major stride". The stride represents the distance in memory between elements in adjacent rows
// (if row-major) or in adjacent columns (if column-major). This means that the stride is usually equal to the number
// of rows/columns in the matrix.
// Matrix A = [1 2 3]
//            [4 5 6]
// Row-major stores values as {1,2,3,4,5,6} Stride here is 3
// Col-major stores values as {1,4,2,5,3,6} Stride here is 2
// (https://www.physicsforums.com/threads/understanding-blas-dgemm-in-c.543110/)



namespace nntl {
namespace math {

	// wrapper around BLAS API. Should at least isolate from double/float differences
	// Also, we are going to use ColMajor ordering in all math libraries (most of them use it by default)
	// EXPECTING data in COL-MAJOR mode!
	struct b_OpenBLAS {

		//TODO: beware that sz_t type used as substitution of blasint can overflow blasint and silencing conversion warnings here can make it difficult to debug!
		//TODO: May be there should be some preliminary check for this condition.

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LEVEL 1
		// AXPY y=a*x+y
		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value > axpy(
			const sz_t n, const fl_t alpha, const fl_t *x, const sz_t incx, fl_t *y, const sz_t incy)
		{	
			cblas_daxpy(static_cast<blasint>(n), alpha, x, static_cast<blasint>(incx), y, static_cast<blasint>(incy));
		}
		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value > axpy(
			const sz_t n, const fl_t alpha, const fl_t *x, const sz_t incx, fl_t *y, const sz_t incy)
		{
			cblas_saxpy(static_cast<blasint>(n), alpha, x, static_cast<blasint>(incx), y, static_cast<blasint>(incy));
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LEVEL 2

		
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LEVEL 3
		// GEMM C=a*`A`*`B`+b*C
		// M - Specifies the number of rows of the matrix op(A) and of the matrix C. The value of m must be at least zero.
		// N - Specifies the number of columns of the matrix op(B) and the number of columns of the matrix C. The value of n must be at least zero.
		// K - Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B). The value of k must be at least zero.
		// A - {transa=CblasNoTrans : Array, size lda*k. Before entry, the leading m-by-k part of the array a must contain the matrix A.
		//		transa=CblasTrans : Array, size lda*m. Before entry, the leading k-by-m part of the array a must contain the matrix A.}
		// lda - Specifies the leading dimension of A as declared in the calling (sub)program.
		//		{transa=CblasNoTrans, lda must be at least max(1, m).
		//		transa=CblasTrans, lda must be at least max(1, k)}
		// B - {transb=CblasNoTrans : Array, size ldb by n. Before entry, the leading k-by-n part of the array b must contain the matrix B.
		//		transb=CblasTrans : Array, size ldb by k. Before entry the leading n-by-k part of the array b must contain the matrix B.}
		// ldb - Specifies the leading dimension of B as declared in the calling (sub)program.
		//		{transb = CblasNoTrans : ldb must be at least max(1, k).
		//		transb=CblasTrans : ldb must be at least max(1, n).}
		// C - Array, size ldc by n. Before entry, the leading m-by-n part of the array c must contain the matrix C, except when beta is
		//		equal to zero, in which case c need not be set on entry.
		// ldc - Specifies the leading dimension of c as declared in the calling (sub)program. ldc must be at least max(1, m).
		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value >
			gemm( //const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const bool bTransposeA, const bool bTransposeB,
			const sz_t M, const sz_t N, const sz_t K, const fl_t alpha, const fl_t *A, const sz_t lda,
			const fl_t *B, const sz_t ldb, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			cblas_dgemm(CblasColMajor, bTransposeA ? CblasTrans : CblasNoTrans, bTransposeB ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K),
				alpha, A, static_cast<blasint>(lda), B, static_cast<blasint>(ldb), beta, C, static_cast<blasint>(ldc));
		}
		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value >
			gemm( //const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const bool bTransposeA, const bool bTransposeB,
			const sz_t M, const sz_t N, const sz_t K, const fl_t alpha, const fl_t *A, const sz_t lda,
			const fl_t *B, const sz_t ldb, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			cblas_sgemm(CblasColMajor, bTransposeA ? CblasTrans: CblasNoTrans, bTransposeB ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K),
				alpha, A, static_cast<blasint>(lda), B, static_cast<blasint>(ldb), beta, C, static_cast<blasint>(ldc));
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Extensions
		

		// The omatcopy routine performs scaling and out-of-place transposition/copying of matrices. A transposition 
		// operation can be a normal matrix copy, a transposition, a conjugate transposition, or just a conjugation.
		// The operation is defined as follows:
		// B : = alpha*op(A)
		// 
		// Parameters
		// rows - The number of rows in the source matrix.
		// cols - The number of columns in the source matrix.
		// alpha - This parameter scales the input matrix by alpha.
		// pA - Array.
		// lda - Distance between the first elements in adjacent columns(in the case of the column - major order)
		//		or rows(in the case of the row - major order) in the source matrix; measured in the number of elements.
		//		This parameter must be at least max(1, rows) if ordering = 'C' or 'c', and max(1, cols) otherwise.
		// b - Array.
		// ldb - Distance between the first elements in adjacent columns(in the case of the column - major order)
		//		or rows(in the case of the row - major order) in the destination matrix; measured in the number of elements.
		//		To determine the minimum value of ldb on output, consider the following guideline :
		//		If ordering = 'C' or 'c', then
		//			If trans = 'T' or 't' or 'C' or 'c', this parameter must be at least max(1, cols)
		//			If trans = 'N' or 'n' or 'R' or 'r', this parameter must be at least max(1, rows)
		//		If ordering = 'R' or 'r', then
		//			If trans = 'T' or 't' or 'C' or 'c', this parameter must be at least max(1, rows)
		//			If trans = 'N' or 'n' or 'R' or 'r', this parameter must be at least max(1, cols)
		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value >
			omatcopy(const bool bTranspose, const sz_t rows, const sz_t cols, const fl_t alpha,
				const fl_t* pA, const sz_t lda, fl_t* pB, const sz_t ldb)
		{
			cblas_domatcopy(CblasColMajor, bTranspose ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(rows), static_cast<blasint>(cols), alpha,
				pA, static_cast<blasint>(lda), pB, static_cast<blasint>(ldb));
		}
		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value >
			omatcopy(const bool bTranspose, const sz_t rows, const sz_t cols, const fl_t alpha,
				const fl_t* pA, const sz_t lda, fl_t* pB, const sz_t ldb)
		{
			cblas_somatcopy(CblasColMajor, bTranspose ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(rows), static_cast<blasint>(cols), alpha,
				pA, static_cast<blasint>(lda), pB, static_cast<blasint>(ldb));
		}
	};

}
}

