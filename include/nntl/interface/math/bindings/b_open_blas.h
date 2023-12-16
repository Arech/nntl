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

#include <cblas.h>
//TODO: function definitions (like dgemm()) conflicts with similar function definitions in ACML. It builds successfully, but
//links to wrong library and access violation happens in run-time.

#include <complex>
#define lapack_complex_float ::std::complex<float>
#define lapack_complex_double ::std::complex<double>
#include <lapacke.h>

//#include <nntl/utils/denormal_floats.h>

#pragma comment(lib,"libopenblas.dll.a")

//what else to do to use OpenBLAS:
// -in the Solution's VC++ Directories property page set parameter Library Directories to point to a folder with the libopenblas.dll.a file
// -copy correct libopenblas.dll and another dlls that the libopenblas.dll require to the debug/release solution's folder
// -if you're going to use any calling convetion except for __cdecl, then most likely, you'll have to update function declarations
//		within the cblas.h and other blas's .h files included to contain the __cdecl keyword (it's absent for some reason).
//		Check the b_OpenBLAS:: methods to find out which function definitions should be changed.


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
	// EXPECTING data to be in COL-MAJOR mode!
	// 
	// NB: Leading dimension is the number of elements in major dimension. We're using Col-Major ordering,
	// therefore it is the number of ROWs of a matrix
	struct b_OpenBLAS {
	private:
		typedef utils::_scoped_restore_FPU _restoreFPU;
	public:
		//TODO: beware that sz_t type used as substitution of blasint can overflow blasint and silencing conversion warnings here can make it difficult to debug!
		//TODO: May be there should be some preliminary check for this condition.

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LEVEL 1
		// AXPY y=a*x+y
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value > axpy(
			const sz_t n, const fl_t alpha, const fl_t *x, const sz_t incx, fl_t *y, const sz_t incy)
		{
			_restoreFPU r;
			cblas_daxpy(static_cast<blasint>(n), alpha, x, static_cast<blasint>(incx), y, static_cast<blasint>(incy));
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value > axpy(
			const sz_t n, const fl_t alpha, const fl_t *x, const sz_t incx, fl_t *y, const sz_t incy)
		{
			_restoreFPU r;
			cblas_saxpy(static_cast<blasint>(n), alpha, x, static_cast<blasint>(incx), y, static_cast<blasint>(incy));
		}

		//////////////////////////////////////////////////////////////////////////
		// cblas_?dot
		// Computes a vector - vector dot product.
		//	Input Parameters
		// n - Specifies the number of elements in vectors x and y.
		// x - Array, size at least(1 + (n - 1)*abs(incx)).
		// incx - Specifies the increment for the elements of x.
		// y - Array, size at least(1 + (n - 1)*abs(incy)).
		// incy - Specifies the increment for the elements of y.
		// Return Values - The result of the dot product of x and y, if n is positive.Otherwise, returns 0.
		//https://software.intel.com/en-us/mkl-developer-reference-c-cblas-dot
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value, double >
			dot(const sz_t n, const fl_t *x, const sz_t incx, const fl_t *y, const sz_t incy)
		{
			_restoreFPU r;
			return cblas_ddot(static_cast<blasint>(n), x, static_cast<blasint>(incx), y, static_cast<blasint>(incy));
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value, float >
			dot(const sz_t n, const fl_t *x, const sz_t incx, const fl_t *y, const sz_t incy)
		{
			_restoreFPU r;
			return cblas_sdot(static_cast<blasint>(n), x, static_cast<blasint>(incx), y, static_cast<blasint>(incy));
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LEVEL 2

		
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LEVEL 3
		// 
		//////////////////////////////////////////////////////////////////////////
		// General matrix multiplication
		// GEMM C := alpha*op(A)*op(B) + beta*C
		// https://software.intel.com/en-us/node/520775
		// where:
		// op(X) is one of op(X) = X, or op(X) = XT, or op(X) = XH,
		// alpha and beta are scalars,
		// A, B and C are matrices :
		// op(A) is an m - by - k matrix,
		// op(B) is a k - by - n matrix,
		// C is an m - by - n matrix.
		// 
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
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value >
			gemm( //const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const bool bTransposeA, const bool bTransposeB,
			const sz_t M, const sz_t N, const sz_t K, const fl_t alpha, const fl_t *A, const sz_t lda,
			const fl_t *B, const sz_t ldb, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			_restoreFPU r;
			cblas_dgemm(CblasColMajor, bTransposeA ? CblasTrans : CblasNoTrans, bTransposeB ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K),
				alpha, A, static_cast<blasint>(lda), B, static_cast<blasint>(ldb), beta, C, static_cast<blasint>(ldc));
			//global_denormalized_floats_mode();
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value >
			gemm( //const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const bool bTransposeA, const bool bTransposeB,
			const sz_t M, const sz_t N, const sz_t K, const fl_t alpha, const fl_t *A, const sz_t lda,
			const fl_t *B, const sz_t ldb, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			_restoreFPU r;
			cblas_sgemm(CblasColMajor, bTransposeA ? CblasTrans: CblasNoTrans, bTransposeB ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K),
				alpha, A, static_cast<blasint>(lda), B, static_cast<blasint>(ldb), beta, C, static_cast<blasint>(ldc));
			//global_denormalized_floats_mode();
		}

		//////////////////////////////////////////////////////////////////////////
		// cblas_?syrk, https://software.intel.com/en-us/node/520780
		// Performs a symmetric rank-k update.
		// The ?syrk routines perform a rank-k matrix-matrix operation for a symmetric matrix C using a general matrix A.
		// The operation is defined as:
		// C := alpha*A*A' + beta*C,
		//		or
		// C : = alpha*A'*A + beta*C,
		// where :
		//		alpha and beta are scalars,
		//		C is an n-by-n symmetric matrix,
		//		A is an n-by-k matrix in the first case and a k-by-n matrix in the second case.
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value >
			syrk( const bool bCLowerTriangl, const bool bFirstATransposed, const sz_t N, const sz_t K, const fl_t alpha
				, const fl_t *A, const sz_t lda, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			_restoreFPU r;
			cblas_dsyrk(CblasColMajor, bCLowerTriangl ? CblasLower : CblasUpper, bFirstATransposed ? CblasTrans : CblasNoTrans
				, static_cast<blasint>(N), static_cast<blasint>(K), alpha, A, static_cast<blasint>(lda), beta, C, static_cast<blasint>(ldc));
			//global_denormalized_floats_mode();
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value >
			syrk(const bool bCLowerTriangl, const bool bFirstATransposed, const sz_t N, const sz_t K, const fl_t alpha
				, const fl_t *A, const sz_t lda, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			_restoreFPU r;
			cblas_ssyrk(CblasColMajor, bCLowerTriangl ? CblasLower : CblasUpper, bFirstATransposed ? CblasTrans : CblasNoTrans
				, static_cast<blasint>(N), static_cast<blasint>(K), alpha, A, static_cast<blasint>(lda), beta, C, static_cast<blasint>(ldc));
			//global_denormalized_floats_mode();
		}

		//////////////////////////////////////////////////////////////////////////
		// cblas_?symm, https://software.intel.com/en-us/node/520779
		// Computes a matrix - matrix product where one input matrix is symmetric.
		// The ?symm routines compute a scalar-matrix-matrix product with one symmetric matrix and add the
		// result to a scalar-matrix product. The operation is defined as
		// C: = alpha*A*B + beta*C,
		//		or
		// C : = alpha*B*A + beta*C,
		// where :
		//		alpha and beta are scalars,
		//		A is a symmetric matrix,
		//		B and C are m - by - n matrices.
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value >
			symm(const bool bSymmAatLeft, const bool bALowerTriangl, const sz_t M, const sz_t N, const fl_t alpha
				, const fl_t *A, const sz_t lda, const fl_t *B, const sz_t ldb, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			_restoreFPU r;
			cblas_dsymm(CblasColMajor, bSymmAatLeft ? CblasLeft : CblasRight, bALowerTriangl ? CblasLower : CblasUpper
				, static_cast<blasint>(M), static_cast<blasint>(N), alpha, A, static_cast<blasint>(lda)
				, B, static_cast<blasint>(ldb), beta, C, static_cast<blasint>(ldc));
			//global_denormalized_floats_mode();
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value >
			symm(const bool bSymmAatLeft, const bool bALowerTriangl, const sz_t M, const sz_t N, const fl_t alpha
				, const fl_t *A, const sz_t lda, const fl_t *B, const sz_t ldb, const fl_t beta, fl_t *C, const sz_t ldc)
		{
			_restoreFPU r;
			cblas_ssymm(CblasColMajor, bSymmAatLeft ? CblasLeft : CblasRight, bALowerTriangl ? CblasLower : CblasUpper
				, static_cast<blasint>(M), static_cast<blasint>(N), alpha, A, static_cast<blasint>(lda)
				, B, static_cast<blasint>(ldb), beta, C, static_cast<blasint>(ldc));
			//global_denormalized_floats_mode();
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// LAPACKE

		// ?gesvd
		// https://software.intel.com/en-us/node/521150
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value, int>
			gesvd(/*int matrix_layout,*/ const char jobu, const char jobvt,
				const sz_t m, const sz_t n, fl_t* A,
				const sz_t lda, fl_t* S, fl_t* U, const sz_t ldu,
				fl_t* Vt, const sz_t ldvt, fl_t* superb)
		{
			_restoreFPU r;
			return static_cast<int>(LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, static_cast<lapack_int>(m), static_cast<lapack_int>(n)
				, A, static_cast<lapack_int>(lda), S, U, static_cast<lapack_int>(ldu), Vt, static_cast<lapack_int>(ldvt), superb));
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value, int>
			gesvd(/*int matrix_layout,*/ const char jobu, const char jobvt,
				const sz_t m, const sz_t n, fl_t* A,
				const sz_t lda, fl_t* S, fl_t* U, const sz_t ldu,
				fl_t* Vt, const sz_t ldvt, fl_t* superb)
		{
			_restoreFPU r;
			return static_cast<int>(LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, static_cast<lapack_int>(m), static_cast<lapack_int>(n)
				, A, static_cast<lapack_int>(lda), S, U, static_cast<lapack_int>(ldu), Vt, static_cast<lapack_int>(ldvt), superb));
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
		//
		// #warning current OpenBLAS implementation is slower, than it can be. See TEST(TestPerfDecisions, mTranspose) in test_perf_decisions.cpp
		//and https://github.com/xianyi/OpenBLAS/issues/1243
		// https://github.com/xianyi/OpenBLAS/issues/2532
		// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c/16743203#16743203
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value >
			omatcopy(const bool bTranspose, const sz_t rows, const sz_t cols, const fl_t alpha,
				const fl_t* pA, const sz_t lda, fl_t* pB, const sz_t ldb)
		{
			_restoreFPU r;
			cblas_domatcopy(CblasColMajor, bTranspose ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(rows), static_cast<blasint>(cols), alpha,
				pA, static_cast<blasint>(lda), pB, static_cast<blasint>(ldb));
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value >
			omatcopy(const bool bTranspose, const sz_t rows, const sz_t cols, const fl_t alpha,
				const fl_t* pA, const sz_t lda, fl_t* pB, const sz_t ldb)
		{
			_restoreFPU r;
			cblas_somatcopy(CblasColMajor, bTranspose ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(rows), static_cast<blasint>(cols), alpha,
				pA, static_cast<blasint>(lda), pB, static_cast<blasint>(ldb));
		}

		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value >
			imatcopy(const bool bTranspose, const sz_t rows, const sz_t cols, const fl_t alpha,
				fl_t* pA, const sz_t lda, const sz_t ldb)
		{
			_restoreFPU r;
			cblas_dimatcopy(CblasColMajor, bTranspose ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(rows), static_cast<blasint>(cols), alpha,
				pA, static_cast<blasint>(lda), static_cast<blasint>(ldb));
		}
		template<typename sz_t, typename fl_t>
		static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value >
			imatcopy(const bool bTranspose, const sz_t rows, const sz_t cols, const fl_t alpha,
				fl_t* pA, const sz_t lda, const sz_t ldb)
		{
			_restoreFPU r;
			cblas_simatcopy(CblasColMajor, bTranspose ? CblasTrans : CblasNoTrans,
				static_cast<blasint>(rows), static_cast<blasint>(cols), alpha,
				pA, static_cast<blasint>(lda), static_cast<blasint>(ldb));
		}
	};

}
}

