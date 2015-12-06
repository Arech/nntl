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

//BTW, it uses enum CBLAS_TRANSPOSE definition from OpenBlas (probably it's the same enum for every BLAS implementation)

#include <acml.h>
//TODO: function definitions (like dgemm()) conflicts with similar function definitions in ACML. It builds successfully, but
//links to wrong library and access violation happens in run-time.
// 

//TODO: should link to different lib versions based on open_mp support setting.
#pragma comment(lib,"libacml_mp_dll.lib")

//TODO: check for matrix ordering here

namespace nntl {
namespace math {

	struct b_ACML {

		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, double>::value > gemm(
			const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const sz_t M, const sz_t N, const sz_t K,
			const fl_t alpha, const fl_t *A, const sz_t lda,
			const fl_t *B, const sz_t ldb,
			const fl_t beta, fl_t *C, const sz_t ldc)
		{
			//TODO: beware that sz_t type can overflow blasint and silencing conversion warnings here can make it difficult to debug!
			//TODO: May be there should be some preliminary check for this condition.
			dgemm(TransA == CblasNoTrans ? 'N' : 'T', TransB == CblasNoTrans ? 'N' : 'T',
				static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
				alpha, consz_tcast<fl_t*>(A), static_cast<int>(lda), consz_tcast<fl_t*>(B), static_cast<int>(ldb), beta, consz_tcast<fl_t*>(C), static_cast<int>(ldc));
		}

		template<typename sz_t, typename fl_t = nntl::math_types::real_ty>
		static typename std::enable_if_t< std::is_same< std::remove_pointer_t<fl_t>, float>::value > gemm(
			const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const sz_t M, const sz_t N, const sz_t K,
			const fl_t alpha, const fl_t *A, const sz_t lda,
			const fl_t *B, const sz_t ldb,
			const fl_t beta, fl_t *C, const sz_t ldc)
		{
			sgemm(TransA == CblasNoTrans ? 'N' : 'T', TransB == CblasNoTrans ? 'N' : 'T',
				static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
				alpha, consz_tcast<fl_t*>(A), static_cast<int>(lda), consz_tcast<fl_t*>(B), static_cast<int>(ldb), beta, consz_tcast<fl_t*>(C), static_cast<int>(ldc));
		}

	};

}
}