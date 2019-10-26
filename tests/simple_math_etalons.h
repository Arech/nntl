/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "asserts.h"

void mTilingRoll_ET(const realmtx_t& src, realmtx_t& dest)noexcept;
void mTilingUnroll_ET(const realmtx_t& src, realmtx_t& dest)noexcept;

real_t ewSumProd_ET(const realmtx_t& A, const realmtx_t& B)noexcept;

void mrwDivideByVec_ET(realmtx_t& A, const real_t* pDiv)noexcept;
void mrwMulByVec_ET(realmtx_t& A, const real_t* pMul)noexcept;

void mrwMax_ET(const realmtx_t& m, real_t* pMax = nullptr, vec_len_t* pColIdxs = nullptr) noexcept;

void mrwSum_ip_ET(realmtx_t& src)noexcept;
void mrwSum_ET(const realmtx_t& src, real_t* pVec)noexcept;

void mCloneCols_ET(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec)noexcept;
void mCloneCol_ET(const realmtx_t& srcCol, realmtx_t& dest)noexcept;

void mrwBinaryOR_ET(const realmtx_t& A, real_t* pVec)noexcept;

real_t ewSumSquares_ET(const realmtx_t& A)noexcept;

//////////////////////////////////////////////////////////////////////////

template<typename T>
bool isMtxRwElmsAreBinEqual(const ::nntl::math::smatrix<T>& A
	, const ::std::vector<typename ::nntl::math::smatrix<T>::vec_len_t>& colIdxs1
	, const ::std::vector<typename ::nntl::math::smatrix<T>::vec_len_t>& colIdxs2)
{
	const auto r = A.rows();
	NNTL_ASSERT(r == colIdxs1.size() && r == colIdxs2.size());

	for (::std::decay_t<decltype(r)> i = 0; i < r; ++i) {
		if (colIdxs1[i] != colIdxs2[i]) {
			if (A.get(i, colIdxs1[i]) != A.get(i, colIdxs2[i])) return false;
		}
	}
	return true;
}

template<bool bLowerTriangl, typename _T>
_T ewSumSquaresTriang_ET(const nntl::math::smatrix<_T>& A) noexcept {
	NNTL_ASSERT(A.rows() == A.cols());

	const vec_len_t n = A.rows();
	_T s(_T(0));

	if (bLowerTriangl) {
		for (vec_len_t ri = 1; ri < n; ++ri) {
			for (vec_len_t ci = 0; ci < ri; ++ci) {
				const auto v = A.get(ri, ci);
				s += v*v;
			}
		}
	} else {
		for (vec_len_t ci = 1; ci < n; ++ci) {
			for (vec_len_t ri = 0; ri < ci; ++ri) {
				const auto v = A.get(ri, ci);
				s += v*v;
			}
		}
	}
	return s;
}

template<typename _T>
void mcwMean_ET(const nntl::math::smatrix<_T>& A, _T*const pVec) noexcept {
	const auto tc = A.cols(), tr=A.rows();
	const _T N = static_cast<_T>(tr);

	for (vec_len_t ci = 0; ci < tc; ++ci) {
		_T v(_T(0));
		const auto* pA = A.colDataAsVec(ci);
		for (vec_len_t ri = 0; ri < tr; ++ri) {
			v += pA[ri] / N;
		}
		pVec[ci] = v;
	}
}

template<typename _T>
void mcwSub_ip_ET(nntl::math::smatrix<_T>& A, const _T* pVec)noexcept {
	const auto tc = A.cols(), tr = A.rows();
	for (vec_len_t ci = 0; ci < tc; ++ci) {
		auto*const pA = A.colDataAsVec(ci);
		const auto v = pVec[ci];
		for (vec_len_t ri = 0; ri < tr; ++ri) {
			pA[ri] -= v;
		}
	}
}

template<typename BaseT>
void mcwMulDiag_ip_ET(nntl::math::smatrix<BaseT>& A, const nntl::math::smatrix<BaseT>& B)noexcept {
	NNTL_ASSERT(B.rows() == B.cols() || !"B must be a square matrix!");
	NNTL_ASSERT(A.cols() == B.cols());
	const auto tc = A.cols(), tr = A.rows();
	for (vec_len_t ci = 0; ci < tc; ++ci) {
		auto*const pA = A.colDataAsVec(ci);
		const auto v = B.get(ci, ci);
		for (vec_len_t ri = 0; ri < tr; ++ri) {
			pA[ri] *= v;
		}
	}
}