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

#include "stdafx.h"
#include "simple_math_etalons.h"


// Transforms a data matrix from a tiled layer format back to normal. For a data with biases it looks like this:
//																	|x1_1...x1_n 1|		:transformed data_x
//																	|........... 1|		:to be fed to the layer
//	data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	<===	|xi_1...xi_n 1|
//																	|........... 1|
//																	|xk_1...xk_n 1|
// For a data without biases the same, just drop all the ones in the picture.
// If src is biased matrix, then src must be a matrix of size [k*m, n+1], dest - [m, k*n+1], also biased.
//		Last column of dest is reserved to contain biases and must be preinitialized to 1s
// If src doesn't have biases, then it's size must be equal to [k*m, n], dest.size() == [k*m, n]
void mTilingUnroll_ET(const realmtx_t& src, realmtx_t& dest)noexcept {
	NNTL_ASSERT(!dest.empty() && !src.empty());
	NNTL_ASSERT(!(dest.emulatesBiases() ^ src.emulatesBiases()));
	NNTL_ASSERT(dest.rows() && (dest.cols() > static_cast<vec_len_t>(dest.emulatesBiases())));
	NNTL_ASSERT(src.rows() && (src.cols() > static_cast<vec_len_t>(src.emulatesBiases())));
	NNTL_ASSERT(dest.rows() < src.rows());
	NNTL_ASSERT(src.cols_no_bias() < dest.cols_no_bias());
	NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_ok());
	NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_ok());

	const vec_len_t m = dest.rows();
	const auto km = src.rows();
	const vec_len_t k = km / m;
	NNTL_ASSERT(km == k*m);//to make sure no rounding happened
	const vec_len_t n = src.cols_no_bias();
	NNTL_ASSERT(dest.cols_no_bias() == k*n);

	const auto pD = dest.data();
	const auto pS = src.data();

	for (vec_len_t ssr = 0; ssr < k; ++ssr) {
		for (vec_len_t sc = 0; sc < n; ++sc) {
			const auto psrc = pS + size_t(sc)*km + size_t(ssr)*m;
			NNTL_ASSERT(psrc + m <= src.colDataAsVec(n));
			const auto pdest = pD + (size_t(ssr)*n + sc)*m;
			NNTL_ASSERT(pdest + m <= dest.colDataAsVec(dest.cols_no_bias()));
			memcpy(pdest, psrc, sizeof(*pdest)*m);
		}
	}

	NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_ok());
	NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_ok());
}



// Transforms a data matrix to be used by tiled layer. For a data with biases it looks like this:
//																	|x1_1...x1_n 1|		:transformed data_x
//																	|........... 1|		:to be fed to the layer
//	data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	===>	|xi_1...xi_n 1|
//																	|........... 1|
//																	|xk_1...xk_n 1|
// For a data without biases the same, just drop all the ones in the picture.
// If src is biased matrix, then src must be a matrix of size [m, k*n+1], dest - [k*m, n+1], also biased.
//		Last column of dest is reserved to contain biases and must be preinitialized to 1s
// If src doesn't have biases, then it's size must be equal to [m, k*n], dest.size() == [k*m, n]
void mTilingRoll_ET(const realmtx_t& src, realmtx_t& dest)noexcept {
	NNTL_ASSERT(!src.empty() && !dest.empty());
	NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
	NNTL_ASSERT(src.rows() && (src.cols() > static_cast<vec_len_t>(src.emulatesBiases())));
	NNTL_ASSERT(dest.rows() && (dest.cols() > static_cast<vec_len_t>(dest.emulatesBiases())));
	NNTL_ASSERT(src.rows() < dest.rows());
	NNTL_ASSERT(dest.cols_no_bias() < src.cols_no_bias());
	NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_ok());
	NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_ok());

	const vec_len_t m = src.rows();
	const auto km = dest.rows();
	const vec_len_t k = km / m;
	NNTL_ASSERT(km == k*m);//to make sure no rounding happened
	const vec_len_t n = dest.cols_no_bias();
	NNTL_ASSERT(src.cols_no_bias() == k*n);

	const auto pS = src.data();
	const auto pD = dest.data();
	
	for (vec_len_t dsr = 0; dsr < k; ++dsr) {
		for (vec_len_t dc = 0; dc < n; ++dc) {
			const auto pdest = pD + size_t(dc)*km + size_t(dsr)*m;
			NNTL_ASSERT(pdest+m <= dest.colDataAsVec(n));
			const auto psrc = pS + (size_t(dsr)*n + dc)*m;
			NNTL_ASSERT(psrc + m <= src.colDataAsVec(src.cols_no_bias()));
			memcpy(pdest, psrc, sizeof(*psrc)*m);
		}
	}

	NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_ok());
	NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_ok());
}

real_t ewSumProd_ET(const realmtx_t& A, const realmtx_t& B)noexcept {
	NNTL_ASSERT(!A.empty() && !B.empty() && B.size() == A.size());
	const auto pA = A.data(), pB = B.data();
	const auto dataCnt = A.numel();
	real_t ret(0.0);
	for (numel_cnt_t i = 0; i < dataCnt; ++i) ret += pA[i] * pB[i];
	return ret;
}

void mrwDivideByVec_ET(realmtx_t& A, const real_t* pDiv)noexcept {
	NNTL_ASSERT(!A.empty() && pDiv);
	const auto rm = A.rows(), cm = A.cols();
	for (vec_len_t c = 0; c < cm; ++c) {
		for (vec_len_t r = 0; r < rm; ++r) {
			A.set(r, c, A.get(r, c) / pDiv[r]);
		}
	}
}

void mrwMulByVec_ET(realmtx_t& A, const real_t* pMul)noexcept {
	NNTL_ASSERT(!A.empty() && pMul);
	const auto rm = A.rows(), cm = A.cols();
	for (vec_len_t c = 0; c < cm; ++c) {
		for (vec_len_t r = 0; r < rm; ++r) {
			A.set(r, c, A.get(r, c) * pMul[r]);
		}
	}
}

// pMax (if specified) is an array of m.rows() real_t and will be filled with maximum value of corresponding m row
// pColIdxs (if specified) is an array of m.rows() vec_len_t and will be filled with indexes of columns with maximum value of corresponding m row
void mrwMax_ET(const realmtx_t& m, real_t* pMax, vec_len_t* pColIdxs) noexcept {
	NNTL_ASSERT(pMax || pColIdxs);

	const auto rm = m.rows(), cm = m.cols();
	for (vec_len_t r = 0; r < rm; ++r) {
		auto mn = ::std::numeric_limits<real_t>::lowest();
		vec_len_t mi = 0;
		for (vec_len_t c = 0; c < cm; ++c) {
			const auto v = m.get(r, c);
			if (v > mn) {
				mn = v;
				mi = c;
			}
		}
		if (pMax) pMax[r] = mn;
		if (pColIdxs) pColIdxs[r] = mi;
	}
}

//Sum matrix src rowwise into first row
void mrwSum_ip_ET(realmtx_t& src)noexcept {
	NNTL_ASSERT(!src.empty() && src.numel() > 0);
	const auto cm = src.cols(), rm = src.rows();
	const auto pSum = src.data();
	for (vec_len_t c = 1; c < cm; ++c) {
		for (vec_len_t r = 0; r < rm; ++r) {
			pSum[r] += src.get(r, c);
		}
	}
}

//Sum matrix src rowwise into pVec
void mrwSum_ET(const realmtx_t& src, real_t* pVec)noexcept {
	NNTL_ASSERT(!src.empty() && src.numel() > 0);
	const auto cm = src.cols(), rm = src.rows();
	memset(pVec, 0, sizeof(*pVec)*rm);
	for (vec_len_t c = 0; c < cm; ++c) {
		for (vec_len_t r = 0; r < rm; ++r) {
			pVec[r] += src.get(r, c);
		}
	}
}

void mCloneCols_ET(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec)noexcept {
	NNTL_ASSERT(!srcCols.empty() && !dest.empty());
	NNTL_ASSERT(pColSpec && srcCols.cols());
	NNTL_ASSERT(srcCols.rows() == dest.rows());
	NNTL_ASSERT(dest.cols() == ::std::accumulate(pColSpec, pColSpec + srcCols.cols(), vec_len_t(0)));

	auto pSrc = srcCols.data();
	auto pDest = dest.data();
	const ptrdiff_t _rows = static_cast<ptrdiff_t> (dest.rows());
	const size_t _csLen = srcCols.cols();
	for (size_t i = 0; i < _csLen; ++i) {
		const auto tCols = pColSpec[i];
		for (vec_len_t c = 0; c < tCols; ++c) {
			memcpy(pDest, pSrc, sizeof(*pSrc)*_rows);
			pDest += _rows;
		}
		pSrc += _rows;
	}
}

void mCloneCol_ET(const realmtx_t& srcCol, realmtx_t& dest)noexcept {
	NNTL_ASSERT(!srcCol.empty() && !dest.empty());
	NNTL_ASSERT(1 == srcCol.cols());
	NNTL_ASSERT(srcCol.rows() == dest.rows());

	const auto pS = srcCol.data();
	auto pD = dest.data();
	const auto pDE = pD + dest.numel();
	const auto _r = dest.rows();
	while (pD != pDE) {
		memcpy(pD, pS, sizeof(*pS)*_r);
		pD += _r;
	}
}

//////////////////////////////////////////////////////////////////////////
// Performs binary OR operation over rows of matrix A and return result to pVec
void mrwBinaryOR_ET(const realmtx_t& A, real_t* pVec)noexcept {
	NNTL_ASSERT(!A.empty() && A.numel() > 0 && !A.emulatesBiases() && A.isBinary());
	NNTL_ASSERT(pVec);

	const auto cm = A.cols(), rm = A.rows();
	memset(pVec, 0, rm * sizeof(real_t));

	for (vec_len_t c = 0; c < cm; ++c) {
		for (vec_len_t r = 0; r < rm; ++r) {
			const auto v = A.get(r, c);
			if (v != 0) {
				pVec[r] = real_t(1.);
			}
		}
	}
}



real_t ewSumSquares_ET(const realmtx_t& A)noexcept {
	const auto dataCnt = A.numel();
	const auto p = A.data();
	real_t ret(0), C(0.), Y, T;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		Y = p[i] * p[i] - C;
		T = ret + Y;
		C = T - ret - Y;
		ret = T;
	}
	return ret;
}
