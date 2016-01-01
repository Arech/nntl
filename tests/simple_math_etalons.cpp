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

#include "stdafx.h"
#include "simple_math_etalons.h"

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void ewBinarize_ET(realmtx_t& A, const real_t frac)noexcept {
	auto pA = A.dataAsVec();
	const auto pAE = pA + A.numel();
	while (pA != pAE) {
		const auto v = *pA;
		NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
		*pA++ = v > frac ? real_t(1.0) : real_t(0.0);
	}
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
		auto mn = std::numeric_limits<real_t>::lowest();
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
	const auto pSum = src.dataAsVec();
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
