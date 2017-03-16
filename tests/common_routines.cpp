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

#include "../nntl/math.h"
#include "../nntl/nntl.h"
#include "asserts.h"
#include "common_routines.h"

using namespace nntl;

void seqFillMtx(realmtx_t& m) {
	NNTL_ASSERT(!m.empty() && m.numel_no_bias());
	const auto p = m.data();
	const auto ne = m.numel_no_bias();
	for (size_t i = 0; i < ne;++i) {
		p[i] = static_cast<real_t>(i+1);
	}
}




//copies srcMask element into mask element if corresponding column element of data_y is nonzero. Column to use is selected by c
void _allowMask(const realmtx_t& srcMask, realmtx_t& mask, const realmtx_t& data_y, const vec_len_t c) {
	auto pSrcM = srcMask.data();
	auto pM = mask.data();
	auto pY = data_y.colDataAsVec(c);
	const size_t _rm = data_y.rows();
	for (size_t r = 0; r < _rm; ++r) {
		if (pY[r]) pM[r] = pSrcM[r];
	}
}

void _maskStat(const realmtx_t& m) {
	auto pD = m.data();
	size_t c = 0;
	const size_t ne = m.numel();
	for (size_t i = 0; i < ne; ++i) {
		if (pD[i] > 0) ++c;
	}
	STDCOUTL("Gating mask opens " << c << " samples. Total samples count is " << ne);
}

void makeTdForGatedSetup(const train_data<real_t>& td, train_data<real_t>& tdGated, const uint64_t seedV,
	const bool bBinarize, const vec_len_t gatesCnt)
{
	SCOPED_TRACE("makeTdForGatedSetup");

	d_interfaces::iMath_t iM;
	d_interfaces::iRng_t iR;
	iR.set_ithreads(iM.ithreads());
	iR.seed64(seedV);

	realmtx_t ntr, nt, ntry, nty;
	if (1 == gatesCnt) {
		makeDataXForGatedSetup(td.train_x(), td.train_y(), iR, iM, bBinarize, ntr);
		makeDataXForGatedSetup(td.test_x(), td.test_y(), iR, iM, bBinarize, nt);
	} else {
		makeDataXForGatedSetup(td.train_x(), td.train_y(), iR, iM, bBinarize, gatesCnt, ntr);
		makeDataXForGatedSetup(td.test_x(), td.test_y(), iR, iM, bBinarize, gatesCnt, nt);
	}
	
	td.train_y().clone_to(ntry);
	td.test_y().clone_to(nty);

	ASSERT_TRUE(tdGated.absorb(std::move(ntr), std::move(ntry), std::move(nt), std::move(nty)));
}