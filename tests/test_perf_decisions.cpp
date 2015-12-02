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

//////////////////////////////////////////////////////////////////////////
// this unit will help to filter just bad decisions from obviously stupid
//////////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "../nntl/interface/math.h"
#include "../nntl/common.h"

#include "../nntl/interface/math/imath_basic.h"
#include "../nntl/nnet_def_interfaces.h"

#include <array>
#include <numeric>

#include "../nntl/utils/chrono.h"
#include "../nntl/utils/prioritize_workers.h"

#include "etalons.h"

using namespace nntl;
using namespace std::chrono;

//////////////////////////////////////////////////////////////////////////
#ifdef _DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
#endif // _DEBUG

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename iMath>
inline void evCMulSub_st(iMath& iM, floatmtx_t& vW, const float_t_ momentum, floatmtx_t& W)noexcept {
	iM.evMulC_ip_st_naive(vW, momentum);
	iM.evSub_ip_st_naive(W, vW);
}
inline void evcombCMulSub(floatmtx_t& vW, const float_t_ momentum, floatmtx_t& W)noexcept {
	NNTL_ASSERT(vW.size() == W.size());
	auto pV = vW.dataAsVec();
	const auto pVE = pV + vW.numel();
	auto pW = W.dataAsVec();
	while (pV != pVE) {
		const auto v = *pV * momentum;
		*pV++ = v;
		*pW++ -= v;
	}
}
template<typename iMath>
void check_evCMulSub(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking evCMulSub() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	//this is to test which implementation of combined operation
	//		vW = momentum.*vW
	//		W = W-vW
	// is better: operation-wise, or combined

	const float momentum = 0.95;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	floatmtx_t vW(rowsCnt, colsCnt), W(colsCnt, rowsCnt), vW2(colsCnt, rowsCnt), W2(colsCnt, rowsCnt);
	ASSERT_TRUE(!vW.isAllocationFailed() && !W.isAllocationFailed() && !vW2.isAllocationFailed() && !W2.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(vW2, 2);
	rg.gen_matrix(W2, 2);
	vW2.cloneTo(vW);
	W2.cloneTo(W);

	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	nanoseconds diffEv(0), diffComb(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		auto bt = steady_clock::now();
		evCMulSub_st(iM, vW, momentum, W);
		diffEv += steady_clock::now() - bt;

		bt = steady_clock::now();
		evcombCMulSub(vW2, momentum, W2);
		diffComb += steady_clock::now() - bt;

		ASSERT_EQ(vW, vW2);
		ASSERT_EQ(W, W2);
	}

	STDCOUTL("ev:\t" << utils::duration_readable(diffEv, maxReps));
	STDCOUTL("comb:\t" << utils::duration_readable(diffComb, maxReps));
}
TEST(TestPerfDecisions, CMulSub) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 50; i <= 10000; i*=2) check_evCMulSub(iM, i,100);
	check_evCMulSub(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	evCMulSub(iM, 1000);
	evCMulSub(iM, 10000);
	evCMulSub(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void mTranspose_ET(const floatmtx_t& src, floatmtx_t& dest) noexcept{
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const auto sRows = src.rows(), sCols = src.cols();
	for (vec_len_t r = 0; r < sRows; ++r) {
		for (vec_len_t c = 0; c < sCols; ++c) {
			dest.set(c,r, src.get(r,c));
		}
	}
}
void mTranspose_seq_read(const floatmtx_t& src, floatmtx_t& dest) noexcept {
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const auto sRows = src.rows(), sCols = src.cols();
	const auto dataCnt = src.numel();
	auto pSrc = src.dataAsVec();
	const auto pSrcE = pSrc + dataCnt;
	auto pDest = dest.dataAsVec();
	
	while (pSrc != pSrcE) {
		auto pD = pDest++;
		auto pS = pSrc;
		pSrc += sRows;
		const auto pSE = pSrc;
		while (pS != pSE) {
			*pD = *pS++;
			pD += sCols;
		}
	}
}
void mTranspose_seq_write(const floatmtx_t& src, floatmtx_t& dest) noexcept {
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const auto sRows = src.rows(), sCols = src.cols();
	const auto dataCnt = src.numel();
	auto pSrc = src.dataAsVec();
	auto pDest = dest.dataAsVec();
	const auto pDestE = pDest + dataCnt;

	while (pDest != pDestE) {
		auto pS = pSrc++;
		auto pD = pDest;		
		pDest += sCols;
		const auto pDE = pDest;
		while (pD != pDE) {
			*pD++ = *pS;
			pS += sRows;
		}
	}
}
void mTranspose_OpenBLAS(const floatmtx_t& src, floatmtx_t& dest) noexcept {
	const auto sRows = src.rows(), sCols = src.cols();
	math::b_OpenBLAS::omatcopy(true, sRows, sCols, float_t_(1.0), src.dataAsVec(), sRows, dest.dataAsVec(), sCols);
}

template<typename iMath>
void check_mTranspose(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking mTranspose() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	floatmtx_t src(rowsCnt, colsCnt), dest(colsCnt, rowsCnt), destEt(colsCnt, rowsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !destEt.isAllocationFailed());
	
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(src, 10);

	mTranspose_ET(src, destEt);

	dest.zeros();
	mTranspose_seq_read(src, dest);
	ASSERT_EQ(destEt, dest) << "mTranspose_seq_read failed";

	dest.zeros();
	mTranspose_seq_write(src, dest);
	ASSERT_EQ(destEt, dest) << "mTranspose_seq_write failed";

	dest.zeros();
	mTranspose_OpenBLAS(src, dest);
	ASSERT_EQ(destEt, dest) << "mTranspose_OpenBLAS failed";

	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	steady_clock::time_point bt;
	nanoseconds diffSR(0), diffSW(0), diffOB(0);

	for (unsigned r = 0; r < maxReps; ++r) {
		dest.zeros();
		bt = steady_clock::now();
		mTranspose_seq_read(src, dest);
		diffSR += steady_clock::now() - bt;

		dest.zeros();
		bt = steady_clock::now();
		mTranspose_seq_write(src, dest);
		diffSW += steady_clock::now() - bt;

		dest.zeros();
		bt = steady_clock::now();
		mTranspose_OpenBLAS(src, dest);
		diffOB += steady_clock::now() - bt;
	}

	STDCOUTL("sread:\t" << utils::duration_readable(diffSR, maxReps));
	STDCOUTL("swrite:\t" << utils::duration_readable(diffSW, maxReps));
	STDCOUTL("OBLAS:\t" << utils::duration_readable(diffOB, maxReps));

	/* Very funny (and consistent through runs) results
	 ******* checking mTranspose() variations over 100x10 matrix (1000 elements) **************
sread:   970.602 ns
swrite:  973.520 ns
OBLAS:   731.237 ns
******* checking mTranspose() variations over 1000x100 matrix (100000 elements) **************
sread:   203.952 mcs
swrite:  221.120 mcs
OBLAS:   202.675 mcs
******* checking mTranspose() variations over 10000x1000 matrix (10000000 elements) **************
sread:   188.957 ms
swrite:  113.841 ms
OBLAS:   190.604 ms

OpenBLAS probably uses something similar to seq_read for bigger matrices and something smarter for smaller (we're 
usually not interested in such sizes)

And by the way - it's a way slower (x4-x5), than whole rowwise_renorm operation on colmajor matrix of the same size.

	 **/
}
TEST(TestPerfDecisions, mTranspose) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 100; i <= 10000; i*=10) check_mTranspose(iM, i,i/10);
	check_mTranspose(iM, 100, 100);
	//check_mTranspose(iM, 10000,1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_mTranspose(iM, 1000);
	check_mTranspose(iM, 10000);
	check_mTranspose(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//normalization of row-vectors of a matrix to max possible length
//static constexpr float_t_ rowvecs_renorm_MULT = float_t_(1.0);
/*
float_t_ rowvecs_renorm_ET(floatmtx_t& m, float_t_* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows(), mCols = m.cols();
	for (vec_len_t r = 0; r < mRows; ++r) {
		pTmp[r] = float_t_(0.0);
		for (vec_len_t c = 0; c < mCols; ++c) {
			auto v = m.get(r, c);
			pTmp[r] += v*v;
		}
	}

	//finding average norm
	float_t_ meanNorm = std::accumulate(pTmp, pTmp+mRows, 0.0) / mRows;

	//test and renormalize
	//const float_t_ newNorm = meanNorm - sqrt(math::float_ty_limits<float_t_>::eps_lower_n(meanNorm, rowvecs_renorm_MULT));
	const float_t_ newNorm = meanNorm - sqrt(math::float_ty_limits<float_t_>::eps_lower(meanNorm));
	for (vec_len_t r = 0; r < mRows; ++r) {
		if (pTmp[r] > meanNorm) {
			const float_t_ normCoeff = sqrt(newNorm / pTmp[r]);
			float_t_ nn = 0;
			for (vec_len_t c = 0; c < mCols; ++c) {
				const auto newV = m.get(r, c)*normCoeff;
				m.set(r, c, newV);
				nn += newV*newV;
			}
			EXPECT_TRUE(nn <= meanNorm);
		}
	}
	return meanNorm;
}*/
//slow
void rowvecs_renorm_naive(floatmtx_t& m, float_t_ maxLenSquared, float_t_* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows();
	memset(pTmp, 0, sizeof(float_t_)*mRows);
	const auto dataCnt = m.numel();
	const float_t_* pCol = m.dataAsVec();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const float_t_* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//test and renormalize
	//const float_t_ newNorm = maxLenSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower_n(maxLenSquared, rowvecs_renorm_MULT));
	const float_t_ newNorm = maxLenSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower(maxLenSquared));
	auto pRow = m.dataAsVec();
	const auto pRowE = pRow + mRows;
	while (pRow!=pRowE) {
		const auto rowNorm = *pTmp++;
		if (rowNorm > maxLenSquared) {
			const float_t_ normCoeff = sqrt(newNorm / rowNorm);
			auto pElm = pRow;
			const auto pElmE = pRow + dataCnt;
			while (pElm!=pElmE){
				*pElm *= normCoeff;
				pElm += mRows;
			}
		}
		++pRow;
	}
}
//best
void rowvecs_renorm_clmnw(floatmtx_t& A, float_t_ maxNormSquared, float_t_* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = A.rows();
	memset(pTmp, 0, sizeof(float_t_)*mRows);
	const auto dataCnt = A.numel();
	float_t_* pCol = A.dataAsVec();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const float_t_* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, that doesn't need.
	//const float_t_ newNorm = maxNormSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower_n(maxNormSquared, rowvecs_renorm_MULT));
	const float_t_ newNorm = maxNormSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower(maxNormSquared));
	auto pCurNorm = pTmp;
	const auto pTmpE = pTmp + mRows;
	while (pCurNorm != pTmpE) {
		const auto rowNorm = *pCurNorm;
		*pCurNorm++ = rowNorm > maxNormSquared ? sqrt(newNorm / rowNorm) : float_t_(1.0);
	}

	//renormalize
	pCol = A.dataAsVec();
	while (pCol != pColE) {
		float_t_* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			*pElm++ *= *pN++;
		}
	}
}
//slower, probably don't vectorize correctly
void rowvecs_renorm_clmnw2(floatmtx_t& m, float_t_ maxLenSquared, float_t_* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows();
	memset(pTmp, 0, sizeof(float_t_)*mRows);
	const auto dataCnt = m.numel();
	float_t_* pCol = m.dataAsVec();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const float_t_* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, that doesn't need.
	//const float_t_ newNorm = maxLenSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower_n(maxLenSquared, rowvecs_renorm_MULT));
	const float_t_ newNorm = maxLenSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower(maxLenSquared));
	auto pCurNorm = pTmp;
	const auto pTmpE = pTmp + mRows;
	while (pCurNorm != pTmpE) {
		const auto rowNorm = *pCurNorm;
		*pCurNorm++ = rowNorm > maxLenSquared ? sqrt(newNorm / rowNorm) : float_t_(1.0);
	}

	//renormalize
	auto pElm = m.dataAsVec();
	const auto pElmE = pElm + dataCnt;
	auto pN = pTmp;
	auto pRowE = pElm + mRows;
	while (pElm != pElmE) {
		if (pElm == pRowE) {
			pN = pTmp;
			pRowE += mRows;
		}
		*pElm++ *= *pN++;
	}
}
//bit slower, than the best
void rowvecs_renorm_clmnw_part(floatmtx_t& m, float_t_ maxLenSquared, float_t_* pTmp, size_t* pOffs)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows();
	memset(pTmp, 0, sizeof(float_t_)*mRows);
	const auto dataCnt = m.numel();
	float_t_* pCol = m.dataAsVec();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const float_t_* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, that doesn't need.
	//memset(pOffs, 0, sizeof(*pOffs)*mRows);
	//const float_t_ newNorm = maxLenSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower_n(maxLenSquared, rowvecs_renorm_MULT));
	const float_t_ newNorm = maxLenSquared - sqrt(math::float_ty_limits<float_t_>::eps_lower(maxLenSquared));
	auto pT = pTmp, pCurNorm = pTmp, pPrevNorm = pTmp;
	const auto pTmpE = pTmp + mRows;
	auto pOE = pOffs;
	while (pT != pTmpE) {
		const auto rowNorm = *pT;
		if (rowNorm > maxLenSquared) {
			*pCurNorm++ = sqrt(newNorm / rowNorm);
			*pOE++ = pT - pPrevNorm;
			pPrevNorm = pT;
		}
		++pT;
	}

	//renormalize
	if (pOE!=pOffs) {
		pCol = m.dataAsVec();
		while (pCol != pColE) {
			float_t_* pElm = pCol;
			pCol += mRows;
			auto pN = pTmp;
			auto pO = pOffs;
			while (pO != pOE) {
				pElm += *pO++;
				*pElm *= *pN++;
			}
		}
	}
}

template<typename iMath>
void check_rowvecs_renorm(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking rowvecs_renorm() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	const float_t_ scale = 5;
	floatmtx_t W(rowsCnt, colsCnt), srcW(rowsCnt, colsCnt), etW(rowsCnt, colsCnt);
	ASSERT_TRUE(!W.isAllocationFailed() && !srcW.isAllocationFailed() && !etW.isAllocationFailed());
	std::vector<float_t_> tmp(rowsCnt);
	std::vector<size_t> ofs(rowsCnt);

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	steady_clock::time_point bt;
	nanoseconds diffNaive(0), diffClmnw(0), diffClmnw2(0), diffClmnwPart(0);

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(srcW, scale);

		srcW.cloneTo(etW);
		const float_t_ meanNorm = rowvecs_renorm_ET(etW, &tmp[0]);

		srcW.cloneTo(W);
		bt = steady_clock::now();
		rowvecs_renorm_naive(W, meanNorm, &tmp[0]);
		diffNaive += steady_clock::now() - bt;
		ASSERT_EQ(etW, W) << "rowvecs_renorm_naive";

		srcW.cloneTo(W);
		bt = steady_clock::now();
		rowvecs_renorm_clmnw(W, meanNorm, &tmp[0]);
		diffClmnw += steady_clock::now() - bt;
		ASSERT_EQ(etW, W) << "rowvecs_renorm_clmnw";

		srcW.cloneTo(W);
		bt = steady_clock::now();
		rowvecs_renorm_clmnw2(W, meanNorm, &tmp[0]);
		diffClmnw2 += steady_clock::now() - bt;
		ASSERT_EQ(etW, W) << "rowvecs_renorm_clmnw2";

		srcW.cloneTo(W);
		bt = steady_clock::now();
		rowvecs_renorm_clmnw_part(W, meanNorm, &tmp[0], &ofs[0]);
		diffClmnwPart += steady_clock::now() - bt;
		ASSERT_EQ(etW, W) << "rowvecs_renorm_clmnw_part";
	}

	STDCOUTL("naive:\t" << utils::duration_readable(diffNaive, maxReps));
	STDCOUTL("clmnw:\t" << utils::duration_readable(diffClmnw, maxReps));
	STDCOUTL("clmnw2:\t" << utils::duration_readable(diffClmnw2, maxReps));
	STDCOUTL("clmnwPart:\t" << utils::duration_readable(diffClmnwPart, maxReps));

}
TEST(TestPerfDecisions, rowvecsRenorm) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

//   	for (unsigned i = 10; i <= 1000; i*=10) check_rowvecs_renorm(iM, i,i);
//   	check_rowvecs_renorm(iM, 4000, 4000);

	check_rowvecs_renorm(iM, 100, 10);
	//check_rowvecs_renorm(iM, 1000, 1000);
	//check_rowvecs_renorm(iM, 10000, 1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_rowvecs_renorm(iM, 1000,100);
	check_rowvecs_renorm(iM, 10000,100);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//calculation of squared norm of row-vectors of a matrix. size(pNorm)==m.rows()
void rowwise_normsq_ET(const floatmtx_t& m, float_t_* pNorm)noexcept {
	const auto mRows = m.rows(), mCols = m.cols();
	for (vec_len_t r = 0; r < mRows; ++r) {
		pNorm[r] = float_t_(0.0);
		for (vec_len_t c = 0; c < mCols; ++c) {
			auto v = m.get(r, c);
			pNorm[r] += v*v;
		}
	}
}
//slow
void rowwise_normsq_naive(const floatmtx_t& m, float_t_* pNorm)noexcept {
	const auto dataCnt = m.numel();
	const auto mRows = m.rows();
	const float_t_* pRow = m.dataAsVec();
	const auto pRowEnd = pRow + mRows;
	while (pRow != pRowEnd) {
		const float_t_* pElm = pRow;
		const auto pElmEnd = pRow++ + dataCnt;
		float_t_ cs = float_t_(0.0);
		while (pElm != pElmEnd) {
			const auto v = *pElm;
			pElm += mRows;
			cs += v*v;
		}
		*pNorm++ = cs;
	}
}
//best
void rowwise_normsq_clmnw(const floatmtx_t& m, float_t_* pNorm)noexcept {
	const auto mRows = m.rows();
	memset(pNorm, 0, sizeof(float_t_)*mRows);

	const auto dataCnt = m.numel();
	const float_t_* pCol = m.dataAsVec();
	const auto pColE = pCol + dataCnt;
	while (pCol!=pColE) {
		const float_t_* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pNorm;
		while (pElm!=pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}
}
template<typename iMath>
void check_rowwiseNormsq(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking rowwise_normsq() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	float_t_ scale = 5;
	floatmtx_t W(rowsCnt, colsCnt);
	ASSERT_TRUE(!W.isAllocationFailed());
	std::vector<float_t_> normvecEt(rowsCnt), normvec(rowsCnt);
	
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	rg.gen_matrix(W, scale);
	rowwise_normsq_ET(W, &normvecEt[0]);
	float_t_ meanNorm = std::accumulate(normvecEt.begin(), normvecEt.end(), 0.0) / rowsCnt;
	STDCOUTL("Mean norm value is "<< meanNorm);

	std::fill(normvec.begin(), normvec.end(), float_t_(10.0));
	rowwise_normsq_naive(W, &normvec[0]);
	ASSERT_TRUE(0 == memcmp(&normvec[0], &normvecEt[0], rowsCnt*sizeof(float_t_))) << "rowwise_normsq_naive wrong implementation";

	std::fill(normvec.begin(), normvec.end(), float_t_(10.0));
	rowwise_normsq_clmnw(W, &normvec[0]);
	ASSERT_TRUE(0 == memcmp(&normvec[0], &normvecEt[0], rowsCnt*sizeof(float_t_))) << "rowwise_normsq_clmnw wrong implementation";

	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	steady_clock::time_point bt;
	nanoseconds diffNaive(0), diffClmnw(0);

	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		rowwise_normsq_naive(W, &normvec[0]);
		diffNaive += steady_clock::now() - bt;

		bt = steady_clock::now();
		rowwise_normsq_clmnw(W, &normvec[0]);
		diffClmnw += steady_clock::now() - bt;
	}

	STDCOUTL("naive:\t" << utils::duration_readable(diffNaive, maxReps));
	STDCOUTL("clmnw:\t" << utils::duration_readable(diffClmnw, maxReps));
}
TEST(TestPerfDecisions, rowwiseNormsq) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 10; i <= 10000; i*=10) check_rowwiseNormsq(iM, i,i);
	check_rowwiseNormsq(iM, 100, 100);
	//check_rowwiseNormsq(iM, 10000, 1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_rowwiseNormsq(iM, 1000);
	check_rowwiseNormsq(iM, 10000);
	check_rowwiseNormsq(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/*%first getting rid of possible NaN's
				a=nn.a{n};
				oma=1-a;
				a(a==0) = realmin;
				oma(oma==0) = realmin;
				%crossentropy for sigmoid E=-y*log(a)-(1-y)log(1-a), dE/dz=a-y
				nn.L = -sum(sum(y.*log(a) + (1-y).*log(oma)))/m;*/
//unaffected by data_y distribution, fastest, when data_y has equal amount of 1 and 0
float_t_ sigm_loss_xentropy_naive(const floatmtx_t& activations, const floatmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.dataAsVec(), ptrY = data_y.dataAsVec();
	constexpr auto log_zero = math::float_ty_limits<float_t_>::log_almost_zero;
	float_t_ ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto a = ptrA[i], y = ptrY[i], oma = float_t_(1.0) - a;
		NNTL_ASSERT(y == float_t_(0.0) || y == float_t_(1.0));
		NNTL_ASSERT(a >= float_t_(0.0) && a <= float_t_(1.0));

		ql += y*(a == float_t_(0.0) ? log_zero : log(a)) + (float_t_(1.0) - y)*(oma == float_t_(0.0) ? log_zero : log(oma));
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
//best when data_y skewed to 1s or 0s. Slightly slower than sigm_loss_xentropy_naive when data_y has equal amount of 1 and 0
float_t_ sigm_loss_xentropy_naive_part(const floatmtx_t& activations, const floatmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.dataAsVec(), ptrY = data_y.dataAsVec();
	constexpr auto log_zero = math::float_ty_limits<float_t_>::log_almost_zero;
	float_t_ ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		auto a = ptrA[i];
		NNTL_ASSERT(y == float_t_(0.0) || y == float_t_(1.0));
		NNTL_ASSERT(a >= float_t_(0.0) && a <= float_t_(1.0));

		if (y > float_t_(0.0)) {
			ql += (a == float_t_(0.0) ? log_zero : log(a));
		} else {
			const auto oma = float_t_(1.0) - a;
			ql += (oma == float_t_(0.0) ? log_zero : log(oma));
		}
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
//unaffected by data_y distribution, but slowest
float_t_ sigm_loss_xentropy_vec(const floatmtx_t& activations, const floatmtx_t& data_y, floatmtx_t& t1, floatmtx_t& t2)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	NNTL_ASSERT(t1.size() == activations.size() && t2.size() == t1.size());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.dataAsVec(), ptrY = data_y.dataAsVec();
	const auto p1 = t1.dataAsVec(), p2 = t2.dataAsVec();
	constexpr auto realmin = std::numeric_limits<float_t_>::min();
	float_t_ ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto a = ptrA[i], oma = float_t_(1.0) - a;
		p1[i] = (a == float_t_(0.0) ? realmin : a);
		p2[i] = (oma == float_t_(0.0) ? realmin : oma);
	}
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		ql += y*log(p1[i]) + (float_t_(1.0) - y)*log(p2[i]);
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
template <typename iRng, typename iMath>
void run_sigm_loss_xentropy(iRng& rg, iMath &iM, floatmtx_t& act, floatmtx_t& data_y, floatmtx_t& t1, floatmtx_t& t2,unsigned maxReps,float_t_ binFrac)noexcept {

	STDCOUTL("binFrac = "<<binFrac);

	steady_clock::time_point bt;
	nanoseconds diffNaive(0), diffPart(0), diffVec(0);
	float_t_ lossNaive, lossPart, lossVec;

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix_norm(act);
		rg.gen_matrix_norm(data_y);
		iM.mBinarize(data_y, binFrac);

		bt = steady_clock::now();
		lossNaive = sigm_loss_xentropy_naive(act, data_y);
		diffNaive += steady_clock::now() - bt;

		bt = steady_clock::now();
		lossPart = sigm_loss_xentropy_naive_part(act, data_y);
		diffPart += steady_clock::now() - bt;

		bt = steady_clock::now();
		lossVec = sigm_loss_xentropy_vec(act, data_y, t1, t2);
		diffVec += steady_clock::now() - bt;

		ASSERT_NEAR(lossNaive, lossPart, 1e-8);
		ASSERT_NEAR(lossNaive, lossVec, 1e-8);
	}

	STDCOUTL("naive:\t" << utils::duration_readable(diffNaive, maxReps));
	STDCOUTL("part:\t" << utils::duration_readable(diffPart, maxReps));
	STDCOUTL("vec:\t" << utils::duration_readable(diffVec, maxReps));

}
template<typename iMath>
void check_sigm_loss_xentropy(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking sigm_loss_xentropy() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	floatmtx_t act(rowsCnt, colsCnt), data_y(rowsCnt, colsCnt), t1(rowsCnt, colsCnt), t2(rowsCnt, colsCnt);
	ASSERT_TRUE(!act.isAllocationFailed() && !data_y.isAllocationFailed() && !t1.isAllocationFailed() && !t2.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	run_sigm_loss_xentropy(rg, iM, act, data_y, t1, t2, maxReps, .5);
	run_sigm_loss_xentropy(rg, iM, act, data_y, t1, t2, maxReps, .1);
	run_sigm_loss_xentropy(rg, iM, act, data_y, t1, t2, maxReps, .9);
}
TEST(TestPerfDecisions, sigmLossXentropy) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 50; i <= 20000; i*=1.5) check_sigm_loss_xentropy(iM, i,100);
	check_sigm_loss_xentropy(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	check_sigm_loss_xentropy(iM, 1000);
	check_sigm_loss_xentropy(iM, 10000);
	check_sigm_loss_xentropy(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void apply_momentum_FOR(floatmtx_t& vW, const float_t_ momentum, const floatmtx_t& dW)noexcept {
	NNTL_ASSERT(vW.size() == dW.size());
	NNTL_ASSERT(!vW.empty() && !dW.empty());

	const auto dataCnt = vW.numel();
	const auto pV = vW.dataAsVec();
	const auto pdW = dW.dataAsVec();
	for (numel_cnt_t i = 0; i < dataCnt;++i) {
		pV[i] = momentum*pV[i] + pdW[i];
	}
}
void apply_momentum_WHILE(floatmtx_t& vW, const float_t_ momentum, const floatmtx_t& dW)noexcept {
	NNTL_ASSERT(vW.size() == dW.size());
	NNTL_ASSERT(!vW.empty() && !dW.empty());

	const auto dataCnt = vW.numel();
	auto pV = vW.dataAsVec();
	const auto pVE = pV + dataCnt;
	auto pdW = dW.dataAsVec();
	while (pV!=pVE) {
		*pV++ = momentum*(*pV) + *pdW++;
	}
}
template<typename iMath>
void check_apply_momentum_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking apply_momentum() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tFOR, tWHILE;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	float_t_ momentum=.9;

	floatmtx_t dW(rowsCnt, colsCnt), vW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !vW.isAllocationFailed());

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(dW, 2);
	rg.gen_matrix(vW, 2);

	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r)  apply_momentum_FOR(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("for:\t" << utils::duration_readable(diff, maxReps, &tFOR));

	rg.gen_matrix(vW, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r)  apply_momentum_WHILE(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("while:\t" << utils::duration_readable(diff, maxReps, &tWHILE));


	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r)  apply_momentum_FOR(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("for2:\t" << utils::duration_readable(diff, maxReps, &tFOR));

	rg.gen_matrix(vW, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r)  apply_momentum_WHILE(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("while2:\t" << utils::duration_readable(diff, maxReps, &tWHILE));
}
TEST(TestPerfDecisions, applyMomentum) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 50; i <= 20000; i*=2) check_apply_momentum_perf(iM, i,100);
	check_apply_momentum_perf(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	check_apply_momentum_perf(iM, 1000);
	check_apply_momentum_perf(iM, 10000);
	check_apply_momentum_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename T> T sgncopysign(T magn, T val) {
	return val == 0 ? T(0) : std::copysign(magn, val);
}
template<typename iMath>
void test_sign_perf(iMath& iM, typename iMath::floatmtx_t::vec_len_t rowsCnt, typename iMath::floatmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking sign() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tSgn, tSgncopysign;// , tBoost;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	floatmtx_t dW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	float_t_ lr = .1;

	float_t_ pz = float_t_(+0.0), nz = float_t_(-0.0), p1 = float_t_(1), n1 = float_t_(-1);

	//boost::sign
	/*{
		auto c_pz = boost::math::sign(pz), c_nz = boost::math::sign(nz), c_p1 = boost::math::sign(p1), c_n1 = boost::math::sign(n1);
		STDCOUTL("boost::sign\t: +0.0=" << c_pz << " -0.0=" << c_nz << " +1=" << c_p1 << " -1=" << c_n1);
		EXPECT_EQ(0, c_pz);
		EXPECT_EQ(0, c_nz);
		EXPECT_EQ(1, c_p1);
		EXPECT_EQ(-1, c_n1);
	}
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 100);
		auto p = dW.dataAsVec();
		const auto pE = p + dW.numel();
		bt = steady_clock::now();
		while (p!=pE) {
			*p++ = lr*boost::math::sign(*p);
		}
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("\t\t" << utils::duration_readable(diff, maxReps, &tBoost));*/
	//decent performance on my hw

	//math::sign
	{
		auto c_pz = math::sign(pz), c_nz = math::sign(nz), c_p1 = math::sign(p1), c_n1 = math::sign(n1);
		STDCOUTL("math::sign\t: +0.0=" << c_pz << " -0.0=" << c_nz << " +1=" << c_p1 << " -1=" << c_n1);
		EXPECT_EQ(0, c_pz);
		EXPECT_EQ(0, c_nz);
		EXPECT_EQ(1, c_p1);
		EXPECT_EQ(-1, c_n1);
	}
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 100);
		auto p = dW.dataAsVec();
		const auto pE = p + dW.numel();
		bt = steady_clock::now();
		while (p != pE) {
			*p++ = lr*math::sign(*p);
		}
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("\t\t" << utils::duration_readable(diff, maxReps, &tSgn));
	//best performance on my hw

	//sgncopysign
	{
		auto c_pz = sgncopysign(p1,pz), c_nz = sgncopysign(p1,nz), c_p1 = sgncopysign(p1, p1), c_n1 = sgncopysign(p1, n1);
		STDCOUTL("sgncopysign\t: +0.0=" << c_pz << " -0.0=" << c_nz << " +1=" << c_p1 << " -1=" << c_n1);
		EXPECT_EQ(0, c_pz);
		EXPECT_EQ(0, c_nz);
		EXPECT_EQ(1, c_p1);
		EXPECT_EQ(-1, c_n1);
	}
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 100);
		auto p = dW.dataAsVec();
		const auto pE = p + dW.numel();
		bt = steady_clock::now();
		while (p != pE) {
			*p++ = sgncopysign(lr,*p);
		}
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("\t\t" << utils::duration_readable(diff, maxReps, &tSgncopysign));
	//worst (x4) performance on my hw
}

TEST(TestPerfDecisions, Sign) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 100; i <= 6400; i*=2) test_sign_perf(iM, i,100);

	test_sign_perf(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_sign_perf(iM, 1000);
	test_sign_perf(iM, 10000);
	test_sign_perf(iM, 100000);
#endif
}


//////////////////////////////////////////////////////////////////////////
/*

template<typename iMath>
void calc_rmsproph_vec(iMath& iM, typename iMath::floatmtx_t& dW, typename iMath::floatmtx_t& rmsF, typename iMath::floatmtx_t::value_type lr,
	typename iMath::floatmtx_t::value_type emaDecay, typename iMath::floatmtx_t::value_type numericStab, typename iMath::floatmtx_t& t1)
{
	typedef typename iMath::floatmtx_t floatmtx_t;
	typedef typename floatmtx_t::value_type float_t_;
	typedef typename floatmtx_t::numel_cnt_t numel_cnt_t;

	NNTL_ASSERT(dW.size() == rmsF.size());

	auto pdW = dW.dataAsVec();
	auto prmsF = rmsF.dataAsVec();
	auto pt = t1.dataAsVec();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto im = dW.numel();
	
	for (numel_cnt_t i = 0; i < im; ++i) {
		const auto w = pdW[i];
		pt[i] = w*w*_1_emaDecay;
	}

	for (numel_cnt_t i = 0; i < im; ++i) {
		const auto rms = emaDecay*prmsF[i] + pt[i];
		prmsF[i] = rms;
		pt[i] = sqrt(rms)+ numericStab;
	}

	for (numel_cnt_t i = 0, im = dW.numel(); i < im; ++i) {
		pdW[i] = (pdW[i] / pt[i] )*lr;
	}
}

template<typename iMath>
void calc_rmsproph_ew(iMath& iM, typename iMath::floatmtx_t& dW, typename iMath::floatmtx_t& rmsF, const typename iMath::floatmtx_t::value_type learningRate,
	const typename iMath::floatmtx_t::value_type emaDecay, const typename iMath::floatmtx_t::value_type numericStabilizer)
{
	typedef typename iMath::floatmtx_t floatmtx_t;
	typedef typename floatmtx_t::value_type float_t_;
	typedef typename floatmtx_t::numel_cnt_t numel_cnt_t;

	NNTL_ASSERT(dW.size() == rmsF.size());

	auto pdW = dW.dataAsVec();
	auto prmsF = rmsF.dataAsVec();
	const auto _1_emaDecay = 1 - emaDecay;
	for (numel_cnt_t i = 0, im = dW.numel(); i < im; ++i) {
		const auto w = pdW[i];
		const auto rms = emaDecay*prmsF[i] + w*w*_1_emaDecay;
		prmsF[i] = rms;
		pdW[i] = learningRate*(w / (sqrt(rms) + numericStabilizer));
	}
}

template<typename iMath>
void test_rmsproph_perf(iMath& iM, typename iMath::floatmtx_t::vec_len_t rowsCnt, typename iMath::floatmtx_t::vec_len_t colsCnt = 10) {
	typedef typename iMath::floatmtx_t floatmtx_t;
	typedef typename floatmtx_t::value_type float_t_;
	typedef typename floatmtx_t::numel_cnt_t numel_cnt_t;

	using namespace std::chrono;
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RMSPropHinton variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tBatch, tEw;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_REPEATS_COUNT;

	float_t_ emaCoeff = .9, lr=.1, numStab=.00001;

	floatmtx_t dW(true, rowsCnt, colsCnt), rms(true, rowsCnt, colsCnt), t1(true, rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed());
	
	rms.zeros();

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 100);
		bt = steady_clock::now();
		calc_rmsproph_ew(iM, dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("ew:\t" << utils::duration_readable(diff, maxReps, &tEw));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 100);
		bt = steady_clock::now();
		calc_rmsproph_vec(iM, dW, rms, lr, emaCoeff, numStab,t1);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("vec:\t" << utils::duration_readable(diff, maxReps, &tBatch));
}

TEST(TestPerfDecisions, RmsPropH) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 100; i <= 20000; i*=1.5) test_rmsproph_perf(iM, i,100);

	test_rmsproph_perf(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_rmsproph_perf(iM, 1000);
	test_rmsproph_perf(iM, 10000);
	test_rmsproph_perf(iM, 100000);
#endif
}
*/


//////////////////////////////////////////////////////////////////////////
// perf test to find out which strategy better to performing dropout - processing elementwise, or 'vectorized'
// Almost no difference for my hardware, so going to implement 'vectorized' version, because batch RNG may provide some benefits
template<typename iRng_t, typename iMath_t>
void dropout_batch(math_types::floatmtx_ty& activs, math_types::float_ty dropoutFraction, math_types::floatmtx_ty& dropoutMask, iRng_t& iR, iMath_t& iM) {
	NNTL_ASSERT(activs.size() == dropoutMask.size());
	NNTL_ASSERT(activs.emulatesBiases() && dropoutMask.emulatesBiases());
	NNTL_ASSERT(dropoutFraction > 0 && dropoutFraction < 1);

	iR.gen_matrix_no_bias_gtz(dropoutMask, 1);
	auto pDM = dropoutMask.dataAsVec();
	const auto pDME = pDM + dropoutMask.numel_no_bias();
	while (pDM != pDME) {
		auto v = *pDM;
		*pDM++ = v > dropoutFraction ? math_types::float_ty(1.0) : math_types::float_ty(0.0);
	}
	iM.evMul_ip_st_naive(activs, dropoutMask);
}

template<typename iRng_t>
void dropout_ew(math_types::floatmtx_ty& activs, math_types::float_ty dropoutFraction, math_types::floatmtx_ty& dropoutMask, iRng_t& iR) {
	NNTL_ASSERT(activs.size() == dropoutMask.size());
	NNTL_ASSERT(activs.emulatesBiases() && dropoutMask.emulatesBiases());
	NNTL_ASSERT(dropoutFraction > 0 && dropoutFraction < 1);

	const auto dataCnt = activs.numel_no_bias();
	auto pDM = dropoutMask.dataAsVec();
	auto pA = activs.dataAsVec();
	for (math_types::floatmtx_ty::numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto rv = iR.gen_f_norm();
		if (rv > dropoutFraction) {
			pDM[i] = 1;
		} else {
			pDM[i] = 0;
			pA[i] = 0;
		}
	}
}

template<typename iMath>
void test_dropout_perf(iMath& iM, typename iMath::floatmtx_t::vec_len_t rowsCnt, typename iMath::floatmtx_t::vec_len_t colsCnt = 10) {
	typedef typename iMath::floatmtx_t floatmtx_t;
	typedef typename floatmtx_t::value_type float_t_;
	typedef typename floatmtx_t::numel_cnt_t numel_cnt_t;

	using namespace std::chrono;
	const auto dataSize = floatmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing dropout variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tBatch, tEw;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	float_t_ dropoutFraction = .5;

	floatmtx_t act(rowsCnt, colsCnt, true), dm(rowsCnt, colsCnt, true);
	ASSERT_TRUE(!act.isAllocationFailed() && !dm.isAllocationFailed());


	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix_no_bias(act, 100);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		dropout_ew(act, dropoutFraction, dm, rg);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("ew:\t" << utils::duration_readable(diff, maxReps, &tEw));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		dropout_batch(act, dropoutFraction, dm, rg, iM);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("batch:\t" << utils::duration_readable(diff, maxReps, &tBatch));
}

TEST(TestPerfDecisions, Dropout) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 100; i <= 140; i+=1) test_dropout_perf(iM, i,100);

	test_dropout_perf(iM, 100, 10);
#ifndef TESTS_SKIP_LONGRUNNING
	test_dropout_perf(iM, 1000);
	test_dropout_perf(iM, 10000);
	test_dropout_perf(iM, 100000);
#endif
}
//////////////////////////////////////////////////////////////////////////