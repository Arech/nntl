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

//////////////////////////////////////////////////////////////////////////
// this unit will help to filter just bad decisions from obviously stupid
//////////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "../nntl/math.h"
#include "../nntl/common.h"

#include "../nntl/interface/math/mathn.h"
#include "../nntl/interfaces.h"

#include <array>
#include <numeric>

#include "../nntl/utils/chrono.h"
#include "../nntl/utils/prioritize_workers.h"

#include "../nntl/utils/tictoc.h"

#include "imath_etalons.h"

using namespace nntl;
using namespace std::chrono;
using namespace nntl::utils;

//////////////////////////////////////////////////////////////////////////
#ifdef NNTL_DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 100;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 500;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 50;
#endif // NNTL_DEBUG

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void softmax_parts_st_cw(const realmtx_t& act, const real_t* pMax, real_t* pDenominator, real_t* pNumerator)noexcept {
	NNTL_ASSERT(pMax && pDenominator && act.numel() > 0);
	const auto rm = act.rows(), cm = act.cols();
	auto pA = act.data();
	const auto pME = pMax + rm;
	std::fill(pDenominator, pDenominator + rm, real_t(0.0));
	for (vec_len_t c = 0; c < cm; ++c) {
		auto pDen = pDenominator;
		auto pM = pMax;
		while (pM != pME) {
			const auto num = std::exp(*pA++ - *pM++);
			*pDen++ += num;
			*pNumerator++ = num;
		}
	}
}
//significantly slower, than cw
void softmax_parts_st_rw(const realmtx_t& act, const real_t* pMax, real_t* pDenominator, real_t* pNumerator)noexcept {
	NNTL_ASSERT(pMax && pDenominator && act.numel() > 0);
	const auto rm = act.rows(), cm = act.cols();
	const auto pA = act.data();
	for (vec_len_t r = 0; r < rm; ++r) {
		auto ofs = r;
		const auto m = pMax[r];
		auto den = real_t(0.0);
		for (vec_len_t c = 0; c < cm; ++c) {
			const auto num = std::exp(pA[ofs] - m);
			den += num;
			pNumerator[ofs] = num;
			ofs += rm;
		}
		pDenominator[r] = den;
	}
}
template<typename base_t> struct softmax_parts_EPS {};
template<> struct softmax_parts_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct softmax_parts_EPS<float> { static constexpr double eps = 1e-5; };
template<typename iMath>
void check_softmax_parts(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** checking softmax_parts() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	constexpr numel_cnt_t maxDataSizeForSt = 50000;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	std::vector<real_t> vec_max(rowsCnt), vec_den(rowsCnt), vec_num(dataSize);

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	{
		std::vector<real_t> vec_den2(rowsCnt), vec_num2(dataSize);

		for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
			rg.gen_matrix(A, 2);
			mrwMax_ET(A, &vec_max[0]);

			softmax_parts_ET(A, &vec_max[0], &vec_den[0], &vec_num[0]);

			std::fill(vec_den2.begin(), vec_den2.end(), real_t(0));
			std::fill(vec_num2.begin(), vec_num2.end(), real_t(0));
			softmax_parts_st_cw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
			ASSERT_VECTOR_NEAR(vec_den, vec_den2, "st_cw() failed denominator vector comparision", softmax_parts_EPS<real_t>::eps);
			ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st_cw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);

			std::fill(vec_den2.begin(), vec_den2.end(), real_t(0));
			std::fill(vec_num2.begin(), vec_num2.end(), real_t(0));
			softmax_parts_st_rw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
			ASSERT_VECTOR_NEAR(vec_den, vec_den2, "st_rw() failed denominator vector comparision", softmax_parts_EPS<real_t>::eps);
			ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st_rw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);
		}
	}

	tictoc tStCw, tStRw;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2);
		mrwMax_ET(A, &vec_max[0]);
		std::fill(vec_den.begin(), vec_den.end(), real_t(0));
		std::fill(vec_num.begin(), vec_num.end(), real_t(0));
		tStCw.tic();
		softmax_parts_st_cw(A, &vec_max[0], &vec_den[0], &vec_num[0]);
		tStCw.toc();

		rg.gen_matrix(A, 2);
		mrwMax_ET(A, &vec_max[0]);
		std::fill(vec_den.begin(), vec_den.end(), real_t(0));
		std::fill(vec_num.begin(), vec_num.end(), real_t(0));
		tStRw.tic();
		softmax_parts_st_rw(A, &vec_max[0], &vec_den[0], &vec_num[0]);
		tStRw.toc();
	}
	tStCw.say("st_cw");
	tStRw.say("st_rw");
}
TEST(TestPerfDecisions, softmaxParts) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;
	check_softmax_parts(iM, 100, 100);

#ifndef TESTS_SKIP_LONGRUNNING
	constexpr vec_len_t maxCol = 10;
	for (unsigned c = 2; c <= maxCol; ++c)check_softmax_parts(iM, 200, c);
	check_softmax_parts(iM, 200, 100);
	check_softmax_parts(iM, 10000, 2);
#endif	
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void mrwIdxsOfMax_st_rw(const realmtx_t& m, vec_len_t* pDest)noexcept {
	const auto rim = m.rows();
	const auto ne = m.numel();
	//NNTL_ASSERT(rows == dest.size());

	auto pD = m.data();
	for (vec_len_t ri = 0; ri < rim; ++ri) {
		auto pV = &pD[ri];
		const auto pVEnd = pV + ne;
		auto m = std::numeric_limits<real_t>::lowest();
		vec_len_t mIdx = 0;
		vec_len_t c = 0;
		while (pV != pVEnd) {
			const auto v = *pV;
			pV += rim;
			if (v > m) {
				m = v;
				mIdx = c;
			}
			c++;
		}
		pDest[ri] = mIdx;
	}
}
//all parameters are mandatory.
//almost always the best (and calc max&idx simultaneously)
void mrwMax_st_memfriendly(const realmtx_t& m, real_t* pMax, vec_len_t* pDest)noexcept {
	const auto rm = m.rows(), cm = m.cols();
	auto p = m.data();
	const auto pE = p + m.numel();
	//treat the first column like the max. Then compare other columns with this column and update max'es
	memset(pDest, 0, sizeof(vec_len_t)*rm);
	memcpy(pMax, p, sizeof(real_t)*rm);
	
	p += rm;
	vec_len_t c = 1;
	while (p != pE) {
		for (vec_len_t r = 0; r < rm; ++r) {
			const auto v = p[r];
			const auto pM = pMax + r;
			if (v > *pM) {
				*pM = v;
				pDest[r] = c;
			}
		}
		++c;
		p += rm;
	}
}

/*
 // works noticeably slower, no need to use this variant.
void mrwMax_st_memfriendly_opt(const realmtx_t& m, real_t* pMax, vec_len_t* pDest=nullptr)noexcept {
	const auto rm = m.rows(), cm = m.cols();
	auto p = m.data();
	const auto pE = p + m.numel();
	//treat the first column like the max. Then compare other columns with this column and update max'es
	if (pDest) memset(pDest, 0, sizeof(vec_len_t)*rm);

	memcpy(pMax, p, sizeof(real_t)*rm);

	p += rm;
	vec_len_t c = 1;
	while (p != pE) {
		for (vec_len_t r = 0; r < rm; ++r) {
			const auto v = p[r];
			const auto pM = pMax + r;
			if (v > *pM) {
				*pM = v;
				if(pDest) pDest[r] = c;
			}
		}
		++c;
		p += rm;
	}
}*/

template<typename iMath>
void check_maxRowwise(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking max_rowwise() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	std::vector<vec_len_t> idxs_st_naive(rowsCnt), idxs_st_memf(rowsCnt);
	std::vector<real_t> max_st_memf(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	//vec_len_t* pDummy = nullptr;
	{
		std::vector<vec_len_t> idxs_et(rowsCnt);
		std::vector<real_t> max_et(rowsCnt);

		for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
			//pDummy++;

			rg.gen_matrix(A, 10);
			mrwMax_ET(A, &max_et[0], &idxs_et[0]);

			mrwIdxsOfMax_st_rw(A, &idxs_st_naive[0]);
			mrwMax_st_memfriendly(A, &max_st_memf[0], &idxs_st_memf[0]);

 			ASSERT_EQ(idxs_et, idxs_st_naive) << "mrwIdxsOfMax_st_rw failed!";
 			ASSERT_EQ(idxs_et, idxs_st_memf) << "mrwMax_st_memfriendly failed idx comparison";
 			ASSERT_EQ(max_et, max_st_memf) << "mrwMax_st_memfriendly failed value comparison";
		}
// 		pDummy -= testCorrRepCnt;
// 		STDCOUTL( "pDummy == " << size_t(pDummy) );
	}

	tictoc tStNaive, tStMemf, tStMemfOpt;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 10);
		tStNaive.tic();
		mrwIdxsOfMax_st_rw(A, &idxs_st_naive[0]);
		tStNaive.toc();

		rg.gen_matrix(A, 10);
		tStMemf.tic();
		mrwMax_st_memfriendly(A, &max_st_memf[0], &idxs_st_memf[0]);
		tStMemf.toc();

// 		rg.gen_matrix(A, 10);
// 		tStMemfOpt.tic();
// 		mrwMax_st_memfriendly_opt(A, &max_st_memf[0], pDummy);
// 		tStMemfOpt.toc();
// 		//works noticeably slower, no need to use this variant.
	}
	tStNaive.say("st_naive");
	tStMemf.say("st_memf");
	//tStMemfOpt.say("st_memfopt");
}
TEST(TestPerfDecisions, MaxRowwise) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;
	NNTL_RUN_TEST2(100*100, 100) check_maxRowwise(iM, i, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	NNTL_RUN_TEST2(10000 * 100, 100) check_maxRowwise(iM, i, 100);
	NNTL_RUN_TEST2(50000 * 10, 10) check_maxRowwise(iM, i, 10);
	NNTL_RUN_TEST2(50000 * 2, 2) check_maxRowwise(iM, i, 2);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename iMath>
inline void evCMulSub_st(iMath& iM, realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
	iM.evMulC_ip_st_naive(vW, momentum);
	iM.evSub_ip_st_naive(W, vW);
}
inline void evcombCMulSub(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
	NNTL_ASSERT(vW.size() == W.size());
	auto pV = vW.data();
	const auto pVE = pV + vW.numel();
	auto pW = W.data();
	while (pV != pVE) {
		const auto v = *pV * momentum;
		*pV++ = v;
		*pW++ -= v;
	}
}
template<typename iMath>
void check_evCMulSub(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking evCMulSub() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	//this is to test which implementation of combined operation
	//		vW = momentum.*vW
	//		W = W-vW
	// is better: operation-wise, or combined

	const real_t momentum = real_t(0.95);
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t vW(rowsCnt, colsCnt), W(colsCnt, rowsCnt), vW2(colsCnt, rowsCnt), W2(colsCnt, rowsCnt);
	ASSERT_TRUE(!vW.isAllocationFailed() && !W.isAllocationFailed() && !vW2.isAllocationFailed() && !W2.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(vW2, 2);
	rg.gen_matrix(W2, 2);
	vW2.clone_to(vW);
	W2.clone_to(W);

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
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 50; i <= 10000; i*=2) check_evCMulSub(iM, i,100);
	check_evCMulSub(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	check_evCMulSub(iM, 1000);
	check_evCMulSub(iM, 10000);
	//check_evCMulSub(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void mTranspose_ET(const realmtx_t& src, realmtx_t& dest) noexcept {
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const auto sRows = src.rows(), sCols = src.cols();
	for (vec_len_t r = 0; r < sRows; ++r) {
		for (vec_len_t c = 0; c < sCols; ++c) {
			dest.set(c, r, src.get(r, c));
		}
	}
}

void mTranspose_seq_read(const realmtx_t& src, realmtx_t& dest) noexcept {
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const auto sRows = src.rows(), sCols = src.cols();
	const auto dataCnt = src.numel();
	auto pSrc = src.data();
	const auto pSrcE = pSrc + dataCnt;
	auto pDest = dest.data();
	
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
void mTranspose_seq_write(const realmtx_t& src, realmtx_t& dest) noexcept {
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const auto sRows = src.rows(), sCols = src.cols();
	const auto dataCnt = src.numel();
	auto pSrc = src.data();
	auto pDest = dest.data();
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
void mTranspose_OpenBLAS(const realmtx_t& src, realmtx_t& dest) noexcept {
	const auto sRows = src.rows(), sCols = src.cols();
	math::b_OpenBLAS::omatcopy(true, sRows, sCols, real_t(1.0), src.data(), sRows, dest.data(), sCols);
}

template<typename iMath>
void check_mTranspose(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking mTranspose() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t src(rowsCnt, colsCnt), dest(colsCnt, rowsCnt), destEt(colsCnt, rowsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !destEt.isAllocationFailed());
	
	d_interfaces::iRng_t rg;
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
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 100; i <= 10000; i*=10) check_mTranspose(iM, i,i/10);
	check_mTranspose(iM, 100, 100);
	//check_mTranspose(iM, 10000,1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_mTranspose(iM, 1000);
	check_mTranspose(iM, 10000);
	//check_mTranspose(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//normalization of row-vectors of a matrix to max possible length
//static constexpr real_t rowvecs_renorm_MULT = real_t(1.0);
/*
real_t rowvecs_renorm_ET(realmtx_t& m, real_t* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows(), mCols = m.cols();
	for (vec_len_t r = 0; r < mRows; ++r) {
		pTmp[r] = real_t(0.0);
		for (vec_len_t c = 0; c < mCols; ++c) {
			auto v = m.get(r, c);
			pTmp[r] += v*v;
		}
	}

	//finding average norm
	real_t meanNorm = std::accumulate(pTmp, pTmp+mRows, 0.0) / mRows;

	//test and renormalize
	//const real_t newNorm = meanNorm - sqrt(math::real_t_limits<real_t>::eps_lower_n(meanNorm, rowvecs_renorm_MULT));
	const real_t newNorm = meanNorm - 2*sqrt(math::real_t_limits<real_t>::eps_lower(meanNorm));
	for (vec_len_t r = 0; r < mRows; ++r) {
		if (pTmp[r] > meanNorm) {
			const real_t normCoeff = sqrt(newNorm / pTmp[r]);
			real_t nn = 0;
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

/*
 *dont need it to run anymore
//slow
void rowvecs_renorm_naive(realmtx_t& m, real_t maxLenSquared, real_t* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows();
	memset(pTmp, 0, sizeof(real_t)*mRows);
	const auto dataCnt = m.numel();
	const real_t* pCol = m.data();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const real_t* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//test and renormalize
	//const real_t newNorm = maxLenSquared - sqrt(math::real_t_limits<real_t>::eps_lower_n(maxLenSquared, rowvecs_renorm_MULT));
	const real_t newNorm = maxLenSquared - 2*sqrt(math::real_t_limits<real_t>::eps_lower(maxLenSquared));
	auto pRow = m.data();
	const auto pRowE = pRow + mRows;
	while (pRow!=pRowE) {
		const auto rowNorm = *pTmp++;
		if (rowNorm > maxLenSquared) {
			const real_t normCoeff = sqrt(newNorm / rowNorm);
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
void rowvecs_renorm_clmnw(realmtx_t& A, real_t maxNormSquared, real_t* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = A.rows();
	memset(pTmp, 0, sizeof(real_t)*mRows);
	const auto dataCnt = A.numel();
	real_t* pCol = A.data();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const real_t* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, that doesn't need.
	//const real_t newNorm = maxNormSquared - sqrt(math::real_t_limits<real_t>::eps_lower_n(maxNormSquared, rowvecs_renorm_MULT));
	const real_t newNorm = maxNormSquared - 2*sqrt(math::real_t_limits<real_t>::eps_lower(maxNormSquared));
	auto pCurNorm = pTmp;
	const auto pTmpE = pTmp + mRows;
	while (pCurNorm != pTmpE) {
		const auto rowNorm = *pCurNorm;
		*pCurNorm++ = rowNorm > maxNormSquared ? sqrt(newNorm / rowNorm) : real_t(1.0);
	}

	//renormalize
	pCol = A.data();
	while (pCol != pColE) {
		real_t* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			*pElm++ *= *pN++;
		}
	}
}
//slower, probably don't vectorize correctly
void rowvecs_renorm_clmnw2(realmtx_t& m, real_t maxLenSquared, real_t* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows();
	memset(pTmp, 0, sizeof(real_t)*mRows);
	const auto dataCnt = m.numel();
	real_t* pCol = m.data();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const real_t* pElm = pCol;
		pCol += mRows;
		const auto pElmE = pCol;
		auto pN = pTmp;
		while (pElm != pElmE) {
			const auto v = *pElm++;
			*pN++ += v*v;
		}
	}

	//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, that doesn't need.
	//const real_t newNorm = maxLenSquared - sqrt(math::real_t_limits<real_t>::eps_lower_n(maxLenSquared, rowvecs_renorm_MULT));
	const real_t newNorm = maxLenSquared - 2*sqrt(math::real_t_limits<real_t>::eps_lower(maxLenSquared));
	auto pCurNorm = pTmp;
	const auto pTmpE = pTmp + mRows;
	while (pCurNorm != pTmpE) {
		const auto rowNorm = *pCurNorm;
		*pCurNorm++ = rowNorm > maxLenSquared ? sqrt(newNorm / rowNorm) : real_t(1.0);
	}

	//renormalize
	auto pElm = m.data();
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
void rowvecs_renorm_clmnw_part(realmtx_t& m, real_t maxLenSquared, real_t* pTmp, size_t* pOffs)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows();
	memset(pTmp, 0, sizeof(real_t)*mRows);
	const auto dataCnt = m.numel();
	real_t* pCol = m.data();
	const auto pColE = pCol + dataCnt;
	while (pCol != pColE) {
		const real_t* pElm = pCol;
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
	//const real_t newNorm = maxLenSquared - sqrt(math::real_t_limits<real_t>::eps_lower_n(maxLenSquared, rowvecs_renorm_MULT));
	const real_t newNorm = maxLenSquared - 2*sqrt(math::real_t_limits<real_t>::eps_lower(maxLenSquared));
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
		pCol = m.data();
		while (pCol != pColE) {
			real_t* pElm = pCol;
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
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking rowvecs_renorm() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	//#TODO: newNormSq might not be good here
	const real_t scale = 5, newNormSq=1;
	realmtx_t W(rowsCnt, colsCnt), srcW(rowsCnt, colsCnt), etW(rowsCnt, colsCnt);
	ASSERT_TRUE(!W.isAllocationFailed() && !srcW.isAllocationFailed() && !etW.isAllocationFailed());
	std::vector<real_t> tmp(rowsCnt);
	std::vector<size_t> ofs(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	steady_clock::time_point bt;
	nanoseconds diffNaive(0), diffClmnw(0), diffClmnw2(0), diffClmnwPart(0);

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(srcW, scale);

		srcW.clone_to(etW);
		const real_t meanNorm = rowvecs_renorm_ET(etW, newNormSq, true, &tmp[0]);

		srcW.clone_to(W);
		bt = steady_clock::now();
		rowvecs_renorm_naive(W, newNormSq, &tmp[0]);
		diffNaive += steady_clock::now() - bt;
		//ASSERT_EQ(etW, W) << "rowvecs_renorm_naive";
		ASSERT_MTX_EQ(etW, W, "rowvecs_renorm_naive");

		srcW.clone_to(W);
		bt = steady_clock::now();
		rowvecs_renorm_clmnw(W, newNormSq, &tmp[0]);
		diffClmnw += steady_clock::now() - bt;
		//ASSERT_EQ(etW, W) << "rowvecs_renorm_clmnw";
		ASSERT_MTX_EQ(etW, W, "rowvecs_renorm_clmnw");

		srcW.clone_to(W);
		bt = steady_clock::now();
		rowvecs_renorm_clmnw2(W, newNormSq, &tmp[0]);
		diffClmnw2 += steady_clock::now() - bt;
		//ASSERT_EQ(etW, W) << "rowvecs_renorm_clmnw2";
		ASSERT_MTX_EQ(etW, W, "rowvecs_renorm_clmnw2");

		srcW.clone_to(W);
		bt = steady_clock::now();
		rowvecs_renorm_clmnw_part(W, newNormSq, &tmp[0], &ofs[0]);
		diffClmnwPart += steady_clock::now() - bt;
		//ASSERT_EQ(etW, W) << "rowvecs_renorm_clmnw_part";
		ASSERT_MTX_EQ(etW, W, "rowvecs_renorm_clmnw_part");
	}

	STDCOUTL("naive:\t" << utils::duration_readable(diffNaive, maxReps));
	STDCOUTL("clmnw:\t" << utils::duration_readable(diffClmnw, maxReps));
	STDCOUTL("clmnw2:\t" << utils::duration_readable(diffClmnw2, maxReps));
	STDCOUTL("clmnwPart:\t" << utils::duration_readable(diffClmnwPart, maxReps));

}
TEST(TestPerfDecisions, rowvecsRenorm) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

//   	for (unsigned i = 10; i <= 1000; i*=10) check_rowvecs_renorm(iM, i,i);
//   	check_rowvecs_renorm(iM, 4000, 4000);

	check_rowvecs_renorm(iM, 100, 10);
	//check_rowvecs_renorm(iM, 1000, 1000);
	//check_rowvecs_renorm(iM, 10000, 1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_rowvecs_renorm(iM, 1000,100);
	//check_rowvecs_renorm(iM, 10000,100);
#endif
}
}*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//calculation of squared norm of row-vectors of a matrix. size(pNorm)==m.rows()
void rowwise_normsq_ET(const realmtx_t& m, real_t* pNorm)noexcept {
	const auto mRows = m.rows(), mCols = m.cols();
	for (vec_len_t r = 0; r < mRows; ++r) {
		pNorm[r] = real_t(0.0);
		for (vec_len_t c = 0; c < mCols; ++c) {
			auto v = m.get(r, c);
			pNorm[r] += v*v;
		}
	}
}
//slow
void rowwise_normsq_naive(const realmtx_t& m, real_t* pNorm)noexcept {
	const auto dataCnt = m.numel();
	const auto mRows = m.rows();
	const real_t* pRow = m.data();
	const auto pRowEnd = pRow + mRows;
	while (pRow != pRowEnd) {
		const real_t* pElm = pRow;
		const auto pElmEnd = pRow++ + dataCnt;
		real_t cs = real_t(0.0);
		while (pElm != pElmEnd) {
			const auto v = *pElm;
			pElm += mRows;
			cs += v*v;
		}
		*pNorm++ = cs;
	}
}
//best
void rowwise_normsq_clmnw(const realmtx_t& m, real_t* pNorm)noexcept {
	const auto mRows = m.rows();
	memset(pNorm, 0, sizeof(real_t)*mRows);

	const auto dataCnt = m.numel();
	const real_t* pCol = m.data();
	const auto pColE = pCol + dataCnt;
	while (pCol!=pColE) {
		const real_t* pElm = pCol;
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
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking rowwise_normsq() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	real_t scale = 5;
	realmtx_t W(rowsCnt, colsCnt);
	ASSERT_TRUE(!W.isAllocationFailed());
	std::vector<real_t> normvecEt(rowsCnt), normvec(rowsCnt);
	
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	rg.gen_matrix(W, scale);
	rowwise_normsq_ET(W, &normvecEt[0]);
	real_t meanNorm = std::accumulate(normvecEt.begin(), normvecEt.end(), real_t(0.0)) / rowsCnt;
	STDCOUTL("Mean norm value is "<< meanNorm);

	std::fill(normvec.begin(), normvec.end(), real_t(10.0));
	rowwise_normsq_naive(W, &normvec[0]);
	ASSERT_TRUE(0 == memcmp(&normvec[0], &normvecEt[0], rowsCnt*sizeof(real_t))) << "rowwise_normsq_naive wrong implementation";

	std::fill(normvec.begin(), normvec.end(), real_t(10.0));
	rowwise_normsq_clmnw(W, &normvec[0]);
	ASSERT_TRUE(0 == memcmp(&normvec[0], &normvecEt[0], rowsCnt*sizeof(real_t))) << "rowwise_normsq_clmnw wrong implementation";

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
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 10; i <= 10000; i*=10) check_rowwiseNormsq(iM, i,i);
	check_rowwiseNormsq(iM, 100, 100);
	//check_rowwiseNormsq(iM, 10000, 1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_rowwiseNormsq(iM, 1000);
	check_rowwiseNormsq(iM, 10000);
	//check_rowwiseNormsq(iM, 100000);
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
real_t sigm_loss_xentropy_naive(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.data(), ptrY = data_y.data();
	constexpr auto log_zero = math::real_t_limits<real_t>::log_almost_zero;
	real_t ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto a = ptrA[i], y = ptrY[i]; // , oma = real_t(1.0) - a;
		NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
		NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

		//ql += y*(a == real_t(0.0) ? log_zero : log(a)) + (real_t(1.0) - y)*(oma == real_t(0.0) ? log_zero : log(oma));
		ql += y*(a == real_t(0.0) ? log_zero : std::log(a)) + (real_t(1.0) - y)*(a == real_t(1.0) ? log_zero : nntl::math::log1p(-a));
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
//best when data_y skewed to 1s or 0s. Slightly slower than sigm_loss_xentropy_naive when data_y has equal amount of 1 and 0
real_t sigm_loss_xentropy_naive_part(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.data(), ptrY = data_y.data();
	constexpr auto log_zero = math::real_t_limits<real_t>::log_almost_zero;
	real_t ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		auto a = ptrA[i];
		NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
		NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

		if (y > real_t(0.0)) {
			ql += (a == real_t(0.0) ? log_zero : std::log(a));
		} else {
			//const auto oma = real_t(1.0) - a;
			//ql += (oma == real_t(0.0) ? log_zero : log(oma));
			ql += (a == real_t(1.0) ? log_zero : nntl::math::log1p(-a));
		}
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
//unaffected by data_y distribution, but slowest
real_t sigm_loss_xentropy_vec(const realmtx_t& activations, const realmtx_t& data_y, realmtx_t& t1, realmtx_t& t2)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	NNTL_ASSERT(t1.size() == activations.size() && t2.size() == t1.size());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.data(), ptrY = data_y.data();
	const auto p1 = t1.data(), p2 = t2.data();
	constexpr auto realmin = std::numeric_limits<real_t>::min();
	real_t ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto a = ptrA[i], oma = real_t(1.0) - a;
		p1[i] = (a == real_t(0.0) ? realmin : a);
		p2[i] = (oma == real_t(0.0) ? realmin : oma);
	}
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		ql += y*std::log(p1[i]) + (real_t(1.0) - y)*std::log(p2[i]);
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
template<typename base_t> struct run_sigm_loss_xentropy_EPS {};
template<> struct run_sigm_loss_xentropy_EPS<double> { static constexpr double eps = 1e-8; };
template<> struct run_sigm_loss_xentropy_EPS<float> { static constexpr double eps = 8e-4; };
template <typename iRng, typename iMath>
void run_sigm_loss_xentropy(iRng& rg, iMath &iM, realmtx_t& act, realmtx_t& data_y, realmtx_t& t1, realmtx_t& t2,unsigned maxReps,real_t binFrac)noexcept {

	STDCOUTL("binFrac = "<<binFrac);

	steady_clock::time_point bt;
	nanoseconds diffNaive(0), diffPart(0), diffVec(0);
	real_t lossNaive, lossPart, lossVec;

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix_norm(act);
		rg.gen_matrix_norm(data_y);
		iM.ewBinarize_ip(data_y, binFrac);

		bt = steady_clock::now();
		lossNaive = sigm_loss_xentropy_naive(act, data_y);
		diffNaive += steady_clock::now() - bt;

		bt = steady_clock::now();
		lossPart = sigm_loss_xentropy_naive_part(act, data_y);
		diffPart += steady_clock::now() - bt;

		bt = steady_clock::now();
		lossVec = sigm_loss_xentropy_vec(act, data_y, t1, t2);
		diffVec += steady_clock::now() - bt;

		ASSERT_NEAR(lossNaive, lossPart, run_sigm_loss_xentropy_EPS<real_t>::eps);
		ASSERT_NEAR(lossNaive, lossVec, run_sigm_loss_xentropy_EPS<real_t>::eps);
	}

	STDCOUTL("naive:\t" << utils::duration_readable(diffNaive, maxReps));
	STDCOUTL("part:\t" << utils::duration_readable(diffPart, maxReps));
	STDCOUTL("vec:\t" << utils::duration_readable(diffVec, maxReps));

}
template<typename iMath>
void check_sigm_loss_xentropy(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking sigm_loss_xentropy() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t act(rowsCnt, colsCnt), data_y(rowsCnt, colsCnt), t1(rowsCnt, colsCnt), t2(rowsCnt, colsCnt);
	ASSERT_TRUE(!act.isAllocationFailed() && !data_y.isAllocationFailed() && !t1.isAllocationFailed() && !t2.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	run_sigm_loss_xentropy(rg, iM, act, data_y, t1, t2, maxReps, real_t(.5));
	run_sigm_loss_xentropy(rg, iM, act, data_y, t1, t2, maxReps, real_t(.1));
	run_sigm_loss_xentropy(rg, iM, act, data_y, t1, t2, maxReps, real_t(.9));
}
TEST(TestPerfDecisions, sigmLossXentropy) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 50; i <= 20000; i*=1.5) check_sigm_loss_xentropy(iM, i,100);
	check_sigm_loss_xentropy(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	check_sigm_loss_xentropy(iM, 1000);
	check_sigm_loss_xentropy(iM, 10000);
	//check_sigm_loss_xentropy(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void apply_momentum_FOR(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
	NNTL_ASSERT(vW.size() == dW.size());
	NNTL_ASSERT(!vW.empty() && !dW.empty());

	const auto dataCnt = vW.numel();
	const auto pV = vW.data();
	const auto pdW = dW.data();
	for (numel_cnt_t i = 0; i < dataCnt;++i) {
		pV[i] = momentum*pV[i] + pdW[i];
	}
}
void apply_momentum_WHILE(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
	NNTL_ASSERT(vW.size() == dW.size());
	NNTL_ASSERT(!vW.empty() && !dW.empty());

	const auto dataCnt = vW.numel();
	auto pV = vW.data();
	const auto pVE = pV + dataCnt;
	auto pdW = dW.data();
	while (pV!=pVE) {
		*pV++ = momentum*(*pV) + *pdW++;
	}
}
template<typename iMath>
void check_apply_momentum_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking apply_momentum() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tFOR, tWHILE;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	real_t momentum= real_t(.9);

	realmtx_t dW(rowsCnt, colsCnt), vW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !vW.isAllocationFailed());

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());

	d_interfaces::iRng_t rg;
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
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 50; i <= 20000; i*=2) check_apply_momentum_perf(iM, i,100);
	check_apply_momentum_perf(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	check_apply_momentum_perf(iM, 1000);
	check_apply_momentum_perf(iM, 10000);
	//check_apply_momentum_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename T> T sgncopysign(T magn, T val) {
	return val == 0 ? T(0) : std::copysign(magn, val);
}
template<typename iMath>
void test_sign_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* checking sign() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tSgn, tSgncopysign;// , tBoost;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t dW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	real_t lr = real_t(.1);

	real_t pz = real_t(+0.0), nz = real_t(-0.0), p1 = real_t(1), n1 = real_t(-1);

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
		auto p = dW.data();
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
		auto p = dW.data();
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
		auto p = dW.data();
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
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 100; i <= 6400; i*=2) test_sign_perf(iM, i,100);

	test_sign_perf(iM, 100, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_sign_perf(iM, 1000);
	test_sign_perf(iM, 10000);
	//test_sign_perf(iM, 100000);
#endif
}


//////////////////////////////////////////////////////////////////////////
/*

template<typename iMath>
void calc_rmsproph_vec(iMath& iM, typename iMath::realmtx_t& dW, typename iMath::realmtx_t& rmsF, value_type lr,
	value_type emaDecay, value_type numericStab, typename iMath::realmtx_t& t1)
{
	typedef typename iMath::realmtx_t realmtx_t;
	typedef typename realmtx_t::value_type real_t;
	typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

	NNTL_ASSERT(dW.size() == rmsF.size());

	auto pdW = dW.data();
	auto prmsF = rmsF.data();
	auto pt = t1.data();
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
void calc_rmsproph_ew(iMath& iM, typename iMath::realmtx_t& dW, typename iMath::realmtx_t& rmsF, const value_type learningRate,
	const value_type emaDecay, const value_type numericStabilizer)
{
	typedef typename iMath::realmtx_t realmtx_t;
	typedef typename realmtx_t::value_type real_t;
	typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

	NNTL_ASSERT(dW.size() == rmsF.size());

	auto pdW = dW.data();
	auto prmsF = rmsF.data();
	const auto _1_emaDecay = 1 - emaDecay;
	for (numel_cnt_t i = 0, im = dW.numel(); i < im; ++i) {
		const auto w = pdW[i];
		const auto rms = emaDecay*prmsF[i] + w*w*_1_emaDecay;
		prmsF[i] = rms;
		pdW[i] = learningRate*(w / (sqrt(rms) + numericStabilizer));
	}
}

template<typename iMath>
void test_rmsproph_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef typename iMath::realmtx_t realmtx_t;
	typedef typename realmtx_t::value_type real_t;
	typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

	using namespace std::chrono;
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RMSPropHinton variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tBatch, tEw;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_REPEATS_COUNT;

	real_t emaCoeff = .9, lr=.1, numStab=.00001;

	realmtx_t dW(true, rowsCnt, colsCnt), rms(true, rowsCnt, colsCnt), t1(true, rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed());
	
	rms.zeros();

	d_interfaces::iRng_t rg;
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
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;

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
// dropout_batch is significantly faster
/*
template<typename iRng_t, typename iMath_t>
void dropout_batch(realmtx_t& activs, real_t dropoutFraction, realmtx_t& dropoutMask, iRng_t& iR, iMath_t& iM) {
	NNTL_ASSERT(activs.size() == dropoutMask.size());
	NNTL_ASSERT(activs.emulatesBiases() && dropoutMask.emulatesBiases());
	NNTL_ASSERT(dropoutFraction > 0 && dropoutFraction < 1);

	iR.gen_matrix_no_bias_gtz(dropoutMask, 1);
	auto pDM = dropoutMask.data();
	const auto pDME = pDM + dropoutMask.numel_no_bias();
	while (pDM != pDME) {
		auto v = *pDM;
		*pDM++ = v > dropoutFraction ? real_t(1.0) : real_t(0.0);
	}
	iM.evMul_ip_st_naive(activs, dropoutMask);
}

template<typename iRng_t>
void dropout_ew(realmtx_t& activs, real_t dropoutFraction, realmtx_t& dropoutMask, iRng_t& iR) {
	NNTL_ASSERT(activs.size() == dropoutMask.size());
	NNTL_ASSERT(activs.emulatesBiases() && dropoutMask.emulatesBiases());
	NNTL_ASSERT(dropoutFraction > 0 && dropoutFraction < 1);

	const auto dataCnt = activs.numel_no_bias();
	auto pDM = dropoutMask.data();
	auto pA = activs.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
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
void test_dropout_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef typename iMath::realmtx_t realmtx_t;
	typedef typename realmtx_t::value_type real_t;
	typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

	using namespace std::chrono;
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing dropout variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tBatch, tEw;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	real_t dropoutFraction = .5;

	realmtx_t act(rowsCnt, colsCnt, true), dm(rowsCnt, colsCnt, true);
	ASSERT_TRUE(!act.isAllocationFailed() && !dm.isAllocationFailed());


	d_interfaces::iRng_t rg;
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
*/

/*
 * this test is a BS
TEST(TestPerfDecisions, Dropout) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 100; i <= 140; i+=1) test_dropout_perf(iM, i,100);

	test_dropout_perf(iM, 100, 10);
#ifndef TESTS_SKIP_LONGRUNNING
	test_dropout_perf(iM, 1000);
	test_dropout_perf(iM, 10000);
	//test_dropout_perf(iM, 100000);
#endif
}*/
//////////////////////////////////////////////////////////////////////////

#include "../nntl/weights_init.h"
#include "../nntl/activation.h"

static void pt_ileakyrelu_st(realmtx_t& srcdest, const real_t leak) noexcept {
	NNTL_ASSERT(!srcdest.empty());
	NNTL_ASSERT(leak > real_t(0.0));
	auto pV = srcdest.data();
	const auto pVE = pV + srcdest.numel();
	while (pV != pVE) {
		const auto v = *pV;
		/*if (v < real_t(+0.0))  *pV = v*leak; //this code doesn't vectorize and work about x10 times slower on my HW
		++pV;*/
		*pV++ = v < real_t(+0.0) ? v*leak : v; //this one however does vectorize
	}
}
template<typename FunctorT>
void pt_iact_asymm_st(realmtx_t& srcdest, FunctorT&& fnc) noexcept {
	NNTL_ASSERT(!srcdest.empty());
	auto pV = srcdest.data();
	const auto pVE = pV + srcdest.numel();
	while (pV != pVE) {
		const auto v = *pV;
		//*pV++ = v < real_t(+0.0) ? FunctorT::f_neg(v) : FunctorT::f_pos(v);
		*pV++ = (std::forward<FunctorT>(fnc)).f( v );
	}
}
//slightly faster (177vs192)
template<typename RealT, size_t LeakKInv100 = 10000, typename WeightsInitScheme = weights_init::He_Zhang<>>
class exp_leaky_relu : public activation::_i_activation<RealT> {
	exp_leaky_relu() = delete;
	~exp_leaky_relu() = delete;
public:
	typedef WeightsInitScheme weights_scheme;
	static constexpr real_t LeakK = real_t(100.0) / real_t(LeakKInv100);

public:
	static void f(realmtx_t& srcdest) noexcept {
		pt_ileakyrelu_st(srcdest, LeakK);
	};
};
//slightly slower (192vs177)
/*
template<typename RealT, size_t LeakKInv100 = 10000, typename WeightsInitScheme = weights_init::He_Zhang<>>
class exp2_leaky_relu : public activation::_i_activation<RealT> {
	exp2_leaky_relu() = delete;
	~exp2_leaky_relu() = delete;
public:
	typedef WeightsInitScheme weights_scheme;
	static constexpr real_t LeakK = real_t(100.0) / real_t(LeakKInv100);

	struct LRFunc {
		static constexpr real_t f_pos(const real_t& x)noexcept { return x; }
		static constexpr real_t f_neg(const real_t& x)noexcept { return x*LeakK; }

		static constexpr real_t df_pos(const real_t& fv)noexcept { return real_t(1.); }
		static constexpr real_t df_neg(const real_t& fv)noexcept { return LeakK; }
	};

public:
	static void f(realmtx_t& srcdest) noexcept {
		pt_iact_asymm_st<LRFunc>(srcdest);
	};
};*/
//well, this one and current pt_iact_asymm_st() is a bit better and approximately as fast as plain version.
//however, better fire me than make me refactor the old code now... Leave it for a future.
template<typename RealT, size_t LeakKInv100 = 10000, typename WeightsInitScheme = weights_init::He_Zhang<>>
class exp3_leaky_relu : public activation::_i_activation<RealT> {
	exp3_leaky_relu() = delete;
	~exp3_leaky_relu() = delete;
public:
	typedef WeightsInitScheme weights_scheme;
	static constexpr real_t LeakK = real_t(100.0) / real_t(LeakKInv100);

	struct LRFunc {
		static constexpr real_t f(const real_t x)noexcept { return x < real_t(+0.0) ? x*LeakK : x; }
	};

public:
	static void f(realmtx_t& srcdest) noexcept {
		pt_iact_asymm_st(srcdest, LRFunc());
	};
};
void test_ActPrmVsNonprm_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** testing test_ActPrmVsNonprm_perf() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	realmtx_t XSrc(rowsCnt, colsCnt), X(rowsCnt, colsCnt), TV(rowsCnt, colsCnt);
	ASSERT_TRUE(!XSrc.isAllocationFailed() && !X.isAllocationFailed() && !TV.isAllocationFailed());
	
	typedef exp_leaky_relu<real_t> BType;
	typedef exp3_leaky_relu<real_t> AType;

	tictoc tA1, tB1, tA2, tB2, tA3, tB3;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, def_threads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix(XSrc, real_t(5.0));

			XSrc.clone_to(X);
			tA1.tic();
			AType::f(X);
			tA1.toc();

			XSrc.clone_to(X);
			tB1.tic();
			BType::f(X);
			tB1.toc();

			XSrc.clone_to(X);
			tA2.tic();
			AType::f(X);
			tA2.toc();

			XSrc.clone_to(X);
			tB2.tic();
			BType::f(X);
			tB2.toc();

			XSrc.clone_to(X);
			tA3.tic();
			AType::f(X);
			tA3.toc();
			X.clone_to(TV);

			XSrc.clone_to(X);
			tB3.tic();
			BType::f(X);
			tB3.toc();

			ASSERT_EQ(TV, X);
	}

	tA1.say("A1");
	tA2.say("A2");
	tA3.say("A3");
	tB1.say("B1");
	tB2.say("B2");
	tB3.say("B3");
}

TEST(TestPerfDecisions, ActPrmVsNonprm) {
	test_ActPrmVsNonprm_perf(100, 5);
	test_ActPrmVsNonprm_perf(100, 50);

	test_ActPrmVsNonprm_perf(10000, 5);
	test_ActPrmVsNonprm_perf(10000, 50);

	test_ActPrmVsNonprm_perf(100000, 5);
	test_ActPrmVsNonprm_perf(100000, 50);
}