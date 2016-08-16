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
#include "../nntl/common.h"

#include "../nntl/interface/math/simple_math.h"
#include "../nntl/nnet_def_interfaces.h"

#include "../nntl/utils/prioritize_workers.h"
#include "../nntl/utils/tictoc.h"

#include "simple_math_etalons.h"
#include "common_routines.h"

using namespace nntl;
using namespace nntl::utils;

typedef nnet_def_interfaces::iThreads_t iThreads_t;
typedef math::simple_math < real_t, iThreads_t> simple_math_t;

static simple_math_t iM;

#ifdef TESTS_SKIP_LONGRUNNING
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 1000;
#endif // NNTL_DEBUG

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mTilingRoll(vec_len_t rowsCnt, vec_len_t colsCnt, vec_len_t k) {
	const auto dataSize = k*realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** testing mTilingRoll() variations over " << rowsCnt << "x" << colsCnt << " matrix, k=" << k << " tiles(" << dataSize << " elements) ****");

	constexpr unsigned maxIntReps = 5;
	constexpr unsigned maxReps = 1*TEST_PERF_REPEATS_COUNT / maxIntReps;

	realmtx_t src(rowsCnt, colsCnt*k), dest(k*rowsCnt, colsCnt), destET(k*rowsCnt, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !destET.isAllocationFailed());

	tictoc tSt, tMt, tB, tStSR, tMtSR, tStSW, tMtSW;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, simple_math_t::ithreads_t> pw(iM.ithreads());

	seqFillMtx(src);
	for (unsigned r = 0; r < maxReps; ++r) {
		if (!r) {
			mTilingRoll_ET(src, destET);
			dest.zeros();
		}

		tStSR.tic();
		for (unsigned ir = 0; ir < maxIntReps;++ir) iM.mTilingRoll_seqread_st(src, dest);
		tStSR.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest,"_seqread_st");
			dest.zeros();
		}

		tStSW.tic();
		for (unsigned ir = 0; ir < maxIntReps; ++ir)iM.mTilingRoll_seqwrite_st(src, dest);
		tStSW.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest, "_seqwrite_st");
			dest.zeros();
		}

		tMtSR.tic();
		for (unsigned ir = 0; ir < maxIntReps; ++ir)iM.mTilingRoll_seqread_mt(src, dest);
		tMtSR.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest, "_seqread_mt");
			dest.zeros();
		}

		tMtSW.tic();
		for (unsigned ir = 0; ir < maxIntReps; ++ir)iM.mTilingRoll_seqwrite_mt(src, dest);
		tMtSW.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest, "_seqwrite_mt");
			dest.zeros();
		}
		
		tSt.tic();
		for (unsigned ir = 0; ir < maxIntReps; ++ir)iM.mTilingRoll_st(src, dest);
		tSt.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest, "_st");
			dest.zeros();
		}

		tMt.tic();
		for (unsigned ir = 0; ir < maxIntReps; ++ir)iM.mTilingRoll_mt(src, dest);
		tMt.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest, "_mt");
			dest.zeros();
		}

		tB.tic();
		for (unsigned ir = 0; ir < maxIntReps; ++ir)iM.mTilingRoll(src, dest);
		tB.toc();
		if (!r) {
			ASSERT_MTX_EQ(destET, dest, "()");
			dest.zeros();
		}
	}
	tStSR.say("StSR");
	tStSW.say("StSW");
	tMtSR.say("MtSR");
	tMtSW.say("MtSW");
	tSt.say("st");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mTilingRoll) {
	for (unsigned k = 2; k < 10; ++k) {
		NNTL_RUN_TEST2(simple_math_t::Thresholds_t::mTilingRoll, k*100) test_mTilingRoll(100, i, k);
	}

// 	test_mTilingRoll(70000, 8, 6);
// 	test_mTilingRoll(70000, 8, 10);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_ewSumProd(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** testing ewSumProd() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt), B(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	real_t s = 0;

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	tictoc tSt, tMt, tB;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, simple_math_t::ithreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2);		rg.gen_matrix(B, 2);
		tSt.tic();
		s += iM.ewSumProd_st(A, B);
		tSt.toc();

		rg.gen_matrix(A, 2);		rg.gen_matrix(B, 2);
		tMt.tic();
		s += iM.ewSumProd_mt(A, B);
		tMt.toc();

		rg.gen_matrix(A, 2);		rg.gen_matrix(B, 2);
		tB.tic();
		s += iM.ewSumProd(A, B);
		tB.toc();
	}
	tSt.say("st");
	tMt.say("mt");
	tB.say("best");
	STDCOUTL(s);
}
TEST(TestSimpleMathThr, ewSumProd) {
	NNTL_RUN_TEST2(simple_math_t::Thresholds_t::ewSumProd, 10) test_ewSumProd(i, 10);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwDivideByVec(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** testing mrwDivideByVec() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT / 3;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	std::vector<real_t> vDiv(rowsCnt);

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	tictoc tStCw, tStRw, tSt, tMtCw, tMtRw, tMt, tB;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, simple_math_t::ithreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tStCw.tic();
		iM.mrwDivideByVec_st_cw(A, &vDiv[0]);
		tStCw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tStRw.tic();
		iM.mrwDivideByVec_st_rw(A, &vDiv[0]);
		tStRw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tSt.tic();
		iM.mrwDivideByVec_st(A, &vDiv[0]);
		tSt.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tMtCw.tic();
		iM.mrwDivideByVec_mt_cw(A, &vDiv[0]);
		tMtCw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tMtRw.tic();
		iM.mrwDivideByVec_mt_rw(A, &vDiv[0]);
		tMtRw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tMt.tic();
		iM.mrwDivideByVec_mt(A, &vDiv[0]);
		tMt.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vDiv[0], rowsCnt, 5);
		tB.tic();
		iM.mrwDivideByVec(A, &vDiv[0]);
		tB.toc();
	}
	tStCw.say("st_cw");
	tStRw.say("st_rw");
	tSt.say("st");
	tMtCw.say("mt_cw");
	tMtRw.say("mt_rw");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mrwDivideByVec) {
	test_mrwDivideByVec(100, 5);
	test_mrwDivideByVec(100, 10);
	test_mrwDivideByVec(200, 5);
	test_mrwDivideByVec(200, 10);
	test_mrwDivideByVec(200, 50);
	test_mrwDivideByVec(100, 100);

#ifndef TESTS_SKIP_LONGRUNNING
	//test_mrwDivideByVec(100000, 4);
	//test_mrwDivideByVec(4, 100000);

	test_mrwDivideByVec(3000, 300);
	test_mrwDivideByVec(300, 3000);

// 	test_mrwDivideByVec(10000, 300);
// 	test_mrwDivideByVec(300, 10000);
	//test_mrwDivideByVec(100, 100000);

	// 	test_mrwDivideByVec(10000, 3000);
	// 	test_mrwDivideByVec(3000, 10000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwMulByVec(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** testing mrwMulByVec() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT / 3;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	std::vector<real_t> vMul(rowsCnt);

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	tictoc tStCw, tStRw, tSt, tMtCw, tMtRw, tMt, tB;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, simple_math_t::ithreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tStCw.tic();
		iM.mrwMulByVec_st_cw(A, &vMul[0]);
		tStCw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tStRw.tic();
		iM.mrwMulByVec_st_rw(A, &vMul[0]);
		tStRw.toc();
		
		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tSt.tic();
		iM.mrwMulByVec_st(A, &vMul[0]);
		tSt.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tMtCw.tic();
		iM.mrwMulByVec_mt_cw(A, &vMul[0]);
		tMtCw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tMtRw.tic();
		iM.mrwMulByVec_mt_rw(A, &vMul[0]);
		tMtRw.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tMt.tic();
		iM.mrwMulByVec_mt(A, &vMul[0]);
		tMt.toc();

		rg.gen_matrix(A, 10);		rg.gen_vector(&vMul[0], rowsCnt, 5);
		tB.tic();
		iM.mrwMulByVec(A, &vMul[0]);
		tB.toc();
	}
	tStCw.say("st_cw");
	tStRw.say("st_rw");
	tSt.say("st");
	tMtCw.say("mt_cw");
	tMtRw.say("mt_rw");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mrwMulByVec) {
	constexpr unsigned maxCols = 10;
	for (unsigned i = 1; i <= maxCols; ++i) test_mrwMulByVec(100, i);
	for (unsigned i = 2; i <= maxCols; ++i) test_mrwMulByVec(400, i);

#ifndef TESTS_SKIP_LONGRUNNING
	test_mrwMulByVec(simple_math_t::Thresholds_t::mrwMulByVec_st_rows - 1, 6);
	test_mrwMulByVec(simple_math_t::Thresholds_t::mrwMulByVec_st_rows - 1, 10);
	//test_mrwMulByVec(simple_math_t::Thresholds_t::mrwMulByVec_st_rows - 1, 64);

	test_mrwMulByVec(simple_math_t::Thresholds_t::mrwMulByVec_st_rows, 6);
	test_mrwMulByVec(simple_math_t::Thresholds_t::mrwMulByVec_st_rows, 10);
	//test_mrwMulByVec(simple_math_t::Thresholds_t::mrwMulByVec_st_rows, 64);

// 	for (unsigned i = 2; i <= maxCols; ++i) test_mrwMulByVec(100000, i);
// 	test_mrwMulByVec(3000, 300);
// 	test_mrwMulByVec(300, 3000);

#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwIdxsOfMax_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	STDCOUTL("**** testing mrwIdxsOfMax() over " << rowsCnt << "x" << colsCnt << " matrix (" << realmtx_t::sNumel(rowsCnt, colsCnt) << " elements) ****");
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	std::vector<vec_len_t> idxs(rowsCnt);

	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	tictoc tStCw, tMtCw, tB, tMtRw, tStRw, tMt, tSt, tStRwSmall, tMtCwSmall;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iThreads_t> pw(iM.ithreads());

	for (unsigned r = 0; r < maxReps; ++r) {
		
		std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
		tStRwSmall.tic();
		iM.mrwIdxsOfMax_st_rw_small(A, &idxs[0]);
		tStRwSmall.toc();
		
		std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
		tStRw.tic();
		iM.mrwIdxsOfMax_st_rw(A, &idxs[0]);
		tStRw.toc();

		std::fill(idxs.begin(), idxs.end(), vec_len_t(0)); 		rg.gen_matrix(A, 10);
		tStCw.tic();
		iM.mrwIdxsOfMax_st_cw(A, &idxs[0]);
		tStCw.toc();

		std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
		tSt.tic();
		iM.mrwIdxsOfMax_st(A, &idxs[0]);
		tSt.toc();

		if (colsCnt > simple_math_t::Thresholds_t::mrwIdxsOfMax_ColsPerThread) {
			std::fill(idxs.begin(), idxs.end(), vec_len_t(0));			rg.gen_matrix(A, 10);
			tMtCw.tic();
			iM.mrwIdxsOfMax_mt_cw(A, &idxs[0]);
			tMtCw.toc();

			std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
			tMtCwSmall.tic();
			iM.mrwIdxsOfMax_mt_cw_small(A, &idxs[0]);
			tMtCwSmall.toc();
		}

		std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
		tMtRw.tic();
		iM.mrwIdxsOfMax_mt_rw(A, &idxs[0]);
		tMtRw.toc();

		std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
		tMt.tic();
		iM.mrwIdxsOfMax_mt(A, &idxs[0]);
		tMt.toc();

		std::fill(idxs.begin(), idxs.end(), vec_len_t(0));		rg.gen_matrix(A, 10);
		tB.tic();
		iM.mrwIdxsOfMax(A, &idxs[0]);
		tB.toc();
	}
	tStCw.say("st_cw");
	tStRwSmall.say("st_rw_small");
	tStRw.say("st_rw");
	tSt.say("st");
	tMtCw.say("mt_cw");
	tMtCwSmall.say("mt_cw_small");
	tMtRw.say("mt_rw");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mrwIdxsOfMax) {
	NNTL_RUN_TEST4(simple_math_t::Thresholds_t::mrwIdxsOfMax, 5, 2.5, simple_math_t::Thresholds_t::mrwIdxsOfMax_ColsPerThread * 6)
		test_mrwIdxsOfMax_perf(i, simple_math_t::Thresholds_t::mrwIdxsOfMax_ColsPerThread * 6);
	NNTL_RUN_TEST4(simple_math_t::Thresholds_t::mrwIdxsOfMax, 5, 2.5, 10) test_mrwIdxsOfMax_perf(i, 10);
	NNTL_RUN_TEST4(simple_math_t::Thresholds_t::mrwIdxsOfMax, 5, 2.5, 2) test_mrwIdxsOfMax_perf(i, 2);
#ifndef TESTS_SKIP_LONGRUNNING
// 	test_mrwIdxsOfMax_perf(300, 3000);
// 	test_mrwIdxsOfMax_perf(3000, 300);
	test_mrwIdxsOfMax_perf(100, 10000);
	test_mrwIdxsOfMax_perf(10000, 100);
// 	test_mrwIdxsOfMax_perf(10, 100000);
//	test_mrwIdxsOfMax_perf(100000, 10);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwMax_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef std::vector<real_t> vec_t;

	STDCOUTL("**** testing mrwMax() over " << rowsCnt << "x" << colsCnt << " matrix (" << realmtx_t::sNumel(rowsCnt, colsCnt) << " elements) ****");
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	vec_t vmax(rowsCnt);

	iM.preinit(m.numel());
	ASSERT_TRUE(iM.init());
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	tictoc tSt, tMt, tB, tMtRw, tMtCw, tStRwSmall, tStCw, tStRw;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iThreads_t> pw(iM.ithreads());

	for (unsigned r = 0; r < maxReps; ++r) {
		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest()); 		rg.gen_matrix(m, 10);
		tStRwSmall.tic();
		iM.mrwMax_st_rw_small(m, &vmax[0]);
		tStRwSmall.toc();

		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest()); 		rg.gen_matrix(m, 10);
		tStRw.tic();
		iM.mrwMax_st_rw(m, &vmax[0]);
		tStRw.toc();
		
		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest()); 		rg.gen_matrix(m, 10);
		tStCw.tic();
		iM.mrwMax_st_cw(m, &vmax[0]);
		tStCw.toc();

		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest()); 		rg.gen_matrix(m, 10);
		tSt.tic();
		iM.mrwMax_st(m, &vmax[0]);
		tSt.toc();

		if (colsCnt > simple_math_t::Thresholds_t::mrwMax_mt_cw_ColsPerThread) {
			std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest());			rg.gen_matrix(m, 10);
			tMtCw.tic();
			iM.mrwMax_mt_cw(m, &vmax[0]);
			tMtCw.toc();
		}

		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest());		rg.gen_matrix(m, 10);
		tMtRw.tic();
		iM.mrwMax_mt_rw(m, &vmax[0]);
		tMtRw.toc();

		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest());		rg.gen_matrix(m, 10);
		tMt.tic();
		iM.mrwMax_mt(m, &vmax[0]);
		tMt.toc();

		std::fill(vmax.begin(), vmax.end(), std::numeric_limits<real_t>::lowest());		rg.gen_matrix(m, 10);
		tB.tic();
		iM.mrwMax(m, &vmax[0]);
		tB.toc();
	}
	tStCw.say("st_cw");
	tStRw.say("st_rw");
	tStRwSmall.say("st_rw_small");
	tSt.say("st");
	tMtRw.say("mt_rw");	
	tMtCw.say("mt_cw");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mrwMax) {
	NNTL_RUN_TEST4(simple_math_t::Thresholds_t::mrwMax, 5, 2.5, simple_math_t::Thresholds_t::mrwMax_mt_cw_ColsPerThread * 6)
		test_mrwMax_perf(i, simple_math_t::Thresholds_t::mrwMax_mt_cw_ColsPerThread * 6);
	NNTL_RUN_TEST4(simple_math_t::Thresholds_t::mrwMax, 5, 2.5, 10) test_mrwMax_perf(i, 10);
	NNTL_RUN_TEST4(simple_math_t::Thresholds_t::mrwMax, 5, 2.5, 2) test_mrwMax_perf(i, 2);
#ifndef TESTS_SKIP_LONGRUNNING
	test_mrwMax_perf(100, 10000);
	test_mrwMax_perf(10000, 100);
// 	test_mrwMax_perf(10, 100000);
// 	test_mrwMax_perf(100000, 10);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_mrwSumIp_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	STDCOUTL("**** testing mrwSum_ip() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << realmtx_t::sNumel(rowsCnt, colsCnt) << " elements) ****");
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	tictoc tStCw, tStRw, tSt, tMtCw, tMtRw, tMt, tB, tStRwSmall;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iThreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		if (colsCnt>1) {
			rg.gen_matrix(A, 10);
			tStCw.tic();
			iM.mrwSum_ip_st_cw(A);
			tStCw.toc();
		}

		rg.gen_matrix(A, 10);
		tStRw.tic();
		iM.mrwSum_ip_st_rw(A);
		tStRw.toc();

		rg.gen_matrix(A, 10);
		tStRwSmall.tic();
		iM.mrwSum_ip_st_rw_small(A);
		tStRwSmall.toc();

		rg.gen_matrix(A, 10);
		tSt.tic();
		iM.mrwSum_ip_st(A);
		tSt.toc();

		if (colsCnt > simple_math_t::Thresholds_t::mrwSum_mt_cw_colsPerThread) {//mrwSum, not _ip_! because it's just a thunk to mrwSum_mt_cw
			rg.gen_matrix(A, 10);
			tMtCw.tic();
			iM.mrwSum_ip_mt_cw(A);
			tMtCw.toc();
		}

		rg.gen_matrix(A, 10);
		tMtRw.tic();
		iM.mrwSum_ip_mt_rw(A);
		tMtRw.toc();

		rg.gen_matrix(A, 10);
		tMt.tic();
		iM.mrwSum_ip_mt(A);
		tMt.toc();

		rg.gen_matrix(A, 10);
		tB.tic();
		iM.mrwSum_ip(A);
		tB.toc();
	}
	tStCw.say("st_cw");
	tStRw.say("st_rw");
	tStRwSmall.say("st_rw_small");
	tSt.say("st");
	tMtRw.say("mt_rw");
	tMtCw.say("mt_cw");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mrwSumIp) {
	const unsigned maxCols = iM.ithreads().workers_count();
	for (unsigned c = 2; c <= maxCols; ++c) test_mrwSumIp_perf(300, c);

#ifndef TESTS_SKIP_LONGRUNNING
	for (unsigned r = 400; r <= 1000; r += 100) test_mrwSumIp_perf(r, 6);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_mrwSum_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	STDCOUTL("**** testing mrwSum() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << realmtx_t::sNumel(rowsCnt, colsCnt) << " elements) ****");
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt);
	std::vector<real_t> vec_test(rowsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	tictoc tStCw, tStRw, tSt, tMtCw, tMtRw, tMt, tB;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iThreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		if (colsCnt > 1) {
			rg.gen_matrix(A, 10);
			tStCw.tic();
			iM.mrwSum_st_cw(A, &vec_test[0]);
			tStCw.toc();

			rg.gen_matrix(A, 10);
			tStRw.tic();
			iM.mrwSum_st_rw(A, &vec_test[0]);
			tStRw.toc();
		}

		rg.gen_matrix(A, 10);
		tSt.tic();
		iM.mrwSum_st(A, &vec_test[0]);
		tSt.toc();

		if (colsCnt > simple_math_t::Thresholds_t::mrwSum_mt_cw_colsPerThread) {
			rg.gen_matrix(A, 10);
			tMtCw.tic();
			iM.mrwSum_mt_cw(A, &vec_test[0]);
			tMtCw.toc();
		}

		if (colsCnt>1) {
			rg.gen_matrix(A, 10);
			tMtRw.tic();
			iM.mrwSum_mt_rw(A, &vec_test[0]);
			tMtRw.toc();
		}

		rg.gen_matrix(A, 10);
		tMt.tic();
		iM.mrwSum_mt(A, &vec_test[0]);
		tMt.toc();

		rg.gen_matrix(A, 10);
		tB.tic();
		iM.mrwSum(A, &vec_test[0]);
		tB.toc();
	}
	tStCw.say("st_cw");
	tStRw.say("st_rw");
	tSt.say("st");
	tMtCw.say("mt_cw");
	tMtRw.say("mt_rw");
	tMt.say("mt");
	tB.say("best");
}
TEST(TestSimpleMathThr, mrwSum) {
	const unsigned maxCols = iM.ithreads().workers_count();
	for (unsigned c = 2; c <= maxCols; ++c) test_mrwSum_perf(300, c);

#ifndef TESTS_SKIP_LONGRUNNING
	for (unsigned r = 400; r <= 1000; r += 100) {
		test_mrwSum_perf(r, 6);
		test_mrwSum_perf(r, 10);
		test_mrwSum_perf(r, 30);
	}
//  	test_mrwSum_perf(3000, 300);
//  	test_mrwSum_perf(300, 3000);
// 
// 	test_mrwSum_perf(100000, 10);
// 	test_mrwSum_perf(10, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

