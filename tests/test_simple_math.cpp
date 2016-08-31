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

// this file implements tests of correctness. Performance tests should be placed at _thr.cpp

#include "../nntl/math.h"
#include "../nntl/common.h"

#include "../nntl/interface/math/simple_math.h"
#include "../nntl/interfaces.h"
#include "../nntl/_supp/io/matfile.h"

#include "simple_math_etalons.h"
#include "common_routines.h"

using namespace nntl;

typedef d_interfaces::iThreads_t iThreads_t;
typedef math::simple_math < real_t, iThreads_t> simple_math_t;

static simple_math_t iM;
const vec_len_t g_MinDataSizeDelta = 2 * iM.ithreads().workers_count() + 2;

#ifdef TESTS_SKIP_LONGRUNNING
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 30, _baseRowsCnt = 30;
#else
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 60, _baseRowsCnt = 300;
#endif // NNTL_DEBUG

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/*
TEST(TestSimpleMath, DumpmTilingRoll) {
	constexpr vec_len_t k = 5, r = 2, c = 3;
	realmtx_t src(r, k*c, true), dest(k*r, c, true);
	ASSERT_TRUE(!src.isAllocationFailed());
	seqFillMtx(src);

	mTilingRoll_ET(src, dest);

	nntl_supp::omatfile<> mf;
	ASSERT_EQ(mf.ErrorCode::Success, mf.openForSave("./test_data/test.mat"));

	mf << NNTL_SERIALIZATION_NVP(src);
	mf << NNTL_SERIALIZATION_NVP(dest);
}*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_mTilingUnroll_corr(const vec_len_t maxSrcRows, const vec_len_t maxSrcCols, const vec_len_t maxK) {
	ASSERT_TRUE(maxSrcRows && maxSrcCols && maxK > 1);

	for (vec_len_t k = 2; k < maxK; ++k) {
		for (vec_len_t srcRows = 1; srcRows < maxSrcRows; ++srcRows) {
			for (vec_len_t _srcCols = 1; _srcCols < maxSrcCols; ++_srcCols) {
				for (unsigned char _b = 0; _b < 2; ++_b) {
					const bool bBiased = !!_b;

					constexpr unsigned _scopeMsgLen = 128; \
						char _scopeMsg[_scopeMsgLen]; \
						sprintf_s(_scopeMsg, "mTilingUnroll (%s) dest(%d,%d)*%d <- src(%d,%d)", bBiased ? "biased" : "not biased"
							, srcRows, k*_srcCols + _b, k, k*srcRows, _srcCols + _b); \
						SCOPED_TRACE(_scopeMsg);

					realmtx_t dest(srcRows, k*_srcCols, bBiased), destET(srcRows, k*_srcCols, bBiased), src(k*srcRows, _srcCols, bBiased);
					ASSERT_TRUE(!dest.isAllocationFailed() && !destET.isAllocationFailed() && !src.isAllocationFailed());

					seqFillMtx(src);

					mTilingUnroll_ET(src, destET);
					if (bBiased) {
						ASSERT_TRUE(src.test_biases_ok());
						ASSERT_TRUE(destET.test_biases_ok());
					}

					dest.zeros();
					iM.mTilingUnroll_seqread_st(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqread_st()");

					dest.zeros();
					iM.mTilingUnroll_seqwrite_st(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqwrite_st()");

					dest.zeros();
					iM.mTilingUnroll_seqread_mt(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqread_mt()");

					dest.zeros();
					iM.mTilingUnroll_seqwrite_mt(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqwrite_mt()");

					dest.zeros();
					iM.mTilingUnroll_st(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_st()");

					dest.zeros();
					iM.mTilingUnroll_mt(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_mt()");

					dest.zeros();
					iM.mTilingUnroll(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "()");
				}
			}
		}
	}
}

void test_mTilingEtalons(const vec_len_t maxSrcRows, const vec_len_t maxSrcCols, const vec_len_t maxK) {
	ASSERT_TRUE(maxSrcRows && maxSrcCols && maxK > 1);

	for (vec_len_t k = 2; k < maxK; ++k) {
		for (vec_len_t srcRows = 1; srcRows < maxSrcRows; ++srcRows) {
			for (vec_len_t _srcCols = 1; _srcCols < maxSrcCols; ++_srcCols) {
				for (unsigned char _b = 0; _b < 2; ++_b) {
					const bool bBiased = !!_b;

					constexpr unsigned _scopeMsgLen = 128; \
						char _scopeMsg[_scopeMsgLen]; \
						sprintf_s(_scopeMsg, "test_mTilingEtalons (%s) src(%d,%d)*%d -> dest(%d,%d)", bBiased ? "biased" : "not biased"
							, srcRows, k*_srcCols + _b, k, k*srcRows, _srcCols + _b); \
						SCOPED_TRACE(_scopeMsg);

					realmtx_t src(srcRows, k*_srcCols, bBiased), destET(k*srcRows, _srcCols, bBiased),
						destET2(k*srcRows, _srcCols, bBiased), src2(srcRows, k*_srcCols, bBiased);
					ASSERT_TRUE(!src.isAllocationFailed() && !destET.isAllocationFailed() 
						&& !destET2.isAllocationFailed() && !src2.isAllocationFailed());

					seqFillMtx(src);

					mTilingRoll_ET(src, destET);
					if (bBiased) {
						ASSERT_TRUE(src.test_biases_ok());
						ASSERT_TRUE(destET.test_biases_ok());
					}
					mTilingUnroll_ET(destET, src2);
					ASSERT_MTX_EQ(src, src2, "Roll/Unroll failed");

					mTilingRoll_ET(src2, destET2);
					ASSERT_MTX_EQ(destET, destET2, "Unroll/roll failed");
				}
			}
		}
	}
}

TEST(TestSimpleMath, mTilingUnroll) {
	ASSERT_NO_FATAL_FAILURE(test_mTilingEtalons(4, 4, 4));
	ASSERT_NO_FATAL_FAILURE(test_mTilingUnroll_corr(2 * g_MinDataSizeDelta, 2 * g_MinDataSizeDelta, g_MinDataSizeDelta));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_mTilingRoll_corr(const vec_len_t maxSrcRows,const vec_len_t maxSrcCols, const vec_len_t maxK) {
	ASSERT_TRUE(maxSrcRows && maxSrcCols && maxK>1);

	for (vec_len_t k = 2; k < maxK;++k) {
		for (vec_len_t srcRows = 1; srcRows < maxSrcRows; ++srcRows) {
			for (vec_len_t _srcCols = 1; _srcCols < maxSrcCols; ++_srcCols) {
				for (unsigned char _b = 0; _b < 2;++_b) {
					const bool bBiased = !!_b;

					constexpr unsigned _scopeMsgLen = 128; \
						char _scopeMsg[_scopeMsgLen]; \
						sprintf_s(_scopeMsg, "mTilingRoll (%s) src(%d,%d)*%d -> dest(%d,%d)", bBiased?"biased":"not biased"
							, srcRows, k*_srcCols + _b, k, k*srcRows, _srcCols + _b); \
						SCOPED_TRACE(_scopeMsg);

					realmtx_t src(srcRows, k*_srcCols, bBiased), destET(k*srcRows,_srcCols,bBiased), dest(k*srcRows, _srcCols, bBiased);
					ASSERT_TRUE(!src.isAllocationFailed() && !destET.isAllocationFailed() && !dest.isAllocationFailed());

					seqFillMtx(src);

					mTilingRoll_ET(src, destET);
					if (bBiased) {
						ASSERT_TRUE(src.test_biases_ok());
						ASSERT_TRUE(destET.test_biases_ok());
					}

					dest.zeros();
					iM.mTilingRoll_seqread_st(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqread_st()");

					dest.zeros();
					iM.mTilingRoll_seqwrite_st(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqwrite_st()");

					dest.zeros();
					iM.mTilingRoll_seqread_mt(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqread_mt()");

					dest.zeros();
					iM.mTilingRoll_seqwrite_mt(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_seqwrite_mt()");

					dest.zeros();
					iM.mTilingRoll_st(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_st()");

					dest.zeros();
					iM.mTilingRoll_mt(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "_mt()");

					dest.zeros();
					iM.mTilingRoll(src, dest);
					if (bBiased) ASSERT_TRUE(dest.test_biases_ok());
					ASSERT_MTX_EQ(destET, dest, "()");
				}
			}
		}
	}
}
TEST(TestSimpleMath, mTilingRoll) {
	ASSERT_NO_FATAL_FAILURE(test_mTilingRoll_corr(2 * g_MinDataSizeDelta, 2 * g_MinDataSizeDelta, g_MinDataSizeDelta));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_mCloneCol_corr(vec_len_t srcRowsCnt, vec_len_t maxCloneCnt = 1, vec_len_t minCloneCnt = 1) {
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t src(srcRowsCnt, 1);
	ASSERT_TRUE(!src.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	const vec_len_t ccSpan = maxCloneCnt - minCloneCnt;

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(src, 10);

		vec_len_t destColsCnt = minCloneCnt + (ccSpan ? static_cast<vec_len_t>(rg.gen_i(ccSpan)) : 0);
		
		constexpr unsigned _scopeMsgLen = 128; \
			char _scopeMsg[_scopeMsgLen]; \
			sprintf_s(_scopeMsg, "mCloneCol src(%d,1)->dest(%d,%d)", srcRowsCnt, srcRowsCnt, destColsCnt); \
			SCOPED_TRACE(_scopeMsg);

		realmtx_t dest(srcRowsCnt, destColsCnt), destET(srcRowsCnt, destColsCnt);
		ASSERT_TRUE(!dest.isAllocationFailed() && !destET.isAllocationFailed());

		mCloneCol_ET(src, destET);

		dest.zeros();
		iM.mCloneCol_st(src, dest);
		ASSERT_MTX_EQ(destET, dest, "st() failed");

		dest.zeros();
		iM.mCloneCol_mt(src, dest);
		ASSERT_MTX_EQ(destET, dest, "mt() failed");

		dest.zeros();
		iM.mCloneCol(src, dest);
		ASSERT_MTX_EQ(destET, dest, "() failed");
	}
}
TEST(TestSimpleMath, mCloneCol) {
	constexpr unsigned rowsCnt = 10, maxCloneCnt = 15;
	for (vec_len_t cc = 1; cc <= maxCloneCnt; ++cc) {
		ASSERT_NO_FATAL_FAILURE(test_mCloneCol_corr(rowsCnt, cc));
	}
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_mCloneCols_corr(vec_len_t srcRowsCnt, vec_len_t srcColsCnt, vec_len_t maxCloneCnt=1, vec_len_t minCloneCnt = 1) {
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t src(srcRowsCnt, srcColsCnt);
	ASSERT_TRUE(!src.isAllocationFailed());
	std::vector<vec_len_t> colSpec(srcColsCnt);
	
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	const vec_len_t ccSpan = maxCloneCnt - minCloneCnt;

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(src, 10);

		//filling colSpec
		vec_len_t destColsCnt = 0;
		for (vec_len_t sc = 0; sc < srcColsCnt; ++sc) {
			const vec_len_t cc = minCloneCnt + (ccSpan ? static_cast<vec_len_t>(rg.gen_i(ccSpan)) : 0);
			destColsCnt += cc;
			colSpec[sc] = cc;
		}

		constexpr unsigned _scopeMsgLen = 128; \
			char _scopeMsg[_scopeMsgLen]; \
			sprintf_s(_scopeMsg, "mCloneCols src(%d,%d)->dest(%d,%d)", srcRowsCnt, srcColsCnt, srcRowsCnt, destColsCnt); \
			SCOPED_TRACE(_scopeMsg);

		realmtx_t dest(srcRowsCnt, destColsCnt), destET(srcRowsCnt, destColsCnt);
		ASSERT_TRUE(!dest.isAllocationFailed() && !destET.isAllocationFailed());

		mCloneCols_ET(src, destET, &colSpec[0]);

		dest.zeros();
		iM.mCloneCols_st(src, dest, &colSpec[0]);
		ASSERT_MTX_EQ(destET, dest, "st() failed");

		dest.zeros();
		iM.mCloneCols_mt(src, dest, &colSpec[0]);
		ASSERT_MTX_EQ(destET, dest, "mt() failed");

		dest.zeros();
		iM.mCloneCols(src, dest, &colSpec[0]);
		ASSERT_MTX_EQ(destET, dest, "() failed");
	}
}
TEST(TestSimpleMath, mCloneCols) {
	constexpr unsigned rowsCnt = 10, maxCloneCnt=15;
	const vec_len_t maxCols = g_MinDataSizeDelta;
	for (vec_len_t c = 1; c < maxCols; ++c) {
		for (vec_len_t cc = 1; cc <= maxCloneCnt; ++cc) {
			ASSERT_NO_FATAL_FAILURE(test_mCloneCols_corr(rowsCnt, c, cc));
		}
	}
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct ewSumProd_EPS {};
template<> struct ewSumProd_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct ewSumProd_EPS<float> { static constexpr float eps = 3e-2f; };
void test_ewSumProd_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "ewSumProd");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt), B(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !B.isAllocationFailed());
	real_t s_et, s;

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 10);
		rg.gen_matrix(B, 10);

		s_et = ewSumProd_ET(A,B);

		s = iM.ewSumProd_st(A,B);
		ASSERT_NEAR(s_et, s, ewSumProd_EPS<real_t>::eps) << "st() failed";

		s = iM.ewSumProd_mt(A, B);
		ASSERT_NEAR(s_et, s, ewSumProd_EPS<real_t>::eps) << "mt() failed";

		s = iM.ewSumProd(A, B);
		ASSERT_NEAR(s_et, s, ewSumProd_EPS<real_t>::eps) << "() failed";
	}
}
TEST(TestSimpleMath, ewSumProd) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_ewSumProd_corr(r, c));
	}
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwDivideByVec_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "mrwDivideByVec");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt), A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !A2.isAllocationFailed() && !A3.isAllocationFailed());
	std::vector<real_t> vDiv(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 10);
		A.cloneTo(A3);
		rg.gen_vector(&vDiv[0], rowsCnt, 5);

		mrwDivideByVec_ET(A, &vDiv[0]);

		A3.cloneTo(A2);
		iM.mrwDivideByVec_st_cw(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "st_cw() failed");

		A3.cloneTo(A2);
		iM.mrwDivideByVec_st_rw(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "st_rw() failed");

		A3.cloneTo(A2);
		iM.mrwDivideByVec_st(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "st() failed");

		A3.cloneTo(A2);
		iM.mrwDivideByVec_mt_cw(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "mt_cw() failed");
				
		A3.cloneTo(A2);
		iM.mrwDivideByVec_mt_rw(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "mt_rw() failed");

		A3.cloneTo(A2);
		iM.mrwDivideByVec_mt(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "mt() failed");

		A3.cloneTo(A2);
		iM.mrwDivideByVec(A2, &vDiv[0]);
		ASSERT_MTX_EQ(A, A2, "() failed");
	}
}
TEST(TestSimpleMath, mrwDivideByVec) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_mrwDivideByVec_corr(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwMulByVec_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "mrwMulByVec");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt), A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !A2.isAllocationFailed() && !A3.isAllocationFailed());
	std::vector<real_t> vMul(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 10);
		A.cloneTo(A3);
		rg.gen_vector(&vMul[0], rowsCnt, 5);

		mrwMulByVec_ET(A, &vMul[0]);

		A3.cloneTo(A2);
		iM.mrwMulByVec_st_cw(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "st_cw() failed");
						
		A3.cloneTo(A2);
		iM.mrwMulByVec_st_rw(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "st_rw() failed");
				
		A3.cloneTo(A2);
		iM.mrwMulByVec_st(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "st() failed");

		A3.cloneTo(A2);
		iM.mrwMulByVec_mt_cw(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "mt_cw() failed");

		A3.cloneTo(A2);
		iM.mrwMulByVec_mt_rw(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "mt_rw() failed");

		A3.cloneTo(A2);
		iM.mrwMulByVec_mt(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "mt() failed");

		A3.cloneTo(A2);
		iM.mrwMulByVec(A2, &vMul[0]);
		ASSERT_MTX_EQ(A, A2, "() failed");
	}
}
TEST(TestSimpleMath, mrwMulByVec) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_mrwMulByVec_corr(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwIdxsOfMaxCorrectness(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef std::vector<realmtx_t::vec_len_t> vec_t;
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "mrwIdxsOfMax");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());

	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	vec_t vec_et(rowsCnt), vec_test(rowsCnt);

	for (unsigned tr = 0; tr < testCorrRepCnt; ++tr) {
		rg.gen_matrix(A, 1000);
		mrwMax_ET(A, nullptr, &vec_et[0]);

		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax_st_cw(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st_cw";
		
		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax_st_rw(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st_rw";

		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax_st_rw_small(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st_rw_small";

		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax_st(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st";

		if (colsCnt > simple_math_t::Thresholds_t::mrwIdxsOfMax_ColsPerThread) {
			std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
			iM.mrwIdxsOfMax_mt_cw(A, &vec_test[0]);
			ASSERT_EQ(vec_et, vec_test) << "mt_cw";

			std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
			iM.mrwIdxsOfMax_mt_cw_small(A, &vec_test[0]);
			ASSERT_EQ(vec_et, vec_test) << "mt_cw_small";
		}

		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax_mt_rw(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "mt_rw";

		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax_mt(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "mt";

		std::fill(vec_test.begin(), vec_test.end(), vec_t::value_type(-1));
		iM.mrwIdxsOfMax(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "()";
	}
}

TEST(TestSimpleMath, mrwIdxsOfMax) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_mrwIdxsOfMaxCorrectness(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mrwMax_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef std::vector<real_t> vec_t;
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "mrwMax");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT/2;
	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());

	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	vec_t vec_et(rowsCnt), vec_test(rowsCnt);

	for (unsigned tr = 0; tr < testCorrRepCnt; ++tr) {
		rg.gen_matrix(A, 1000);
		mrwMax_ET(A, &vec_et[0]);

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax_st_cw(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st_cw";

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax_st_rw_small(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st_rw_small";

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax_st_rw(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st_rw";

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax_st(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "st";

		if (colsCnt > simple_math_t::Thresholds_t::mrwMax_mt_cw_ColsPerThread) {
			std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
			iM.mrwMax_mt_cw(A, &vec_test[0]);
			ASSERT_EQ(vec_et, vec_test) <<"mt_cw";
		}

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax_mt_rw(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "mt_rw";

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax_mt(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) << "mt";

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::lowest());
		iM.mrwMax(A, &vec_test[0]);
		ASSERT_EQ(vec_et, vec_test) <<"()";
	}
}

TEST(TestSimpleMath, mrwMax) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta*simple_math_t::Thresholds_t::mrwMax_mt_cw_ColsPerThread, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_mrwMax_corr(r, c));
	}
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct mrwSumIp_EPS {};
template<> struct mrwSumIp_EPS<double> { static constexpr double eps = 1e-12; };
template<> struct mrwSumIp_EPS<float> { static constexpr float eps = 5e-5f; };
void test_mrwSumIp_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "mrwSum_ip");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt), A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !A2.isAllocationFailed() && !A3.isAllocationFailed());
	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 10);
		A.cloneTo(A3);

		mrwSum_ip_ET(A);

		if (colsCnt>1) {
			A3.cloneTo(A2);
			iM.mrwSum_ip_st_cw(A2);
			ASSERT_REALMTX_NEAR(A, A2, "st_cw() failed", mrwSumIp_EPS<real_t>::eps);
		}

		A3.cloneTo(A2);
		iM.mrwSum_ip_st_rw(A2);
		ASSERT_REALMTX_NEAR(A, A2, "st_rw() failed", mrwSumIp_EPS<real_t>::eps);

		A3.cloneTo(A2);
		iM.mrwSum_ip_st_rw_small(A2);
		ASSERT_REALMTX_NEAR(A, A2, "st_rw_small() failed", mrwSumIp_EPS<real_t>::eps);

		A3.cloneTo(A2);
		iM.mrwSum_ip_st(A2);
		ASSERT_REALMTX_NEAR(A, A2, "st() failed", mrwSumIp_EPS<real_t>::eps);

		if (colsCnt > simple_math_t::Thresholds_t::mrwSum_mt_cw_colsPerThread) {//mrwSum, not _ip_! because it's just a thunk to mrwSum_mt_cw
			A3.cloneTo(A2);
			iM.mrwSum_ip_mt_cw(A2);
			ASSERT_REALMTX_NEAR(A, A2, "mt_cw() failed", mrwSumIp_EPS<real_t>::eps);
		}

		A3.cloneTo(A2);
		iM.mrwSum_ip_mt_rw(A2);
		ASSERT_REALMTX_NEAR(A, A2, "mt_rw() failed", mrwSumIp_EPS<real_t>::eps);

		A3.cloneTo(A2);
		iM.mrwSum_ip_mt(A2);
		ASSERT_REALMTX_NEAR(A, A2, "mt() failed", mrwSumIp_EPS<real_t>::eps);

		A3.cloneTo(A2);
		iM.mrwSum_ip(A2);
		ASSERT_REALMTX_NEAR(A, A2, "() failed", mrwSumIp_EPS<real_t>::eps);
	}
}
TEST(TestSimpleMath, mrwSumIp) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_mrwSumIp_corr(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct mrwSum_EPS {};
template<> struct mrwSum_EPS<double> { static constexpr double eps = 1e-12; };
template<> struct mrwSum_EPS<float> { static constexpr float eps = 8e-5f; };
void test_mrwSum_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "mrwSum");
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	std::vector<real_t> vec_et(rowsCnt), vec_test(rowsCnt);
	d_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 10);
		mrwSum_ET(A, &vec_et[0]);

		if (colsCnt>1) {
			std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
			iM.mrwSum_st_cw(A, &vec_test[0]);
			ASSERT_VECTOR_NEAR(vec_et, vec_test, "st_cw() failed", mrwSum_EPS<real_t>::eps);

			std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
			iM.mrwSum_st_rw(A, &vec_test[0]);
			ASSERT_VECTOR_NEAR(vec_et, vec_test, "st_rw() failed", mrwSum_EPS<real_t>::eps);
		}

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
		iM.mrwSum_st(A, &vec_test[0]);
		ASSERT_VECTOR_NEAR(vec_et, vec_test, "st() failed", mrwSum_EPS<real_t>::eps);

		if (colsCnt > simple_math_t::Thresholds_t::mrwSum_mt_cw_colsPerThread) {
			std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
			iM.mrwSum_mt_cw(A, &vec_test[0]);
			ASSERT_VECTOR_NEAR(vec_et, vec_test, "mt_cw() failed", mrwSum_EPS<real_t>::eps);
		}

		if (colsCnt>1) {
			std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
			iM.mrwSum_mt_rw(A, &vec_test[0]);
			ASSERT_VECTOR_NEAR(vec_et, vec_test, "mt_rw() failed", mrwSum_EPS<real_t>::eps);
		}

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
		iM.mrwSum_mt(A, &vec_test[0]);
		ASSERT_VECTOR_NEAR(vec_et, vec_test, "mt() failed", mrwSum_EPS<real_t>::eps);

		std::fill(vec_test.begin(), vec_test.end(), std::numeric_limits<real_t>::infinity());
		iM.mrwSum(A, &vec_test[0]);
		ASSERT_VECTOR_NEAR(vec_et, vec_test, "() failed", mrwSum_EPS<real_t>::eps);
	}
}
TEST(TestSimpleMath, mrwSum) {
	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta*simple_math_t::Thresholds_t::mrwSum_mt_cw_colsPerThread
		, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_mrwSum_corr(r, c));
	}
}