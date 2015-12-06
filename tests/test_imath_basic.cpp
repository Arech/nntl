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

#include "../nntl/interface/math.h"
#include "../nntl/common.h"

#include "../nntl/interface/math/imath_basic.h"
#include "../nntl/nnet_def_interfaces.h"

#include "../nntl/_supp/jsonreader.h"

#include <array>
#include <numeric>

#include "../nntl/utils/chrono.h"
#include "../nntl/utils/prioritize_workers.h"

#include "etalons.h"

using namespace nntl;
using namespace std::chrono;

#ifdef _DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 100;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 50;
#endif // _DEBUG

void ASSERT_FLOATMTX_EQ(const realmtx_t& c1, const realmtx_t& c2, const char* descr, const real_t eps) noexcept{
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;

	const auto p1 = c1.dataAsVec(), p2 = c2.dataAsVec();
	const auto im = c1.numel();
	if (eps <= 0) {
		if (std::is_same<real_t,float>::value) {
			for (numel_cnt_t i = 0; i < im; ++i) {
				ASSERT_FLOAT_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
			}
		} else {
			for (numel_cnt_t i = 0; i < im; ++i) {
				ASSERT_DOUBLE_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
			}
		}
	} else {
		for (numel_cnt_t i = 0; i < im; ++i) {
			ASSERT_NEAR(p1[i], p2[i], eps) << "Mismatches element #" << i << " @ " << descr;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void evAdd_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept {
	NNTL_ASSERT(A.size() == B.size());

	const auto dataCnt = A.numel();
	const auto pA = A.dataAsVec();
	const auto pB = B.dataAsVec();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += pB[i];
}
template<typename iMath>
void test_evAdd_ip(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evAdd_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest; //, tstVec, tmtVec;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(B, 2);

	{
		realmtx_t A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
		ASSERT_TRUE(!A2.isAllocationFailed() && !A3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(A, 2);
			A.cloneTo(A2);
			A.cloneTo(A3);

			evAdd_ip_ET(A2, B);

			iM.evAdd_ip_st(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evAdd_ip_st failed correctness test");

			A3.cloneTo(A);
			iM.evAdd_ip_mt(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evAdd_ip_mt failed correctness test");

			A3.cloneTo(A);
			iM.evAdd_ip(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evAdd_ip failed correctness test");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evAdd_ip_st(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evAdd_ip_mt(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evAdd_ip(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evAddIp) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 160; i <= 240; i += 5) test_evAdd_ip(iM, i,100);
	test_evAdd_ip(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evAdd_ip(iM, 1000);
	test_evAdd_ip(iM, 10000);
	test_evAdd_ip(iM, 100000);
#endif
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void evCMulSub_ET(iMath& iM, realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
	iM.evMulC_ip_st_naive(vW, momentum);
	iM.evSub_ip_st_naive(W, vW);
}
template<typename iMath>
void test_evMulCipSubip(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing evMulC_ip_Sub_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	nanoseconds diffSt(0), diffMt(0), diffB(0);
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t momentum = .9;
	realmtx_t vW(rowsCnt, colsCnt), W(colsCnt, rowsCnt), vW2(colsCnt, rowsCnt), W2(colsCnt, rowsCnt), vW3(colsCnt, rowsCnt), W3(colsCnt, rowsCnt);
	ASSERT_TRUE(!vW.isAllocationFailed() && !W.isAllocationFailed() && !vW2.isAllocationFailed()
		&& !W2.isAllocationFailed() && !vW3.isAllocationFailed() && !W3.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(vW2, 2);
	rg.gen_matrix(W2, 2);
	vW2.cloneTo(vW);
	W2.cloneTo(W);
	vW2.cloneTo(vW3);
	W2.cloneTo(W3);

	for (unsigned r = 0; r < testCorrRepCnt; ++r) {
		evCMulSub_ET(iM, vW3, momentum, W3);
			
		iM.evMulC_ip_Sub_ip_st(vW, momentum, W);
		ASSERT_FLOATMTX_EQ(vW3, vW, "evMulC_ip_Sub_ip_st failed correctness test on vW");
		ASSERT_FLOATMTX_EQ(W3, W, "evMulC_ip_Sub_ip_st failed correctness test on W");

		iM.evMulC_ip_Sub_ip_mt(vW2, momentum, W2);
		ASSERT_FLOATMTX_EQ(vW3, vW2, "evMulC_ip_Sub_ip_mt failed correctness test on vW");
		ASSERT_FLOATMTX_EQ(W3, W2, "evMulC_ip_Sub_ip_mt failed correctness test on W");
	}

	rg.gen_matrix(vW2, 2);
	rg.gen_matrix(W2, 2);
	vW2.cloneTo(vW);
	W2.cloneTo(W);
	vW2.cloneTo(vW3);
	W2.cloneTo(W3);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		auto bt = steady_clock::now();
		iM.evMulC_ip_Sub_ip_st(vW, momentum, W);
		diffSt += steady_clock::now() - bt;

		bt = steady_clock::now();
		iM.evMulC_ip_Sub_ip_mt(vW2, momentum, W2);
		diffMt += steady_clock::now() - bt;

		bt = steady_clock::now();
		iM.evMulC_ip_Sub_ip(vW3, momentum, W3);
		diffB += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diffSt, maxReps));
	STDCOUTL("mt:\t" << utils::duration_readable(diffMt, maxReps));
	STDCOUTL("best:\t" << utils::duration_readable(diffB, maxReps));
}

TEST(TestIMathBasic, evMulCipSubip) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 100; i <= 200; i += 10) test_evMulCipSubip(iM, i, 100);
	test_evMulCipSubip(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evMulCipSubip(iM, 1000);
	test_evMulCipSubip(iM, 10000);
#endif
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
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
	real_t meanNorm = std::accumulate(pTmp, pTmp + mRows, 0.0) / mRows;

	//test and renormalize
	//const real_t newNorm = meanNorm - sqrt(math::real_ty_limits<real_t>::eps_lower_n(meanNorm, rowvecs_renorm_MULT));
	const real_t newNorm = meanNorm - sqrt(math::real_ty_limits<real_t>::eps_lower(meanNorm));
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
}
template<typename base_t> struct mCheck_normalize_rows_EPS {};
template<> struct mCheck_normalize_rows_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct mCheck_normalize_rows_EPS<float> { static constexpr double eps = 1e-5; };

template<typename iMath>
void test_mCheck_normalize_rows(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing mCheck_normalize_rows() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	steady_clock::time_point bt;
	nanoseconds diffSt(0), diffMt(0), diffB(0);
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t scale = 5;
	real_t renormTo = 0;
	realmtx_t W(rowsCnt, colsCnt), srcW(rowsCnt, colsCnt);
	ASSERT_TRUE(!W.isAllocationFailed() && !srcW.isAllocationFailed());

	iM.preinit(W.numel());
	ASSERT_TRUE(iM.init());
	
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	{
		realmtx_t etW(rowsCnt, colsCnt);
		ASSERT_TRUE(!etW.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(srcW, scale);

			srcW.cloneTo(etW);
			auto renormVal = rowvecs_renorm_ET(etW, iM._get_thread_temp_raw_storage(etW.numel()));
			renormTo += renormVal;

			srcW.cloneTo(W);
			iM.mCheck_normalize_rows_st(W, renormVal);
			ASSERT_FLOATMTX_EQ(etW, W, "mCheck_normalize_rows_st failed correctness test", mCheck_normalize_rows_EPS<real_t>::eps);

			srcW.cloneTo(W);
			iM.mCheck_normalize_rows_mt(W, renormVal);
			ASSERT_FLOATMTX_EQ(etW, W, "mCheck_normalize_rows_mt failed correctness test", mCheck_normalize_rows_EPS<real_t>::eps);
		}
		renormTo /= testCorrRepCnt;
	}

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(srcW, scale);
		
		srcW.cloneTo(W);
		bt = steady_clock::now();
		iM.mCheck_normalize_rows_st(W, renormTo);
		diffSt += steady_clock::now() - bt;

		srcW.cloneTo(W);
		bt = steady_clock::now();
		iM.mCheck_normalize_rows_mt(W, renormTo);
		diffMt += steady_clock::now() - bt;

		srcW.cloneTo(W);
		bt = steady_clock::now();
		iM.mCheck_normalize_rows(W, renormTo);
		diffB += steady_clock::now() - bt;
	}
	STDCOUTL("st_naivepart:\t" << utils::duration_readable(diffSt, maxReps));
	STDCOUTL("mt_naivepart:\t" << utils::duration_readable(diffMt, maxReps));
	STDCOUTL("best:\t\t" << utils::duration_readable(diffB, maxReps));
}

TEST(TestIMathBasic, mCheckNormalizeRows) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

// 	for (unsigned i = 1400; i <= 1425; i += 5) {
// 		test_mCheck_normalize_rows(iM, i, i / 16);
// 		test_mCheck_normalize_rows(iM, i / 4, i / 4);
// 		test_mCheck_normalize_rows(iM, i / 16, i);
// 	}
	test_mCheck_normalize_rows(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_mCheck_normalize_rows(iM, 1000);
	test_mCheck_normalize_rows(iM, 10000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


real_t loss_sigm_xentropy_ET(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.dataAsVec(), ptrY = data_y.dataAsVec();
	constexpr auto log_zero = math::real_ty_limits<real_t>::log_almost_zero;
	real_t ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		auto a = ptrA[i];
		NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
		NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

		if (y > real_t(0.0)) {
			ql += (a == real_t(0.0) ? log_zero : log(a));
		} else {
			const auto oma = real_t(1.0) - a;
			ql += (oma == real_t(0.0) ? log_zero : log(oma));
		}
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
template<typename base_t> struct loss_sigm_xentropy_EPS {};
template<> struct loss_sigm_xentropy_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct loss_sigm_xentropy_EPS<float> { static constexpr double eps = 2e-5; };
template<typename iMath>
void test_loss_sigm_xentropy(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing loss_sigm_xentropy() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());

	double tmtNaive, tstNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diffSt(0),diffMt(0),diffB(0);
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	const real_t frac = .5;
	realmtx_t A(rowsCnt, colsCnt), Y(rowsCnt, colsCnt);
	real_t etLoss = 0, loss = 0;
	ASSERT_EQ(dataSize, A.numel());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix_norm(A);
		rg.gen_matrix_norm(Y);
		iM.mBinarize(Y, frac);

		etLoss = loss_sigm_xentropy_ET(A, Y);

		bt = steady_clock::now();
		loss = iM.loss_sigm_xentropy_st_naivepart(A, Y);
		diffSt += steady_clock::now() - bt;
		ASSERT_NEAR(etLoss, loss, loss_sigm_xentropy_EPS<real_t>::eps);

		bt = steady_clock::now();
		loss = iM.loss_sigm_xentropy_mt_naivepart(A, Y);
		diffMt += steady_clock::now() - bt;
		ASSERT_NEAR(etLoss, loss, loss_sigm_xentropy_EPS<real_t>::eps);

		bt = steady_clock::now();
		loss = iM.loss_sigm_xentropy(A, Y);
		diffB += steady_clock::now() - bt;
		ASSERT_NEAR(etLoss, loss, loss_sigm_xentropy_EPS<real_t>::eps);
	}
	STDCOUTL("st_naivepart:\t" << utils::duration_readable(diffSt, maxReps, &tstNaive));
	STDCOUTL("mt_naivepart:\t" << utils::duration_readable(diffMt, maxReps, &tmtNaive));
	STDCOUTL("best:\t\t" << utils::duration_readable(diffB, maxReps, &tBest));
}

TEST(TestIMathBasic, lossSigmXentropy) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 100; i <= 130; i += 5 ) test_loss_sigm_xentropy(iM, i, 10);
	test_loss_sigm_xentropy(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_loss_sigm_xentropy(iM, 1000);
	test_loss_sigm_xentropy(iM, 10000);
	//test_loss_sigm_xentropy(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void mBinarize_ET(realmtx_t& A, const real_t frac)noexcept {
	auto pA = A.dataAsVec();
	const auto pAE = pA + A.numel();
	while (pA != pAE) {
		const auto v = *pA;
		NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
		*pA++ = v > frac ? real_t(1.0) : real_t(0.0);
	}
}
template<typename iMath>
void test_mBinarize(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing mBinarize() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t frac = .5;

	realmtx_t A(rowsCnt, colsCnt), AS(rowsCnt, colsCnt);
	ASSERT_TRUE(!AS.isAllocationFailed() && !A.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix_norm(AS);

	{
		realmtx_t A2(rowsCnt, colsCnt);
		ASSERT_TRUE(!A2.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			AS.cloneTo(A2);
			mBinarize_ET(A2, frac);

			AS.cloneTo(A);
			iM.mBinarize_st(A, frac);
			ASSERT_FLOATMTX_EQ(A2, A, "mBinarize_st failed correctness test");

			AS.cloneTo(A);
			iM.mBinarize_mt(A, frac);
			ASSERT_FLOATMTX_EQ(A2, A, "mBinarize_mt failed correctness test");

			AS.cloneTo(A);
			iM.mBinarize(A, frac);
			ASSERT_FLOATMTX_EQ(A2, A, "mBinarize failed correctness test");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		AS.cloneTo(A);
		bt = steady_clock::now();
		iM.mBinarize_st(A, frac);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		AS.cloneTo(A);
		bt = steady_clock::now();
		iM.mBinarize_mt(A, frac);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		AS.cloneTo(A);
		bt = steady_clock::now();
		iM.mBinarize(A, frac);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, mBinarize) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 130; i <= 144; i += 2) test_mBinarize(iM, i,1000);
	test_mBinarize(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_mBinarize(iM, 1000);
	test_mBinarize(iM, 10000);
	test_mBinarize(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void evSub_ET(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
	NNTL_ASSERT(A.size() == B.size() && A.size() == C.size());

	const auto dataCnt = A.numel();
	const auto pA = A.dataAsVec(), pB = B.dataAsVec();
	const auto pC = C.dataAsVec();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pC[i] = pA[i] - pB[i];
}
template<typename iMath>
void test_evSub(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evSub() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt), C(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed() && !C.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(A, 2);
	rg.gen_matrix(B, 2);

	{
		realmtx_t C2(rowsCnt, colsCnt);
		ASSERT_TRUE(!C2.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			evSub_ET(A, B, C2);

			iM.evSub_st_naive(A, B, C);
			ASSERT_FLOATMTX_EQ(C2, C, "evSub_st_naive failed correctness test");

			iM.evSub_mt_naive(A, B, C);
			ASSERT_FLOATMTX_EQ(C2, C, "evSub_mt_naive failed correctness test");

			iM.evSub(A, B, C);
			ASSERT_FLOATMTX_EQ(C2, C, "evSub failed correctness test");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_st_naive(A, B, C);
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_mt_naive(A, B, C);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub(A, B, C);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evSub) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 100; i <= 170; i += 5) test_evSub(iM, i,100);
	test_evSub(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evSub(iM, 1000);
	test_evSub(iM, 10000);
	test_evSub(iM, 100000);
#endif
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void evSub_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept {
	NNTL_ASSERT(A.size() == B.size());

	const auto dataCnt = A.numel();
	const auto pA = A.dataAsVec();
	const auto pB = B.dataAsVec();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] -= pB[i];
}
template<typename iMath>
void test_evSub_ip(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evSub_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest; //, tstVec, tmtVec;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(B, 2);

	{
		realmtx_t A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
		ASSERT_TRUE(!A2.isAllocationFailed() && !A3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(A, 2);
			A.cloneTo(A2);
			A.cloneTo(A3);

			evSub_ip_ET(A2, B);

			iM.evSub_ip_st_naive(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evSub_ip_st_naive failed correctness test");

			A3.cloneTo(A);
			iM.evSub_ip_mt_naive(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evSub_ip_mt_naive failed correctness test");

			/*iM.evSub_ip_st_vec(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evSub_ip_st_vec failed correctness test");

			A3.cloneTo(A);
			iM.evSub_ip_mt_vec(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evSub_ip_mt_vec failed correctness test");*/

			A3.cloneTo(A);
			iM.evSub_ip(A, B);
			ASSERT_FLOATMTX_EQ(A2, A, "evSub_ip failed correctness test");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	
	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_ip_st_naive(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_ip_mt_naive(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	/*rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_ip_st_vec(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("st_vec:\t" << utils::duration_readable(diff, maxReps, &tstVec));

	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_ip_mt_vec(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_vec:\t" << utils::duration_readable(diff, maxReps, &tmtVec));*/

	rg.gen_matrix(A, 2);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSub_ip(A, B);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evSubIp) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 160; i <= 240; i += 5) test_evSub_ip(iM, i,100);
	test_evSub_ip(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evSub_ip(iM, 1000);
	test_evSub_ip(iM, 10000);
	test_evSub_ip(iM, 100000);
#endif
}



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void apply_momentum_ET(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
	NNTL_ASSERT(vW.size() == dW.size());
	NNTL_ASSERT(!vW.empty() && !dW.empty());

	const auto dataCnt = vW.numel();
	const auto pV = vW.dataAsVec();
	const auto pdW = dW.dataAsVec();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		pV[i] = momentum*pV[i] + pdW[i];
	}
}
template<typename iMath>
void test_apply_momentum(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing apply_momentum() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t momentum = 0.9;
	realmtx_t dW(rowsCnt, colsCnt), vW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !vW.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(dW, 2);

	{
		realmtx_t vW2(rowsCnt, colsCnt), vW3(rowsCnt, colsCnt);
		ASSERT_TRUE(!vW2.isAllocationFailed() && !vW3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(vW, 2);
			vW.cloneTo(vW2);
			vW.cloneTo(vW3);

			apply_momentum_ET(vW2, momentum, dW);

			iM.apply_momentum_st(vW, momentum, dW);
			ASSERT_FLOATMTX_EQ(vW2, vW, "apply_momentum_st failed correctness test");

			vW3.cloneTo(vW);
			iM.apply_momentum_mt(vW,momentum, dW);
			ASSERT_FLOATMTX_EQ(vW2, vW, "apply_momentum_mt failed correctness test");

			vW3.cloneTo(vW);
			iM.apply_momentum(vW, momentum, dW);
			ASSERT_FLOATMTX_EQ(vW2, vW, "apply_momentum failed correctness test");
		}
	}
	rg.gen_matrix(vW, 2);

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.apply_momentum_st(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.apply_momentum_mt(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.apply_momentum(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, applyMomentum) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 190; i <= 250; i += 5) test_apply_momentum(iM, i,100);
	test_apply_momentum(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_apply_momentum(iM, 1000);
	test_apply_momentum(iM, 10000);
	test_apply_momentum(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//apply individual learning rate to dLdW
void apply_ILR_ET(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
	const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
{
	ASSERT_EQ(dLdW.size(), prevdLdW.size());
	ASSERT_EQ(dLdW.size(), ILRGain.size());
	ASSERT_TRUE(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

	const auto dataCnt = dLdW.numel();
	auto pdW = dLdW.dataAsVec();
	const auto prevdW = prevdLdW.dataAsVec();
	auto pGain = ILRGain.dataAsVec();

	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto cond = pdW[i] * prevdW[i];
		auto gain = pGain[i];

		if (cond > 0) {
			gain *= incr;
			if (gain > capHigh)gain = capHigh;
		} else if (cond < 0) {
			gain *= decr;
			if (gain < capLow)gain = capLow;
		}
		pGain[i] = gain;
		pdW[i] *= gain;
	}
}

template<typename iMath>
void test_applyILR_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing apply_ILR() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tstVec, tmtNaive, tmtVec, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t decr = .9, incr = (1/0.9), capH = 9.9, capL = 0.1;

	realmtx_t dW(rowsCnt, colsCnt), prevdW(rowsCnt, colsCnt), gain(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !prevdW.isAllocationFailed() && !gain.isAllocationFailed() );

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(prevdW, 10);

	//////////////////////////////////////////////////////////////////////////
	//testing correctness
	{
		realmtx_t dW2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt), gain2(rowsCnt, colsCnt), gain3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !dW3.isAllocationFailed() && !gain2.isAllocationFailed() && !gain3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.cloneTo(dW2);
			dW.cloneTo(dW3);
			rg.gen_matrix_gtz(gain, 10);
			gain.cloneTo(gain2);
			gain.cloneTo(gain3);

			apply_ILR_ET(dW, prevdW, gain, decr, incr, capL, capH);

			iM.apply_ILR_st_naive(dW2, prevdW, gain2, decr, incr, capL, capH);
			ASSERT_FLOATMTX_EQ(dW2, dW, "apply_ILR_st_naive: wrong dLdW matrix content!");
			ASSERT_FLOATMTX_EQ(gain2, gain, "apply_ILR_st_naive: wrong ILRGain matrix content!");

			dW3.cloneTo(dW2);
			gain3.cloneTo(gain2);
			iM.apply_ILR_st_vec(dW2, prevdW, gain2, decr, incr, capL, capH);
			ASSERT_FLOATMTX_EQ(dW2, dW, "apply_ILR_st_vec: wrong dLdW matrix content!");
			ASSERT_FLOATMTX_EQ(gain2, gain, "apply_ILR_st_vec: wrong ILRGain matrix content!");

			dW3.cloneTo(dW2);
			gain3.cloneTo(gain2);
			iM.apply_ILR_mt_naive(dW2, prevdW, gain2, decr, incr, capL, capH);
			ASSERT_FLOATMTX_EQ(dW2, dW, "apply_ILR_mt_naive: wrong dLdW matrix content!");
			ASSERT_FLOATMTX_EQ(gain2, gain, "apply_ILR_mt_naive: wrong ILRGain matrix content!");

			dW3.cloneTo(dW2);
			gain3.cloneTo(gain2);
			iM.apply_ILR_mt_vec(dW2, prevdW, gain2, decr, incr, capL, capH);
			ASSERT_FLOATMTX_EQ(dW2, dW, "apply_ILR_mt_vec: wrong dLdW matrix content!");
			ASSERT_FLOATMTX_EQ(gain2, gain, "apply_ILR_mt_vec: wrong ILRGain matrix content!");

			dW3.cloneTo(dW2);
			gain3.cloneTo(gain2);
			iM.apply_ILR(dW2, prevdW, gain2, decr, incr, capL, capH);
			ASSERT_FLOATMTX_EQ(dW2, dW, "apply_ILR: wrong dLdW matrix content!");
			ASSERT_FLOATMTX_EQ(gain2, gain, "apply_ILR: wrong ILRGain matrix content!");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		rg.gen_matrix_gtz(gain, 10);
		bt = steady_clock::now();
		iM.apply_ILR_st_naive(dW,prevdW,gain,decr,incr,capL,capH);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		rg.gen_matrix_gtz(gain, 10);
		bt = steady_clock::now();
		iM.apply_ILR_st_vec(dW, prevdW, gain, decr, incr, capL, capH);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps, &tstVec));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		rg.gen_matrix_gtz(gain, 10);
		bt = steady_clock::now();
		iM.apply_ILR_mt_naive(dW, prevdW, gain, decr, incr, capL, capH);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		rg.gen_matrix_gtz(gain, 10);
		bt = steady_clock::now();
		iM.apply_ILR_mt_vec(dW, prevdW, gain, decr, incr, capL, capH);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps, &tmtVec));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		rg.gen_matrix_gtz(gain, 10);
		bt = steady_clock::now();
		iM.apply_ILR(dW, prevdW, gain, decr, incr, capL, capH);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));

	iM.deinit();
}

TEST(TestIMathBasic, ApplyILRPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 300; i <= 370; i += 10) test_applyILR_perf(iM, i, 1000);
	test_applyILR_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_applyILR_perf(iM, 1000);
	test_applyILR_perf(iM, 10000);
	test_applyILR_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void evAbs_ET(realmtx_t& dest, const realmtx_t& src)noexcept {
	ASSERT_EQ(dest.size(), src.size());
	const auto pS = src.dataAsVec();
	auto pD = dest.dataAsVec();
	const auto dataCnt = src.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i)  pD[i] = abs(pS[i]);
}
template<typename iMath>
void test_evAbs_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evAbs() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), dest(rowsCnt, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());	

	{
		realmtx_t dest2(rowsCnt, colsCnt);
		ASSERT_TRUE(!dest2.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(src, 10);
			evAbs_ET(dest2, src);

			iM.evAbs_st(dest, src);
			ASSERT_FLOATMTX_EQ(dest2, dest, "evAbs_st failed correctness test");

			iM.evAbs_mt(dest, src);
			ASSERT_FLOATMTX_EQ(dest2, dest, "evAbs_mt failed correctness test");

			iM.evAbs(dest, src);
			ASSERT_FLOATMTX_EQ(dest2, dest, "evAbs failed correctness test");
		}
	}
	rg.gen_matrix(src, 10);

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r)  iM.evAbs_st(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r)  iM.evAbs_mt(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evAbs(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evAbsPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 200; i <= 230; i+=5) test_evAbs_perf(iM, i,100);
	test_evAbs_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evAbs_perf(iM, 1000);
	test_evAbs_perf(iM, 10000);
	test_evAbs_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void evSquare_ET(realmtx_t& dest, const realmtx_t& src)noexcept {
	ASSERT_EQ(dest.size(), src.size());

	const auto pS = src.dataAsVec();
	auto pD = dest.dataAsVec();
	const auto dataCnt = src.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto s = pS[i];
		pD[i] = s*s;
	}
}
template<typename iMath>
void test_evSquare_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evSquare() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), dest(rowsCnt, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	
	{
		realmtx_t dest2(rowsCnt, colsCnt);
		ASSERT_TRUE(!dest2.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(src, 10);
			evSquare_ET(dest2, src);

			iM.evSquare_st(dest, src);
			ASSERT_FLOATMTX_EQ(dest2, dest, "evSquare_st failed correctness test");

			iM.evSquare_mt(dest, src);
			ASSERT_FLOATMTX_EQ(dest2, dest, "evSquare_mt failed correctness test");

			iM.evSquare(dest, src);
			ASSERT_FLOATMTX_EQ(dest2, dest, "evSquare failed correctness test");
		}
	}
	rg.gen_matrix(src, 10);

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSquare_st(dest,src);
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSquare_mt(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.evSquare(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evSquarePerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 230; i <= 270; i+=5) test_evSquare_perf(iM, i,100);
	test_evSquare_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evSquare_perf(iM, 1000);
	test_evSquare_perf(iM, 10000);
	test_evSquare_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void ModProp_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept {
	ASSERT_EQ(dW.size(), rmsF.size());
	ASSERT_TRUE(emaDecay > 0 && emaDecay < 1);
	ASSERT_TRUE(numericStabilizer > 0 && numericStabilizer < 1);

	auto pdW = dW.dataAsVec();
	auto prmsF = rmsF.dataAsVec();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto dataCnt = dW.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto rms = prmsF[i]* emaDecay + abs(pdW[i])*_1_emaDecay;
		prmsF[i] = rms;
		pdW[i] *= learningRate / (rms + numericStabilizer);
	}
}
template<typename iMath>
void test_modprop_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing ModProp() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t emaCoeff = .9, lr = .1, numStab = .00001;

	realmtx_t dW(rowsCnt, colsCnt), rms(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed());

	rms.zeros();

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//testing correctness
	{
		realmtx_t dW2(rowsCnt, colsCnt), rms2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt), rms3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !rms2.isAllocationFailed() && !dW3.isAllocationFailed() && !rms3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.cloneTo(dW2);
			dW.cloneTo(dW3);
			rg.gen_matrix_gtz(rms, 10);
			rms.cloneTo(rms2);
			rms.cloneTo(rms3);

			ModProp_ET(dW2, rms2, lr, emaCoeff, numStab);

			iM.ModProp_st(dW, rms, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "ModProp_st: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "ModProp_st: wrong rms");

			dW3.cloneTo(dW);
			rms3.cloneTo(rms);
			iM.ModProp_mt(dW, rms, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "ModProp_mt: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "ModProp_mt: wrong rms");

			dW3.cloneTo(dW);
			rms3.cloneTo(rms);
			iM.ModProp(dW, rms, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "ModProp: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "ModProp: wrong rms");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.ModProp_st(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.ModProp_mt(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.ModProp(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, ModPropPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 450; i <= 600; i+=10) test_modprop_perf(iM, i,10);
	test_modprop_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_modprop_perf(iM, 1000);
	test_modprop_perf(iM, 10000);
	test_modprop_perf(iM, 100000);
#endif
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void RProp_ET(realmtx_t& dW, const real_t learningRate)noexcept {
	auto p = dW.dataAsVec();
	const auto im=dW.numel();
	for (numel_cnt_t i = 0; i < im; ++i) {
		const auto w = p[i];
		if (w > real_t(0)) {
			p[i] = learningRate;
		} else if (w < real_t(0)) {
			p[i] = -learningRate;
		} else p[i] = real_t(0);
	}
}
template<typename iMath>
void test_rprop_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RProp() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t lr = .1;

	realmtx_t dW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed());
	
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	{
		realmtx_t dW2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !dW3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.cloneTo(dW2);
			dW.cloneTo(dW3);

			RProp_ET(dW2, lr);

			iM.RProp_st(dW, lr);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RProp_st: wrong dW");

			dW3.cloneTo(dW);
			iM.RProp_mt(dW, lr);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RProp_mt: wrong dW");

			dW3.cloneTo(dW);
			iM.RProp(dW, lr);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RProp: wrong dW");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RProp_st(dW, lr);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RProp_mt(dW, lr);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RProp(dW, lr);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, RPropPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 400; i <= 800; i+=50) test_rprop_perf(iM, i,10);
	test_rprop_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_rprop_perf(iM, 1000);
	test_rprop_perf(iM, 10000);
	test_rprop_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
void RMSProp_Graves_ET(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
	const real_t emaDecay, const real_t numericStabilizer)noexcept
{
	auto pdW = dW.dataAsVec();
	auto prmsF = rmsF.dataAsVec();
	auto prmsG = rmsG.dataAsVec();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto dataCnt = dW.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		prmsF[i] = emaDecay*prmsF[i] + _1_emaDecay*pdW[i] * pdW[i];
		prmsG[i] = emaDecay*prmsG[i] + _1_emaDecay*pdW[i];
		pdW[i] *= learningRate / (sqrt(prmsF[i] - prmsG[i] * prmsG[i] + numericStabilizer));
	}
}
template<typename iMath>
void test_rmspropgraves_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RMSProp_Graves() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t emaCoeff = .9, lr = .1, numStab = .00001;

	realmtx_t dW(rowsCnt, colsCnt), rms(rowsCnt, colsCnt), rmsG(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed() && !rmsG.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	{
		realmtx_t dW2(rowsCnt, colsCnt), rms2(rowsCnt, colsCnt), rmsG2(rowsCnt, colsCnt),
			dW3(rowsCnt, colsCnt), rms3(rowsCnt, colsCnt), rmsG3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !rms2.isAllocationFailed() && !dW3.isAllocationFailed() && !rms3.isAllocationFailed()
			&& !rmsG2.isAllocationFailed() && !rmsG3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.cloneTo(dW2);
			dW.cloneTo(dW3);
			evSquare_ET(rms, dW);
			rms.cloneTo(rms2);
			rms.cloneTo(rms3);
			dW.cloneTo(rmsG);
			rmsG.cloneTo(rmsG2);
			rmsG.cloneTo(rmsG3);

			RMSProp_Graves_ET(dW2, rms2, rmsG2, lr, emaCoeff, numStab);

			iM.RMSProp_Graves_st(dW, rms, rmsG, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RMSProp_Graves_st: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "RMSProp_Graves_st: wrong rms");
			ASSERT_FLOATMTX_EQ(rmsG2, rmsG, "RMSProp_Graves_st: wrong rmsG");

			dW3.cloneTo(dW);
			rms3.cloneTo(rms);
			rmsG3.cloneTo(rmsG);
			iM.RMSProp_Graves_mt(dW, rms, rmsG, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RMSProp_Graves_mt: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "RMSProp_Graves_mt: wrong rms");
			ASSERT_FLOATMTX_EQ(rmsG2, rmsG, "RMSProp_Graves_mt: wrong rmsG");

			dW3.cloneTo(dW);
			rms3.cloneTo(rms);
			rmsG3.cloneTo(rmsG);
			iM.RMSProp_Graves(dW, rms, rmsG, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RMSProp_Graves: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "RMSProp_Graves: wrong rms");
			ASSERT_FLOATMTX_EQ(rmsG2, rmsG, "RMSProp_Graves: wrong rmsG");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Graves_st(dW, rms, rmsG, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Graves_mt(dW, rms, rmsG, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Graves(dW, rms, rmsG, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, RMSPropGravesPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 250; i <= 350; i+=10) test_rmspropgraves_perf(iM, i,10);
	test_rmspropgraves_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_rmspropgraves_perf(iM, 1000);
	test_rmspropgraves_perf(iM, 10000);
	test_rmspropgraves_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
void RMSProp_Hinton_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
	const real_t emaDecay, const real_t numericStabilizer)noexcept
{
	auto pdW = dW.dataAsVec();
	auto prmsF = rmsF.dataAsVec();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto dataCnt = dW.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		prmsF[i] = emaDecay*prmsF[i] + _1_emaDecay*pdW[i] * pdW[i];
		pdW[i] *= learningRate / (sqrt(prmsF[i]) + numericStabilizer);
	}
}
template<typename iMath>
void test_rmsprophinton_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RMSProp_Hinton() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t emaCoeff = .9, lr = .1, numStab = .00001;

	realmtx_t dW(rowsCnt, colsCnt), rms(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed());

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	{
		realmtx_t dW2(rowsCnt, colsCnt), rms2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt), rms3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !rms2.isAllocationFailed() && !dW3.isAllocationFailed() && !rms3.isAllocationFailed());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.cloneTo(dW2);
			dW.cloneTo(dW3);
			evSquare_ET(rms, dW);
			rms.cloneTo(rms2);
			rms.cloneTo(rms3);

			RMSProp_Hinton_ET(dW2, rms2, lr, emaCoeff, numStab);

			iM.RMSProp_Hinton_st(dW, rms, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RMSProp_Hinton_st: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "RMSProp_Hinton_st: wrong rms");

			dW3.cloneTo(dW);
			rms3.cloneTo(rms);
			iM.RMSProp_Hinton_mt(dW, rms, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RMSProp_Hinton_mt: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "RMSProp_Hinton_mt: wrong rms");

			dW3.cloneTo(dW);
			rms3.cloneTo(rms);
			iM.RMSProp_Hinton(dW, rms, lr, emaCoeff, numStab);
			ASSERT_FLOATMTX_EQ(dW2, dW, "RMSProp_Hinton: wrong dW");
			ASSERT_FLOATMTX_EQ(rms2, rms, "RMSProp_Hinton: wrong rms");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Hinton_st(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Hinton_mt(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Hinton(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, RMSPropHintonPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 200; i <= 300; i+=10) test_rmsprophinton_perf(iM, i,10);
	test_rmsprophinton_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_rmsprophinton_perf(iM, 1000);
	test_rmsprophinton_perf(iM, 10000);
	test_rmsprophinton_perf(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
void make_dropout_ET(realmtx_t& act, real_t dfrac, realmtx_t& dropoutMask)noexcept {
	const auto dataCnt = act.numel_no_bias();
	auto pDM = dropoutMask.dataAsVec();
	const auto pA = act.dataAsVec();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		if (pDM[i] > dfrac) {
			pDM[i] = real_t(1);
		} else {
			pDM[i] = real_t(0);
			pA[i] = real_t(0);
		}
	}
}
template<typename iMath>
void test_make_dropout_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing make_dropout() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t dfrac = .5;

	realmtx_t act(rowsCnt, colsCnt, true), dm(rowsCnt, colsCnt);
	ASSERT_TRUE(!act.isAllocationFailed() && !dm.isAllocationFailed());
	
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	{
		realmtx_t act2(rowsCnt, colsCnt, true), dm2(rowsCnt, colsCnt), act3(rowsCnt, colsCnt, true), dm3(rowsCnt, colsCnt);
		ASSERT_TRUE(!act2.isAllocationFailed() && !dm2.isAllocationFailed()&& !act3.isAllocationFailed() && !dm3.isAllocationFailed());
		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix_no_bias(act, 5);
			act.assert_biases_ok();
			act.cloneTo(act2);
			act.cloneTo(act3);
			rg.gen_matrix_norm(dm);
			dm.cloneTo(dm2);
			dm.cloneTo(dm3);

			make_dropout_ET(act2, dfrac, dm2);
			act2.assert_biases_ok();

			iM.make_dropout_st(act, dfrac, dm);
			ASSERT_FLOATMTX_EQ(act2, act, "make_dropout_st: wrong act");
			ASSERT_FLOATMTX_EQ(dm2, dm, "make_dropout_st: wrong dm");

			act3.cloneTo(act);
			dm3.cloneTo(dm);
			iM.make_dropout_mt(act, dfrac, dm);
			ASSERT_FLOATMTX_EQ(act2, act, "make_dropout_mt: wrong act");
			ASSERT_FLOATMTX_EQ(dm2, dm, "make_dropout_mt: wrong dm");

			act3.cloneTo(act);
			dm3.cloneTo(dm);
			iM.make_dropout(act, dfrac, dm);
			ASSERT_FLOATMTX_EQ(act2, act, "make_dropout: wrong act");
			ASSERT_FLOATMTX_EQ(dm2, dm, "make_dropout: wrong dm");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	rg.gen_matrix_no_bias(act, 5);
	act.assert_biases_ok();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix_norm(dm);
		bt = steady_clock::now();
		iM.make_dropout_st(act, dfrac, dm);
		diff += steady_clock::now() - bt;
		act.assert_biases_ok();
	}
	STDCOUTL("st:\t\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	rg.gen_matrix_no_bias(act, 5);
	act.assert_biases_ok();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix_norm(dm);
		bt = steady_clock::now();
		iM.make_dropout_mt(act, dfrac, dm);
		diff += steady_clock::now() - bt;
		act.assert_biases_ok();
	}
	STDCOUTL("mt:\t\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	rg.gen_matrix_no_bias(act, 5);
	act.assert_biases_ok();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix_norm(dm);
		bt = steady_clock::now();
		iM.make_dropout(act, dfrac, dm);
		diff += steady_clock::now() - bt;
		act.assert_biases_ok();
	}
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, MakeDropoutPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 9000; i <= 11000; i+=100) test_make_dropout_perf(iM, 1,i);
	test_make_dropout_perf(iM, 1, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_make_dropout_perf(iM, 1,1000);
	test_make_dropout_perf(iM, 1,10000);
	test_make_dropout_perf(iM, 1,100000);
#endif
}

//////////////////////////////////////////////////////////////////////////

TEST(TestIMathBasic, vCountSameNaive) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	constexpr unsigned dataCnt = 9;
	const std::array<unsigned, dataCnt> src1 = { 3,55,32, 35,63,5, 2,400,6 };
	const std::array<unsigned, dataCnt> src2 = { 3,55,33, 35,63,5, 4,400,6 };

	iMB iM;
	ASSERT_EQ(iM.vCountSame_st_naive(src1, src2), dataCnt-2);
}

TEST(TestIMathBasic, vCountSameMtCorrectness) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	typedef std::vector<realmtx_t::vec_len_t> vec_t;

#ifdef _DEBUG
	constexpr unsigned rowsCnt = 100;
#else
	constexpr unsigned rowsCnt = 100000;
#endif

	vec_t v1(rowsCnt), v2(rowsCnt);

	iMB iM;
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	rg.gen_vector_gtz(&v1[0], rowsCnt, (vec_t::value_type)5);
	rg.gen_vector_gtz(&v2[0], rowsCnt, (vec_t::value_type)5);

	ASSERT_EQ(iM.vCountSame_st_naive(v1, v2), iM.vCountSame_mt_naive(v1, v2));
}

template<typename iMath>
void test_vCountSame_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt) {
	typedef std::vector<realmtx_t::vec_len_t> vec_t;
	STDCOUTL("******* testing vCountSame() over " << rowsCnt << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	size_t vv;

	vec_t v1(rowsCnt), v2(rowsCnt);
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	rg.gen_vector_gtz(&v1[0], rowsCnt, (vec_t::value_type)5);
	rg.gen_vector_gtz(&v2[0], rowsCnt, (vec_t::value_type)5);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	iM.vCountSame_st_naive(v1, v2);
	vv = 0;
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		vv += iM.vCountSame_st_naive(v1, v2);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive) << "\t\tvv=" << vv);

	iM.vCountSame_mt_naive(v1, v2);;
	vv = 0;
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		vv += iM.vCountSame_mt_naive(v1, v2);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive) << "\t\tvv=" << vv);

	iM.vCountSame(v1, v2);
	vv = 0;
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		vv += iM.vCountSame(v1, v2);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest) << "\t\tvv=" << vv);
}

TEST(TestIMathBasic, vCountSamePerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 1; i <= 128; i*=2) test_vCountSame_perf(iM, i*100000);

	test_vCountSame_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_vCountSame_perf(iM, 1000);
	test_vCountSame_perf(iM, 10000);
	test_vCountSame_perf(iM, 100000);
#endif

}


//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_evClamp_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	typedef std::vector<realmtx_t::vec_len_t> vec_t;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evClamp() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	real_t lo = -50, hi = 50;

	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	vec_t vec(rowsCnt);

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(m, 100);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	iM.evClamp_st(m, lo,hi);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.evClamp_st(m, lo, hi);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	iM.evClamp_mt(m, lo, hi);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.evClamp_mt(m, lo, hi);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	iM.evClamp(m, lo, hi);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.evClamp(m, lo, hi);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evClampPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 100; i <= 140; i+=1) test_evClamp_perf(iM, i,100);

	test_evClamp_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evClamp_perf(iM, 1000);
	test_evClamp_perf(iM, 10000);
	test_evClamp_perf(iM, 100000);
#endif
}


//////////////////////////////////////////////////////////////////////////
TEST(TestIMathBasic, mExtractRowsCorrectness) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	
	constexpr vec_len_t rowsCnt = 2000, colsCnt = 50, extrCnt = 1000;

	realmtx_t src(rowsCnt, colsCnt), destSt(extrCnt, colsCnt), destMt(extrCnt, colsCnt);;
	ASSERT_TRUE(!src.isAllocationFailed() && !destSt.isAllocationFailed() && !destMt.isAllocationFailed());
	auto pSrc = src.dataAsVec();
	for (numel_cnt_t i = 0, im = src.numel(); i < im; ++i) pSrc[i] = static_cast<real_t>(i);

	std::vector<vec_len_t> vec(extrCnt);
	iMB iM;
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());

	rg.gen_vector_gtz(&vec[0], vec.size(), rowsCnt - 1);

	iM.mExtractRows_st_naive(src, vec.begin(), extrCnt, destSt);
	iM.mExtractRows_mt_naive(src, vec.begin(), extrCnt, destMt);

	ASSERT_EQ(destSt, destMt);
	for (vec_len_t r = 0; r < extrCnt; ++r) {
		for (vec_len_t c = 0; c < colsCnt; ++c) {
			ASSERT_DOUBLE_EQ(destSt.get(r, c), src.get(vec[r], c));
			//ASSERT_DOUBLE_EQ(destMt.get(r, c), src.get(vec[r], c));
		}
	}
}

template<typename iMath>
void test_mExtractRows_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t extrCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	typedef std::vector<realmtx_t::vec_len_t> vec_t;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing mExtractRows() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elems) ExtractRows="<< extrCnt 
		<<" -> "<< realmtx_t::sNumel(extrCnt,colsCnt) << " elems *********");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), dest(extrCnt, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed());
	vec_t vec(extrCnt);

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(src, 1000);
	rg.gen_vector_gtz(&vec[0], vec.size(), rowsCnt - 1);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	iM.mExtractRows_st_naive(src, vec.begin(), extrCnt, dest);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.mExtractRows_st_naive(src, vec.begin(), extrCnt, dest);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	iM.mExtractRows_mt_naive(src, vec.begin(), extrCnt, dest);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.mExtractRows_mt_naive(src, vec.begin(), extrCnt, dest);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	iM.mExtractRows(src, vec.begin(), extrCnt, dest);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.mExtractRows(src, vec.begin(), extrCnt, dest);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, mExtractRowsPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

/*
	for (unsigned r = 8; r <= 64; r *= 2) {
		for (unsigned c = 10; c <= 800; c += 790) {
			for (unsigned e = 1; e <= 8; e *= 2) {
				test_mExtractRows_perf(iM, r * 1000, e * 100, c);
			}
		}
	}*/

	constexpr unsigned batchSize = 100;

	test_mExtractRows_perf(iM, 200, batchSize, 10);

#ifndef TESTS_SKIP_LONGRUNNING
	test_mExtractRows_perf(iM, 60000, batchSize, 800);
	test_mExtractRows_perf(iM, 60000, batchSize, 10);
	test_mExtractRows_perf(iM, 40000, batchSize, 800);
	test_mExtractRows_perf(iM, 40000, batchSize, 10);
#endif

}

//////////////////////////////////////////////////////////////////////////
TEST(TestIMathBasic, mFindIdxsOfMaxRowwiseNaive) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	constexpr unsigned dataCnt = 9;
	const real_t _src[dataCnt]{ 3,55,32,35,63,5,-2,400,6 };

	realmtx_t m(3, 3);
	ASSERT_TRUE(!m.isAllocationFailed());
	ASSERT_EQ(dataCnt, m.numel());

	memcpy(m.dataAsVec(), _src, m.byte_size());

	std::vector<realmtx_t::vec_len_t> vec(3);

	iMB iM;
	iM.mFindIdxsOfMaxRowwise_st_naive(m,vec);

	ASSERT_EQ(vec[0], 1);
	ASSERT_EQ(vec[1], 2);
	ASSERT_EQ(vec[2], 0);
}

TEST(TestIMathBasic, mFindIdxsOfMaxRowwiseMtCorrectness) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

#ifdef _DEBUG
	constexpr unsigned rowsCnt = 100;
#else
	constexpr unsigned rowsCnt = 10000;
#endif

	realmtx_t m(rowsCnt, 100);
	ASSERT_TRUE(!m.isAllocationFailed());

	iMB iM;
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(m, 1000);

	std::vector<realmtx_t::vec_len_t> vec_st(rowsCnt), vec_mt(rowsCnt);

	iM.mFindIdxsOfMaxRowwise_st_naive(m, vec_st);
	iM.mFindIdxsOfMaxRowwise_mt_naive(m, vec_mt);

	for (unsigned r = 0; r < rowsCnt; ++r) {
		ASSERT_EQ(vec_st[r], vec_mt[r]);
	}
}

template<typename iMath>
void test_mFindIdxsOfMaxRowwise_perf(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	typedef std::vector<realmtx_t::vec_len_t> vec_t;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing mFindIdxsOfMaxRowwise() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	vec_t vec(rowsCnt);

	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(m, 1000);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	iM.mFindIdxsOfMaxRowwise_st_naive(m,vec);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.mFindIdxsOfMaxRowwise_st_naive(m, vec);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	iM.mFindIdxsOfMaxRowwise_mt_naive(m, vec);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.mFindIdxsOfMaxRowwise_mt_naive(m, vec);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	iM.mFindIdxsOfMaxRowwise(m, vec);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		iM.mFindIdxsOfMaxRowwise(m, vec);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, mFindIdxsOfMaxRowwisePerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 1; i <= 10; i+=1) test_mFindIdxsOfMaxRowwise_perf(iM, i*100,10);
	test_mFindIdxsOfMaxRowwise_perf(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_mFindIdxsOfMaxRowwise_perf(iM, 1000);
	test_mFindIdxsOfMaxRowwise_perf(iM, 10000);
	test_mFindIdxsOfMaxRowwise_perf(iM, 100000);
#endif

}

TEST(TestIMathBasic, mMulABt_Cnb) {
	using namespace nntl_supp;

	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	
	using ErrCode = jsonreader::ErrorCode;
	using realmtx_t = train_data::realmtx_t;
	using mtx_size_t = realmtx_t::mtx_size_t;

	realmtx_t A,B,C, etA, etB, etC;
	jsonreader reader;

	ErrCode ec = reader.read(NNTL_STRING("./test_data/mtx4-2.json"), etA);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etA.empty());
	etA.cloneTo(A);

	ec = reader.read(NNTL_STRING("./test_data/mtx3-2.json"), etB);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etB.empty());
	etB.cloneTo(B);

	ec = reader.read(NNTL_STRING("./test_data/mtx4-3.json"), etC);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etC.empty());

	C.resize(etC.size());
	C.zeros();

	iMB iM;

	iM.mMulABt_Cnb(A, B, C);
	EXPECT_EQ(A, etA);
	EXPECT_EQ(B, etB);
	EXPECT_EQ(C, etC);
}

TEST(TestIMathBasic, mMulABt_Cnb_biased) {
	using namespace nntl_supp;

	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	using ErrCode = jsonreader::ErrorCode;
	using realmtx_t = train_data::realmtx_t;
	using mtx_size_t = realmtx_t::mtx_size_t;

	realmtx_t A, B, C, etA, etB, etC;
	jsonreader reader;

	ErrCode ec = reader.read(NNTL_STRING("./test_data/mtx4-2.json"), etA);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etA.empty());
	etA.cloneTo(A);

	ec = reader.read(NNTL_STRING("./test_data/mtx3-2.json"), etB);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etB.empty());
	etB.cloneTo(B);

	ec = reader.read(NNTL_STRING("./test_data/mtx4-3.json"), etC);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etC.empty());

	C.will_emulate_biases();
	C.resize(etC.size());
	C.zeros();

	iMB iM;

	iM.mMulABt_Cnb(A, B, C);
	EXPECT_EQ(A, etA);
	EXPECT_EQ(B, etB);

	auto ptrC = C.dataAsVec(), ptrEt = etC.dataAsVec();
	auto cnt = etC.numel(), bcnt = C.numel();
	ASSERT_TRUE(cnt < bcnt);
	for (numel_cnt_t i = 0; i < cnt; ++i) {
		ASSERT_DOUBLE_EQ(ptrC[i], ptrEt[i]) << "offset "<<i;
	}
	for (numel_cnt_t i = cnt; i < bcnt; ++i) {
		ASSERT_DOUBLE_EQ(ptrC[i], real_t(1.0)) << "offset " << i;
	}
}


//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_evMul_ip(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing evMul_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tstNaive, tmtNaive, tBest; //tmtVect, tstVect,
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t m, etM(rowsCnt, colsCnt), etDest(rowsCnt, colsCnt), etB(rowsCnt, colsCnt), B;
	ASSERT_EQ(dataSize, etM.numel());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etM, 5);
	rg.gen_matrix(etB, 5);
	ASSERT_TRUE(etB.cloneTo(B));
	auto ptrEtM = etM.dataAsVec(), ptrDest = etDest.dataAsVec(), ptretB=etB.dataAsVec();
	for (unsigned i = 0; i < dataSize; ++i) ptrDest[i] = ptrEtM[i]* ptretB[i];

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	ASSERT_TRUE(etM.cloneTo(m));
	iM.evMul_ip_st_naive(m, B);
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMul_ip_st_naive(m, B);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest) << "evMul_ip_st_naive";
	ASSERT_EQ(B, etB) << "evMul_ip_st_naive";
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	ASSERT_TRUE(etM.cloneTo(m));
	iM.evMul_ip_mt_naive(m, B);
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMul_ip_mt_naive(m, B);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest) << "evMul_ip_mt_naive";
	ASSERT_EQ(B, etB) << "evMul_ip_mt_naive";
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	/*ASSERT_TRUE(etM.cloneTo(m));
	iM.evMul_ip_st_vec(m, B);
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMul_ip_st_vec(m, B);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest) << "evMul_ip_st_vec";
	ASSERT_EQ(B, etB) << "evMul_ip_st_vec";
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps, &tstVect));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	ASSERT_TRUE(etM.cloneTo(m));
	iM.evMul_ip_mt_vec(m, B);
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMul_ip_mt_vec(m, B);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest) << "evMul_ip_mt_vec";
	ASSERT_EQ(B, etB) << "evMul_ip_mt_vec";
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps, &tmtVect));
*/

	//////////////////////////////////////////////////////////////////////////
	//best guess
	ASSERT_TRUE(etM.cloneTo(m));
	iM.evMul_ip(m, B);
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMul_ip(m, B);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest) << "evMul_ip";
	ASSERT_EQ(B, etB) << "evMul_ip";
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evMulIp) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 350; i <= 380; i+=5) test_evMul_ip(iM, i*100,1);
	test_evMul_ip(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evMul_ip(iM, 1000);
	test_evMul_ip(iM, 10000);
	test_evMul_ip(iM, 100000);
#endif
}



//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_evMulC_ip(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing evMulC_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tmtNaive, tstNaive, tBest; //tstVect, tmtVect
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	const real_t mulC = 0.01;
	realmtx_t m, etM(rowsCnt, colsCnt), etDest(rowsCnt, colsCnt);
	ASSERT_EQ(dataSize, etM.numel());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etM, 5);
	auto ptrEtM = etM.dataAsVec(), ptrDest=etDest.dataAsVec();
	for (unsigned i = 0; i < dataSize; ++i) ptrDest[i] = mulC*ptrEtM[i];
	
	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMulC_ip_st_naive(m,mulC);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest);
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMulC_ip_mt_naive(m, mulC);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest);
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	/*//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMulC_ip_st_vec(m, mulC);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest);
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps, &tstVect));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMulC_ip_mt_vec(m, mulC);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest);
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps, &tmtVect));*/

	//////////////////////////////////////////////////////////////////////////
	//best guess
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		ASSERT_TRUE(etM.cloneTo(m));
		bt = steady_clock::now();
		iM.evMulC_ip(m, mulC);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etDest);
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, evMulC_ip) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 60; i <= 70; i+=1) test_evMulC_ip(iM, i*10,100);	
	test_evMulC_ip(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_evMulC_ip(iM, 1000);
	test_evMulC_ip(iM, 10000);
	test_evMulC_ip(iM, 100000);
#endif
}


//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct sigm_EPS {};
template<> struct sigm_EPS<double> { static constexpr double eps = 1e-12; };
template<> struct sigm_EPS<float> { static constexpr double eps = 1e-6; };

template<typename iMath>
void test_sigm(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	typedef typename iMath::ithreads_t threads_t;
	typedef math_types::realmtxdef_ty realmtxdef_t;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing sigm() over ~" << dataSize << " elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tmtNaive, tstNaive, tBest; //tstVect, tmtVect
	steady_clock::time_point bt;
	nanoseconds diff;
	const unsigned maxReps = ceil(((real_t)TEST_PERF_REPEATS_COUNT)/25.0);


	const auto threadsCount = iM.ithreads().workers_count();
	ASSERT_TRUE(threadsCount > 0);
	const auto biggestDataSize = static_cast<realmtx_t::vec_len_t>(dataSize + threadsCount);

	realmtxdef_t m, etDest(biggestDataSize, 1);
	realmtx_t etM(biggestDataSize, 1);
	ASSERT_TRUE(biggestDataSize == etM.numel());

	iM.preinit(biggestDataSize);
	ASSERT_TRUE(iM.init());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etM, 2);
	auto ptrEtM = etM.dataAsVec(), ptrDest = etDest.dataAsVec();
	for (unsigned i = 0; i < biggestDataSize; ++i) {
		ptrDest[i] = real_t(1.0) / (real_t(1.0) + std::exp(-ptrEtM[i]));
	}
	ASSERT_TRUE(m.cloneFrom(etM));
	ASSERT_TRUE(etM == m);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());
	
	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.sigm_st_naive(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		if (std::is_same<real_t, float>::value) {
			for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_FLOAT_EQ(ptrDest[i], ptr[i]);
		} else {
			for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
		}
	}
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps*threadsCount, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.sigm_mt_naive(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		if (std::is_same<real_t, float>::value) {
			for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_FLOAT_EQ(ptrDest[i], ptr[i]);
		} else {
			for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
		}
	}
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps*threadsCount, &tmtNaive));
/*

	//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.sigm_st_vec(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps*threadsCount, &tstVect));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.sigm_mt_vec(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps*threadsCount, &tmtVect));
*/

	//////////////////////////////////////////////////////////////////////////
	//best
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.sigm(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		if (std::is_same<real_t, float>::value) {
			for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_FLOAT_EQ(ptrDest[i], ptr[i]);
		} else {
			for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
		}
	}
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps*threadsCount, &tBest));
}

TEST(TestIMathBasic, Sigm) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	
	iMB iM;

	//for (unsigned i = 20; i <= 30; i += 1) test_sigm(iM, i * 100, 1);

	test_sigm(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_sigm(iM, 1000);
	test_sigm(iM, 10000);
	test_sigm(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_dsigm(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing dsigm() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tmtNaive, tstNaive, tBest; //tstVect, tmtVect
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t m, etM(rowsCnt, colsCnt), etDest(rowsCnt, colsCnt), dest(rowsCnt, colsCnt);
	ASSERT_EQ(dataSize, etM.numel());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etM, 10);
	auto ptrEtM = etM.dataAsVec(), ptrDest = etDest.dataAsVec();
	for (unsigned i = 0; i < dataSize; ++i) ptrDest[i] = ptrEtM[i] * (1 - ptrEtM[i]);
	ASSERT_TRUE(etM.cloneTo(m));

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.dsigm_st_naive(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.dsigm_mt_naive(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));
/*

	//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.dsigm_st_vec(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps, &tstVect));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.dsigm_mt_vec(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps, &tmtVect));
*/

	//////////////////////////////////////////////////////////////////////////
	//best guess
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.dsigm(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
	// If it's not the best, then tune iM.loss_quadratic() appropriately


	//double rat = static_cast<double>(tmtNaive) / tmtVect;
	//STDCOUTL("Vectorized version is " << (rat > 1 ? "faster " : "SLOWER ") << std::setprecision(2) << (rat > 1 ? rat : 1 / rat) << " times");
}

TEST(TestIMathBasic, dsigm) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 220; i <= 300; i+=5) test_dsigm(iM, i*100,1);

	test_dsigm(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_dsigm(iM, 1000);
	test_dsigm(iM, 10000);
	test_dsigm(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_relu(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	typedef typename iMath::ithreads_t threads_t;
	typedef math_types::realmtxdef_ty realmtxdef_t;
	
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing relu() over ~" << dataSize << " elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tmtNaive, tstNaive, tBest;//tstVect, tmtVect
	steady_clock::time_point bt;
	nanoseconds diff;
	const unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	const auto threadsCount = iM.ithreads().workers_count();
	ASSERT_TRUE(threadsCount > 0);
	const auto biggestDataSize = static_cast<realmtx_t::vec_len_t>(dataSize + threadsCount);

	realmtxdef_t m, etDest(biggestDataSize, 1);
	realmtx_t etM(biggestDataSize, 1);
	ASSERT_TRUE(biggestDataSize == etM.numel());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etM, 2);
	auto ptrEtM = etM.dataAsVec(), ptrDest = etDest.dataAsVec();
	for (unsigned i = 0; i < biggestDataSize; ++i) {
		ptrDest[i] = (ptrEtM[i] < 0) ? 0 : ptrEtM[i];
	}
	ASSERT_TRUE(m.cloneFrom(etM));
	ASSERT_TRUE(etM == m);

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.relu_st_naive(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps*threadsCount, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.relu_mt_naive(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps*threadsCount, &tmtNaive));

	/*//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.relu_st_vec(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps*threadsCount, &tstVect));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.relu_mt_vec(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps*threadsCount, &tmtVect));*/

	//////////////////////////////////////////////////////////////////////////
	//best
	diff = nanoseconds(0);
	for (threads_t::thread_id_t t = 0; t < threadsCount; ++t) {
		const unsigned iMax = static_cast<realmtx_t::vec_len_t>(dataSize + t);
		for (unsigned r = 0; r < maxReps; ++r) {
			m.deform_rows(biggestDataSize);
			ASSERT_TRUE(m.cloneFrom(etM));
			m.deform_rows(iMax);
			bt = steady_clock::now();
			iM.relu(m);
			diff += steady_clock::now() - bt;
		}
		const auto ptr = m.dataAsVec();
		for (numel_cnt_t i = 0; i < iMax; ++i) ASSERT_DOUBLE_EQ(ptrDest[i], ptr[i]);
	}
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps*threadsCount, &tBest));
}

TEST(TestIMathBasic, Relu) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 37; i <= 47; i += 1) test_relu(iM, i * 100, 1);

	test_relu(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_relu(iM, 1000);
	test_relu(iM, 10000);
	test_relu(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_drelu(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing drelu() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t m, etM(rowsCnt, colsCnt), etDest(rowsCnt, colsCnt), dest(rowsCnt, colsCnt);
	ASSERT_EQ(dataSize, etM.numel());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etM, 10);
	auto ptrEtM = etM.dataAsVec(), ptrDest = etDest.dataAsVec();
	for (unsigned i = 0; i < dataSize; ++i) ptrDest[i] = ptrEtM[i]>0 ? 1 : 0;
	ASSERT_TRUE(etM.cloneTo(m));

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.drelu_st_naive(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.drelu_mt_naive(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	//////////////////////////////////////////////////////////////////////////
	//best guess
	dest.zeros();
	diff = nanoseconds(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		bt = steady_clock::now();
		iM.drelu(m, dest);
		diff += steady_clock::now() - bt;
	}
	ASSERT_EQ(m, etM);
	ASSERT_EQ(dest, etDest);
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestIMathBasic, drelu) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

	//for (unsigned i = 200; i <= 250; i+=5) test_drelu(iM, i*100,1);

	test_drelu(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_drelu(iM, 1000);
	test_drelu(iM, 10000);
	test_drelu(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct loss_quadratic_EPS {};
template<> struct loss_quadratic_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct loss_quadratic_EPS<float> { static constexpr double eps = 3e-5; };
template<typename iMath>
void test_loss_quadratic(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt=10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing loss_quadratic() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());

	EXPECT_TRUE(steady_clock::is_steady);
	double tmtNaive, tstNaive, tBest;//tstVect, tmtVect
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t A, etA(rowsCnt, colsCnt), etY(rowsCnt, colsCnt), Y;
	real_t etQuadLoss = 0, quadLoss = 0;
	ASSERT_EQ(dataSize, etA.numel());

	//filling etalon
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etA, 5);
	rg.gen_matrix(etY, 5);
	ASSERT_TRUE(etA.cloneTo(A));
	ASSERT_TRUE(etY.cloneTo(Y));
	ASSERT_TRUE(etA == A && etY==Y);
	auto ptrEtA = etA.dataAsVec(), ptrEtY = etY.dataAsVec();

	for (unsigned i = 0; i < dataSize; ++i) {
		const real_t v = ptrEtA[i]- ptrEtY[i];
		etQuadLoss += v*v;
	}
	etQuadLoss = etQuadLoss / (2*etA.rows());

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) quadLoss = iM.loss_quadratic_st_naive(A,Y);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_NEAR(etQuadLoss, quadLoss, loss_quadratic_EPS<real_t>::eps);
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) quadLoss = iM.loss_quadratic_mt_naive(A,Y);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_NEAR(etQuadLoss, quadLoss, loss_quadratic_EPS<real_t>::eps);
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	/*//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) quadLoss = iM.loss_quadratic_st_vec(A,Y);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_NEAR(etQuadLoss, quadLoss, loss_quadratic_EPS<real_t>::eps);
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps, &tstVect));
	
	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) quadLoss = iM.loss_quadratic_mt_vec(A,Y);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_NEAR(etQuadLoss, quadLoss, loss_quadratic_EPS<real_t>::eps);
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps, &tmtVect));*/

	//////////////////////////////////////////////////////////////////////////
	//best guess
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) quadLoss = iM.loss_quadratic(A,Y);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_NEAR(etQuadLoss, quadLoss, loss_quadratic_EPS<real_t>::eps);
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
	// If it's not the best, then tune iM.loss_quadratic() appropriately


	//double rat = static_cast<double>(tmtNaive) / tmtVect;
	//STDCOUTL("Vectorized version is " << (rat > 1 ? "faster " : "SLOWER ") << std::setprecision(2) << (rat > 1 ? rat : 1 / rat) << " times");
}

TEST(TestIMathBasic, LossQuadratic) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;
	iMB iM;

	//for (unsigned i = 200; i <= 280; i+=5) test_loss_quadratic(iM, i*100,1);
	test_loss_quadratic(iM, 100);
#ifndef TESTS_SKIP_LONGRUNNING
	test_loss_quadratic(iM, 1000);
	test_loss_quadratic(iM, 10000);
	test_loss_quadratic(iM, 100000);
#endif
}

//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_dSigmQuadLoss_dZ(iMath& iM, typename iMath::realmtx_t::vec_len_t rowsCnt, typename iMath::realmtx_t::vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing dSigmQuadLoss_dZ() over " << rowsCnt << "x" << colsCnt << " matrix ("<< dataSize <<" elements) **************");

	EXPECT_TRUE(steady_clock::is_steady);
	double tmtNaive, tstNaive, tBest; //tstVect, tmtVect
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());
	realmtx_t etA(rowsCnt, colsCnt), etY(rowsCnt, colsCnt), etdLdZ(rowsCnt, colsCnt);
	realmtx_t A, Y, dLdZ;
	
	//filling etalons
	nnet_def_interfaces::iRng_t rg;
	rg.set_ithreads(iM.ithreads());
	rg.gen_matrix(etA, 5);
	rg.gen_matrix(etY, 5);
	ASSERT_TRUE(etA.cloneTo(A) && etY.cloneTo(Y) && dLdZ.resize(etdLdZ));
	ASSERT_TRUE(etY == Y && etA == A);
	const auto ptretA = etA.dataAsVec(), ptretY = etY.dataAsVec(), ptretdLdZ = etdLdZ.dataAsVec();
	for (numel_cnt_t i = 0; i < dataSize; ++i) {
		const auto a = ptretA[i];
		ptretdLdZ[i] = (a - ptretY[i])*a*(real_t(1.0) - a);
	}

	//testing performance
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iMath::ithreads_t> pw(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//single threaded naive
	dLdZ.zeros();
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.dSigmQuadLoss_dZ_st_naive(A,Y,dLdZ);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_EQ(dLdZ, etdLdZ);
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff,maxReps,&tstNaive));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded naive
	dLdZ.zeros();
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.dSigmQuadLoss_dZ_mt_naive(A, Y, dLdZ);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_EQ(dLdZ, etdLdZ);
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

/*
	//////////////////////////////////////////////////////////////////////////
	//single threaded vectorized
	dLdZ.zeros();
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.dSigmQuadLoss_dZ_st_vec(A,Y,dLdZ);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_EQ(dLdZ, etdLdZ);
	STDCOUTL("st_vec:\t\t" << utils::duration_readable(diff, maxReps, &tstVect));

	//////////////////////////////////////////////////////////////////////////
	//multi threaded vectorized
	dLdZ.zeros();
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.dSigmQuadLoss_dZ_mt_vec(A, Y, dLdZ);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_EQ(dLdZ, etdLdZ);
	STDCOUTL("mt_vec:\t\t" << utils::duration_readable(diff, maxReps, &tmtVect));*/

	//////////////////////////////////////////////////////////////////////////
	//best guess
	dLdZ.zeros();
	diff = nanoseconds(0);
	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) iM.dSigmQuadLoss_dZ(A, Y, dLdZ);
	diff += steady_clock::now() - bt;
	ASSERT_EQ(A, etA);
	ASSERT_EQ(Y, etY);
	ASSERT_EQ(dLdZ, etdLdZ);
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
	// If it's not the best, then tune iM.loss_quadratic() appropriately


	//double rat = static_cast<double>(tmtNaive) / tmtVect;
	//STDCOUTL("Vectorized version is " << (rat > 1 ? "faster " : "SLOWER ") << std::setprecision(2) << (rat > 1 ? rat : 1 / rat) << " times");
}

TEST(TestIMathBasic, dSigmQuadLoss_dZ) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	typedef math::iMath_basic<def_threads_t> iMB;

	iMB iM;

 	test_dSigmQuadLoss_dZ(iM, 500);

#ifndef TESTS_SKIP_LONGRUNNING
 	test_dSigmQuadLoss_dZ(iM, 2000);
	test_dSigmQuadLoss_dZ(iM, 20000);
 	test_dSigmQuadLoss_dZ(iM, 70000);
#endif
}


