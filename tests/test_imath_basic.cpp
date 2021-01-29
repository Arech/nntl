/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
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

//TODO: cleanup all this mess. Move all performance testing code into separate module and leave only correctness tests.

#include "../nntl/math.h"
#include "../nntl/common.h"

#include "../nntl/interface/math/mathn.h"
#include "../nntl/common_nn_data.h"

#include "../nntl/_supp/io/jsonreader.h"

#include <array>
#include <numeric>

#include "../nntl/utils/tictoc.h"
#include "imath_etalons.h"

#include "../nntl/_SNN_common.h"

#include "../nntl/_test/functions.h"

#if NNTL_MATLAB_AVAILABLE
#include "../nntl/_supp/io/matfile.h"
//using namespace nntl_supp;
#endif


using namespace nntl;
using namespace nntl::utils;
using namespace nntl::math_etalons;

typedef d_interfaces::real_t real_t;
typedef math::smatrix<real_t> realmtx_t;
typedef math::smatrix_deform<real_t> realmtxdef_t;

typedef d_interfaces::iThreads_t iThreads_t;
typedef math::MathN<real_t, iThreads_t> imath_basic_t;


template<typename RealT>
using def_keeper_tpl = ::nntl::_impl::interfaces_keeper<dt_interfaces<RealT>>;


static imath_basic_t iM;
const vec_len_t g_MinDataSizeDelta = 2 * iM.ithreads().workers_count() + 2;


using namespace ::std::chrono;

#ifdef TESTS_SKIP_LONGRUNNING
constexpr vec_len_t TEST_PERF_REPEATS_COUNT = 10;
//constexpr vec_len_t TEST_CORRECTN_REPEATS_COUNT = 100;
constexpr vec_len_t TEST_CORRECTN_REPEATS_COUNT = 5, _baseRowsCnt = 20;
#else
constexpr vec_len_t TEST_PERF_REPEATS_COUNT = 500;
//constexpr vec_len_t TEST_CORRECTN_REPEATS_COUNT = 50;
constexpr vec_len_t TEST_CORRECTN_REPEATS_COUNT = 10, _baseRowsCnt = 200;
#endif // NNTL_DEBUG

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/*
#if NNTL_MATLAB_AVAILABLE

template<typename RealT, bool bLowerTriangl, bool bNumStab>
void dump_deCov(const char* pFileName, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef RealT real_t;
	typedef nntl::math::smatrix<real_t> realmtx_t;
	typedef d_int_nI<real_t>::iThreads_t iThreads_t;
	typedef nntl::math::MathN<real_t, iThreads_t> imath_basic_t;
	imath_basic_t iM;
	
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "dump_deCov");

	realmtx_t A(rowsCnt, colsCnt), dL(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !dL.isAllocationFailed());

	iM.preinit(iM.loss_DeCov_needTempMem(true, A));
	ASSERT_TRUE(iM.init());
	d_int_nI<real_t>::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	rg.gen_matrix(A, real_t(5));
	const auto loss = iM.loss_deCov<bLowerTriangl, bNumStab>(A);
	iM.dLoss_deCov<bLowerTriangl, bNumStab>(A, dL);

	using namespace nntl_supp;
	omatfile<> mf;
	ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName));

	mf << NNTL_SERIALIZATION_NVP(A);
	ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

	mf << NNTL_SERIALIZATION_NVP(loss);
	ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

	mf << NNTL_SERIALIZATION_NVP(dL);
	ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
}

TEST(TestMathN, DumpDeCov) {
	dump_deCov<double, false, true>("./test_data/deCov.mat", 300, 70);
}

#endif*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct dLoss_deCov_EPS {};
template<> struct dLoss_deCov_EPS<double> { static constexpr double eps = 6e-5; };
template<> struct dLoss_deCov_EPS<float> { static constexpr float eps = 1; };

template<typename RealT, bool bLowerTriangl, bool bNumStab>
void test_dLoss_deCov(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
#pragma warning(disable:4459)
	typedef RealT real_t;
	typedef nntl::math::smatrix<real_t> realmtx_t;
	typedef d_int_nI<real_t>::iThreads_t iThreads_t;
	typedef nntl::math::MathN<real_t, iThreads_t> imath_basic_t;
	imath_basic_t iM;
#pragma warning(default:4459)


	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "dLoss_deCov");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt, true), A2(rowsCnt, colsCnt, true), tDM(rowsCnt, colsCnt), tCov(colsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !A2.isAllocationFailed() && !tDM.isAllocationFailed() && !tCov.isAllocationFailed());
	realmtx_t dL_ET(rowsCnt, colsCnt), dL(rowsCnt, colsCnt);
	ASSERT_TRUE(!dL_ET.isAllocationFailed() && !dL.isAllocationFailed());
	::std::vector<real_t> tMean(colsCnt);

	iM.preinit(iM.loss_DeCov_needTempMem(true, A.size_no_bias()));
	ASSERT_TRUE(iM.init());
	d_int_nI<real_t>::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(A, real_t(5));
		A.copy_data_skip_bias(A2);

		dLoss_deCov_ET<bLowerTriangl>(A, dL_ET, tDM, tCov, tMean, iM);
		ASSERT_MTX_EQ(A, A2, "_ET has changed const A!!");

		// 		loss = iM.dLoss_deCov_st(A, Y);
		// 		ASSERT_NEAR(etLoss, loss, dLoss_deCov_EPS<real_t>::eps);
		// 
		// 		loss = iM.dLoss_deCov_mt(A, Y);
		// 		ASSERT_NEAR(etLoss, loss, dLoss_deCov_EPS<real_t>::eps);

		iM.dLoss_deCov<bLowerTriangl, bNumStab>(A, dL);
		ASSERT_MTX_EQ(A, A2, "() has changed const A!!");
		ASSERT_REALMTX_NEAR(dL_ET, dL, "() failed!", dLoss_deCov_EPS<real_t>::eps);
		//ASSERT_NEAR(etLoss, loss, dLoss_deCov_EPS<real_t>::eps) << "<" << bLowerTriangl << "," << bNumStab << "> failed";
	}
}

TEST(TestMathN, dLoss_deCov) {
	//testing with double datatype. float reduces accuracy extremely (and that's totally expectable and shouldn't hurt a nn learning process).
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 2; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, false, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, false, true>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, true, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, true, true>(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 2; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, false, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, false, true>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, true, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_dLoss_deCov<double, true, true>(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct loss_deCov_EPS {};
template<> struct loss_deCov_EPS<double> { static constexpr double eps = 1e-5; };
template<> struct loss_deCov_EPS<float> { static constexpr float eps = 1e-1f; };

template<typename RealT, bool bLowerTriangl, bool bNumStab>
void test_loss_deCov(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
#pragma warning(disable:4459)
	typedef RealT real_t;
	typedef nntl::math::smatrix<real_t> realmtx_t;
	typedef d_int_nI<real_t>::iThreads_t iThreads_t;
	typedef nntl::math::MathN<real_t, iThreads_t> imath_basic_t;
	imath_basic_t iM;
#pragma warning(default:4459)


	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "loss_deCov");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt, true), A2(rowsCnt, colsCnt, true), tDM(rowsCnt, colsCnt), tCov(colsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !A2.isAllocationFailed() && !tDM.isAllocationFailed() && !tCov.isAllocationFailed());
	::std::vector<real_t> tMean(colsCnt);

	iM.preinit(iM.loss_DeCov_needTempMem(false, A.size_no_bias()));
	ASSERT_TRUE(iM.init());
	d_int_nI<real_t>::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(A, real_t(5));
		A.copy_data_skip_bias(A2);

		real_t loss, etLoss = loss_deCov_ET<bLowerTriangl>(A, tDM, tCov, tMean, iM);
		ASSERT_MTX_EQ(A, A2, "_ET has changed const A!!");

// 		loss = iM.loss_deCov_st(A, Y);
// 		ASSERT_NEAR(etLoss, loss, loss_deCov_EPS<real_t>::eps);
// 
// 		loss = iM.loss_deCov_mt(A, Y);
// 		ASSERT_NEAR(etLoss, loss, loss_deCov_EPS<real_t>::eps);

		loss = iM.loss_deCov<bLowerTriangl, bNumStab>(A);
		ASSERT_MTX_EQ(A, A2, "() has changed const A!!");
		ASSERT_NEAR(etLoss / rowsCnt, loss / rowsCnt, loss_deCov_EPS<real_t>::eps) << "<" << bLowerTriangl << "," << bNumStab << "> failed";

	}
}

TEST(TestMathN, loss_deCov) {
	//testing with double datatype. float reduces accuracy extremely (and that's totally expectable and shouldn't hurt a nn learning process).
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 2; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, false, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, false, true>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, true, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, true, true>(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 2; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, false, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, false, true>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, true, false>(r, c)));
			ASSERT_NO_FATAL_FAILURE((test_loss_deCov<double, true, true>(r, c)));
		}
	}
}

TEST(TestMathN, loss_deCovVisually) {
#pragma warning(disable:4459)
	typedef float real_t;
	typedef nntl::math::smatrix<real_t> realmtx_t;
	typedef d_int_nI<real_t>::iThreads_t iThreads_t;
	typedef nntl::math::MathN<real_t, iThreads_t> imath_basic_t;
	imath_basic_t iM;
#pragma warning(default:4459)
	
	const vec_len_t rowsCnt = 5, colsCnt = 4;
	static constexpr bool bLowerTriangl = false, bNumStab = false;

	realmtx_t A(rowsCnt, colsCnt), dL(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !dL.isAllocationFailed());
	
	iM.preinit(iM.loss_DeCov_needTempMem(false, A.size_no_bias()));
	ASSERT_TRUE(iM.init());

	A.set(0, 0, real_t(10));
	A.set(1, 0, real_t(11));
	A.set(2, 0, real_t(9));
	A.set(3, 0, real_t(10));
	A.set(4, 0, real_t(9));

	for (vec_len_t r = 0; r < rowsCnt; ++r) A.set(r, 1, A.get(r, 0) + real_t(2));
	for (vec_len_t r = 0; r < rowsCnt; ++r) A.set(r, 2, -A.get(r, 0) + real_t(20));
	for (vec_len_t r = 0; r < rowsCnt; ++r) A.set(r, 3, real_t(0));
	
	dbg_show_matrix(A, "A", 5, 0);
	STDCOUTL("Loss = " << (iM.loss_deCov<bLowerTriangl, bNumStab>(A)));
	
	iM.dLoss_deCov<bLowerTriangl, bNumStab>(A, dL);
	dbg_show_matrix(dL, "dL", 10, 3);

	STDCOUTL("===========================");

	realmtx_t A2(A.data(), A.rows(), A.cols() - 1), dL2(dL.data(), dL.rows(), dL.cols()-1);
	dbg_show_matrix(A2, "A 3cols", 5, 0);
	STDCOUTL("Loss = " << (iM.loss_deCov<bLowerTriangl, bNumStab>(A2)));

	iM.dLoss_deCov<bLowerTriangl, bNumStab>(A2, dL2);
	dbg_show_matrix(dL2, "dL 3 cols", 10, 3);

	for (vec_len_t r = 0; r < rowsCnt; ++r) A.set(r, 3, A.get(r, 0) + real_t(2));
	dbg_show_matrix(A, "A 4cols", 5, 0);
	STDCOUTL("Loss = " << (iM.loss_deCov<bLowerTriangl, bNumStab>(A)));

	iM.dLoss_deCov<bLowerTriangl, bNumStab>(A, dL);
	dbg_show_matrix(dL, "dL 4 cols", 10, 3);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct loss_xentropy_EPS {};
template<> struct loss_xentropy_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct loss_xentropy_EPS<float> { static constexpr float eps = 7e-5f; };
void test_loss_xentropy(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "loss_xentropy");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	const real_t frac = .5;
	realmtx_t A(rowsCnt, colsCnt), Y(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !Y.isAllocationFailed());

	iM.preinit(A.numel());
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	auto pA = A.data();
	auto pY = Y.data();
	const auto anum = A.numel();
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_norm(A);
		rg.gen_matrix_norm(Y);
		iM.ewBinarize_ip(Y, frac);

		pA[0] = real_t(0.);
		pY[0] = real_t(0.);
		if (anum > 1) {
			pA[1] = real_t(1.);
			pY[1] = real_t(1.);
			if (anum > 2) {
				pA[2] = real_t(0.);
				pY[2] = real_t(1.);
				if (anum > 3) {
					pA[3] = real_t(1.);
					pY[3] = real_t(0.);
				}
			}
		}

		real_t loss, etLoss = loss_xentropy_ET(A, Y);

		loss = iM.loss_xentropy_st(A, Y);
		ASSERT_NEAR(etLoss / rowsCnt, loss / rowsCnt, loss_xentropy_EPS<real_t>::eps);

		loss = iM.loss_xentropy_mt(A, Y);
		ASSERT_NEAR(etLoss / rowsCnt, loss / rowsCnt, loss_xentropy_EPS<real_t>::eps);

		loss = iM.loss_xentropy(A, Y);
		ASSERT_NEAR(etLoss / rowsCnt, loss / rowsCnt, loss_xentropy_EPS<real_t>::eps);
	}
}

TEST(TestMathN, lossXentropy) {
	const numel_cnt_t elmsMax = g_MinDataSizeDelta;
	for (numel_cnt_t e = 1; e < elmsMax; ++e) {
		ASSERT_NO_FATAL_FAILURE(test_loss_xentropy(static_cast<vec_len_t>(e), 1));
	}
	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_loss_xentropy(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_ewBinarize_ip_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10, const real_t frac = .5) {
	MTXSIZE_SCOPED_TRACE1(rowsCnt, colsCnt, "ewBinarize_ip, frac=", frac);
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt), A_orig(rowsCnt, colsCnt), A_ET(rowsCnt, colsCnt);
	ASSERT_TRUE(!A_orig.isAllocationFailed() && !A.isAllocationFailed() && !A_ET.isAllocationFailed());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_norm(A_orig);

		A_orig.clone_to(A_ET);
		ewBinarize_ip_ET(A_ET, frac);

		A_orig.clone_to(A);
		iM.ewBinarize_ip_st(A, frac);
		ASSERT_MTX_EQ(A_ET, A, "st() failed correctness test");

		/*A_orig.clone_to(A);
		iM.ex_ewBinarize_ip_st(A, frac);
		ASSERT_MTX_EQ(A_ET, A, "ex_st() failed correctness test");

		A_orig.clone_to(A);
		iM.ex2_ewBinarize_ip_st(A, frac);
		ASSERT_MTX_EQ(A_ET, A, "ex2_st() failed correctness test");*/

		A_orig.clone_to(A);
		iM.ewBinarize_ip_mt(A, frac);
		ASSERT_MTX_EQ(A_ET, A, "mt() failed correctness test");

		A_orig.clone_to(A);
		iM.ewBinarize_ip(A, frac);
		ASSERT_MTX_EQ(A_ET, A, "() failed correctness test");
	}
}

TEST(TestMathN, ewBinarizeIp) {
	const numel_cnt_t elmsMax = 3*g_MinDataSizeDelta;
	for (numel_cnt_t e = 1; e < elmsMax; ++e) {
		ASSERT_NO_FATAL_FAILURE(test_ewBinarize_ip_corr(static_cast<vec_len_t>(e), 1, real_t(.5)));
		ASSERT_NO_FATAL_FAILURE(test_ewBinarize_ip_corr(static_cast<vec_len_t>(e), 1, real_t(.1)));
		ASSERT_NO_FATAL_FAILURE(test_ewBinarize_ip_corr(static_cast<vec_len_t>(e), 1, real_t(.9)));
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_ewBinarize_ip_corr(r, c, real_t(.5)));
	}
}

void test_ewBinarize_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10, const real_t frac = real_t(.5)) {
	MTXSIZE_SCOPED_TRACE1(rowsCnt, colsCnt, "ewBinarize, frac=", frac);
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	typedef math::smatrix<char> binmtx_t;

	realmtx_t A(rowsCnt, colsCnt);
	binmtx_t DestET(rowsCnt, colsCnt), Dest(rowsCnt, colsCnt);

	ASSERT_TRUE(!A.isAllocationFailed() && !DestET.isAllocationFailed() && !Dest.isAllocationFailed());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_norm(A);

		ewBinarize_ET(DestET, A, frac);

		::std::fill(Dest.begin(), Dest.end(), binmtx_t::value_type(-1));
		iM.ewBinarize_st(Dest, A, frac);
		ASSERT_MTX_EQ(DestET, Dest, "st() failed correctness test");

		::std::fill(Dest.begin(), Dest.end(), binmtx_t::value_type(-1));
		iM.ewBinarize_mt(Dest, A, frac);
		ASSERT_MTX_EQ(DestET, Dest, "mt() failed correctness test");

		::std::fill(Dest.begin(), Dest.end(), binmtx_t::value_type(-1));
		iM.ewBinarize(Dest, A, frac);
		ASSERT_MTX_EQ(DestET, Dest, "() failed correctness test");
	}
}

TEST(TestMathN, ewBinarize) {
	const vec_len_t elmsMax = g_MinDataSizeDelta;
	for (vec_len_t e = 1; e < elmsMax; ++e) {
		ASSERT_NO_FATAL_FAILURE(test_ewBinarize_corr(e, 1, real_t(.5)));
		ASSERT_NO_FATAL_FAILURE(test_ewBinarize_corr(e, 1, real_t(.1)));
		ASSERT_NO_FATAL_FAILURE(test_ewBinarize_corr(e, 1, real_t(.9)));
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_ewBinarize_corr(r, c, real_t(.5)));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////




template<typename T, typename TD, typename RngT = dt_interfaces<T>::iRng_t>
void test_ewBinarizeBatch_corr2(const T frac, ::nntl::math::smatrix_deform<T>& src, ::nntl::math::smatrix_deform<T>& src_ETHlpr
	, ::nntl::math::smatrix_deform<TD>& dest, ::nntl::math::smatrix_deform<TD>& destET, RngT& rg, const ::std::vector<vec_len_t>& idxs_ETHlpr)
{
	ASSERT_SUPPORTED_REAL_T(T);
	//in fact, there's no reason to restrict to floating point, it's just a bit easier to make test with it.
	//but if it works with any fp - it works with anything

	if (src.emulatesBiases()) {
		ASSERT_TRUE(src_ETHlpr.emulatesBiases() && dest.emulatesBiases() && destET.emulatesBiases());
		ASSERT_TRUE(src.test_biases_strict() && src_ETHlpr.test_biases_strict() && dest.test_biases_strict() && destET.test_biases_strict());
	} else {
		ASSERT_TRUE(!src_ETHlpr.emulatesBiases() && !dest.emulatesBiases() && !destET.emulatesBiases());
	}
	ASSERT_TRUE(dest.size() == destET.size());
	ASSERT_TRUE(dest.cols() == src.cols() && src.rows() <= dest.rows());
	ASSERT_TRUE(src_ETHlpr.size() == dest.size());

	const auto maxSrcRows = src.rows();
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	vec_len_t dr;
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias_norm(src_ETHlpr);
		
		//computing destET
		ewBinarize_ET(destET, src_ETHlpr, frac);

		//computing dest with _st()
		dest.nans_no_bias();
		dr = 0;
		while (dr < dest.rows()) {
			//preparing src matrix
			const vec_len_t rtp = ::std::min(maxSrcRows, dest.rows() - dr);
			src.deform_rows(rtp);

			//extracting current src from src_ETHlpr
			mExtractRows_ET(src_ETHlpr, &idxs_ETHlpr[dr], src);

			//executing function to test
			iM.ewBinarizeBatch_st(dest, dr, src, frac);

			//updating dr
			dr += rtp;
		}
		ASSERT_MTX_EQt(TD, destET, dest, "ewBinarizeBatch_st failed!");

		dest.nans_no_bias();
		dr = 0;
		while (dr < dest.rows()) {
			const vec_len_t rtp = ::std::min(maxSrcRows, dest.rows() - dr);
			src.deform_rows(rtp);
			mExtractRows_ET(src_ETHlpr, &idxs_ETHlpr[dr], src);
			iM.ewBinarizeBatch_mt(dest, dr, src, frac);
			dr += rtp;
		}
		ASSERT_MTX_EQt(TD, destET, dest, "ewBinarizeBatch_mt failed!");

		dest.nans_no_bias();
		dr = 0;
		while (dr < dest.rows()) {
			const vec_len_t rtp = ::std::min(maxSrcRows, dest.rows() - dr);
			src.deform_rows(rtp);
			mExtractRows_ET(src_ETHlpr, &idxs_ETHlpr[dr], src);
			iM.ewBinarizeBatch(dest, dr, src, frac);
			dr += rtp;
		}
		ASSERT_MTX_EQt(TD, destET, dest, "ewBinarizeBatch failed!");
	}
}

template<typename T, typename TD = char>
void test_ewBinarizeBatch_corr(vec_len_t destRows, vec_len_t srcRows, vec_len_t colsCnt, const T frac) {
	//MTXSIZE_SCOPED_TRACE1d2f(srcRows, colsCnt, "ewBinarizeBatch, (destRows,frac)", destRows, frac);
	ASSERT_TRUE(srcRows <= destRows) << "Invalid params for test_ewBinarizeBatch_corr";

	dt_interfaces<T>::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	::nntl::math::smatrix_deform<T> src(srcRows, colsCnt), src_ETHlpr(destRows, colsCnt);
	::nntl::math::smatrix_deform<TD> dest(destRows, colsCnt), destET(destRows, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !src_ETHlpr.isAllocationFailed() && !dest.isAllocationFailed() && !destET.isAllocationFailed());

	::std::vector<vec_len_t> idxs_ETHlpr(destRows);
	::std::iota(idxs_ETHlpr.begin(), idxs_ETHlpr.end(), vec_len_t(0));

	//no biases mode
	ASSERT_TRUE(!src.emulatesBiases() && !src_ETHlpr.emulatesBiases() && !dest.emulatesBiases() && !destET.emulatesBiases());
	
	{
		MTXSIZE_SCOPED_TRACE_TYPED_1d(src.rows(), src.cols_no_bias(), "test_ewBinarizeBatch_corr no bias, dest rows=", dest.rows());
		ASSERT_NO_FATAL_FAILURE(test_ewBinarizeBatch_corr2(frac, src, src_ETHlpr, dest, destET, rg, idxs_ETHlpr));
	}

	if (colsCnt > 1) {
		//biases mode
		src._enforce_biases(); src.set_biases();
		src_ETHlpr._enforce_biases(); src_ETHlpr.set_biases();
		dest._enforce_biases(); dest.set_biases();
		destET._enforce_biases(); destET.set_biases();

		MTXSIZE_SCOPED_TRACE_TYPED_1d(src.rows(), src.cols_no_bias(), "test_ewBinarizeBatch_corr, with bias, dest rows=", dest.rows());
		ASSERT_NO_FATAL_FAILURE(test_ewBinarizeBatch_corr2(frac, src, src_ETHlpr, dest, destET, rg, idxs_ETHlpr));
	}
}

TEST(TestMathN, ewBinarizeBatch) {
	const vec_len_t maxCols = g_MinDataSizeDelta, maxSrcRows = g_MinDataSizeDelta;

	const auto fCheck = [maxCols,maxSrcRows](const vec_len_t destRows) {
		for (vec_len_t sr = 1; sr < maxSrcRows; ++sr) {
			if (sr <= destRows) {
				for (vec_len_t c = 1; c < maxCols; ++c) {
					ASSERT_NO_FATAL_FAILURE(test_ewBinarizeBatch_corr(destRows, sr, c, real_t(.5)));
					ASSERT_NO_FATAL_FAILURE(test_ewBinarizeBatch_corr(destRows, sr, c, real_t(.1)));
					ASSERT_NO_FATAL_FAILURE(test_ewBinarizeBatch_corr(destRows, sr, c, real_t(.9)));
				}
			}
		}
	};

	for (vec_len_t r = 1; r < 5 * g_MinDataSizeDelta; r += g_MinDataSizeDelta - 1) ASSERT_NO_FATAL_FAILURE(fCheck(r));

	const vec_len_t maxRows = _baseRowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = _baseRowsCnt; r < maxRows; ++r) ASSERT_NO_FATAL_FAILURE(fCheck(r));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct softmax_parts_EPS {};
template<> struct softmax_parts_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct softmax_parts_EPS<float> { static constexpr float eps = 1e-5f; };

void test_softmax_parts(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "softmax_parts");
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);

	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	const auto denominatorElmsMax = realmtx_t::sNumel(rowsCnt, iM.ithreads().workers_count());
	::std::vector<real_t> vec_max(rowsCnt), vec_den(denominatorElmsMax), vec_num(dataSize), vec_den2(denominatorElmsMax), vec_num2(dataSize);

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 2);
		mrwMax_ET(A, &vec_max[0]);

		softmax_parts_ET(A, &vec_max[0], &vec_den[0], &vec_num[0]);

		::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
		::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
		iM.softmax_parts_st_rw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
		//real denominator takes only a first row of vec_den
		for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "st_rw() failed denominator vector comparision @ " << i;
		ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st_rw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);

		::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
		::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
		iM.softmax_parts_st_cw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
		//real denominator takes only a first row of vec_den
		for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "st_cw() failed denominator vector comparision @ " << i;
		ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st_cw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);

		::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
		::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
		iM.softmax_parts_st(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
		//real denominator takes only a first row of vec_den
		for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "st() failed denominator vector comparision @ " << i;
		ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);

		if (colsCnt > imath_basic_t::Thresholds_t::softmax_parts_mt_cw_ColsPerThread) {
			::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
			::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
			iM.softmax_parts_mt_cw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
			//real denominator takes only a first row of vec_den
			for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "mt_cw() failed denominator vector comparision @ " << i;
			ASSERT_VECTOR_NEAR(vec_num, vec_num2, "mt_cw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);
		}
		
		::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
		::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
		iM.softmax_parts_mt_rw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
		//real denominator takes only a first row of vec_den
		for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "mt_rw() failed denominator vector comparision @ " << i;
		ASSERT_VECTOR_NEAR(vec_num, vec_num2, "mt_rw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);
				
		::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
		::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
		iM.softmax_parts_mt(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
		//real denominator takes only a first row of vec_den
		for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "mt() failed denominator vector comparision @ " << i;
		ASSERT_VECTOR_NEAR(vec_num, vec_num2, "mt() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);

		::std::fill(vec_den2.begin(), vec_den2.end(), real_t(-1));
		::std::fill(vec_num2.begin(), vec_num2.end(), real_t(-1));
		iM.softmax_parts(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
		//real denominator takes only a first row of vec_den
		for (vec_len_t i = 0; i < rowsCnt; ++i) ASSERT_NEAR(vec_den[i], vec_den2[i], softmax_parts_EPS<real_t>::eps) << "() failed denominator vector comparision @ " << i;
		ASSERT_VECTOR_NEAR(vec_num, vec_num2, "() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);
	}
}
TEST(TestMathN, SoftmaxParts) {
	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_softmax_parts(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct softmax_EPS {};
template<> struct softmax_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct softmax_EPS<float> { static constexpr double eps = 1e-5; };

template<bool bHasBiases>
void test_softmax(vec_len_t rowsCnt, vec_len_t colsCnt) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, bHasBiases ? "softmax with biases" : "softmax");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtxdef_t A(rowsCnt, colsCnt, bHasBiases), A_ET(rowsCnt, colsCnt, bHasBiases), A_orig(rowsCnt, colsCnt, bHasBiases);
	ASSERT_TRUE(!A.isAllocationFailed() && !A_ET.isAllocationFailed() && !A_orig.isAllocationFailed());

	const auto maxSoftmaxMemSize = iM.softmax_needTempMem(A);
	iM.preinit(maxSoftmaxMemSize);
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t rr = 0; rr < testCorrRepCnt; ++rr) {
		if (bHasBiases) rg.gen_matrix_no_bias(A_orig, 5);
		else rg.gen_matrix(A_orig, 5);

		A_orig.clone_to(A_ET);
		auto pTmp = iM._istor_alloc(maxSoftmaxMemSize);
		softmax_ET(A_ET, pTmp);
		iM._istor_free(pTmp, maxSoftmaxMemSize);
		
		A_orig.clone_to(A);
		iM.softmax_st(A);
		ASSERT_REALMTX_NEAR(A_ET, A, "st() failed", softmax_EPS<real_t>::eps);

		A_orig.clone_to(A);
		iM.softmax_mt(A);
		ASSERT_REALMTX_NEAR(A_ET, A, "mt() failed", softmax_EPS<real_t>::eps);

		A_orig.clone_to(A);
		iM.softmax(A);
		ASSERT_REALMTX_NEAR(A_ET, A, "() failed", softmax_EPS<real_t>::eps);
	}
}
TEST(TestMathN, Softmax) {
	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_softmax<false>(r, c));
			ASSERT_NO_FATAL_FAILURE(test_softmax<true>(r, c));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct loss_softmax_xentropy_EPS {};
template<> struct loss_softmax_xentropy_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct loss_softmax_xentropy_EPS<float> { static constexpr float eps = 4e-5f; };
void test_loss_softmax_xentropy(vec_len_t rowsCnt, vec_len_t colsCnt) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "loss_softmax_xentropy");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	realmtx_t A(rowsCnt, colsCnt), Y(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed() && !Y.isAllocationFailed());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	real_t et, l;
	for (vec_len_t rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix_norm(A);
		rg.gen_matrix_norm(Y);

		et = loss_softmax_xentropy_ET(A, Y);

		l = iM.loss_softmax_xentropy_st(A, Y);
		ASSERT_NEAR(et / rowsCnt, l / rowsCnt, loss_softmax_xentropy_EPS<real_t>::eps) << "st failed";

		l = iM.loss_softmax_xentropy_mt(A, Y);
		ASSERT_NEAR(et / rowsCnt, l / rowsCnt, loss_softmax_xentropy_EPS<real_t>::eps) << "mt failed";

		l = iM.loss_softmax_xentropy(A, Y);
		ASSERT_NEAR(et / rowsCnt, l / rowsCnt, loss_softmax_xentropy_EPS<real_t>::eps) << "() failed";
	}
}
TEST(TestMathN, LossSoftmaxXentropy) {
	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) ASSERT_NO_FATAL_FAILURE(test_loss_softmax_xentropy(r, c));
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


template<typename base_t> struct vSumAbs_EPS {};
template<> struct vSumAbs_EPS<double> { static constexpr double eps = 3e-8; };
template<> struct vSumAbs_EPS<float> { static constexpr float eps = .5f; };
template<typename iMath>
void test_vSumAbs(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing vSumAbs() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix(A, 1);

		const auto vss = vSumAbs_ET(A);

		auto v = iM.vSumAbs_st(A);
		ASSERT_NEAR(vss, v, vSumAbs_EPS<real_t>::eps) << "vSumAbs_st failed correctness test";

		v = iM.vSumAbs_mt(A);
		ASSERT_NEAR(vss, v, vSumAbs_EPS<real_t>::eps) << "vSumAbs_mt failed correctness test";

		v = iM.vSumAbs(A);
		ASSERT_NEAR(vss, v, vSumAbs_EPS<real_t>::eps) << "vSumAbs failed correctness test";
	}

	tictoc tst, tmt, tb;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	real_t vv = 0;
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2);
		tst.tic();
		vv += iM.vSumAbs_st(A);
		tst.toc();

		rg.gen_matrix(A, 2);
		tmt.tic();
		vv += iM.vSumAbs_mt(A);
		tmt.toc();

		rg.gen_matrix(A, 2);
		tb.tic();
		vv += iM.vSumAbs(A);
		tb.toc();
	}
	tst.say("st");
	tmt.say("mt");
	tb.say("best");
	STDCOUTL(vv);
}
TEST(TestMathN, vSumAbs) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::vSumAbs, 100) test_vSumAbs(iM, i, 100);
}



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evAddScaledSign_ip(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evAddScaledSign_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed());

	real_t scaleCoeff = .5;

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(B, 2);

	{
		realmtx_t A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
		ASSERT_TRUE(!A2.isAllocationFailed() && !A3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(A, 2);
			A.clone_to(A2);
			A.clone_to(A3);

			evAddScaledSign_ip_ET(A2, scaleCoeff, B);

			iM.evAddScaledSign_ip_st(A, scaleCoeff, B);
			ASSERT_MTX_EQ(A2, A, "evAddScaledSign_ip_st failed correctness test");

			A3.clone_to(A);
			iM.evAddScaledSign_ip_mt(A, scaleCoeff, B);
			ASSERT_MTX_EQ(A2, A, "evAddScaledSign_ip_mt failed correctness test");

			A3.clone_to(A);
			iM.evAddScaledSign_ip(A, scaleCoeff, B);
			ASSERT_MTX_EQ(A2, A, "evAddScaledSign_ip failed correctness test");
		}
	}

	tictoc tst, tmt, tb;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tst.tic();
		iM.evAddScaledSign_ip_st(A, scaleCoeff, B);
		tst.toc();

		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tmt.tic();
		iM.evAddScaledSign_ip_mt(A, scaleCoeff, B);
		tmt.toc();

		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tb.tic();
		iM.evAddScaledSign_ip(A, scaleCoeff, B);
		tb.toc();
	}
	tst.say("st");
	tmt.say("mt");
	tb.say("best");
}
TEST(TestMathN, evAddScaledSign_ip) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evAddScaledSign_ip, 100) test_evAddScaledSign_ip(iM, i, 100);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_evAddScaled_ip_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "evAddScaled_ip");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed());

	real_t scaleCoeff = .5;

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(B, 2);

	realmtx_t A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
	ASSERT_TRUE(!A2.isAllocationFailed() && !A3.isAllocationFailed());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix(A, 2);
		A.clone_to(A2);
		A.clone_to(A3);

		evAddScaled_ip_ET(A2, scaleCoeff, B);

		iM.evAddScaled_ip_st(A, scaleCoeff, B);
		ASSERT_MTX_EQ(A2, A, "evAddScaled_ip_st failed correctness test");

		A3.clone_to(A);
		iM.evAddScaled_ip_mt(A, scaleCoeff, B);
		ASSERT_MTX_EQ(A2, A, "evAddScaled_ip_mt failed correctness test");

		A3.clone_to(A);
		iM.evAddScaled_ip(A, scaleCoeff, B);
		ASSERT_MTX_EQ(A2, A, "evAddScaled_ip failed correctness test");
	}
}
TEST(TestMathN, evAddScaled_ip) {
	for (vec_len_t r = 1; r < 2*g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < 2*g_MinDataSizeDelta; ++c) {
			test_evAddScaled_ip_corr(r, c);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evAdd_ip(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evAdd_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(B, 2);

	{
		realmtx_t A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
		ASSERT_TRUE(!A2.isAllocationFailed() && !A3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(A, 2);
			A.clone_to(A2);
			A.clone_to(A3);

			evAdd_ip_ET(A2, B);

			iM.evAdd_ip_st(A, B);
			ASSERT_MTX_EQ(A2, A, "evAdd_ip_st failed correctness test");

			A3.clone_to(A);
			iM.evAdd_ip_mt(A, B);
			ASSERT_MTX_EQ(A2, A, "evAdd_ip_mt failed correctness test");

			A3.clone_to(A);
			iM.evAdd_ip(A, B);
			ASSERT_MTX_EQ(A2, A, "evAdd_ip failed correctness test");
		}
	}
	
	tictoc tst, tmt, tb;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tst.tic();
		iM.evAdd_ip_st(A, B);
		tst.toc();

		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tmt.tic();
		iM.evAdd_ip_mt(A, B);
		tmt.toc();

		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tb.tic();
		iM.evAdd_ip(A, B);
		tb.toc();
	}
	tst.say("st");
	tmt.say("mt");
	tb.say("best");
}
TEST(TestMathN, evAddIp) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evAdd_ip, 100) test_evAdd_ip(iM, i, 100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evMulCipSubip(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing evMulC_ip_Sub_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t momentum(real_t(.9));
	realmtx_t vW(rowsCnt, colsCnt), W(colsCnt, rowsCnt), vW2(colsCnt, rowsCnt), W2(colsCnt, rowsCnt), vW3(colsCnt, rowsCnt), W3(colsCnt, rowsCnt);
	ASSERT_TRUE(!vW.isAllocationFailed() && !W.isAllocationFailed() && !vW2.isAllocationFailed()
		&& !W2.isAllocationFailed() && !vW3.isAllocationFailed() && !W3.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(vW2, 2);
	rg.gen_matrix(W2, 2);
	vW2.clone_to(vW);
	W2.clone_to(W);
	vW2.clone_to(vW3);
	W2.clone_to(W3);

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		evCMulSub_ET(iM, vW3, momentum, W3);
			
		iM.evMulC_ip_Sub_ip_st(vW, momentum, W);
		ASSERT_MTX_EQ(vW3, vW, "evMulC_ip_Sub_ip_st failed correctness test on vW");
		ASSERT_MTX_EQ(W3, W, "evMulC_ip_Sub_ip_st failed correctness test on W");

		iM.evMulC_ip_Sub_ip_mt(vW2, momentum, W2);
		ASSERT_MTX_EQ(vW3, vW2, "evMulC_ip_Sub_ip_mt failed correctness test on vW");
		ASSERT_MTX_EQ(W3, W2, "evMulC_ip_Sub_ip_mt failed correctness test on W");
	}

// 	rg.gen_matrix(vW2, 2);
// 	rg.gen_matrix(W2, 2);
// 	vW2.clone_to(vW);
// 	W2.clone_to(W);
// 	vW2.clone_to(vW3);
// 	W2.clone_to(W3);

	tictoc tst, tmt, tb;
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(vW, 2);
		rg.gen_matrix(W, 2);
		tst.tic();
		iM.evMulC_ip_Sub_ip_st(vW, momentum, W);
		tst.toc();

		rg.gen_matrix(vW2, 2);
		rg.gen_matrix(W2, 2);
		tmt.tic();
		iM.evMulC_ip_Sub_ip_mt(vW2, momentum, W2);
		tmt.toc();

		rg.gen_matrix(vW3, 2);
		rg.gen_matrix(W3, 2);
		tb.tic();
		iM.evMulC_ip_Sub_ip(vW3, momentum, W3);
		tb.toc();
	}
	tst.say("st");
	tmt.say("mt");
	tb.say("best");
}

TEST(TestMathN, evMulCipSubip) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evMulC_ip_Sub_ip, 100) test_evMulCipSubip(iM, i, 100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct mCheck_normalize_rows_EPS {};
template<> struct mCheck_normalize_rows_EPS<double> { static constexpr double eps = 1e-12; };
template<> struct mCheck_normalize_rows_EPS<float> { static constexpr float eps = 1e-6f; };

void test_mCheck_normalize_rows(vec_len_t rowsCnt, vec_len_t colsCnt, const bool bNormIncludesBias) {
	ASSERT_GT(int(colsCnt), 1) << "There must be at least 2 columns in weight matrix";

	MTXSIZE_SCOPED_TRACE1(rowsCnt, colsCnt, "mCheck_normalize_rows, bNormIncludesBias=", real_t(bNormIncludesBias));

	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t scale = 3;
	const real_t newNormSq = 1;//3*3/sqrt(colsCnt - (!bNormIncludesBias));
	realmtxdef_t W(rowsCnt, colsCnt), srcW(rowsCnt, colsCnt), ones(rowsCnt,colsCnt);
	ASSERT_TRUE(!W.isAllocationFailed() && !srcW.isAllocationFailed() && !ones.isAllocationFailed());
	ones.ones();

	iM.preinit(iM.mCheck_normalize_rows_needTempMem(W));
	ASSERT_TRUE(iM.init());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	realmtxdef_t etW(rowsCnt, colsCnt);
	ASSERT_TRUE(!etW.isAllocationFailed());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_gtz(srcW, scale);
		iM.evAdd_ip(srcW, ones);//to make sure norms will be greater than 1

		srcW.clone_to(etW);
		auto pTmp = iM._istor_alloc(rowsCnt);
		auto meanNorm = rowvecs_renorm_ET(etW, newNormSq, bNormIncludesBias, pTmp);
		iM._istor_free(pTmp, rowsCnt);
		ASSERT_LT(newNormSq, meanNorm) << "Mean norm should be greater than a new norm";

		srcW.clone_to(W);
		iM.mCheck_normalize_rows_st(W, newNormSq, bNormIncludesBias);
		ASSERT_REALMTX_NEAR(etW, W, "st failed correctness test", mCheck_normalize_rows_EPS<real_t>::eps);

		srcW.clone_to(W);
		iM.mCheck_normalize_rows_mt(W, newNormSq, bNormIncludesBias);
		ASSERT_REALMTX_NEAR(etW, W, "mt failed correctness test", mCheck_normalize_rows_EPS<real_t>::eps);

		srcW.clone_to(W);
		iM.mCheck_normalize_rows(W, newNormSq, bNormIncludesBias);
		ASSERT_REALMTX_NEAR(etW, W, "() failed correctness test", mCheck_normalize_rows_EPS<real_t>::eps);
	}
}

TEST(TestMathN, mCheckNormalizeRows) {
	const vec_len_t maxCols = 2*g_MinDataSizeDelta, maxRows = 2*g_MinDataSizeDelta;
	for (vec_len_t r = 1; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_mCheck_normalize_rows(r, c + 1, false));
			ASSERT_NO_FATAL_FAILURE(test_mCheck_normalize_rows(r, c + 1, true));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evSub(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evSub() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt), C(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed() && !C.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(A, 2);
	rg.gen_matrix(B, 2);

	{
		realmtx_t C2(rowsCnt, colsCnt);
		ASSERT_TRUE(!C2.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			evSub_ET(A, B, C2);

			iM.evSub_st_naive(A, B, C);
			ASSERT_MTX_EQ(C2, C, "evSub_st_naive failed correctness test");

			iM.evSub_mt_naive(A, B, C);
			ASSERT_MTX_EQ(C2, C, "evSub_mt_naive failed correctness test");

			iM.evSub(A, B, C);
			ASSERT_MTX_EQ(C2, C, "evSub failed correctness test");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evSub_st_naive(A, B, C);
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evSub_mt_naive(A, B, C);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evSub(A, B, C);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, evSub) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evSub, 10) test_evSub(iM, i, 10);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evSub_ip(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evSub_ip() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t B(rowsCnt, colsCnt), A(rowsCnt, colsCnt);
	ASSERT_TRUE(!B.isAllocationFailed() && !A.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(B, 2);

	{
		realmtx_t A2(rowsCnt, colsCnt), A3(rowsCnt, colsCnt);
		ASSERT_TRUE(!A2.isAllocationFailed() && !A3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(A, 2);
			A.clone_to(A2);
			A.clone_to(A3);

			evSub_ip_ET(A2, B);

			iM.evSub_ip_st_naive(A, B);
			ASSERT_MTX_EQ(A2, A, "evSub_ip_st_naive failed correctness test");

			A3.clone_to(A);
			iM.evSub_ip_mt_naive(A, B);
			ASSERT_MTX_EQ(A2, A, "evSub_ip_mt_naive failed correctness test");

			/*iM.evSub_ip_st_vec(A, B);
			ASSERT_MTX_EQ(A2, A, "evSub_ip_st_vec failed correctness test");

			A3.clone_to(A);
			iM.evSub_ip_mt_vec(A, B);
			ASSERT_MTX_EQ(A2, A, "evSub_ip_mt_vec failed correctness test");*/

			A3.clone_to(A);
			iM.evSub_ip(A, B);
			ASSERT_MTX_EQ(A2, A, "evSub_ip failed correctness test");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());
	
	utils::tictoc tS, tM, tB;

	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tS.tic();
		iM.evSub_ip_st_naive(A, B);
		tS.toc();

		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tM.tic();
		iM.evSub_ip_mt_naive(A, B);
		tM.toc();

		rg.gen_matrix(A, 2); rg.gen_matrix(B, 2);
		tB.tic();
		iM.evSub_ip(A, B);
		tB.toc();
	}

	tS.say("st_naive");
	tM.say("mt_naive");
	tB.say("best");
}

TEST(TestMathN, evSubIp) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evSub_ip, 100) test_evSub_ip(iM, i,100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_apply_momentum(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing apply_momentum() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t momentum(real_t(0.9));
	realmtx_t dW(rowsCnt, colsCnt), vW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !vW.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(dW, 2);

	{
		realmtx_t vW2(rowsCnt, colsCnt), vW3(rowsCnt, colsCnt);
		ASSERT_TRUE(!vW2.isAllocationFailed() && !vW3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(vW, 2);
			vW.clone_to(vW2);
			vW.clone_to(vW3);

			apply_momentum_ET(vW2, momentum, dW);

			iM.apply_momentum_st(vW, momentum, dW);
			ASSERT_MTX_EQ(vW2, vW, "apply_momentum_st failed correctness test");

			vW3.clone_to(vW);
			iM.apply_momentum_mt(vW,momentum, dW);
			ASSERT_MTX_EQ(vW2, vW, "apply_momentum_mt failed correctness test");

			vW3.clone_to(vW);
			iM.apply_momentum(vW, momentum, dW);
			ASSERT_MTX_EQ(vW2, vW, "apply_momentum failed correctness test");
		}
	}
	rg.gen_matrix(vW, 2);

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.apply_momentum_st(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.apply_momentum_mt(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.apply_momentum(vW, momentum, dW);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, applyMomentum) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::apply_momentum, 100) test_apply_momentum(iM, i,100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_ApplyILR_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "ApplyILR");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);

	real_t decr = real_t(.8), incr = real_t(1.3), capH = real_t(9.9), capL = real_t(0.1);

	realmtx_t dW(rowsCnt, colsCnt), prevdW(rowsCnt, colsCnt), gain(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !prevdW.isAllocationFailed() && !gain.isAllocationFailed());

	iM.preinit(iM.apply_ILR_needTempMem<real_t>(nntl::math::smatrix_td::mtx_size_t(rowsCnt, colsCnt)));
	ASSERT_TRUE(iM.init());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(prevdW, 10);

	//////////////////////////////////////////////////////////////////////////
	//testing correctness
	realmtx_t dW2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt), gain2(rowsCnt, colsCnt), gain3(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW2.isAllocationFailed() && !dW3.isAllocationFailed() && !gain2.isAllocationFailed() && !gain3.isAllocationFailed());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix(dW, 10);
		auto pDD = dW.data();
		pDD[0] = 0;
		pDD[rowsCnt - 1] = 0;
		pDD[dataSize - rowsCnt + 1] = 0;

		dW.clone_to(dW2);
		dW.clone_to(dW3);
		rg.gen_matrix_gtz(gain, 10);
		gain.clone_to(gain2);
		gain.clone_to(gain3);

		apply_ILR_ET(dW, prevdW, gain, decr, incr, capL, capH);

		iM.apply_ILR_st_naive(dW2, prevdW, gain2, decr, incr, capL, capH);
		ASSERT_MTX_EQ(dW2, dW, "apply_ILR_st_naive: wrong dLdW matrix content!");
		ASSERT_MTX_EQ(gain2, gain, "apply_ILR_st_naive: wrong ILRGain matrix content!");

		dW3.clone_to(dW2);
		gain3.clone_to(gain2);
		iM.apply_ILR_st_vec(dW2, prevdW, gain2, decr, incr, capL, capH);
		ASSERT_MTX_EQ(dW2, dW, "apply_ILR_st_vec: wrong dLdW matrix content!");
		ASSERT_MTX_EQ(gain2, gain, "apply_ILR_st_vec: wrong ILRGain matrix content!");

		dW3.clone_to(dW2);
		gain3.clone_to(gain2);
		iM.apply_ILR_mt_naive(dW2, prevdW, gain2, decr, incr, capL, capH);
		ASSERT_MTX_EQ(dW2, dW, "apply_ILR_mt_naive: wrong dLdW matrix content!");
		ASSERT_MTX_EQ(gain2, gain, "apply_ILR_mt_naive: wrong ILRGain matrix content!");

		dW3.clone_to(dW2);
		gain3.clone_to(gain2);
		iM.apply_ILR_mt_vec(dW2, prevdW, gain2, decr, incr, capL, capH);
		ASSERT_MTX_EQ(dW2, dW, "apply_ILR_mt_vec: wrong dLdW matrix content!");
		ASSERT_MTX_EQ(gain2, gain, "apply_ILR_mt_vec: wrong ILRGain matrix content!");

		dW3.clone_to(dW2);
		gain3.clone_to(gain2);
		iM.apply_ILR(dW2, prevdW, gain2, decr, incr, capL, capH);
		ASSERT_MTX_EQ(dW2, dW, "apply_ILR: wrong dLdW matrix content!");
		ASSERT_MTX_EQ(gain2, gain, "apply_ILR: wrong ILRGain matrix content!");
	}
}

TEST(TestMathN, ApplyILR) {
	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_ApplyILR_corr(r, c));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evAbs_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evAbs() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), dest(rowsCnt, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());	

	{
		realmtx_t dest2(rowsCnt, colsCnt);
		ASSERT_TRUE(!dest2.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(src, 10);
			evAbs_ET(dest2, src);

			iM.evAbs_st(dest, src);
			ASSERT_MTX_EQ(dest2, dest, "evAbs_st failed correctness test");

			iM.evAbs_mt(dest, src);
			ASSERT_MTX_EQ(dest2, dest, "evAbs_mt failed correctness test");

			iM.evAbs(dest, src);
			ASSERT_MTX_EQ(dest2, dest, "evAbs failed correctness test");
		}
	}
	rg.gen_matrix(src, 10);

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r)  iM.evAbs_st(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r)  iM.evAbs_mt(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evAbs(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, evAbsPerf) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evAbs, 100) test_evAbs_perf(iM, i,100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_evSquare_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evSquare() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), dest(rowsCnt, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	
	{
		realmtx_t dest2(rowsCnt, colsCnt);
		ASSERT_TRUE(!dest2.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(src, 10);
			evSquare_ET(dest2, src);

			iM.evSquare_st(dest, src);
			ASSERT_MTX_EQ(dest2, dest, "evSquare_st failed correctness test");

			iM.evSquare_mt(dest, src);
			ASSERT_MTX_EQ(dest2, dest, "evSquare_mt failed correctness test");

			iM.evSquare(dest, src);
			ASSERT_MTX_EQ(dest2, dest, "evSquare failed correctness test");
		}
	}
	rg.gen_matrix(src, 10);

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());
	
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evSquare_st(dest,src);
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evSquare_mt(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) iM.evSquare(dest, src);
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, evSquarePerf) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evSquare, 100) test_evSquare_perf(iM, i,100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_modprop_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing ModProp() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t emaCoeff = real_t(.9), lr = real_t(.1), numStab = real_t(.00001);

	realmtx_t dW(rowsCnt, colsCnt), rms(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed());

	rms.zeros();

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//testing correctness
	{
		realmtx_t dW2(rowsCnt, colsCnt), rms2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt), rms3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !rms2.isAllocationFailed() && !dW3.isAllocationFailed() && !rms3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.clone_to(dW2);
			dW.clone_to(dW3);
			rg.gen_matrix_gtz(rms, 10);
			rms.clone_to(rms2);
			rms.clone_to(rms3);

			ModProp_ET(dW2, rms2, lr, emaCoeff, numStab);

			iM.ModProp_st(dW, rms, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "ModProp_st: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "ModProp_st: wrong rms");

			dW3.clone_to(dW);
			rms3.clone_to(rms);
			iM.ModProp_mt(dW, rms, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "ModProp_mt: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "ModProp_mt: wrong rms");

			dW3.clone_to(dW);
			rms3.clone_to(rms);
			iM.ModProp(dW, rms, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "ModProp: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "ModProp: wrong rms");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.ModProp_st(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.ModProp_mt(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.ModProp(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, ModPropPerf) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::ModProp, 1) test_modprop_perf(iM, i,1);
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_rprop_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RProp() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t lr = real_t(.1);

	realmtx_t dW(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed());
	
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	{
		realmtx_t dW2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !dW3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.clone_to(dW2);
			dW.clone_to(dW3);

			RProp_ET(dW2, lr);

			iM.RProp_st(dW, lr);
			ASSERT_MTX_EQ(dW2, dW, "RProp_st: wrong dW");

			dW3.clone_to(dW);
			iM.RProp_mt(dW, lr);
			ASSERT_MTX_EQ(dW2, dW, "RProp_mt: wrong dW");

			dW3.clone_to(dW);
			iM.RProp(dW, lr);
			ASSERT_MTX_EQ(dW2, dW, "RProp: wrong dW");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RProp_st(dW, lr);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RProp_mt(dW, lr);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RProp(dW, lr);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, RPropPerf) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::RProp, 1) test_rprop_perf(iM, i,1);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_rmspropgraves_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RMSProp_Graves() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t emaCoeff = real_t(.9), lr = real_t(.1), numStab = real_t(.00001);

	realmtx_t dW(rowsCnt, colsCnt), rms(rowsCnt, colsCnt), rmsG(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed() && !rmsG.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	{
		realmtx_t dW2(rowsCnt, colsCnt), rms2(rowsCnt, colsCnt), rmsG2(rowsCnt, colsCnt),
			dW3(rowsCnt, colsCnt), rms3(rowsCnt, colsCnt), rmsG3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !rms2.isAllocationFailed() && !dW3.isAllocationFailed() && !rms3.isAllocationFailed()
			&& !rmsG2.isAllocationFailed() && !rmsG3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.clone_to(dW2);
			dW.clone_to(dW3);
			evSquare_ET(rms, dW);
			rms.clone_to(rms2);
			rms.clone_to(rms3);
			dW.clone_to(rmsG);
			rmsG.clone_to(rmsG2);
			rmsG.clone_to(rmsG3);

			RMSProp_Graves_ET(dW2, rms2, rmsG2, lr, emaCoeff, numStab);

			iM.RMSProp_Graves_st(dW, rms, rmsG, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "RMSProp_Graves_st: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "RMSProp_Graves_st: wrong rms");
			ASSERT_MTX_EQ(rmsG2, rmsG, "RMSProp_Graves_st: wrong rmsG");

			dW3.clone_to(dW);
			rms3.clone_to(rms);
			rmsG3.clone_to(rmsG);
			iM.RMSProp_Graves_mt(dW, rms, rmsG, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "RMSProp_Graves_mt: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "RMSProp_Graves_mt: wrong rms");
			ASSERT_MTX_EQ(rmsG2, rmsG, "RMSProp_Graves_mt: wrong rmsG");

			dW3.clone_to(dW);
			rms3.clone_to(rms);
			rmsG3.clone_to(rmsG);
			iM.RMSProp_Graves(dW, rms, rmsG, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "RMSProp_Graves: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "RMSProp_Graves: wrong rms");
			ASSERT_MTX_EQ(rmsG2, rmsG, "RMSProp_Graves: wrong rmsG");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Graves_st(dW, rms, rmsG, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Graves_mt(dW, rms, rmsG, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Graves(dW, rms, rmsG, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, RMSProp_Graves) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::RMSProp_Graves, 10) test_rmspropgraves_perf(iM, i,10);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename iMath>
void test_rmsprophinton_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing RMSProp_Hinton() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	real_t emaCoeff = real_t(.9), lr = real_t(.1), numStab = real_t(.00001);

	realmtx_t dW(rowsCnt, colsCnt), rms(rowsCnt, colsCnt);
	ASSERT_TRUE(!dW.isAllocationFailed() && !rms.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	{
		realmtx_t dW2(rowsCnt, colsCnt), rms2(rowsCnt, colsCnt), dW3(rowsCnt, colsCnt), rms3(rowsCnt, colsCnt);
		ASSERT_TRUE(!dW2.isAllocationFailed() && !rms2.isAllocationFailed() && !dW3.isAllocationFailed() && !rms3.isAllocationFailed());

		for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix(dW, 10);
			dW.clone_to(dW2);
			dW.clone_to(dW3);
			evSquare_ET(rms, dW);
			rms.clone_to(rms2);
			rms.clone_to(rms3);

			RMSProp_Hinton_ET(dW2, rms2, lr, emaCoeff, numStab);

			iM.RMSProp_Hinton_st(dW, rms, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "RMSProp_Hinton_st: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "RMSProp_Hinton_st: wrong rms");

			dW3.clone_to(dW);
			rms3.clone_to(rms);
			iM.RMSProp_Hinton_mt(dW, rms, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "RMSProp_Hinton_mt: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "RMSProp_Hinton_mt: wrong rms");

			dW3.clone_to(dW);
			rms3.clone_to(rms);
			iM.RMSProp_Hinton(dW, rms, lr, emaCoeff, numStab);
			ASSERT_MTX_EQ(dW2, dW, "RMSProp_Hinton: wrong dW");
			ASSERT_MTX_EQ(rms2, rms, "RMSProp_Hinton: wrong rms");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Hinton_st(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Hinton_mt(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	diff = nanoseconds(0);
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dW, 10);
		bt = steady_clock::now();
		iM.RMSProp_Hinton(dW, rms, lr, emaCoeff, numStab);
		diff += steady_clock::now() - bt;
	}
	STDCOUTL("best:\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, RMSProp_Hinton) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::RMSProp_Hinton, 10) test_rmsprophinton_perf(iM, i,10);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_Adam_corr(const numel_cnt_t epochs, const vec_len_t maxRowsCnt, const vec_len_t maxColsCnt = 10) {
	const real_t beta1 = real_t(.9), beta2=real_t(.999), learningRate=real_t(.001), numStab=real_t(1e-8);
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 1; r < maxRowsCnt; ++r)	{
		for (vec_len_t c = 1; c < maxColsCnt; ++c) {
			MTXSIZE_SCOPED_TRACE(r, c, "test_Adam_corr");

			realmtx_t dW_ET(r, c), Mt_ET(r, c), Vt_ET(r, c);
			ASSERT_TRUE(!dW_ET.isAllocationFailed() && !Mt_ET.isAllocationFailed() && !Vt_ET.isAllocationFailed());
			realmtx_t dW_st(r, c), Mt_st(r, c), Vt_st(r, c);
			ASSERT_TRUE(!dW_st.isAllocationFailed() && !Mt_st.isAllocationFailed() && !Vt_st.isAllocationFailed());
			realmtx_t dW_mt(r, c), Mt_mt(r, c), Vt_mt(r, c);
			ASSERT_TRUE(!dW_mt.isAllocationFailed() && !Mt_mt.isAllocationFailed() && !Vt_mt.isAllocationFailed());
			realmtx_t dW_(r, c), Mt_(r, c), Vt_(r, c);
			ASSERT_TRUE(!dW_.isAllocationFailed() && !Mt_.isAllocationFailed() && !Vt_.isAllocationFailed());

			Mt_ET.zeros(); Mt_st.zeros(); Mt_mt.zeros(); Mt_.zeros();
			Vt_ET.zeros(); Vt_st.zeros(); Vt_mt.zeros(); Vt_.zeros();

			real_t beta1t_ET = real_t(1.), beta2t_ET = real_t(1.);
			real_t beta1t_st = real_t(1.), beta2t_st = real_t(1.);
			real_t beta1t_mt = real_t(1.), beta2t_mt = real_t(1.);
			real_t beta1t_ = real_t(1.), beta2t_ = real_t(1.);
			
			for (numel_cnt_t e = 0; e < epochs; ++e) {
				rg.gen_matrix(dW_ET, real_t(3.0));
				ASSERT_TRUE(dW_ET.clone_to(dW_st)); ASSERT_TRUE(dW_ET.clone_to(dW_mt)); ASSERT_TRUE(dW_ET.clone_to(dW_));
				
				Adam_ET(dW_ET, Mt_ET, Vt_ET, beta1t_ET, beta2t_ET, learningRate, beta1, beta2, numStab);

				iM.Adam_st(dW_st, Mt_st, Vt_st, beta1t_st, beta2t_st, learningRate, beta1, beta2, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_st, "dW @ _st");
				ASSERT_MTX_EQ(Mt_ET, Mt_st, "Mt @ _st");
				ASSERT_MTX_EQ(Vt_ET, Vt_st, "Vt @ _st");
				ASSERT_EQ(beta1t_ET, beta1t_st) << "_st";
				ASSERT_EQ(beta2t_ET, beta2t_st) << "_st";

				iM.Adam_mt(dW_mt, Mt_mt, Vt_mt, beta1t_mt, beta2t_mt, learningRate, beta1, beta2, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_mt, "dW @ _mt");
				ASSERT_MTX_EQ(Mt_ET, Mt_mt, "Mt @ _mt");
				ASSERT_MTX_EQ(Vt_ET, Vt_mt, "Vt @ _mt");
				ASSERT_EQ(beta1t_ET, beta1t_mt) << "_mt";
				ASSERT_EQ(beta2t_ET, beta2t_mt) << "_mt";

				iM.Adam(dW_, Mt_, Vt_, beta1t_, beta2t_, learningRate, beta1, beta2, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_, "dW @ _");
				ASSERT_MTX_EQ(Mt_ET, Mt_, "Mt @ _");
				ASSERT_MTX_EQ(Vt_ET, Vt_, "Vt @ _");
				ASSERT_EQ(beta1t_ET, beta1t_) << "_";
				ASSERT_EQ(beta2t_ET, beta2t_) << "_";
			}
		}
	}
}

TEST(TestMathN, Adam) {
	test_Adam_corr(10, g_MinDataSizeDelta, g_MinDataSizeDelta);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void test_AdaMax_corr(const numel_cnt_t epochs, const vec_len_t maxRowsCnt, const vec_len_t maxColsCnt = 10) {
	const real_t beta1 = real_t(.9), beta2 = real_t(.999), learningRate = real_t(.001), numStab = real_t(1e-8);
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 1; r < maxRowsCnt; ++r) {
		for (vec_len_t c = 1; c < maxColsCnt; ++c) {
			MTXSIZE_SCOPED_TRACE(r, c, "test_AdaMax_corr");

			realmtx_t dW_ET(r, c), Mt_ET(r, c), Vt_ET(r, c);
			ASSERT_TRUE(!dW_ET.isAllocationFailed() && !Mt_ET.isAllocationFailed() && !Vt_ET.isAllocationFailed());
			realmtx_t dW_st(r, c), Mt_st(r, c), Vt_st(r, c);
			ASSERT_TRUE(!dW_st.isAllocationFailed() && !Mt_st.isAllocationFailed() && !Vt_st.isAllocationFailed());
			realmtx_t dW_mt(r, c), Mt_mt(r, c), Vt_mt(r, c);
			ASSERT_TRUE(!dW_mt.isAllocationFailed() && !Mt_mt.isAllocationFailed() && !Vt_mt.isAllocationFailed());
			realmtx_t dW_(r, c), Mt_(r, c), Vt_(r, c);
			ASSERT_TRUE(!dW_.isAllocationFailed() && !Mt_.isAllocationFailed() && !Vt_.isAllocationFailed());

			Mt_ET.zeros(); Mt_st.zeros(); Mt_mt.zeros(); Mt_.zeros();
			Vt_ET.zeros(); Vt_st.zeros(); Vt_mt.zeros(); Vt_.zeros();

			real_t beta1t_ET = real_t(1.), beta1t_st = real_t(1.), beta1t_mt = real_t(1.), beta1t_ = real_t(1.);

			for (numel_cnt_t e = 0; e < epochs; ++e) {
				rg.gen_matrix(dW_ET, real_t(3.0));
				ASSERT_TRUE(dW_ET.clone_to(dW_st)); ASSERT_TRUE(dW_ET.clone_to(dW_mt)); ASSERT_TRUE(dW_ET.clone_to(dW_));

				AdaMax_ET(dW_ET, Mt_ET, Vt_ET, beta1t_ET, learningRate, beta1, beta2, numStab);

				iM.AdaMax_st(dW_st, Mt_st, Vt_st, beta1t_st, learningRate, beta1, beta2, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_st, "dW @ _st");
				ASSERT_MTX_EQ(Mt_ET, Mt_st, "Mt @ _st");
				ASSERT_MTX_EQ(Vt_ET, Vt_st, "Vt @ _st");
				ASSERT_EQ(beta1t_ET, beta1t_st) << "_st";

				iM.AdaMax_mt(dW_mt, Mt_mt, Vt_mt, beta1t_mt, learningRate, beta1, beta2, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_mt, "dW @ _mt");
				ASSERT_MTX_EQ(Mt_ET, Mt_mt, "Mt @ _mt");
				ASSERT_MTX_EQ(Vt_ET, Vt_mt, "Vt @ _mt");
				ASSERT_EQ(beta1t_ET, beta1t_mt) << "_mt";

				iM.AdaMax(dW_, Mt_, Vt_, beta1t_, learningRate, beta1, beta2, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_, "dW @ _");
				ASSERT_MTX_EQ(Mt_ET, Mt_, "Mt @ _");
				ASSERT_MTX_EQ(Vt_ET, Vt_, "Vt @ _");
				ASSERT_EQ(beta1t_ET, beta1t_) << "_";
			}
		}
	}
}

TEST(TestMathN, AdaMax) {
	test_AdaMax_corr(10, g_MinDataSizeDelta, g_MinDataSizeDelta);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_Nadam_corr(const numel_cnt_t epochs, const vec_len_t maxRowsCnt, const vec_len_t maxColsCnt = 10) {
	const real_t mu = real_t(.9), eta = real_t(.999), learningRate = real_t(.001), numStab = real_t(1e-8);
	const real_t _g = real_t(0);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 1; r < maxRowsCnt; ++r) {
		for (vec_len_t c = 1; c < maxColsCnt; ++c) {
			MTXSIZE_SCOPED_TRACE(r, c, "test_Nadam_corr");

			realmtx_t dW_ET(r, c), Mt_ET(r, c), Vt_ET(r, c);
			ASSERT_TRUE(!dW_ET.isAllocationFailed() && !Mt_ET.isAllocationFailed() && !Vt_ET.isAllocationFailed());
			realmtx_t dW_st(r, c), Mt_st(r, c), Vt_st(r, c);
			ASSERT_TRUE(!dW_st.isAllocationFailed() && !Mt_st.isAllocationFailed() && !Vt_st.isAllocationFailed());
			realmtx_t dW_mt(r, c), Mt_mt(r, c), Vt_mt(r, c);
			ASSERT_TRUE(!dW_mt.isAllocationFailed() && !Mt_mt.isAllocationFailed() && !Vt_mt.isAllocationFailed());
			realmtx_t dW_(r, c), Mt_(r, c), Vt_(r, c);
			ASSERT_TRUE(!dW_.isAllocationFailed() && !Mt_.isAllocationFailed() && !Vt_.isAllocationFailed());

			Mt_ET.zeros(); Mt_st.zeros(); Mt_mt.zeros(); Mt_.zeros();
			Vt_ET.zeros(); Vt_st.zeros(); Vt_mt.zeros(); Vt_.zeros();

			real_t mu_t_ET = real_t(1.), eta_t_ET = real_t(1.);
			real_t mu_t_st = real_t(1.), eta_t_st = real_t(1.);
			real_t mu_t_mt = real_t(1.), eta_t_mt = real_t(1.);
			real_t mu_t_ = real_t(1.), eta_t_ = real_t(1.);

			for (numel_cnt_t e = 0; e < epochs; ++e) {
				rg.gen_matrix(dW_ET, real_t(3.0));
				ASSERT_TRUE(dW_ET.clone_to(dW_st)); ASSERT_TRUE(dW_ET.clone_to(dW_mt)); ASSERT_TRUE(dW_ET.clone_to(dW_));

				Nadam_ET(dW_ET, Mt_ET, Vt_ET, mu_t_ET, eta_t_ET, learningRate, mu, eta, numStab);

				iM.RNadam_st(dW_st, Mt_st, Vt_st, mu_t_st, eta_t_st, learningRate, mu, eta, _g, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_st, "dW @ _st");
				ASSERT_MTX_EQ(Mt_ET, Mt_st, "Mt @ _st");
				ASSERT_MTX_EQ(Vt_ET, Vt_st, "Vt @ _st");
				ASSERT_EQ(mu_t_ET, mu_t_st) << "_st";
				ASSERT_EQ(eta_t_ET, eta_t_st) << "_st";

				iM.RNadam_mt(dW_mt, Mt_mt, Vt_mt, mu_t_mt, eta_t_mt, learningRate, mu, eta, _g, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_mt, "dW @ _mt");
				ASSERT_MTX_EQ(Mt_ET, Mt_mt, "Mt @ _mt");
				ASSERT_MTX_EQ(Vt_ET, Vt_mt, "Vt @ _mt");
				ASSERT_EQ(mu_t_ET, mu_t_mt) << "_mt";
				ASSERT_EQ(eta_t_ET, eta_t_mt) << "_mt";

				iM.RNadam(dW_, Mt_, Vt_, mu_t_, eta_t_, learningRate, mu, eta, _g, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_, "dW @ _");
				ASSERT_MTX_EQ(Mt_ET, Mt_, "Mt @ _");
				ASSERT_MTX_EQ(Vt_ET, Vt_, "Vt @ _");
				ASSERT_EQ(mu_t_ET, mu_t_) << "_";
				ASSERT_EQ(eta_t_ET, eta_t_) << "_";
			}
		}
	}
}

TEST(TestMathN, Nadam) {
	test_Nadam_corr(10, g_MinDataSizeDelta, g_MinDataSizeDelta);
}


void test_Radam_corr(const numel_cnt_t epochs, const vec_len_t maxRowsCnt, const vec_len_t maxColsCnt = 10) {
	const real_t mu = real_t(.9), eta = real_t(.999), learningRate = real_t(.001), numStab = real_t(1e-8);
	const real_t gamma = real_t(.1);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 1; r < maxRowsCnt; ++r) {
		for (vec_len_t c = 1; c < maxColsCnt; ++c) {
			MTXSIZE_SCOPED_TRACE(r, c, "test_Radam_corr");

			realmtx_t dW_ET(r, c), Mt_ET(r, c), Vt_ET(r, c);
			ASSERT_TRUE(!dW_ET.isAllocationFailed() && !Mt_ET.isAllocationFailed() && !Vt_ET.isAllocationFailed());
			realmtx_t dW_st(r, c), Mt_st(r, c), Vt_st(r, c);
			ASSERT_TRUE(!dW_st.isAllocationFailed() && !Mt_st.isAllocationFailed() && !Vt_st.isAllocationFailed());
			realmtx_t dW_mt(r, c), Mt_mt(r, c), Vt_mt(r, c);
			ASSERT_TRUE(!dW_mt.isAllocationFailed() && !Mt_mt.isAllocationFailed() && !Vt_mt.isAllocationFailed());
			realmtx_t dW_(r, c), Mt_(r, c), Vt_(r, c);
			ASSERT_TRUE(!dW_.isAllocationFailed() && !Mt_.isAllocationFailed() && !Vt_.isAllocationFailed());

			Mt_ET.zeros(); Mt_st.zeros(); Mt_mt.zeros(); Mt_.zeros();
			Vt_ET.zeros(); Vt_st.zeros(); Vt_mt.zeros(); Vt_.zeros();

			real_t mu_t_ET = real_t(1.), eta_t_ET = real_t(1.);
			real_t mu_t_st = real_t(1.), eta_t_st = real_t(1.);
			real_t mu_t_mt = real_t(1.), eta_t_mt = real_t(1.);
			real_t mu_t_ = real_t(1.), eta_t_ = real_t(1.);

			for (numel_cnt_t e = 0; e < epochs; ++e) {
				rg.gen_matrix(dW_ET, real_t(3.0));
				ASSERT_TRUE(dW_ET.clone_to(dW_st)); ASSERT_TRUE(dW_ET.clone_to(dW_mt)); ASSERT_TRUE(dW_ET.clone_to(dW_));

				Radam_ET(dW_ET, Mt_ET, Vt_ET, mu_t_ET, eta_t_ET, learningRate, mu, eta, gamma, numStab);

				iM.RNadam_st(dW_st, Mt_st, Vt_st, mu_t_st, eta_t_st, learningRate, mu, eta, gamma, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_st, "dW @ _st");
				ASSERT_MTX_EQ(Mt_ET, Mt_st, "Mt @ _st");
				ASSERT_MTX_EQ(Vt_ET, Vt_st, "Vt @ _st");
				ASSERT_EQ(mu_t_ET, mu_t_st) << "_st";
				ASSERT_EQ(eta_t_ET, eta_t_st) << "_st";

				iM.RNadam_mt(dW_mt, Mt_mt, Vt_mt, mu_t_mt, eta_t_mt, learningRate, mu, eta, gamma, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_mt, "dW @ _mt");
				ASSERT_MTX_EQ(Mt_ET, Mt_mt, "Mt @ _mt");
				ASSERT_MTX_EQ(Vt_ET, Vt_mt, "Vt @ _mt");
				ASSERT_EQ(mu_t_ET, mu_t_mt) << "_mt";
				ASSERT_EQ(eta_t_ET, eta_t_mt) << "_mt";

				iM.RNadam(dW_, Mt_, Vt_, mu_t_, eta_t_, learningRate, mu, eta, gamma, numStab);
				ASSERT_MTX_EQ(dW_ET, dW_, "dW @ _");
				ASSERT_MTX_EQ(Mt_ET, Mt_, "Mt @ _");
				ASSERT_MTX_EQ(Vt_ET, Vt_, "Vt @ _");
				ASSERT_EQ(mu_t_ET, mu_t_) << "_";
				ASSERT_EQ(eta_t_ET, eta_t_) << "_";
			}
		}
	}
}

TEST(TestMathN, Radam) {
	test_Radam_corr(10, g_MinDataSizeDelta, g_MinDataSizeDelta);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_make_dropout_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10, const real_t dpa = real_t(.5)) {
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t act(rowsCnt, colsCnt, true), dm(rowsCnt, colsCnt);
	ASSERT_TRUE(!act.isAllocationFailed() && !dm.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	realmtx_t act2(rowsCnt, colsCnt, true), dm2(rowsCnt, colsCnt), act3(rowsCnt, colsCnt, true), dm3(rowsCnt, colsCnt);
	ASSERT_TRUE(!act2.isAllocationFailed() && !dm2.isAllocationFailed() && !act3.isAllocationFailed() && !dm3.isAllocationFailed());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(act, 5);
		ASSERT_TRUE(act.test_biases_strict());
		act.clone_to(act2);
		act.clone_to(act3);
		rg.gen_matrix_norm(dm);
		dm.clone_to(dm2);
		dm.clone_to(dm3);

		make_dropout_ET(act2, dpa, dm2);
		ASSERT_TRUE(act2.test_biases_strict());

		iM.make_dropout_st(act, dpa, dm);
		ASSERT_MTX_EQ(act2, act, "make_dropout_st: wrong act");
		ASSERT_MTX_EQ(dm2, dm, "make_dropout_st: wrong dm");

		act3.clone_to(act);
		dm3.clone_to(dm);
		iM.make_dropout_mt(act, dpa, dm);
		ASSERT_MTX_EQ(act2, act, "make_dropout_mt: wrong act");
		ASSERT_MTX_EQ(dm2, dm, "make_dropout_mt: wrong dm");

		act3.clone_to(act);
		dm3.clone_to(dm);
		iM.make_dropout(act, dpa, dm);
		ASSERT_MTX_EQ(act2, act, "make_dropout: wrong act");
		ASSERT_MTX_EQ(dm2, dm, "make_dropout: wrong dm");
	}
}

TEST(TestMathN, make_dropout) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_make_dropout_corr(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_make_dropout_corr(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_apply_dropout_mask_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10, const real_t dpa = real_t(.5)) {
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t act(rowsCnt, colsCnt, true), dm(rowsCnt, colsCnt);
	ASSERT_TRUE(!act.isAllocationFailed() && !dm.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	realmtx_t actET(rowsCnt, colsCnt, true), act3(rowsCnt, colsCnt, true);
	ASSERT_TRUE(!actET.isAllocationFailed() && !act3.isAllocationFailed());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(act, 5);
		ASSERT_TRUE(act.test_biases_strict());
		act.clone_to(actET);
		act.clone_to(act3);
		rg.gen_matrix_norm(dm);

		apply_dropout_mask_ET(actET, dpa, dm);
		ASSERT_TRUE(actET.test_biases_strict());

		iM.apply_dropout_mask_st(act, dpa, dm);
		ASSERT_MTX_EQ(actET, act, "apply_dropout_mask_st: wrong act");

		act3.clone_to(act);
		iM.apply_dropout_mask_mt(act, dpa, dm);
		ASSERT_MTX_EQ(actET, act, "apply_dropout_mask_mt: wrong act");

		act3.clone_to(act);
		iM.apply_dropout_mask(act, dpa, dm);
		ASSERT_MTX_EQ(actET, act, "apply_dropout_mask: wrong act");
	}
}

TEST(TestMathN, apply_dropout_mask) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_apply_dropout_mask_corr(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_apply_dropout_mask_corr(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////

TEST(TestMathN, vCountSameNaive) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;

	constexpr vec_len_t dataCnt = 9;
	const ::std::array<vec_len_t, dataCnt> src1 = { 3,55,32, 35,63,5, 2,400,6 };
	const ::std::array<vec_len_t, dataCnt> src2 = { 3,55,33, 35,63,5, 4,400,6 };

	//iMB iM;
	ASSERT_EQ(iM.vCountSame_st_naive(src1, src2), dataCnt-2);
}

TEST(TestMathN, vCountSameMtCorrectness) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	typedef ::std::vector<vec_len_t> vec_t;

#ifdef NNTL_DEBUG
	constexpr vec_len_t rowsCnt = 100;
#else
	constexpr vec_len_t rowsCnt = 100000;
#endif

	vec_t v1(rowsCnt), v2(rowsCnt);

//	iMB iM;
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	rg.gen_vector_gtz(&v1[0], rowsCnt, (vec_t::value_type)5);
	rg.gen_vector_gtz(&v2[0], rowsCnt, (vec_t::value_type)5);

	ASSERT_EQ(iM.vCountSame_st_naive(v1, v2), iM.vCountSame_mt_naive(v1, v2));
}

template<typename iMath>
void test_vCountSame_perf(iMath& iM, vec_len_t rowsCnt) {
	typedef ::std::vector<vec_len_t> vec_t;
	STDCOUTL("******* testing vCountSame() over " << rowsCnt << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT;
	numel_cnt_t vv;

	vec_t v1(rowsCnt), v2(rowsCnt);
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	rg.gen_vector_gtz(&v1[0], rowsCnt, (vec_t::value_type)5);
	rg.gen_vector_gtz(&v2[0], rowsCnt, (vec_t::value_type)5);

	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	iM.vCountSame_st_naive(v1, v2);
	vv = 0;
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) {
		vv += iM.vCountSame_st_naive(v1, v2);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st_naive:\t" << utils::duration_readable(diff, maxReps, &tstNaive) << "\t\tvv=" << vv);

	iM.vCountSame_mt_naive(v1, v2);;
	vv = 0;
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) {
		vv += iM.vCountSame_mt_naive(v1, v2);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt_naive:\t" << utils::duration_readable(diff, maxReps, &tmtNaive) << "\t\tvv=" << vv);

	iM.vCountSame(v1, v2);
	vv = 0;
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) {
		vv += iM.vCountSame(v1, v2);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest) << "\t\tvv=" << vv);
}

TEST(TestMathN, vCountSamePerf) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST4(100000, 75, 25, 1) test_vCountSame_perf(iM, i);
}


//////////////////////////////////////////////////////////////////////////
template<typename iMath>
void test_evClamp_perf(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	typedef ::std::vector<vec_len_t> vec_t;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing evClamp() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	double tstNaive, tmtNaive, tBest;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT;

	real_t lo = -50, hi = 50;

	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	vec_t vec(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(m, 100);

	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	iM.evClamp_st(m, lo,hi);
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) {
		iM.evClamp_st(m, lo, hi);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t\t" << utils::duration_readable(diff, maxReps, &tstNaive));

	iM.evClamp_mt(m, lo, hi);
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) {
		iM.evClamp_mt(m, lo, hi);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t\t" << utils::duration_readable(diff, maxReps, &tmtNaive));

	iM.evClamp(m, lo, hi);
	bt = steady_clock::now();
	for (vec_len_t r = 0; r < maxReps; ++r) {
		iM.evClamp(m, lo, hi);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best:\t\t" << utils::duration_readable(diff, maxReps, &tBest));
}

TEST(TestMathN, evClampPerf) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::evClamp, 10) test_evClamp_perf(iM, i,10);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename IntfT>
void test_mExtractCols_corr(typename IntfT::iMath_t& iM, typename IntfT::iRng_t& iR
	, const bool bBiases, const bool bBatchsInRows, const vec_len_t totalBatches, const vec_len_t sampleSize, const vec_len_t batchSize)
{
	typedef typename IntfT::real_t real_t;
	typedef typename IntfT::iMath_t::realmtx_t realmtx_t;
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	
	STDCOUTL("bBiases="<< bBiases<< ", batchesInRows="<< bBatchsInRows
		<< ", totalBatches=" << totalBatches << ", sampleSize=" << sampleSize << ", batchSize=" << batchSize);

	realmtx_t src(sampleSize, totalBatches, bBiases, bBatchsInRows)
		, dest(sampleSize, batchSize, bBiases, bBatchsInRows)
		, destET(sampleSize, batchSize, bBiases, bBatchsInRows);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !destET.isAllocationFailed());

	::std::vector<vec_len_t> vec(totalBatches);
	::std::iota(vec.begin(), vec.end(), 0);

	const vec_len_t bcnt = totalBatches / batchSize;

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		iR.gen_matrixAny(src, real_t(2.));
		::std::random_shuffle(vec.begin(), vec.end(), iR);

		auto it = vec.begin();

		for (vec_len_t ib = 0; ib < bcnt; ++ib) {
			const auto& batchIt = it;
			mExtractCols_ET(src, batchIt, destET);

			dest.ones();
			iM.mExtractCols_st(src, batchIt, dest);
			ASSERT_EQ(dest, destET) << "mExtractCols_st failed";

			dest.ones();
			iM.mExtractCols_mt(src, batchIt, dest);
			ASSERT_EQ(dest, destET) << "mExtractCols_mt failed";

			it += batchSize;
		}
	}
}

TEST(TestMathN, mExtractColsCorr) {
	typedef dt_interfaces<real_t> myInterfaces_t;
	typename myInterfaces_t::iRng_t iR;
	iR.init_ithreads(iM.ithreads());

	const vec_len_t tbatch = 500, sc = 10, batchs = 50;

	ASSERT_NO_FATAL_FAILURE(test_mExtractCols_corr<myInterfaces_t>(iM, iR, false, false, tbatch, sc, batchs));
	ASSERT_NO_FATAL_FAILURE(test_mExtractCols_corr<myInterfaces_t>(iM, iR, false, true, tbatch, sc, batchs));
	ASSERT_NO_FATAL_FAILURE(test_mExtractCols_corr<myInterfaces_t>(iM, iR, true, false, tbatch, sc, batchs));
	ASSERT_NO_FATAL_FAILURE(test_mExtractCols_corr<myInterfaces_t>(iM, iR, true, true, tbatch, sc, batchs));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mExtractRows_corr(const bool bBiases, const vec_len_t rowsCnt, const vec_len_t colsCnt, const vec_len_t extrCnt) {
	realmtx_t src(rowsCnt, colsCnt, bBiases), destSt(extrCnt, colsCnt, bBiases), destMt(extrCnt, colsCnt, bBiases);
	ASSERT_TRUE(!src.isAllocationFailed() && !destSt.isAllocationFailed() && !destMt.isAllocationFailed());
	
	auto pSrc = src.data();
	for (numel_cnt_t i = 0, im = src.numel_no_bias(); i < im; ++i) pSrc[i] = static_cast<real_t>(i);
	ASSERT_TRUE(!src.emulatesBiases() || src.test_biases_strict());

	::std::vector<vec_len_t> vec(extrCnt);
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	::std::iota(vec.begin(), vec.end(), 0);
	::std::random_shuffle(vec.begin(), vec.end(), rg);

	iM.mExtractRows_seqWrite_st(src, vec.begin(), destSt);
	iM.mExtractRows_seqWrite_mt(src, vec.begin(), destMt);

	ASSERT_EQ(destSt, destMt);
	ASSERT_TRUE(!destSt.emulatesBiases() || destSt.test_biases_strict());

	for (vec_len_t r = 0; r < extrCnt; ++r) {
		for (vec_len_t c = 0, cm = src.cols(); c < cm; ++c) { //using src.cols() to take biases into account
			ASSERT_EQ(destSt.get(r, c), src.get(vec[r], c));//must be binary equal
		}
	}
}

TEST(TestMathN, mExtractRowsCorrectness) {
	ASSERT_NO_FATAL_FAILURE(test_mExtractRows_corr(false, 3000, 50, 1000));
	ASSERT_NO_FATAL_FAILURE(test_mExtractRows_corr(true, 3000, 50, 1000));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

TEST(TestMathN, mMulABt_Cnb) {
	using namespace nntl_supp;

	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	
	using ErrCode = jsonreader::ErrorCode;
	typedef math::smatrix<real_t> realmtx_t;
	using mtx_size_t = realmtx_t::mtx_size_t;

	realmtx_t A,B,C, etA, etB, etC;
	jsonreader reader;

	ErrCode ec = reader.read(NNTL_STRING("./test_data/mtx4-2.json"), etA);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etA.empty());
	etA.clone_to(A);

	ec = reader.read(NNTL_STRING("./test_data/mtx3-2.json"), etB);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etB.empty());
	etB.clone_to(B);

	ec = reader.read(NNTL_STRING("./test_data/mtx4-3.json"), etC);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etC.empty());

	C.resize(etC.size());
	C.zeros();

//	iMB iM;

	iM.mMulABt_Cnb(A, B, C);
	EXPECT_EQ(A, etA);
	EXPECT_EQ(B, etB);
	EXPECT_EQ(C, etC);
}

TEST(TestMathN, mMulABt_Cnb_biased) {
	using namespace nntl_supp;

	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;

	using ErrCode = jsonreader::ErrorCode;

	typedef math::smatrix<real_t> realmtx_t;
	using mtx_size_t = realmtx_t::mtx_size_t;

	realmtx_t A, B, C, etA, etB, etC;
	jsonreader reader;

	ErrCode ec = reader.read(NNTL_STRING("./test_data/mtx4-2.json"), etA);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etA.empty());
	etA.clone_to(A);

	ec = reader.read(NNTL_STRING("./test_data/mtx3-2.json"), etB);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etB.empty());
	etB.clone_to(B);

	ec = reader.read(NNTL_STRING("./test_data/mtx4-3.json"), etC);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();
	ASSERT_TRUE(!etC.empty());

	C.will_emulate_biases();
	C.resize(etC.size());
	C.zeros();

//	iMB iM;

	iM.mMulABt_Cnb(A, B, C);
	EXPECT_EQ(A, etA);
	EXPECT_EQ(B, etB);

	auto ptrC = C.data(), ptrEt = etC.data();
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
void test_evMul_ip(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t M(rowsCnt, colsCnt), etM(rowsCnt, colsCnt), testM(rowsCnt, colsCnt), etB(rowsCnt, colsCnt), B(rowsCnt, colsCnt);
	
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(M, real_t(5));
		rg.gen_matrix(B, real_t(5));
		ASSERT_TRUE(M.clone_to(etM));
		ASSERT_TRUE(B.clone_to(etB));

		evMul_ip_ET(etM, B);
		ASSERT_MTX_EQ(B, etB, "changed B!");

		ASSERT_TRUE(M.clone_to(testM));
		iM.evMul_ip_st(testM, B);
		ASSERT_MTX_EQ(B, etB, "_st changed B!");
		ASSERT_MTX_EQ(testM, etM, "_st wrong M!");

		ASSERT_TRUE(M.clone_to(testM));
		iM.evMul_ip_mt(testM, B);
		ASSERT_MTX_EQ(B, etB, "_mt changed B!");
		ASSERT_MTX_EQ(testM, etM, "_mt wrong M!");

		ASSERT_TRUE(M.clone_to(testM));
		iM.evMul_ip(testM, B);
		ASSERT_MTX_EQ(B, etB, "() changed B!");
		ASSERT_MTX_EQ(testM, etM, "() wrong M!");
	}
}

TEST(TestMathN, evMul_ip) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_evMul_ip(r, c));
		}
	}
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////// 
template<typename CommonDataT>
void test_evMulC_ip(const CommonDataT& cd, vec_len_t batchSiz, vec_len_t sampleSiz, const bool bBiases, const bool bBiR) {
	typedef typename CommonDataT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;

	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	const auto bIgnoreBias = bBiases;

	constexpr unsigned _scopeMsgLen = 256;
	char _scopeMsg[_scopeMsgLen];
	sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements), bBiases=%d, bBiR=%d", "mTranspose_corr", batchSiz, sampleSiz
		, ::nntl::math::smatrix_td::sNumel(batchSiz, sampleSiz), int(bBiases), int(bBiR));
	SCOPED_TRACE(_scopeMsg);

	realmtx_t srcET(bBiR, batchSiz, sampleSiz, bBiases), src_dest(bBiR, batchSiz, sampleSiz, bBiases)
		, destET(bBiR, batchSiz, sampleSiz, bBiases);
	ASSERT_TRUE(!srcET.isAllocationFailed() && !src_dest.isAllocationFailed() && !destET.isAllocationFailed());
	
#pragma warning(push)
#pragma warning(disable:4459)
	auto& iM = cd.get_iMath();
#pragma warning(pop)
	auto& iR = cd.get_iRng();

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		iR.gen_matrixAny(srcET, real_t(5));
		ASSERT_TRUE(srcET.if_biases_test_strict());
		ASSERT_TRUE(srcET.copy_to(destET));

		const real_t C = iR.gen_f(real_t(10)) - real_t(5);

		evMulC_ip_ET(destET, C, bIgnoreBias);
		ASSERT_TRUE(destET.if_biases_test_strict());

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evMulC_ip_st(src_dest, C, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evMulC_ip_st() failed");

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evMulC_ip_mt(src_dest, C, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evMulC_ip_mt() failed");

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evMulC_ip(src_dest, C, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evMulC_ip() failed");
	}
}

TEST(TestMathN, evMulC_ip) {
	typedef def_keeper_tpl<real_t> IKeeper_t;

	IKeeper_t keeper;
	auto& cd = keeper.get_const_common_data();

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			for (int f = 0; f < 4; ++f) {
				ASSERT_NO_FATAL_FAILURE(test_evMulC_ip(cd, r, c, !(f & 1), !(f & 2)));
			}
		}
	}

	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = 3 * g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			for (int f = 0; f < 4; ++f) {
				ASSERT_NO_FATAL_FAILURE(test_evMulC_ip(cd, r, c, !(f & 1), !(f & 2)));
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////
template<typename CommonDataT>
void test_evAddC_ip(const CommonDataT& cd, vec_len_t batchSiz, vec_len_t sampleSiz, const bool bBiases, const bool bBiR) {
	typedef typename CommonDataT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;

	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
		const auto bIgnoreBias = bBiases;

	constexpr unsigned _scopeMsgLen = 256;
	char _scopeMsg[_scopeMsgLen];
	sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements), bBiases=%d, bBiR=%d", "mTranspose_corr", batchSiz, sampleSiz
		, ::nntl::math::smatrix_td::sNumel(batchSiz, sampleSiz), int(bBiases), int(bBiR));
	SCOPED_TRACE(_scopeMsg);

	realmtx_t srcET(bBiR, batchSiz, sampleSiz, bBiases), src_dest(bBiR, batchSiz, sampleSiz, bBiases)
		, destET(bBiR, batchSiz, sampleSiz, bBiases);
	ASSERT_TRUE(!srcET.isAllocationFailed() && !src_dest.isAllocationFailed() && !destET.isAllocationFailed());

#pragma warning(push)
#pragma warning(disable:4459)
	auto& iM = cd.get_iMath();
#pragma warning(pop)
	auto& iR = cd.get_iRng();

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		iR.gen_matrixAny(srcET, real_t(5));
		ASSERT_TRUE(srcET.if_biases_test_strict());
		ASSERT_TRUE(srcET.copy_to(destET));

		const real_t C = iR.gen_f(real_t(10)) - real_t(5);

		evAddC_ip_ET(destET, C, bIgnoreBias);
		ASSERT_TRUE(destET.if_biases_test_strict());

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evAddC_ip_st(src_dest, C, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evAddC_ip_st() failed");

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evAddC_ip_mt(src_dest, C, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evAddC_ip_mt() failed");

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evAddC_ip(src_dest, C, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evAddC_ip() failed");
	}
}

TEST(TestMathN, evAddC_ip) {
	typedef def_keeper_tpl<real_t> IKeeper_t;

	IKeeper_t keeper;
	auto& cd = keeper.get_const_common_data();

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			for (int f = 0; f < 4; ++f) {
				ASSERT_NO_FATAL_FAILURE(test_evAddC_ip(cd, r, c, !(f & 1), !(f & 2)));
			}
		}
	}

	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = 3 * g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			for (int f = 0; f < 4; ++f) {
				ASSERT_NO_FATAL_FAILURE(test_evAddC_ip(cd, r, c, !(f & 1), !(f & 2)));
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////

template<typename CommonDataT>
void test_evMulCAddC_ip(const CommonDataT& cd, vec_len_t batchSiz, vec_len_t sampleSiz, const bool bBiases, const bool bBiR) {
	typedef typename CommonDataT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;

	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	const auto bIgnoreBias = bBiases;

	constexpr unsigned _scopeMsgLen = 256;
	char _scopeMsg[_scopeMsgLen];
	sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements), bBiases=%d, bBiR=%d", "mTranspose_corr", batchSiz, sampleSiz
		, ::nntl::math::smatrix_td::sNumel(batchSiz, sampleSiz), int(bBiases), int(bBiR));
	SCOPED_TRACE(_scopeMsg);

	realmtx_t srcET(bBiR, batchSiz, sampleSiz, bBiases), src_dest(bBiR, batchSiz, sampleSiz, bBiases)
		, destET(bBiR, batchSiz, sampleSiz, bBiases);
	ASSERT_TRUE(!srcET.isAllocationFailed() && !src_dest.isAllocationFailed() && !destET.isAllocationFailed());

#pragma warning(push)
#pragma warning(disable:4459)
	auto& iM = cd.get_iMath();
#pragma warning(pop)
	auto& iR = cd.get_iRng();

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		iR.gen_matrixAny(srcET, real_t(5));
		ASSERT_TRUE(srcET.if_biases_test_strict());
		ASSERT_TRUE(srcET.copy_to(destET));

		const real_t sc = iR.gen_f(real_t(6)) - real_t(6), ofs = iR.gen_f(real_t(6)) - real_t(6);;

		evMulCAddC_ip_ET(destET, sc, ofs, bIgnoreBias);
		ASSERT_TRUE(destET.if_biases_test_strict());

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evMulCAddC_ip_st(src_dest, sc, ofs, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evMulCAddC_ip_st() failed");

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evMulCAddC_ip_mt(src_dest, sc, ofs, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evMulCAddC_ip_mt() failed");

		ASSERT_TRUE(srcET.copy_to(src_dest));
		iM.evMulCAddC_ip(src_dest, sc, ofs, bIgnoreBias);
		ASSERT_MTX_EQ(destET, src_dest, "evMulCAddC_ip() failed");
	}
}

TEST(TestMathN, evMulCAddC_ip) {
	typedef def_keeper_tpl<real_t> IKeeper_t;

	IKeeper_t keeper;
	auto& cd = keeper.get_const_common_data();

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			for (int f = 0; f < 4; ++f) {
				ASSERT_NO_FATAL_FAILURE(test_evMulCAddC_ip(cd, r, c, !(f & 1), !(f & 2)));
			}
		}
	}

	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = 3 * g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			for (int f = 0; f < 4; ++f) {
				ASSERT_NO_FATAL_FAILURE(test_evMulCAddC_ip(cd, r, c, !(f & 1), !(f & 2)));
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/*
template<typename base_t> struct sigm_EPS {};
template<> struct sigm_EPS<double> { static constexpr double eps = 1e-12; };
template<> struct sigm_EPS<float> { static constexpr float eps = 1e-6f; };*/


TEST(TestMathN, Sigm) {
	const auto fst = [](realmtx_t& X) { iM.sigm_st(X); };
	const auto fmt = [](realmtx_t& X) { iM.sigm_mt(X); };
	const auto fb = [](realmtx_t& X) { iM.sigm(X); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_f_x_corr<true>(sigm_ET<real_t>, fst, fmt, fb, "sigm", r, c));
		}
	}
}

TEST(TestMathN, DSigm) {
	const auto fst = [](realmtx_t& X) { iM.dsigm_st(X); };
	const auto fmt = [](realmtx_t& X) { iM.dsigm_mt(X); };
	const auto fb = [](realmtx_t& X) { iM.dsigm(X); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE( (test_f_x_corr<false, true>(dsigm_ET<real_t>, fst, fmt, fb, "dsigm", r, c)) );
		}
	}
}

TEST(TestMathN, Relu) {
	const auto fst = [](realmtx_t& X) { iM.relu_st(X); };
	const auto fmt = [](realmtx_t& X) { iM.relu_mt(X); };
	const auto fb = [](realmtx_t& X) { iM.relu(X); };
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_f_x_corr<true>(relu_ET<real_t>, fst, fmt, fb, "relu", r, c));
		}
	}
}

TEST(TestMathN, DRelu) {
	const auto fst = [](realmtx_t& X) { iM.drelu_st(X); };
	const auto fmt = [](realmtx_t& X) { iM.drelu_mt(X); };
	const auto fb = [](realmtx_t& X) { iM.drelu(X); };
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_f_x_corr<false>(drelu_ET<real_t>, fst, fmt, fb, "drelu", r, c)));
		}
	}
}

TEST(TestMathN, LeakyRelu) {
	const real_t leak = real_t(.001);
	const auto fst = [leak](realmtx_t& X) { iM.leakyrelu_st(X, leak); };
	const auto fmt = [leak](realmtx_t& X) { iM.leakyrelu_mt(X, leak); };
	const auto fb = [leak](realmtx_t& X) { iM.leakyrelu(X, leak); };
	const auto fet = [leak](realmtx_t& X) { leakyrelu_ET(X, leak); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_f_x_corr<true>(fet, fst, fmt, fb, "leakyrelu", r, c));
		}
	}
}

TEST(TestMathN, DLeakyRelu) {
	const real_t leak = real_t(.001);
	const auto fst = [leak](realmtx_t& X) { iM.dleakyrelu_st(X, leak); };
	const auto fmt = [leak](realmtx_t& X) { iM.dleakyrelu_mt(X, leak); };
	const auto fb = [leak](realmtx_t& X) { iM.dleakyrelu(X, leak); };
	const auto fet = [leak](realmtx_t& X) { dleakyrelu_ET(X, leak); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_f_x_corr<false>(fet, fst, fmt, fb, "dleakyrelu", r, c));
		}
	}
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct selu_EPS {};
template<> struct selu_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct selu_EPS <float> { static constexpr float eps = 1e-6f; };
TEST(TestMathN, SELU) {
	const real_t lambda = real_t(1.050700), alpha = real_t(1.6732632), a_t_l = alpha*lambda;
	const auto fst = [&a_t_l, &lambda](realmtx_t& X) { iM.selu_st(X, a_t_l, lambda); };
	const auto fmt = [&a_t_l, &lambda](realmtx_t& X) { iM.selu_mt(X, a_t_l, lambda); };
	const auto fb = [&a_t_l, &lambda](realmtx_t& X) { iM.selu(X, a_t_l, lambda); };
	const auto fet = [&alpha, &lambda](realmtx_t& X) { selu_ET(X, alpha, lambda); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_f_x_corr_eps<selu_EPS<real_t>,true>(fet, fst, fmt, fb, "selu", r, c)));
		}
	}
}

TEST(TestMathN, DSELU) {
	const real_t lambda = real_t(1.050700), alpha = real_t(1.6732632), a_t_l = alpha*lambda;
	const auto fst = [&a_t_l, &lambda](realmtx_t& X) { iM.dselu_st(X, a_t_l, lambda); };
	const auto fmt = [&a_t_l, &lambda](realmtx_t& X) { iM.dselu_mt(X, a_t_l, lambda); };
	const auto fb = [&a_t_l, &lambda](realmtx_t& X) { iM.dselu(X, a_t_l, lambda); };
	const auto fet = [&alpha, &lambda](realmtx_t& X) { dselu_ET(X, alpha, lambda); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_f_x_corr_eps<selu_EPS<real_t>,false>(fet, fst, fmt, fb, "dselu", r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct elu_EPS {};
template<> struct elu_EPS   <double> { static constexpr double eps = 1e-12; };
template<> struct elu_EPS  <float> { static constexpr float eps = 1e-6f; };
void test_elu_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_elu_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t src(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true), F_ET(rowsCnt, colsCnt, true), FU_ET(rowsCnt, colsCnt, true);
	ASSERT_TRUE(!src.isAllocationFailed() && !F.isAllocationFailed() && !F_ET.isAllocationFailed() && !FU_ET.isAllocationFailed());

	const real_t alpha = real_t(2.5);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(src, 2);
		ASSERT_TRUE(src.test_biases_strict());

		src.clone_to(F_ET);
		elu_ET(F_ET, alpha);
		ASSERT_TRUE(F_ET.test_biases_strict());
		src.clone_to(FU_ET);
		elu_unitalpha_ET(FU_ET);
		ASSERT_TRUE(FU_ET.test_biases_strict());
		//ASSERT_TRUE(FU_ET != F_ET);

		src.clone_to(F);
		iM.elu_st(F, alpha);
		ASSERT_REALMTX_NEAR(F, F_ET, "elu_st() failed", elu_EPS<real_t>::eps);
		src.clone_to(F);
		iM.elu_unitalpha_st(F);
		ASSERT_REALMTX_NEAR(F, FU_ET, "elu_unitalpha_st() failed", elu_EPS<real_t>::eps);

		src.clone_to(F);
		iM.elu_mt(F, alpha);
		ASSERT_REALMTX_NEAR(F, F_ET, "elu_mt() failed", elu_EPS<real_t>::eps);
		src.clone_to(F);
		iM.elu_unitalpha_mt(F);
		ASSERT_REALMTX_NEAR(F, FU_ET, "elu_unitalpha_mt() failed", elu_EPS<real_t>::eps);

		src.clone_to(F);
		iM.elu(F, alpha);
		ASSERT_REALMTX_NEAR(F, F_ET, "elu() failed", elu_EPS<real_t>::eps);
		src.clone_to(F);
		iM.elu_unitalpha(F);
		ASSERT_REALMTX_NEAR(F, FU_ET, "elu_unitalpha() failed", elu_EPS<real_t>::eps);
	}
}
TEST(TestMathN, ELU) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_elu_corr(r, c);
		}
	}
}
void test_delu_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_delu_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t df_ET(rowsCnt, colsCnt), F(rowsCnt, colsCnt), f_df(rowsCnt, colsCnt), dfU_ET(rowsCnt, colsCnt);
	ASSERT_TRUE(!df_ET.isAllocationFailed() && !F.isAllocationFailed() && !f_df.isAllocationFailed() && !dfU_ET.isAllocationFailed());
	
	const real_t alpha = real_t(2.5);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix(F, 2);
		
		F.clone_to(df_ET);
		delu_ET(df_ET, alpha);

		F.clone_to(dfU_ET);
		delu_unitalpha_ET(dfU_ET);

		F.clone_to(f_df);
		iM.delu_st(f_df, alpha);
		ASSERT_MTX_EQ(df_ET, f_df, "delu_st() failed");
		F.clone_to(f_df);
		iM.delu_unitalpha_st(f_df);
		ASSERT_MTX_EQ(dfU_ET, f_df, "delu_unitalpha_st() failed");

		F.clone_to(f_df);
		iM.delu_mt(f_df, alpha);
		ASSERT_MTX_EQ(df_ET, f_df, "delu_mt() failed");
		F.clone_to(f_df);
		iM.delu_unitalpha_mt(f_df);
		ASSERT_MTX_EQ(dfU_ET, f_df, "delu_unitalpha_mt() failed");

		F.clone_to(f_df);
		iM.delu(f_df, alpha);
		ASSERT_MTX_EQ(df_ET, f_df, "delu() failed");
		F.clone_to(f_df);
		iM.delu_unitalpha(f_df);
		ASSERT_MTX_EQ(dfU_ET, f_df, "delu_unitalpha() failed");
	}
}
TEST(TestMathN, DELU) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_delu_corr(r, c);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct elogu_EPS {};
template<> struct elogu_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct elogu_EPS <float> { static constexpr float eps = 1e-6f; };
void test_elogu_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_elogu_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true)
		, F_ET(rowsCnt, colsCnt, true)
		, FUA_ET(rowsCnt, colsCnt, true)
		, FNB_ET(rowsCnt, colsCnt, true)
		, FUANB_ET(rowsCnt, colsCnt, true);

	ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !F_ET.isAllocationFailed() 
		&& !FUA_ET.isAllocationFailed() && !FNB_ET.isAllocationFailed() && !FUANB_ET.isAllocationFailed());

	constexpr real_t alpha = real_t(2.5), b=real_t(2.);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(X, 5);
		ASSERT_TRUE(X.test_biases_strict());

		elogu_ET(X, F_ET, alpha, b);
		ASSERT_TRUE(F_ET.test_biases_strict());
		elogu_ua_ET(X, FUA_ET, b);
		ASSERT_TRUE(FUA_ET.test_biases_strict());
		elogu_nb_ET(X, FNB_ET, alpha);
		ASSERT_TRUE(FNB_ET.test_biases_strict());
		elogu_ua_nb_ET(X, FUANB_ET);
		ASSERT_TRUE(FUANB_ET.test_biases_strict());


		X.clone_to(F);
		iM.elogu_st(F, alpha, b);
		ASSERT_REALMTX_NEAR(F, F_ET, "elogu_st() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_ua_st(F, b);
		ASSERT_REALMTX_NEAR(F, FUA_ET, "elogu_ua_st() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_nb_st(F, alpha);
		ASSERT_REALMTX_NEAR(F, FNB_ET, "elogu_nb_st() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_ua_nb_st(F);
		ASSERT_REALMTX_NEAR(F, FUANB_ET, "elogu_ua_nb_st() failed", elogu_EPS<real_t>::eps);


		X.clone_to(F);
		iM.elogu_mt(F, alpha, b);
		ASSERT_REALMTX_NEAR(F, F_ET, "elogu_mt() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_ua_mt(F, b);
		ASSERT_REALMTX_NEAR(F, FUA_ET, "elogu_ua_mt() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_nb_mt(F, alpha);
		ASSERT_REALMTX_NEAR(F, FNB_ET, "elogu_nb_mt() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_ua_nb_mt(F);
		ASSERT_REALMTX_NEAR(F, FUANB_ET, "elogu_ua_nb_mt() failed", elogu_EPS<real_t>::eps);


		X.clone_to(F);
		iM.elogu(F, alpha, b);
		ASSERT_REALMTX_NEAR(F, F_ET, "elogu() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_ua(F, b);
		ASSERT_REALMTX_NEAR(F, FUA_ET, "elogu_ua() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_nb(F, alpha);
		ASSERT_REALMTX_NEAR(F, FNB_ET, "elogu_nb() failed", elogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.elogu_ua_nb(F);
		ASSERT_REALMTX_NEAR(F, FUANB_ET, "elogu_ua_nb() failed", elogu_EPS<real_t>::eps);
	}
}
TEST(TestMathN, ELogU_family) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_elogu_corr(r, c);
		}
	}
}
template<typename base_t> struct delogu_EPS {};
template<> struct delogu_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct delogu_EPS <float> { static constexpr float eps = 1e-6f; };
void test_delogu_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_delogu_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true), DF(rowsCnt, colsCnt, false)
		, df_ET(rowsCnt, colsCnt, false), dfUA_ET(rowsCnt, colsCnt, false)
		, dfNB_ET(rowsCnt, colsCnt, false), dfUANB_ET(rowsCnt, colsCnt, false);
	ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !DF.isAllocationFailed() 
		&& !df_ET.isAllocationFailed() && !dfUA_ET.isAllocationFailed()
		&& !dfNB_ET.isAllocationFailed() && !dfUANB_ET.isAllocationFailed());

	constexpr real_t alpha = real_t(2.5), b = real_t(2.);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(X, 5);
		ASSERT_TRUE(X.test_biases_strict());

		delogu_ET(X, df_ET, alpha, b);
		delogu_ua_ET(X, dfUA_ET, b);
		delogu_nb_ET(X, dfNB_ET, alpha);
		delogu_ua_nb_ET(X, dfUANB_ET);

		elogu_ET(X, F, alpha, b);
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_st(DF, alpha, b);
		ASSERT_REALMTX_NEAR(df_ET, DF, "delogu_st() failed", delogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_mt(DF, alpha, b);
		ASSERT_REALMTX_NEAR(df_ET, DF, "delogu_mt() failed", delogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu(DF, alpha, b);
		ASSERT_REALMTX_NEAR(df_ET, DF, "delogu() failed", delogu_EPS<real_t>::eps);

		elogu_ua_ET(X, F, b);
		ASSERT_TRUE(F.test_biases_strict());
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_ua_st(DF, b);
		ASSERT_REALMTX_NEAR(dfUA_ET, DF, "delogu_ua_st() failed", delogu_EPS<real_t>::eps);
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_ua_mt(DF, b);
		ASSERT_REALMTX_NEAR(dfUA_ET, DF, "delogu_ua_mt() failed", delogu_EPS<real_t>::eps);
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_ua(DF, b);
		ASSERT_REALMTX_NEAR(dfUA_ET, DF, "delogu_ua() failed", delogu_EPS<real_t>::eps);

		elogu_nb_ET(X, F, alpha);
		ASSERT_TRUE(F.test_biases_strict());
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_nb_st(DF, alpha);
		ASSERT_REALMTX_NEAR(dfNB_ET, DF, "delogu_nb_st() failed", delogu_EPS<real_t>::eps);
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_nb_mt(DF, alpha);
		ASSERT_REALMTX_NEAR(dfNB_ET, DF, "delogu_nb_mt() failed", delogu_EPS<real_t>::eps);
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_nb(DF, alpha);
		ASSERT_REALMTX_NEAR(dfNB_ET, DF, "delogu_nb() failed", delogu_EPS<real_t>::eps);

		elogu_ua_nb_ET(X, F);
		ASSERT_TRUE(F.test_biases_strict());
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_ua_nb_st(DF);
		ASSERT_REALMTX_NEAR(dfUANB_ET, DF, "delogu_ua_nb_st() failed", delogu_EPS<real_t>::eps);
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_ua_nb_mt(DF);
		ASSERT_REALMTX_NEAR(dfUANB_ET, DF, "delogu_ua_nb_mt() failed", delogu_EPS<real_t>::eps);
		
		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.delogu_ua_nb(DF);
		ASSERT_REALMTX_NEAR(dfUANB_ET, DF, "delogu_ua_nb() failed", delogu_EPS<real_t>::eps);
	}
}
TEST(TestMathN, DELogU_family) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_delogu_corr(r, c);
		}
	}
}
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct loglogu_EPS {};
template<> struct loglogu_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct loglogu_EPS <float> { static constexpr float eps = 1e-6f; };
void test_loglogu_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_loglogu_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true)
		, F_ET(rowsCnt, colsCnt, true)
		, FUA_ET(rowsCnt, colsCnt, true)
		, FNB_ET(rowsCnt, colsCnt, true)
		, FUANB_ET(rowsCnt, colsCnt, true);

	ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !F_ET.isAllocationFailed()
		&& !FUA_ET.isAllocationFailed() && !FNB_ET.isAllocationFailed() && !FUANB_ET.isAllocationFailed());

	constexpr real_t b_neg = real_t(3), b_pos = real_t(2.);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(X, 5);
		ASSERT_TRUE(X.test_biases_strict());

		loglogu_ET(X, F_ET, b_neg, b_pos);
		ASSERT_TRUE(F_ET.test_biases_strict());
		loglogu_nbn_ET(X, FUA_ET, b_pos);
		ASSERT_TRUE(FUA_ET.test_biases_strict());
		loglogu_nbp_ET(X, FNB_ET, b_neg);
		ASSERT_TRUE(FNB_ET.test_biases_strict());
		loglogu_nbn_nbp_ET(X, FUANB_ET);
		ASSERT_TRUE(FUANB_ET.test_biases_strict());


		X.clone_to(F);
		iM.loglogu_st(F, b_neg, b_pos);
		ASSERT_REALMTX_NEAR(F, F_ET, "loglogu_st() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbn_st(F, b_pos);
		ASSERT_REALMTX_NEAR(F, FUA_ET, "loglogu_nbn_st() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbp_st(F, b_neg);
		ASSERT_REALMTX_NEAR(F, FNB_ET, "loglogu_nbp_st() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbn_nbp_st(F);
		ASSERT_REALMTX_NEAR(F, FUANB_ET, "loglogu_nbn_nbp_st() failed", loglogu_EPS<real_t>::eps);


		X.clone_to(F);
		iM.loglogu_mt(F, b_neg, b_pos);
		ASSERT_REALMTX_NEAR(F, F_ET, "loglogu_mt() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbn_mt(F, b_pos);
		ASSERT_REALMTX_NEAR(F, FUA_ET, "loglogu_nbn_mt() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbp_mt(F, b_neg);
		ASSERT_REALMTX_NEAR(F, FNB_ET, "loglogu_nbp_mt() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbn_nbp_mt(F);
		ASSERT_REALMTX_NEAR(F, FUANB_ET, "loglogu_nbn_nbp_mt() failed", loglogu_EPS<real_t>::eps);


		X.clone_to(F);
		iM.loglogu(F, b_neg, b_pos);
		ASSERT_REALMTX_NEAR(F, F_ET, "loglogu() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbn(F, b_pos);
		ASSERT_REALMTX_NEAR(F, FUA_ET, "loglogu_nbn() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbp(F, b_neg);
		ASSERT_REALMTX_NEAR(F, FNB_ET, "loglogu_nbp() failed", loglogu_EPS<real_t>::eps);
		X.clone_to(F);
		iM.loglogu_nbn_nbp(F);
		ASSERT_REALMTX_NEAR(F, FUANB_ET, "loglogu_nbn_nbp() failed", loglogu_EPS<real_t>::eps);
	}
}
TEST(TestMathN, LogLogU_family) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_loglogu_corr(r, c);
		}
	}
}
template<typename base_t> struct dloglogu_EPS {};
template<> struct dloglogu_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct dloglogu_EPS <float> { static constexpr float eps = 1e-6f; };
void test_dloglogu_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_dloglogu_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true), DF(rowsCnt, colsCnt, false)
		, df_ET(rowsCnt, colsCnt, false), dfUA_ET(rowsCnt, colsCnt, false)
		, dfNB_ET(rowsCnt, colsCnt, false), dfUANB_ET(rowsCnt, colsCnt, false);
	ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !DF.isAllocationFailed()
		&& !df_ET.isAllocationFailed() && !dfUA_ET.isAllocationFailed()
		&& !dfNB_ET.isAllocationFailed() && !dfUANB_ET.isAllocationFailed());

	constexpr real_t b_neg = real_t(3), b_pos = real_t(2.);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(X, 5);
		ASSERT_TRUE(X.test_biases_strict());

		dloglogu_ET(X, df_ET, b_neg, b_pos);
		dloglogu_nbn_ET(X, dfUA_ET, b_pos);
		dloglogu_nbp_ET(X, dfNB_ET, b_neg);
		dloglogu_nbn_nbp_ET(X, dfUANB_ET);

		loglogu_ET(X, F, b_neg, b_pos);
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_st(DF, b_neg, b_pos);
		ASSERT_REALMTX_NEAR(df_ET, DF, "dloglogu_st() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_mt(DF, b_neg, b_pos);
		ASSERT_REALMTX_NEAR(df_ET, DF, "dloglogu_mt() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu(DF, b_neg, b_pos);
		ASSERT_REALMTX_NEAR(df_ET, DF, "dloglogu() failed", dloglogu_EPS<real_t>::eps);

		loglogu_nbn_ET(X, F, b_pos);
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbn_st(DF, b_pos);
		ASSERT_REALMTX_NEAR(dfUA_ET, DF, "dloglogu_nbn_st() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbn_mt(DF, b_pos);
		ASSERT_REALMTX_NEAR(dfUA_ET, DF, "dloglogu_nbn_mt() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbn(DF, b_pos);
		ASSERT_REALMTX_NEAR(dfUA_ET, DF, "dloglogu_nbn() failed", dloglogu_EPS<real_t>::eps);

		loglogu_nbp_ET(X, F, b_neg);
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbp_st(DF, b_neg);
		ASSERT_REALMTX_NEAR(dfNB_ET, DF, "dloglogu_nbp_st() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbp_mt(DF, b_neg);
		ASSERT_REALMTX_NEAR(dfNB_ET, DF, "dloglogu_nbp_mt() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbp(DF, b_neg);
		ASSERT_REALMTX_NEAR(dfNB_ET, DF, "dloglogu_nbp() failed", dloglogu_EPS<real_t>::eps);

		loglogu_nbn_nbp_ET(X, F);
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbn_nbp_st(DF);
		ASSERT_REALMTX_NEAR(dfUANB_ET, DF, "dloglogu_nbn_nbp_st() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbn_nbp_mt(DF);
		ASSERT_REALMTX_NEAR(dfUANB_ET, DF, "dloglogu_nbn_nbp_mt() failed", dloglogu_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dloglogu_nbn_nbp(DF);
		ASSERT_REALMTX_NEAR(dfUANB_ET, DF, "dloglogu_nbn_nbp() failed", dloglogu_EPS<real_t>::eps);
	}
}
TEST(TestMathN, DLogLogU_family) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_dloglogu_corr(r, c);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct softsign_EPS {};
template<> struct softsign_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct softsign_EPS <float> { static constexpr float eps = 1e-6f; };
void test_softsign_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_softsign_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true)
		, F_ET(rowsCnt, colsCnt, true)
		, FUC_ET(rowsCnt, colsCnt, true);

	ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !F_ET.isAllocationFailed()
		&& !FUC_ET.isAllocationFailed());

	constexpr real_t c = real_t(1.7), a = real_t(2.3);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(X, 5);
		ASSERT_TRUE(X.test_biases_strict());

		softsign_ET(X, F_ET, a, c);
		ASSERT_TRUE(F_ET.test_biases_strict());
		softsign_ET(X, FUC_ET, a, real_t(1));
		ASSERT_TRUE(FUC_ET.test_biases_strict());
		
		X.clone_to(F);
		iM.softsign_st(F, a, c);
		ASSERT_REALMTX_NEAR(F, F_ET, "softsign_st(ua) failed", softsign_EPS<real_t>::eps);

		X.clone_to(F);
		iM.softsign_uc_st(F, a);
		ASSERT_REALMTX_NEAR(F, FUC_ET, "softsign_uc_st() failed", softsign_EPS<real_t>::eps);

		X.clone_to(F);
		iM.softsign_mt(F, a, c);
		ASSERT_REALMTX_NEAR(F, F_ET, "softsign_mt() failed", softsign_EPS<real_t>::eps);

		X.clone_to(F);
		iM.softsign_uc_mt(F, a);
		ASSERT_REALMTX_NEAR(F, FUC_ET, "softsign_uc_mt() failed", softsign_EPS<real_t>::eps);

		X.clone_to(F);
		iM.softsign(F, a, c);
		ASSERT_REALMTX_NEAR(F, F_ET, "softsign() failed", softsign_EPS<real_t>::eps);

		X.clone_to(F);
		iM.softsign_uc(F, a);
		ASSERT_REALMTX_NEAR(F, FUC_ET, "softsign_uc() failed", softsign_EPS<real_t>::eps);
	}
}
TEST(TestMathN, SoftSign_family) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_softsign_corr(r, c);
		}
	}
}
template<typename base_t> struct dsoftsign_EPS {};
template<> struct dsoftsign_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct dsoftsign_EPS <float> { static constexpr float eps = 1e-6f; };
void test_dsoftsign_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_dsoftsign_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true), DF(rowsCnt, colsCnt, false)
		, df_ET(rowsCnt, colsCnt, false), dfUAUC_ET(rowsCnt, colsCnt, false);
	ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !DF.isAllocationFailed()
		&& !df_ET.isAllocationFailed() && !dfUAUC_ET.isAllocationFailed());

	constexpr real_t a = real_t(2.3), c = real_t(1.7);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(X, 5);
		ASSERT_TRUE(X.test_biases_strict());

		dsoftsign_ET(X, df_ET, a, c);
		softsign_ET(X, F, a, c);
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dsoftsign_st(DF, a, c);
		ASSERT_REALMTX_NEAR(df_ET, DF, "dsoftsign_st() failed", dsoftsign_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dsoftsign_mt(DF, a, c);
		ASSERT_REALMTX_NEAR(df_ET, DF, "dsoftsign_mt() failed", dsoftsign_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dsoftsign(DF, a, c);
		ASSERT_REALMTX_NEAR(df_ET, DF, "dsoftsign() failed", dsoftsign_EPS<real_t>::eps);

		dsoftsign_ET(X, dfUAUC_ET, real_t(1), real_t(1));
		softsign_ET(X, F, real_t(1), real_t(1));
		ASSERT_TRUE(F.test_biases_strict());

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dsoftsign_ua_uc_st(DF);
		ASSERT_REALMTX_NEAR(dfUAUC_ET, DF, "dsoftsign_ua_uc_st() failed", dsoftsign_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dsoftsign_ua_uc_mt(DF);
		ASSERT_REALMTX_NEAR(dfUAUC_ET, DF, "dsoftsign_ua_uc_mt() failed", dsoftsign_EPS<real_t>::eps);

		ASSERT_TRUE(F.clone_to_no_bias(DF));
		iM.dsoftsign_ua_uc(DF);
		ASSERT_REALMTX_NEAR(dfUAUC_ET, DF, "dsoftsign_ua_uc() failed", dsoftsign_EPS<real_t>::eps);
	}
}
TEST(TestMathN, DSoftSign_family) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			test_dsoftsign_corr(r, c);
		}
	}
}

template<typename base_t> struct softsigm_EPS {};
template<> struct softsigm_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct softsigm_EPS <float> { static constexpr float eps = 1e-6f; };
TEST(TestMathN, SoftSigm) {
	constexpr real_t a = real_t(2.);
	const auto fst = [a](realmtx_t& X) { iM.softsigm_st(X,a); };
	const auto fmt = [a](realmtx_t& X) { iM.softsigm_mt(X,a); };
	const auto fb = [a](realmtx_t& X) { iM.softsigm(X,a); };
	const auto fet = [a](const realmtx_t& X, realmtx_t& F) { softsigm_ET(X, F, a); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_f_x_xbasedET_corr<true, false, softsigm_EPS<real_t>>(fet, fst, fmt, fb, "softsigm", r, c)));
		}
	}
}

template<typename base_t> struct dsoftsigm_EPS {};
template<> struct dsoftsigm_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct dsoftsigm_EPS <float> { static constexpr float eps = 1e-6f; };
TEST(TestMathN, DSoftSigm) {
	constexpr real_t a = real_t(2.);
	const auto dfst = [a](realmtx_t& X) { iM.dsoftsigm_st(X, a); };
	const auto dfmt = [a](realmtx_t& X) { iM.dsoftsigm_mt(X, a); };
	const auto dfb = [a](realmtx_t& X) { iM.dsoftsigm(X, a); };
	const auto fet = [a](const realmtx_t& X, realmtx_t& F) { softsigm_ET(X, F, a); };
	const auto dfet = [a](const realmtx_t& X, realmtx_t& DF) { dsoftsigm_ET(X, DF, a); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_df_x_xbasedET_corr<dsoftsigm_EPS<real_t>>(fet, dfet, dfst, dfmt, dfb, "dsoftsigm", r, c)));
		}
	}
}



////////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct loss_quadratic_EPS {};
template<> struct loss_quadratic_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct loss_quadratic_EPS<float> { static constexpr float eps = 2e-2f; };
template<typename iMath>
void test_loss_quadratic(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
#pragma warning(disable:4459)
	typedef typename iMath::real_t real_t;
	typedef typename iMath::realmtx_t realmtx_t;
#pragma warning(default:4459)

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("********* testing loss_quadratic() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());

	constexpr vec_len_t maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t A, etA(rowsCnt, colsCnt), etY(rowsCnt, colsCnt), Y;
	real_t etQuadLoss = 0, quadLoss = 0;
	ASSERT_EQ(dataSize, etA.numel());

	rng::AFRand_mt<real_t, AFog::CRandomSFMT0, typename iMath::iThreads_t> rg;
	rg.init_ithreads(iM.ithreads());

	typedef activation::Loss_quadratic<real_t> Loss_t;

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix(etA, 5);
		rg.gen_matrix(etY, 5);
		ASSERT_TRUE(etA.clone_to(A));
		ASSERT_TRUE(etY.clone_to(Y));
		ASSERT_TRUE(etA == A && etY == Y);

		etQuadLoss = loss_quadratic_ET(etA, etY);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);

		quadLoss = iM.loss_quadratic_st_naive(A, Y);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);
		ASSERT_NEAR(etQuadLoss / rowsCnt, quadLoss / rowsCnt, loss_quadratic_EPS<real_t>::eps) << "_st";

		quadLoss = iM.loss_quadratic_mt_naive(A, Y);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);
		ASSERT_NEAR(etQuadLoss / rowsCnt, quadLoss / rowsCnt, loss_quadratic_EPS<real_t>::eps) << "_mt";

		quadLoss = iM.loss_quadratic(A, Y);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);
		ASSERT_NEAR(etQuadLoss / rowsCnt, quadLoss / rowsCnt, loss_quadratic_EPS<real_t>::eps) << "()";

		quadLoss = iM.compute_loss_st<Loss_t>(A, Y);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);
		ASSERT_NEAR(etQuadLoss / rowsCnt, quadLoss / rowsCnt, loss_quadratic_EPS<real_t>::eps) << "_st";

		quadLoss = iM.compute_loss_mt<Loss_t>(A, Y);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);
		ASSERT_NEAR(etQuadLoss / rowsCnt, quadLoss / rowsCnt, loss_quadratic_EPS<real_t>::eps) << "_mt";

		quadLoss = iM.compute_loss<Loss_t>(A, Y);
		ASSERT_EQ(A, etA);
		ASSERT_EQ(Y, etY);
		ASSERT_NEAR(etQuadLoss / rowsCnt, quadLoss / rowsCnt, loss_quadratic_EPS<real_t>::eps) << "()";
	}

	quadLoss = real_t(0);
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	utils::tictoc tS, tM, tB, tSt, tMt, tBt;
	for (vec_len_t r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2);		rg.gen_matrix(Y, 2);
		tS.tic();
		quadLoss += iM.loss_quadratic_st_naive(A, Y);
		tS.toc();

		rg.gen_matrix(A, 2);		rg.gen_matrix(Y, 2);
		tM.tic();
		quadLoss += iM.loss_quadratic_mt_naive(A, Y);
		tM.toc();

		rg.gen_matrix(A, 2);		rg.gen_matrix(Y, 2);
		tB.tic();
		quadLoss += iM.loss_quadratic(A, Y);
		tB.toc();


		rg.gen_matrix(A, 2);		rg.gen_matrix(Y, 2);
		tSt.tic();
		quadLoss += iM.compute_loss_st<Loss_t>(A, Y);
		tSt.toc();

		rg.gen_matrix(A, 2);		rg.gen_matrix(Y, 2);
		tMt.tic();
		quadLoss += iM.compute_loss_mt<Loss_t>(A, Y);
		tMt.toc();

		rg.gen_matrix(A, 2);		rg.gen_matrix(Y, 2);
		tBt.tic();
		quadLoss += iM.compute_loss<Loss_t>(A, Y);
		tBt.toc();
	}
	tS.say("st");
	tM.say("mt");
	tB.say("best");

	tSt.say("st_t");
	tMt.say("mt_t");
	tBt.say("best_t");
	STDCOUTL(quadLoss);
}
TEST(TestMathN, LossQuadratic) {
// 	typedef nntl::d_interfaces::iThreads_t def_threads_t;
// 	typedef math::MathN<real_t, def_threads_t> iMB;
// 	iMB iM;
	NNTL_RUN_TEST2(imath_basic_t::Thresholds_t::loss_quadratic, 100) test_loss_quadratic(iM, i, 100);
}

TEST(TestMathN, dSigmQuadLoss_dZ) {
	const auto fst = [](const realmtx_t& data_y, realmtx_t& act_dLdZ) { iM.dSigmQuadLoss_dZ_st(data_y, act_dLdZ); };
	const auto fmt = [](const realmtx_t& data_y, realmtx_t& act_dLdZ) { iM.dSigmQuadLoss_dZ_mt(data_y, act_dLdZ); };
	const auto fb = [](const realmtx_t& data_y, realmtx_t& act_dLdZ) { iM.dSigmQuadLoss_dZ(data_y, act_dLdZ); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_dLdZ_corr<true>(dSigmQuadLoss_dZ_ET<real_t>, fst, fmt, fb, "dSigmQuadLoss_dZ", r, c));
		}
	}
}

TEST(TestMathN, dLoss_dZ) {
	typedef activation::Linear_Loss_quadWeighted_FP<real_t> WL_FP;

	const auto fst = [](const realmtx_t& data_y, realmtx_t& act_dLdZ) { iM.dLoss_dZ_st<WL_FP>(data_y, act_dLdZ); };
	const auto fmt = [](const realmtx_t& data_y, realmtx_t& act_dLdZ) { iM.dLoss_dZ_mt<WL_FP>(data_y, act_dLdZ); };
	const auto fb = [](const realmtx_t& data_y, realmtx_t& act_dLdZ) { iM.dLoss_dZ<WL_FP>(data_y, act_dLdZ); };

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE(test_dLdZ_corr<false>(dLoss_dZ_ET<WL_FP, real_t>, fst, fmt, fb, "dLoss_dZ<WeightedLoss_FP>", r, c));
		}
	}
}

#if NNTL_MATLAB_AVAILABLE

TEST(TestMathN, _mIsOrthogonal) {
	{
		SCOPED_TRACE("Random matrix shouldb't be orthogonal");

		realmtx_t M(20, 30);
		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());

		rg.gen_matrix(M, 5);
		ASSERT_FALSE(iM._mIsOrthogonal(M, true));
		ASSERT_FALSE(iM._mIsOrthogonal(M, false));
	}
	{
		SCOPED_TRACE("Pregenerated orthogonal matrices");
		const char *const pFileName = "./test_data/test_orthogonal.mat";

		nntl_supp::imatfile<> mf;
		ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName));

		realmtx_t O1;
		mf >> serialization::make_nvp("O1", O1);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_TRUE(iM._mIsOrthogonal(O1, true));
		ASSERT_FALSE(iM._mIsOrthogonal(O1, false));

		realmtx_t O2;
		mf >> serialization::make_nvp("O2", O2);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_TRUE(iM._mIsOrthogonal(O2, false));
		ASSERT_FALSE(iM._mIsOrthogonal(O2, true));
	}
}

TEST(TestMathN, mSVD_Orthogonalize_ss) {
	const char *const pFileName = "./test_data/test_orthogonal.mat";

	nntl_supp::imatfile<> mf;
	ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName));

	{
		SCOPED_TRACE("Pregenerated matrix rows>cols");

		realmtx_t mO, W1;

		mf >> serialization::make_nvp("W1", W1);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_FALSE(iM._mIsOrthogonal(W1, true));
		ASSERT_FALSE(iM._mIsOrthogonal(W1, false));

		mf >> serialization::make_nvp("O1", mO);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_TRUE(iM._mIsOrthogonal(mO, true));
		ASSERT_FALSE(iM._mIsOrthogonal(mO, false));

		ASSERT_EQ(W1.size(), mO.size());
		ASSERT_GT(W1.rows(), W1.cols());

		ASSERT_TRUE(iM.mSVD_Orthogonalize_ss(W1));
		ASSERT_TRUE(iM._mIsOrthogonal(W1, true));
		ASSERT_FALSE(iM._mIsOrthogonal(W1, false));
	}

	{
		SCOPED_TRACE("Pregenerated matrix rows<cols");

		realmtx_t mO, W2;

		mf >> serialization::make_nvp("W2", W2);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_FALSE(iM._mIsOrthogonal(W2, true));
		ASSERT_FALSE(iM._mIsOrthogonal(W2, false));

		mf >> serialization::make_nvp("O2", mO);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_TRUE(iM._mIsOrthogonal(mO, false));
		ASSERT_FALSE(iM._mIsOrthogonal(mO, true));

		ASSERT_EQ(W2.size(), mO.size());
		ASSERT_LT(W2.rows(), W2.cols());

		ASSERT_TRUE(iM.mSVD_Orthogonalize_ss(W2));
		ASSERT_TRUE(iM._mIsOrthogonal(W2, false));
		ASSERT_FALSE(iM._mIsOrthogonal(W2, true));
	}
}

#else
TEST(TestMathN, _mIsOrthogonal) {
	realmtx_t M(20, 30);
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	rg.gen_matrix(M, 5);
	ASSERT_FALSE(iM._mIsOrthogonal(M, true));
	ASSERT_FALSE(iM._mIsOrthogonal(M, false));

	ADD_FAILURE() << "Unfortunately, this test requires a working Matlab support";
}

TEST(TestMathN, mSVD_Orthogonalize_ss) {
	GTEST_FAIL() << "Unfortunately, this test requires a working Matlab support";
}
#endif

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////// 
template<typename base_t> struct mColumnsCov_EPS {};
template<> struct mColumnsCov_EPS <double> { static constexpr double eps = 1e-12; };
template<> struct mColumnsCov_EPS <float> { static constexpr float eps = 1e-6f; };
void test_mColumnsCov_corr(vec_len_t rowsCnt, vec_len_t colsCnt) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "test_mColumnsCov_corr");
	constexpr vec_len_t testCorrRepCnt = 10;
	realmtx_t A(rowsCnt, colsCnt), A2(rowsCnt, colsCnt)
		, C_ET(colsCnt, colsCnt)
		, C(colsCnt, colsCnt);

	ASSERT_TRUE(!A.isAllocationFailed() && !A2.isAllocationFailed() && !C_ET.isAllocationFailed() && !C.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	for (vec_len_t rr = 0; rr < testCorrRepCnt; ++rr) {
		rg.gen_matrix(A, 5);
		A.clone_to(A2);

		mColumnsCov_ET(A, C_ET, iM);
		ASSERT_EQ(A, A2);
		
		C.zeros();
		iM.mColumnsCov(A, C, false);
		ASSERT_EQ(A, A2);

		ASSERT_NO_FATAL_FAILURE({
			SCOPED_TRACE("upper triangular");
			for (vec_len_t c = 0; c < colsCnt; ++c) {
				for (vec_len_t r = 0; r <= c; ++r) {
					ASSERT_NEAR(C_ET.get(r, c), C.get(r, c), mColumnsCov_EPS<real_t>::eps) << "element (" << r << "," << c << ") differs";
				}
			}
		});

		C.zeros();
		iM.mColumnsCov(A, C, true);
		ASSERT_EQ(A, A2);

		ASSERT_NO_FATAL_FAILURE({
			SCOPED_TRACE("lower triangular");
			for (vec_len_t r = 0; r < colsCnt; ++r) {
				for (vec_len_t c = 0; c <= r; ++c) {
					ASSERT_NEAR(C_ET.get(r, c), C.get(r, c), mColumnsCov_EPS<real_t>::eps) << "element (" << r << "," << c << ") differs";
				}
			}
		});
	}
}

TEST(TestMathN, mColumnsCov) {
	for (vec_len_t r = 1; r < 2*g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 2; c < 2*g_MinDataSizeDelta; ++c) {
			test_mColumnsCov_corr(r, c);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct make_alphaDropout_EPS {};
template<> struct make_alphaDropout_EPS<double> { static constexpr double eps = 1e-18; };
template<> struct make_alphaDropout_EPS<float> { static constexpr float eps = 1e-12f; };

void test_make_alphaDropout(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "make_alphaDropout");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t dpa = real_t(.6), a = real_t(2), b = real_t(-3), c = real_t(4);

	realmtx_t A(rowsCnt, colsCnt, true), A_ET(rowsCnt, colsCnt, true), As(rowsCnt, colsCnt, true)
		, DM(rowsCnt, colsCnt), DM_ET(rowsCnt, colsCnt), DMs(rowsCnt, colsCnt);

	ASSERT_TRUE(!A.isAllocationFailed() && !A_ET.isAllocationFailed() && !As.isAllocationFailed()
		&& !DM.isAllocationFailed() && !DM_ET.isAllocationFailed() && !DMs.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(As, real_t(5));
		rg.gen_matrix_norm(DMs);

		As.copy_data_skip_bias(A_ET);
		DMs.copy_to(DM_ET);
		make_alphaDropout_ET(A_ET, dpa, a, b, c, DM_ET);
		ASSERT_TRUE(A_ET.test_biases_strict());

		As.copy_data_skip_bias(A);
		DMs.copy_to(DM);
		iM.make_alphaDropout_st(A, dpa, a, b, c, DM);
		ASSERT_TRUE(A.test_biases_strict());
		ASSERT_REALMTX_NEAR(A, A_ET, "_st() computes different A!!", make_alphaDropout_EPS<real_t>::eps);
		ASSERT_REALMTX_NEAR(DM, DM_ET, "_st() computes different DM!!", make_alphaDropout_EPS<real_t>::eps);

		As.copy_data_skip_bias(A);
		DMs.copy_to(DM);
		iM.make_alphaDropout_mt(A, dpa, a, b, c, DM);
		ASSERT_TRUE(A.test_biases_strict());
		ASSERT_REALMTX_NEAR(A, A_ET, "_mt() computes different A!!", make_alphaDropout_EPS<real_t>::eps);
		ASSERT_REALMTX_NEAR(DM, DM_ET, "_mt() computes different DM!!", make_alphaDropout_EPS<real_t>::eps);

		As.copy_data_skip_bias(A);
		DMs.copy_to(DM);
		iM.make_alphaDropout(A, dpa, a, b, c, DM);
		ASSERT_TRUE(A.test_biases_strict());
		ASSERT_REALMTX_NEAR(A, A_ET, "() computes different A!!", make_alphaDropout_EPS<real_t>::eps);
		ASSERT_REALMTX_NEAR(DM, DM_ET, "() computes different DM!!", make_alphaDropout_EPS<real_t>::eps);
	}
}

TEST(TestMathN, make_alphaDropout) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_make_alphaDropout(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_make_alphaDropout(r, c)));
		}
	}
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct evSubMtxMulC_ip_nb_EPS {};
template<> struct evSubMtxMulC_ip_nb_EPS<double> { static constexpr double eps = 1e-18; };
template<> struct evSubMtxMulC_ip_nb_EPS<float> { static constexpr float eps = 1e-12f; };

void test_evSubMtxMulC_ip_nb(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, "evSubMtxMulC_ip_nb");
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const real_t c = real_t(4);

	realmtx_t A(rowsCnt, colsCnt, true), A_ET(rowsCnt, colsCnt, true), As(rowsCnt, colsCnt, true)
		, mB(rowsCnt, colsCnt), mBs(rowsCnt, colsCnt);

	ASSERT_TRUE(!A.isAllocationFailed() && !A_ET.isAllocationFailed() && !As.isAllocationFailed()
		&& !mB.isAllocationFailed() && !mBs.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias(As, real_t(5));
		rg.gen_matrix(mBs, real_t(3));

		As.copy_data_skip_bias(A_ET);
		mBs.copy_to(mB);
		evSubMtxMulC_ip_nb_ET(A_ET, mB, c);
		ASSERT_TRUE(A_ET.test_biases_strict());
		ASSERT_MTX_EQ(mB, mBs, "_ET changes const mtxM!");

		As.copy_data_skip_bias(A);
		iM.evSubMtxMulC_ip_nb_st(A, mB, c);
		ASSERT_TRUE(A.test_biases_strict());
		ASSERT_MTX_EQ(mB, mBs, "_st changes const mtxM!");
		ASSERT_REALMTX_NEAR(A, A_ET, "_st() computes different A!!", evSubMtxMulC_ip_nb_EPS<real_t>::eps);

		As.copy_data_skip_bias(A);
		iM.evSubMtxMulC_ip_nb_mt(A, mB, c);
		ASSERT_TRUE(A.test_biases_strict());
		ASSERT_MTX_EQ(mB, mBs, "_mt changes const mtxM!");
		ASSERT_REALMTX_NEAR(A, A_ET, "_mt() computes different A!!", evSubMtxMulC_ip_nb_EPS<real_t>::eps);

		As.copy_data_skip_bias(A);
		iM.evSubMtxMulC_ip_nb(A, mB, c);
		ASSERT_TRUE(A.test_biases_strict());
		ASSERT_MTX_EQ(mB, mBs, "() changes const mtxM!");
		ASSERT_REALMTX_NEAR(A, A_ET, "() computes different A!!", evSubMtxMulC_ip_nb_EPS<real_t>::eps);
	}
}

TEST(TestMathN, evSubMtxMulC_ip_nb) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_evSubMtxMulC_ip_nb(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_evSubMtxMulC_ip_nb(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////

void test_vCountNonZeros_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t dat(rowsCnt, colsCnt);
	ASSERT_TRUE(!dat.isAllocationFailed());
	const numel_cnt_t ne = dat.numel();

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix(dat, 1);
		iM.ewBinarize_ip(dat, real_t(0.5), real_t(0), real_t(1));

		real_t* ptr = dat.data();
		ptr[0] = real_t(-0.0);
		if (ne > 1) {
			ptr[1] = real_t(+0.0);
		}

		const auto et = vCountNonZeros_ET(ptr, ne);

		auto v = iM.vCountNonZeros(ptr, ne);
		ASSERT_EQ(et, v) << "vCountNonZeros failed!";

		v = vCountNonZeros_naive(ptr, ne);
		ASSERT_EQ(et, v) << "vCountNonZeros_naive failed!";

		ptr[0] = real_t(0);
		v = iM.vCountNonZerosStrict(ptr, ne);
		ASSERT_EQ(et, v) << "vCountNonZeros2 failed!";
	}
}

TEST(TestMathN, vCountNonZeros) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_vCountNonZeros_corr(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_vCountNonZeros_corr(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////

void test_evOneCompl_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t gate(rowsCnt, colsCnt), gcompl(rowsCnt, colsCnt), gcomplET(rowsCnt, colsCnt);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_norm(gate);
		iM.ewBinarize_ip(gate, real_t(0.5), real_t(0), real_t(1));

		evOneCompl_ET(gate, gcomplET);

		gcompl.zeros();
		iM.evOneCompl_st(gate, gcompl);
		ASSERT_EQ(gcomplET, gcompl) << "evOneCompl_st failed!";

		gcompl.zeros();
		iM.evOneCompl_mt(gate, gcompl);
		ASSERT_EQ(gcomplET, gcompl) << "evOneCompl_mt failed!";

		gcompl.zeros();
		iM.evOneCompl(gate, gcompl);
		ASSERT_EQ(gcomplET, gcompl) << "evOneCompl failed!";
	}
}

TEST(TestMathN, evOneCompl) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_evOneCompl_corr(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_evOneCompl_corr(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename T, typename RngT = dt_interfaces<T>::iRng_t>
void test_mExtractRows_corr2(::nntl::math::smatrix<T>& src, ::nntl::math::smatrix<T>& dest, ::nntl::math::smatrix<T>& destET
	, RngT& rg, ::std::vector<vec_len_t>& idxs)
{
	ASSERT_SUPPORTED_REAL_T(T);
	//in fact, there's no reason to restrict to floating point, it's just a bit easier to make test with it.
	//but if it works with any fp - it works with anything

	if (src.emulatesBiases()) {
		ASSERT_TRUE(dest.emulatesBiases() && destET.emulatesBiases());
		ASSERT_TRUE(src.test_biases_strict() && dest.test_biases_strict() && destET.test_biases_strict());
	} else {
		ASSERT_TRUE(!dest.emulatesBiases() && !destET.emulatesBiases());
	}
	ASSERT_TRUE(dest.size() == destET.size());
	ASSERT_TRUE(dest.cols() == src.cols() && src.rows() >= dest.rows());
	ASSERT_TRUE(src.rows() == ::nntl::math::Numel(idxs));

	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		::std::random_shuffle(idxs.begin(), idxs.end(), rg);

		rg.gen_matrix_no_bias_norm(src);

		const auto pIdxs = idxs.cbegin();
		destET.zeros();
		mExtractRows_ET(src, pIdxs, destET);

		dest.nans_no_bias();
		iM.mExtractRows_seqWrite_st(src, pIdxs, dest);
		ASSERT_MTX_EQ(destET, dest, "mExtractRows_seqWrite_st failed!");

		dest.nans_no_bias();
		iM.mExtractRows_seqWrite_mt(src, pIdxs, dest);
		ASSERT_MTX_EQ(destET, dest, "mExtractRows_seqWrite_mt failed!");

		dest.nans_no_bias();
		iM.mExtractRows(src, pIdxs, dest);
		ASSERT_MTX_EQ(destET, dest, "mExtractRows failed!");
	}
}

template<typename T>
void test_mExtractRows_corr(const vec_len_t srcRows, const vec_len_t destRows, const vec_len_t colsCnt) {
	ASSERT_TRUE(srcRows <= destRows) << "Invalid params for test_mExtractRows_corr";

	dt_interfaces<T>::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	
	::nntl::math::smatrix_deform<T> src(srcRows, colsCnt), dest(destRows, colsCnt), destET(destRows, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !destET.isAllocationFailed());

	::std::vector<vec_len_t> idxs(srcRows);
	::std::iota(idxs.begin(), idxs.end(), vec_len_t(0));

	//no biases mode
	ASSERT_TRUE(!src.emulatesBiases() && !dest.emulatesBiases() && !destET.emulatesBiases());
	ASSERT_NO_FATAL_FAILURE(test_mExtractRows_corr2<T>(src, dest, destET, rg, idxs));

	if (colsCnt > 1) {
		//biases mode
		src._enforce_biases(); src.set_biases();
		dest._enforce_biases(); dest.set_biases();
		destET._enforce_biases(); destET.set_biases();
		ASSERT_NO_FATAL_FAILURE(test_mExtractRows_corr2<T>(src, dest, destET, rg, idxs));
	}
}

TEST(TestMathN, mExtractRows) {
	for (vec_len_t dr = 1; dr < g_MinDataSizeDelta; ++dr) {
		for (vec_len_t sr = dr; sr < 4 * g_MinDataSizeDelta; sr += g_MinDataSizeDelta-1) {
			if (dr >= sr) {
				for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
					ASSERT_NO_FATAL_FAILURE(test_mExtractRows_corr<real_t>(sr, dr, c));
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename T, typename RngT = dt_interfaces<T>::iRng_t>
void test_mExtractRowsSeq_corr2(::nntl::math::smatrix<T>& src, ::nntl::math::smatrix<T>& dest, ::nntl::math::smatrix<T>& destET
	, RngT& rg, const ::std::vector<vec_len_t>& idxs)
{
	ASSERT_SUPPORTED_REAL_T(T);
	//in fact, there's no reason to restrict to floating point, it's just a bit easier to make test with it.
	//but if it works with any fp - it works with anything

	if (src.emulatesBiases()) {
		ASSERT_TRUE(dest.emulatesBiases() && destET.emulatesBiases());
		ASSERT_TRUE(src.test_biases_strict() && dest.test_biases_strict() && destET.test_biases_strict());
	} else {
		ASSERT_TRUE(!dest.emulatesBiases() && !destET.emulatesBiases());
	}
	ASSERT_TRUE(dest.size() == destET.size());
	ASSERT_TRUE(dest.cols() == src.cols() && src.rows() >= dest.rows());
	ASSERT_TRUE(src.rows() == ::nntl::math::Numel(idxs));

	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		rg.gen_matrix_no_bias_norm(src);

		auto rdif = src.rows() - dest.rows();
		if (0 != rdif) rdif = rg(rdif);
		const vec_len_t rowOfs = rdif;
		ASSERT_TRUE(rowOfs >= 0 && rowOfs <= (src.rows() - dest.rows())) << "Invalid row index generated";

		destET.zeros();
		mExtractRows_ET(src, &idxs[rowOfs], destET);

		dest.nans_no_bias();
		iM.mExtractRowsSeq_st(src, rowOfs, dest);
		ASSERT_MTX_EQ(destET, dest, "mExtractRowsSeq_st failed!");

		dest.nans_no_bias();
		iM.mExtractRowsSeq_mt(src, rowOfs, dest);
		ASSERT_MTX_EQ(destET, dest, "mExtractRowsSeq_mt failed!");

		dest.nans_no_bias();
		iM.mExtractRowsSeq(src, rowOfs, dest);
		ASSERT_MTX_EQ(destET, dest, "mExtractRowsSeq failed!");
	}
}

template<typename T>
void test_mExtractRowsSeq_corr(const vec_len_t srcRows, const vec_len_t destRows, const vec_len_t colsCnt) {
	ASSERT_TRUE(srcRows >= destRows) << "Invalid params for test_mExtractRowsSeq_corr";

	dt_interfaces<T>::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	::nntl::math::smatrix_deform<T> src(srcRows, colsCnt), dest(destRows, colsCnt), destET(destRows, colsCnt);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !destET.isAllocationFailed());

	::std::vector<vec_len_t> idxs(srcRows);
	::std::iota(idxs.begin(), idxs.end(), vec_len_t(0));

	//no biases mode
	ASSERT_TRUE(!src.emulatesBiases() && !dest.emulatesBiases() && !destET.emulatesBiases());
	ASSERT_NO_FATAL_FAILURE(test_mExtractRowsSeq_corr2<T>(src, dest, destET, rg, idxs));

	if (colsCnt > 1) {
		//biases mode
		src._enforce_biases(); src.set_biases();
		dest._enforce_biases(); dest.set_biases();
		destET._enforce_biases(); destET.set_biases();
		ASSERT_NO_FATAL_FAILURE(test_mExtractRowsSeq_corr2<T>(src, dest, destET, rg, idxs));
	}
}

TEST(TestMathN, mExtractRowsSeq) {
	for (vec_len_t dr = 1; dr < g_MinDataSizeDelta; ++dr) {
		for (vec_len_t sr = dr; sr < 4 * g_MinDataSizeDelta; sr += g_MinDataSizeDelta - 1) {
			if (sr >= dr) {
				for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
					ASSERT_NO_FATAL_FAILURE(test_mExtractRowsSeq_corr<real_t>(sr, dr, c));
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mExtractRowsByMask_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), mask(rowsCnt, 1);
	realmtxdef_t dest(rowsCnt, colsCnt), destET(rowsCnt, colsCnt);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		vec_len_t nzc;
		do {
			rg.gen_matrix_norm(mask);
			iM.ewBinarize_ip(mask, real_t(0.3), real_t(0), real_t(1));
			nzc = static_cast<vec_len_t>(iM.vCountNonZeros(mask.data(), mask.rows()));
		} while (0 == nzc);
		dest.deform_rows(nzc);
		destET.deform_rows(nzc);

		rg.gen_matrix(src, real_t(5));

		destET.zeros();
		mExtractRowsByMask_ET(src, mask.data(), destET);

		dest.zeros();
		iM.mExtractRowsByMask_st(src, mask.data(), dest);
		ASSERT_EQ(destET, dest) << "mExtractRowsByMask_st failed!";

		dest.zeros();
		iM.mExtractRowsByMask_mt(src, mask.data(), dest);
		ASSERT_EQ(destET, dest) << "mExtractRowsByMask_mt failed!";

		dest.zeros();
		iM.mExtractRowsByMask(src, mask.data(), dest);
		ASSERT_EQ(destET, dest) << "mExtractRowsByMask() failed!";
	}
}

TEST(TestMathN, mExtractRowsByMask) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_mExtractRowsByMask_corr(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_mExtractRowsByMask_corr(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void test_mFillRowsByMask_corr(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	constexpr vec_len_t testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	realmtx_t dest(rowsCnt, colsCnt), mask(rowsCnt, 1), destET(rowsCnt, colsCnt);
	realmtxdef_t src(rowsCnt, colsCnt);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	for (vec_len_t r = 0; r < testCorrRepCnt; ++r) {
		numel_cnt_t nzc;
		do {
			rg.gen_matrix_norm(mask);
			iM.ewBinarize_ip(mask, real_t(0.3), real_t(0), real_t(1));
			nzc = iM.vCountNonZeros(mask.data(), mask.rows());
		} while (0 == nzc);
		src.deform_rows(static_cast<neurons_count_t>(nzc));
		rg.gen_matrix(src, real_t(5));

		destET.ones();
		mFillRowsByMask_ET(src, mask.data(), destET);

		dest.ones();
		iM.mFillRowsByMask_st(src, mask.data(), dest);
		ASSERT_EQ(destET, dest) << "mFillRowsByMask_st failed!";

		dest.ones();
		iM.mFillRowsByMask_mt(src, mask.data(), dest);
		ASSERT_EQ(destET, dest) << "mFillRowsByMask_mt failed!";

		dest.ones();
		iM.mFillRowsByMask(src, mask.data(), dest);
		ASSERT_EQ(destET, dest) << "mFillRowsByMask() failed!";
	}
}

TEST(TestMathN, mFillRowsByMask) {
	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_mFillRowsByMask_corr(r, c)));
		}
	}

	constexpr vec_len_t rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			ASSERT_NO_FATAL_FAILURE((test_mFillRowsByMask_corr(r, c)));
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//NOTE: libxsmm code below are fine, but I see no sense in integrating it in the project, b/c
//it helps (on my hw-architecture) mostly in very narrow case of small matrices (see below a perf check) and unclogged cache
// Don't want to delete the code here, may be helpful later, so leaving it commented out
/*
#pragma warning(disable: 4996) //This function or variable may be unsafe.
#pragma warning(disable: 4752) //found Intel(R) Advanced Vector Extensions
#define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE3
//#define LIBXSMM_XCOPY_JIT 3
#include <libxsmm_source.h>
#pragma warning(default: 4752)
#pragma warning(default: 4996)

void my_libxsmm_init()noexcept {
	libxsmm_init();
	libxsmm_set_verbosity(5);

	int l=0;
	auto arch = libxsmmf_get_target_arch(&l);
	STDCOUTL("libxsmm preset arch = " << arch << "; cpuid = "<< libxsmm_cpuid_name(libxsmm_cpuid()));
	
// 	libxsmm_set_target_arch("sse3");
// 	arch = libxsmmf_get_target_arch(&l);
// 	STDCOUTL("libxsmm new arch = " << arch);
}
void my_libxsmm_deinit()noexcept {
	libxsmm_finalize();
}

template<typename T>
void my_libxsmm_transpose_ignore_bias(const math::smatrix<T>& src, math::smatrix<T>& dest)noexcept {
	NNTL_ASSERT(src.cols_no_bias() == dest.rows() && src.rows() == dest.cols_no_bias());
	libxsmm_otrans(dest.data(), src.data(), sizeof(T), src.rows(), src.cols_no_bias(), src.ldim(), dest.ldim());
}
*/

template<typename CommonDataT>
void mTranspose_corr(const CommonDataT& cd, vec_len_t batchSiz, vec_len_t sampleSiz
	, const bool bSrcBias, const bool bDestBias, const bool bSrcBiR, const bool _bIgnoreBias)
{
	typedef typename CommonDataT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;
	typedef math::smatrix_deform<real_t> realmtxdef_t;

	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

	const auto bSameBias = (bSrcBias == bDestBias);
	const auto bIgnoreBias = (_bIgnoreBias || !bSameBias);

	constexpr unsigned _scopeMsgLen = 256;
	char _scopeMsg[_scopeMsgLen];
	sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements), bSrcB=%d, bDestB=%d, bSrcBiR=%d, bIgnoreBias=%d"
		, "mTranspose_corr", batchSiz, sampleSiz
		, ::nntl::math::smatrix_td::sNumel(batchSiz, sampleSiz)
		, int(bSrcBias), int(bDestBias), int(bSrcBiR), int(bIgnoreBias)
	);
	SCOPED_TRACE(_scopeMsg);

	realmtx_t srcET(bSrcBiR, batchSiz, sampleSiz, bSrcBias), src2(bSrcBiR, batchSiz, sampleSiz, bSrcBias)
		, destT(!bSrcBiR, batchSiz, sampleSiz, bDestBias), destTET(!bSrcBiR, batchSiz, sampleSiz, bDestBias);
	ASSERT_TRUE(!srcET.isAllocationFailed() && !src2.isAllocationFailed() && !destT.isAllocationFailed() && !destTET.isAllocationFailed());
	
	realmtxdef_t mdef(realmtx_t::sNumel(batchSiz, sampleSiz) + numel_cnt_t(bSrcBias)*::std::max(batchSiz, sampleSiz), bSrcBias);
	ASSERT_TRUE(!mdef.isAllocationFailed());

	//realmtx_t destLXS(bSrcBiR, colsCnt, rowsCnt, bSrcBias);
	//ASSERT_TRUE(!destLXS.isAllocationFailed());

#pragma warning(push)
#pragma warning(disable:4459)
	auto& iM = cd.get_iMath();
#pragma warning(pop)
	auto& iR = cd.get_iRng();

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		iR.gen_matrixAny(srcET, real_t(5));
		ASSERT_TRUE(srcET.if_biases_test_strict());

		mTranspose_ET(srcET, destTET, bIgnoreBias);
		ASSERT_TRUE(destTET.if_biases_test_strict());

		destT.ones();
		iM.mTranspose_seq_write(srcET, destT, bIgnoreBias);
		ASSERT_MTX_EQ(destTET, destT, "mTranspose_seq_write() failed!");
		if (bSameBias) {
			src2.ones();
			iM.mTranspose_seq_write(destT, src2, bIgnoreBias);
			ASSERT_MTX_EQ(srcET, src2, "mTranspose_seq_write() failed double application!");
		}
		
		destT.ones();
		iM.mTranspose_seq_read(srcET, destT, bIgnoreBias);
		ASSERT_MTX_EQ(destTET, destT, "mTranspose_seq_read() failed!");
		if (bSameBias) {
			src2.ones();
			iM.mTranspose_seq_read(destT, src2, bIgnoreBias);
			ASSERT_MTX_EQ(srcET, src2, "mTranspose_seq_read() failed double application!");
		}

		destT.ones();
		iM.mTranspose(srcET, destT, bIgnoreBias);
		ASSERT_MTX_EQ(destTET, destT, "mTranspose(bIgnoreBias) failed!");
		if (bSameBias) {
			src2.ones();
			iM.mTranspose(destT, src2, bIgnoreBias);
			ASSERT_MTX_EQ(srcET, src2, "mTranspose(bIgnoreBias) failed double application!");
		}

		if (bIgnoreBias) {
			destT.ones();
			iM.mTranspose_ignore_bias(srcET, destT);
			ASSERT_MTX_EQ(destTET, destT, "mTranspose_ignore_bias() failed!");
			if (bSameBias) {
				src2.ones();
				iM.mTranspose_ignore_bias(destT, src2);
				ASSERT_MTX_EQ(srcET, src2, "mTranspose_ignore_bias() failed double application!");
			}
		}

		/*iM.mTranspose_ignore_bias_BLAS(srcET, destT);
		ASSERT_MTX_EQ(destTET, destT, "mTranspose_ignore_bias_BLAS() failed!");
		iM.mTranspose_ignore_bias_BLAS(destT, src2);
		ASSERT_MTX_EQ(srcET, src2, "mTranspose_ignore_bias_BLAS() failed double application!");*/

		if (bSameBias) {
			mdef.set_batchInRow(bSrcBiR); mdef.deform(1, 1);
			if (mdef.emulatesBiases()) mdef.set_biases();//must do it, b/c set_batchInRow() may change row/col relationship and destroy biases
			mdef.deform_dataset(srcET.batch_size(), srcET.sample_size() + bSrcBias);
			ASSERT_EQ(srcET.size(), mdef.size());
			srcET.copy_to(mdef);
			iM.s_mTranspose_ip_BLAS(mdef);
			ASSERT_MTX_EQ(destTET, static_cast<realmtx_t&>(mdef), "s_mTranspose_ip_BLAS() failed!");

			mdef.set_batchInRow(bSrcBiR); mdef.deform(1, 1);
			if (mdef.emulatesBiases()) mdef.set_biases();//must do it, b/c set_batchInRow() may change row/col relationship and destroy biases
			mdef.deform_dataset(srcET.batch_size(), srcET.sample_size() + bSrcBias);
			ASSERT_EQ(srcET.size(), mdef.size());
			srcET.copy_to(mdef);
			iM.s_mTranspose_ip(mdef);
			ASSERT_MTX_EQ(destTET, static_cast<realmtx_t&>(mdef), "s_mTranspose_ip() failed!");
		}

		//my_libxsmm_transpose_ignore_bias(srcET, destLXS);
		//ASSERT_MTX_EQ(destTET, destLXS, "my_libxsmm_transpose_ignore_bias() failed!");
	}
}

TEST(TestMathN, mTranspose) {
	typedef def_keeper_tpl<real_t> IKeeper_t;

	IKeeper_t keeper;
	auto& cd = keeper.get_const_common_data();

	STDCOUTL("sizeof(real_t) = " << sizeof(real_t));

	//my_libxsmm_init();
	//const vec_len_t g_MinDataSizeDelta = 2 * iM.ithreads().workers_count() + 2;

	for (vec_len_t r = 1; r < g_MinDataSizeDelta; ++r) {
		for (vec_len_t c = 1; c < g_MinDataSizeDelta; ++c) {
			for (int bin = 0; bin < (1 << 4); ++bin) {
				ASSERT_NO_FATAL_FAILURE(mTranspose_corr(cd, r, c, !(bin & 1), !(bin & 2), !(bin & 4), !(bin & 8)));
			}
		}
	}

	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t r = rowsCnt; r < maxRows; ++r) {
		for (vec_len_t c = 1; c < maxCols; ++c) {
			for (int bin = 0; bin < (1 << 4); ++bin) {
				ASSERT_NO_FATAL_FAILURE(mTranspose_corr(cd, r, c, !(bin & 1), !(bin & 2), !(bin & 4), !(bin & 8)));
			}
		}
	}

#ifndef TESTS_SKIP_LONGRUNNING
	for (int bin = 0; bin < (1 << 4); ++bin) {
		ASSERT_NO_FATAL_FAILURE(mTranspose_corr(cd, 17, 1291, !(bin & 1), !(bin & 2), !(bin & 4), !(bin & 8)));
		ASSERT_NO_FATAL_FAILURE(mTranspose_corr(cd, 1291, 17, !(bin & 1), !(bin & 2), !(bin & 4), !(bin & 8)));
	}
#endif

	//my_libxsmm_deinit();
}

/*
template<typename IntfT>
void mTranspose_perf_wLIBXSMM(typename IntfT::iMath_t& iM, vec_len_t rowsCnt, vec_len_t colsCnt) {
	typedef typename IntfT::real_t real_t;
	typedef typename IntfT::iMath_t::realmtx_t realmtx_t;
	typedef typename IntfT::iMath_t::realmtxdef_t realmtxdef_t;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing mTranspose() over " << rowsCnt << "x" << colsCnt << (rowsCnt > colsCnt ? " TALL" : " wide")
		<< " matrix (" << dataSize
		<< " elements). **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	realmtx_t dest(colsCnt, rowsCnt), src(rowsCnt, colsCnt);

	typename IntfT::iRng_t rg;
	rg.init_ithreads(iM.ithreads());
	
	realmtxdef_t cClog;
	realmtx_t cClogDest;
	constexpr numel_cnt_t CACHE_SIZE = 6 * 1024 * 1024;
	auto bClog = false; // dest.byte_size() < CACHE_SIZE;
	constexpr auto bClogOnlyTheRest = true;
	
	if (bClog) {
		numel_cnt_t clogNumel = CACHE_SIZE / sizeof(real_t);
		if (bClogOnlyTheRest) clogNumel -= src.numel();

		if (clogNumel > 0) {
			cClog.resize(clogNumel);
			cClog.deform(static_cast<vec_len_t>(clogNumel), 1);
			cClogDest.resize(cClog.size());
			rg.gen_matrix(cClog, real_t(10));
		} else bClog = false;
	}

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, imath_basic_t::iThreads_t> pw(iM.ithreads());

	utils::tictoc tOBlas1, tLXS1, tOBlas2, tLXS2, tSR1, tSR2, tSW1, tSW2;
	real_t v = real_t(0.);

	//code warmup
	iM.mTranspose_ignore_bias_BLAS(src, dest);
	my_libxsmm_transpose_ignore_bias(src, dest);

	for (unsigned r = 0; r < maxReps; ++r) {
		
		rg.gen_matrix(src, real_t(5));
		if (bClog){
			ASSERT_TRUE(cClog.copy_to(cClogDest));
			for (const auto e : cClogDest) v += r*e;
		}
		tOBlas1.tic();
		iM.mTranspose_ignore_bias_BLAS(src, dest);
		tOBlas1.toc();
		for (const auto e : dest) v += e;

		rg.gen_matrix(src, real_t(5));
		if (bClog) {
			ASSERT_TRUE(cClog.copy_to(cClogDest));
			for (const auto e : cClogDest) v += r*e;
		}
		tSR1.tic();
		iM.mTranspose_seq_read(src, dest, true);
		tSR1.toc();
		for (const auto e : dest) v += e;

		rg.gen_matrix(src, real_t(5));
		if (bClog) {
			ASSERT_TRUE(cClog.copy_to(cClogDest));
			for (const auto e : cClogDest) v += r*e;
		}
		tLXS1.tic();
		my_libxsmm_transpose_ignore_bias(src, dest);
		tLXS1.toc();
		for (const auto e : dest) v += e;

		rg.gen_matrix(src, real_t(5));
		if (bClog) {
			ASSERT_TRUE(cClog.copy_to(cClogDest));
			for (const auto e : cClogDest) v += r*e;
		}
		tSW1.tic();
		iM.mTranspose_seq_write(src, dest, true);
		tSW1.toc();
		for (const auto e : dest) v += e;
	}

	tOBlas1.say(" OBl");
	//tOBlas2.say(" OBl2");
	tLXS1.say(" LXS");
	//tLXS2.say(" LXS2");

	tSR1.say("SeqR");
	//tSR2.say("SeqR2");
	tSW1.say("SeqW");
	//tSW2.say("SeqW2");
	STDCOUTL(v);
}

TEST(TestMathN, mTransposePerfWLIBXSMM) {
	//typedef float real_t;
	typedef dt_interfaces<real_t> myInterfaces_t;

	//typename myInterfaces_t::iMath_t iM;
	//const vec_len_t g_MinDataSizeDelta = 2 * iM.ithreads().workers_count() + 2;

	STDCOUTL("sizeof(real_t) = " << sizeof(real_t));

	//my_libxsmm_init();

#ifdef TESTS_SKIP_LONGRUNNING
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 100, 10));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 10, 100));
	
#else TESTS_SKIP_LONGRUNNING
	
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 128, 16));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 16, 128));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 128, 32));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 32, 128));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 128, 64));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 64, 128));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 128, 128));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 256, 16));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 16, 256));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 256, 32));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 32, 256));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 256, 64));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 64, 256));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 256, 128));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 128, 256));


	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 512, 16));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 16, 512));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 512, 32));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 32, 512));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 512, 64));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 64, 512));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 512, 128));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 128, 512));


	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 1024, 16));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 16, 1024));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 1024, 32));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 32, 1024));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 1024, 64));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 64, 1024));
	

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 1000, 100));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 100, 1000));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 10000, 10));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 10, 10000));

	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 10000, 100));
	ASSERT_NO_FATAL_FAILURE(mTranspose_perf_wLIBXSMM<myInterfaces_t>(iM, 100, 10000));
#endif

	//my_libxsmm_deinit();
}*/

// LIBXSMM vs. OpenBLAS vs. naive results
// bClog=true, clogRest = false
// ******* testing mTranspose() over 128x16 TALL matrix(2048 elements). **************
// OBl              4.137 us     3.310 us     4.056 us
// LXS              5.240 us     3.586 us     4.321 us
// SeqR              3.861 us     3.034 us     3.631 us  <<-------------------
// SeqW             52.133 us     3.861 us     4.667 us
// - 5.15281e+08
// ******* testing mTranspose() over 16x128 wide matrix(2048 elements). **************
// OBl              4.138 us     4.137 us     4.931 us
// LXS              4.137 us     3.034 us     4.002 us
// SeqR              4.413 us     3.861 us     4.531 us
// SeqW              4.138 us     3.034 us     3.882 us  <<-------------------
// 2.60802e+09
// ******* testing mTranspose() over 128x32 TALL matrix(4096 elements). **************
// OBl              7.723 us     6.620 us     7.673 us
// LXS              7.723 us     6.895 us     7.970 us
// SeqR              6.620 us     6.069 us     7.103 us  <<-------------------
// SeqW             14.895 us     7.447 us     8.269 us
// 3.38434e+09
// ******* testing mTranspose() over 32x128 wide matrix(4096 elements). **************
// OBl              7.999 us     7.723 us     9.222 us
// LXS              6.896 us     6.068 us     7.437 us
// SeqR              8.551 us     7.447 us     8.628 us
// SeqW              7.172 us     6.345 us     7.260 us  <<-------------------
// - 2.43793e+09
// ******* testing mTranspose() over 128x64 TALL matrix(8192 elements). **************
// OBl             15.447 us    13.792 us    15.561 us
// LXS             14.619 us    12.964 us    14.893 us
// SeqR             14.068 us    12.964 us    14.116 us  <<-------------------
// SeqW             15.171 us    14.343 us    15.267 us
// - 2.47644e+09
// ******* testing mTranspose() over 64x128 wide matrix(8192 elements). **************
// OBl             16.275 us    14.895 us    16.726 us
// LXS             13.516 us    12.413 us    13.803 us  <<-------------------
// SeqR             52.410 us    13.792 us    15.382 us
// SeqW             13.792 us    13.516 us    14.732 us  <<-------------------
// 3.76955e+09
// ******* testing mTranspose() over 128x128 wide matrix(16384 elements). **************
// OBl             31.722 us    29.238 us    31.943 us
// LXS             27.308 us    25.928 us    27.786 us  <<------<<-------------------
// SeqR             29.791 us    28.135 us    30.410 us
// SeqW             30.894 us    28.135 us    30.334 us
// 2.96691e+09
// ******* testing mTranspose() over 256x16 TALL matrix(4096 elements). **************
// OBl              6.620 us     6.344 us     7.227 us
// LXS              7.723 us     6.620 us     7.647 us
// SeqR              6.068 us     5.792 us     6.545 us  <<-------------------
// SeqW              8.276 us     7.172 us     8.167 us
// 7.72818e+09
// ******* testing mTranspose() over 16x256 wide matrix(4096 elements). **************
// OBl              8.000 us     7.723 us     8.844 us
// LXS              6.896 us     5.517 us     6.540 us  <<-------------------
// SeqR              8.275 us     7.447 us     8.334 us
// SeqW              6.896 us     6.068 us     6.849 us  <<-------------------
// - 8.00898e+09
// ******* testing mTranspose() over 256x32 TALL matrix(8192 elements). **************
// OBl             14.344 us    13.516 us    15.119 us
// LXS             16.550 us    13.791 us    15.243 us
// SeqR             13.792 us    12.137 us    13.738 us  <<-------------------
// SeqW             15.998 us    15.447 us    16.712 us
// - 3.4285e+09
// ******* testing mTranspose() over 32x256 wide matrix(8192 elements). **************
// OBl             17.653 us    16.274 us    18.120 us
// LXS             13.516 us    11.861 us    13.214 us  <<-------------------
// SeqR             16.550 us    14.895 us    16.887 us
// SeqW             13.792 us    12.964 us    13.876 us  <<-------------------
// 3.12838e+09
// ******* testing mTranspose() over 256x64 TALL matrix(16384 elements). **************
// OBl             29.791 us    27.584 us    30.230 us
// LXS             30.618 us    28.687 us    31.523 us
// SeqR             26.756 us    26.480 us    28.277 us  <<-------------------
// SeqW             32.549 us    30.894 us    33.018 us
// 2.94909e+08
// ******* testing mTranspose() over 64x256 wide matrix(16384 elements). **************
// OBl             33.376 us    31.997 us    34.395 us
// LXS             26.205 us    25.377 us    27.334 us  <<-------------------
// SeqR             34.480 us    31.721 us    34.858 us
// SeqW             28.136 us    27.860 us    29.166 us  <<-------------------
// 9.51248e+08
// ******* testing mTranspose() over 256x128 TALL matrix(32768 elements). **************
// OBl            112.266 us   108.681 us   114.508 us
// LXS            110.059 us   107.025 us   111.633 us
// SeqR            112.266 us   109.232 us   114.957 us  <<-------------------
// SeqW            108.680 us   105.094 us   110.119 us  <<-------------------!!!
// 9.32529e+08
// ******* testing mTranspose() over 128x256 wide matrix(32768 elements). **************
// OBl            121.645 us   119.162 us   124.821 us
// LXS            130.195 us   103.991 us   108.335 us
// SeqR            159.711 us   117.783 us   125.703 us
// SeqW            104.818 us   103.163 us   107.575 us  <<-------------------
// 9.99285e+08
// ******* testing mTranspose() over 512x16 TALL matrix(8192 elements). **************
// OBl             14.344 us    12.964 us    14.213 us
// LXS             25.929 us    24.549 us    26.206 us
// SeqR             12.689 us    11.585 us    12.969 us  <<-------------------
// SeqW             17.377 us    14.619 us    16.320 us
// - 2.80236e+09
// ******* testing mTranspose() over 16x512 wide matrix(8192 elements). **************
// OBl             16.274 us    15.171 us    16.876 us
// LXS             12.413 us    11.310 us    12.607 us  <<-------------------
// SeqR             17.377 us    15.171 us    16.731 us
// SeqW             12.689 us    12.412 us    13.404 us  <<-------------------
// 2.13746e+09
// ******* testing mTranspose() over 512x32 TALL matrix(16384 elements). **************
// OBl             32.273 us    29.515 us    31.613 us
// LXS             53.237 us    52.133 us    54.473 us
// SeqR             29.790 us    28.135 us    30.139 us  <<-------------------
// SeqW             36.962 us    33.652 us    38.264 us
// 9.03974e+09
// ******* testing mTranspose() over 32x512 wide matrix(16384 elements). **************
// OBl             40.549 us    38.618 us    42.325 us
// LXS             28.687 us    26.480 us    28.672 us  <<-------------------
// SeqR             43.583 us    41.376 us    44.670 us
// SeqW             29.514 us    28.411 us    30.515 us  <<-------------------
// - 8.56488e+08
// ******* testing mTranspose() over 512x64 TALL matrix(32768 elements). **************
// OBl            105.922 us   103.715 us   107.816 us
// LXS            165.227 us   132.678 us   139.325 us
// SeqR            107.853 us   105.921 us   111.048 us  <<-------------------
// SeqW            110.059 us   107.853 us   113.166 us
// 6.21846e+09
// ******* testing mTranspose() over 64x512 wide matrix(32768 elements). **************
// OBl            136.540 us   131.299 us   138.383 us
// LXS            105.646 us   102.888 us   107.104 us  <<-------------------
// SeqR            128.264 us   125.782 us   131.862 us
// SeqW            102.887 us   101.508 us   106.007 us  <<-------------------
// 2.80015e+09
// ******* testing mTranspose() over 512x128 TALL matrix(65536 elements). **************
// OBl            223.981 us   220.671 us   230.068 us
// LXS            277.493 us   263.150 us   282.661 us
// SeqR            221.498 us   215.154 us   223.548 us  <<-------------------
// SeqW            213.774 us   211.016 us   218.246 us  <<-------------------!!
// 2.79574e+09
// ******* testing mTranspose() over 128x512 wide matrix(65536 elements). **************
// OBl            402.999 us   384.794 us   399.827 us
// LXS            206.052 us   204.120 us   212.026 us
// SeqR            354.177 us   346.177 us   358.622 us
// SeqW            208.809 us   201.086 us   208.122 us  <<-------------------
// 1.53352e+09
// ******* testing mTranspose() over 1024x16 TALL matrix(16384 elements). **************
// OBl             34.479 us    32.824 us    34.719 us
// LXS             48.823 us    47.168 us    49.711 us
// SeqR             37.790 us    31.721 us    33.864 us  <<-------------------
// SeqW             45.237 us    31.997 us    45.664 us
// 2.06203e+09
// ******* testing mTranspose() over 16x1024 wide matrix(16384 elements). **************
// OBl             43.030 us    39.445 us    43.113 us
// LXS             28.687 us    26.204 us    28.573 us  <<-------------------
// SeqR             55.996 us    49.099 us    53.401 us
// SeqW             30.066 us    28.687 us    30.564 us  <<-------------------
// - 3.22816e+09
// ******* testing mTranspose() over 1024x32 TALL matrix(32768 elements). **************
// OBl            119.438 us   104.267 us   108.697 us  <<-------------------
// LXS             98.199 us    95.716 us   100.344 us  <<-------------------
// SeqR            115.300 us   106.473 us   111.793 us
// SeqW            114.197 us   113.645 us   117.587 us
// 2.14471e+09
// ******* testing mTranspose() over 32x1024 wide matrix(32768 elements). **************
// OBl            304.526 us   270.321 us   284.202 us
// LXS            104.267 us   102.336 us   107.142 us
// SeqR            185.639 us   182.329 us   191.350 us
// SeqW            102.060 us   100.405 us   104.908 us  <<-------------------
// - 8.24001e+09
// ******* testing mTranspose() over 1024x64 TALL matrix(65536 elements). **************
// OBl            231.980 us   206.051 us   213.530 us  <<-------------------
// LXS            237.773 us   233.635 us   243.740 us
// SeqR            230.325 us   207.431 us   214.984 us  <<-------------------
// SeqW            239.979 us   238.876 us   247.670 us
// - 2.62368e+09
// ******* testing mTranspose() over 64x1024 wide matrix(65536 elements). **************
// OBl            913.301 us   882.959 us   903.852 us
// LXS            204.672 us   203.292 us   210.344 us
// SeqR            898.681 us   863.650 us   896.595 us
// SeqW            201.638 us   200.259 us   207.737 us  <<-------------------
// 6.31126e+09
// ******* testing mTranspose() over 1000x100 TALL matrix(100000 elements). **************
// OBl            273.080 us   203.845 us   211.585 us
// LXS            324.110 us   321.351 us   333.134 us
// SeqR            201.637 us   197.500 us   206.526 us  <<-------------------
// SeqW            271.701 us   252.668 us   262.737 us
// - 3.85931e+09
// ******* testing mTranspose() over 100x1000 wide matrix(100000 elements). **************
// OBl            227.291 us   223.153 us   232.394 us
// LXS            204.672 us   199.431 us   208.029 us  <<-------------------!!!
// SeqR            247.702 us   225.084 us   233.522 us
// SeqW            257.081 us   214.878 us   263.197 us >>>>
// 8.71402e+08
// ******* testing mTranspose() over 10000x10 TALL matrix(100000 elements). **************
// OBl            316.663 us   310.870 us   320.837 us
// LXS            298.181 us   282.458 us   291.837 us
// SeqR            298.181 us   294.320 us   305.605 us
// SeqW            140.954 us   138.195 us   145.402 us  <<----<<-------------------
// 9.62054e+08
// ******* testing mTranspose() over 10x10000 wide matrix(100000 elements). **************
// OBl            154.469 us   141.781 us   149.610 us  <<-------------------
// LXS            233.083 us   229.773 us   237.356 us
// SeqR            151.435 us   147.022 us   153.741 us  <<---<<-------------------
// SeqW            256.530 us   229.497 us   238.425 us
// - 1.04417e+09
// ******* testing mTranspose() over 10000x100 TALL matrix(1000000 elements). **************
// OBl              7.669 ms     7.448 ms     7.644 ms
// LXS              3.297 ms     3.235 ms     3.317 ms
// SeqR              7.639 ms     7.485 ms     7.658 ms
// SeqW              2.415 ms     2.356 ms     2.417 ms  <<---<<-------------------
// - 2.02927e+09
// ******* testing mTranspose() over 100x10000 wide matrix(1000000 elements). **************
// OBl              2.314 ms     2.277 ms     2.334 ms
// LXS              6.780 ms     6.507 ms     6.676 ms
// SeqR              2.368 ms     2.303 ms     2.364 ms  <<---<<-------------------
// SeqW              6.969 ms     6.832 ms     6.938 ms
// - 1.09005e+09
//basically, before 100000 elements works general pattern wide->write, tall->read. After that it inverts.
// libxsmm can't do anything with clogged cache-induced delays.

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// clogRest=true - differs visibly from false
// ******* testing mTranspose() over 128x16 TALL matrix(2048 elements). **************
// OBl              3.862 us     3.034 us     3.691 us
// LXS              4.414 us     3.310 us     4.022 us
// SeqR              3.862 us     3.034 us     3.602 us  <<-------------------
// SeqW              4.138 us     3.310 us     4.064 us
// 1.78561e+09
// ******* testing mTranspose() over 16x128 wide matrix(2048 elements). **************
// OBl              4.138 us     3.310 us     4.165 us
// LXS              3.586 us     2.758 us     3.440 us  <<-------------------
// SeqR              3.862 us     3.310 us     4.063 us
// SeqW              3.310 us     3.034 us     3.507 us  <<-------------------
// - 5.40063e+09
// ******* testing mTranspose() over 128x32 TALL matrix(4096 elements). **************
// OBl              7.172 us     5.793 us     6.983 us
// LXS              7.723 us     5.516 us     6.883 us  <<-------------------
// SeqR              6.621 us     5.792 us     6.814 us  <<-------------------
// SeqW              6.896 us     6.068 us     7.108 us
// - 2.22991e+09
// ******* testing mTranspose() over 32x128 wide matrix(4096 elements). **************
// OBl              8.551 us     6.896 us     8.468 us
// LXS              6.620 us     5.241 us     6.778 us  <<-------------------
// SeqR              7.724 us     6.620 us     7.597 us
// SeqW              6.896 us     5.792 us     6.504 us  <<-------------------
// 9.68361e+08
// ******* testing mTranspose() over 128x64 TALL matrix(8192 elements). **************
// OBl             13.516 us    12.413 us    13.993 us
// LXS             15.723 us    11.585 us    13.384 us  <<-------------------
// SeqR             13.792 us    12.688 us    13.737 us  <<-------------------
// SeqW             13.516 us    12.136 us    14.128 us
// - 2.97645e+09
// ******* testing mTranspose() over 64x128 wide matrix(8192 elements). **************
// OBl             15.171 us    12.689 us    14.949 us
// LXS             12.137 us    10.757 us    12.535 us  <<-------------------
// SeqR             14.344 us    12.964 us    14.440 us
// SeqW             13.516 us    11.585 us    13.399 us  <<-------------------
// - 6.22572e+09
// ******* testing mTranspose() over 128x128 wide matrix(16384 elements). **************
// OBl             30.066 us    26.480 us    29.116 us
// LXS             28.135 us    22.895 us    26.145 us  <<-------------------
// SeqR             27.860 us    26.480 us    29.080 us
// SeqW             27.032 us    24.550 us    26.894 us  <<-------------------
// - 1.03796e+08
// ******* testing mTranspose() over 128x128 wide matrix(16384 elements). **************
// OBl             30.893 us    25.928 us    29.001 us
// LXS             26.205 us    21.791 us    25.493 us  <<-------------------
// SeqR             30.343 us    25.929 us    28.536 us
// SeqW             28.135 us    24.273 us    26.960 us  <<-------------------
// 2.02847e+09
// ******* testing mTranspose() over 256x16 TALL matrix(4096 elements). **************
// OBl              7.448 us     6.068 us     6.777 us
// LXS              6.621 us     5.792 us     6.879 us  <<-------------------
// SeqR              6.345 us     5.792 us     6.882 us  <<-------------------
// SeqW              7.172 us     6.344 us     7.378 us
// 2.17661e+09
// ******* testing mTranspose() over 16x256 wide matrix(4096 elements). **************
// OBl              7.724 us     6.344 us     7.784 us
// LXS              6.344 us     4.689 us     5.918 us  <<-------------------
// SeqR              7.999 us     6.620 us     7.834 us
// SeqW              6.896 us     5.516 us     6.559 us  <<-------------------
// - 3.94272e+09
// ******* testing mTranspose() over 256x32 TALL matrix(8192 elements). **************
// OBl             14.620 us    12.413 us    13.869 us
// LXS             14.344 us    11.309 us    13.870 us  <<-------------------
// SeqR             13.792 us    12.412 us    13.568 us  <<-------------------
// SeqW             15.447 us    12.964 us    14.902 us
// - 3.54141e+09
// ******* testing mTranspose() over 32x256 wide matrix(8192 elements). **************
// OBl             19.309 us    14.067 us    15.738 us
// LXS             11.861 us    10.206 us    12.525 us  <<-------------------
// SeqR             15.998 us    13.791 us    15.396 us
// SeqW             12.964 us    11.861 us    13.021 us  <<-------------------
// 3.95841e+08
// ******* testing mTranspose() over 256x64 TALL matrix(16384 elements). **************
// OBl             34.204 us    25.929 us    28.520 us
// LXS             30.067 us    24.550 us    28.254 us  <<-------------------
// SeqR             28.136 us    25.377 us    28.346 us  <<-------------------
// SeqW             29.515 us    26.205 us    29.078 us
// 5.70597e+08
// ******* testing mTranspose() over 64x256 wide matrix(16384 elements). **************
// OBl             31.722 us    28.136 us    31.336 us
// LXS             25.102 us    22.619 us    25.678 us  <<-------------------
// SeqR             31.445 us    28.412 us    31.048 us
// SeqW             27.584 us    24.550 us    27.302 us  <<-------------------
// - 5.6073e+09
// ******* testing mTranspose() over 256x128 TALL matrix(32768 elements). **************
// OBl            107.852 us   106.198 us   115.031 us
// LXS            124.127 us    98.198 us   104.240 us  <<-------------------
// SeqR            110.887 us   101.508 us   114.793 us
// SeqW            101.509 us    96.819 us   102.743 us  <<---<<-------------------
// 1.49674e+09
// ******* testing mTranspose() over 128x256 wide matrix(32768 elements). **************
// OBl            120.542 us   111.715 us   121.610 us
// LXS            103.163 us    97.647 us   102.984 us  <<-------------------
// SeqR            116.128 us   110.335 us   123.771 us
// SeqW            100.681 us    96.820 us   101.893 us  <<-------------------
// 4.6203e+09
// ******* testing mTranspose() over 512x16 TALL matrix(8192 elements). **************
// OBl             13.792 us    12.412 us    14.009 us
// LXS             23.446 us    22.343 us    24.040 us
// SeqR             13.240 us    12.136 us    13.304 us  <<-------------------
// SeqW             14.895 us    12.413 us    14.499 us
// - 2.62916e+09
// ******* testing mTranspose() over 16x512 wide matrix(8192 elements). **************
// OBl             15.447 us    13.240 us    14.752 us
// LXS             12.965 us     9.930 us    12.067 us  <<-------------------
// SeqR             14.895 us    13.241 us    15.114 us
// SeqW             12.688 us    11.310 us    12.888 us  <<-------------------
// 1.10801e+09
// ******* testing mTranspose() over 512x32 TALL matrix(16384 elements). **************
// OBl             30.066 us    27.032 us    29.729 us
// LXS             50.479 us    46.065 us    49.217 us
// SeqR             28.963 us    27.308 us    29.634 us  <<-------------------
// SeqW             31.997 us    27.308 us    31.842 us
// 1.50959e+09
// ******* testing mTranspose() over 32x512 wide matrix(16384 elements). **************
// OBl             36.687 us    29.791 us    33.140 us
// LXS             26.205 us    23.997 us    26.460 us  <<-------------------
// SeqR             53.512 us    33.928 us    36.787 us
// SeqW             28.687 us    25.929 us    29.109 us  <<-------------------
// - 3.87571e+09
// ******* testing mTranspose() over 512x64 TALL matrix(32768 elements). **************
// OBl            105.646 us   102.888 us   111.725 us
// LXS            121.921 us   110.060 us   119.820 us
// SeqR            131.574 us   103.991 us   113.317 us
// SeqW            128.265 us    99.302 us   104.678 us
// - 6.09414e+09
// ******* testing mTranspose() over 64x512 wide matrix(32768 elements). **************
// OBl            143.160 us   119.714 us   129.543 us
// LXS             99.302 us    95.440 us   100.611 us  <<-------------------
// SeqR            126.610 us   116.128 us   129.727 us
// SeqW             99.854 us    94.337 us    99.664 us  <<-------------------
// - 1.14001e+09
// ******* testing mTranspose() over 512x128 TALL matrix(65536 elements). **************
// OBl            222.050 us   205.500 us   215.448 us
// LXS            262.598 us   232.256 us   246.497 us
// SeqR            214.879 us   199.983 us   212.552 us  <<-------------------
// SeqW            207.706 us   196.948 us   207.266 us  <<----<<-------------------
// - 4.904e+09
// ******* testing mTranspose() over 128x512 wide matrix(65536 elements). **************
// OBl            396.380 us   334.040 us   356.701 us
// LXS            203.845 us   193.086 us   203.109 us
// SeqR            308.111 us   274.184 us   293.274 us
// SeqW            197.500 us   190.880 us   199.746 us  <<-------------------
// - 6.34919e+09
// ******* testing mTranspose() over 1024x16 TALL matrix(16384 elements). **************
// OBl             38.065 us    31.445 us    33.662 us
// LXS             52.961 us    43.307 us    46.451 us
// SeqR             32.549 us    31.446 us    33.507 us  <<-------------------
// SeqW             41.651 us    26.205 us    39.728 us
// 3.768e+09
// ******* testing mTranspose() over 16x1024 wide matrix(16384 elements). **************
// OBl             40.824 us    35.307 us    38.732 us
// LXS             33.100 us    24.825 us    27.954 us  <<-------------------
// SeqR             45.238 us    39.445 us    42.911 us
// SeqW             29.238 us    27.584 us    29.888 us  <<-------------------
// 3.54753e+09
// ******* testing mTranspose() over 1024x32 TALL matrix(32768 elements). **************
// OBl            106.750 us   102.888 us   111.762 us
// LXS             94.889 us    87.992 us    94.819 us  <<-------<<-------------------
// SeqR            113.093 us   104.819 us   114.334 us  <<-------------------
// SeqW            111.439 us   101.784 us   111.432 us  <<------<<-------------------
// 2.73579e+09
// ******* testing mTranspose() over 32x1024 wide matrix(32768 elements). **************
// OBl            234.462 us   212.395 us   226.053 us
// LXS            102.336 us    96.268 us   100.952 us  <<-------------------
// SeqR            194.467 us   154.469 us   169.983 us
// SeqW             99.301 us    95.716 us   100.510 us  <<-------------------
// - 1.35643e+09
// ******* testing mTranspose() over 1024x64 TALL matrix(65536 elements). **************
// OBl            210.189 us   199.155 us   211.512 us
// LXS            235.015 us   222.325 us   243.460 us
// SeqR            207.707 us   201.086 us   215.412 us  <<-------------------
// SeqW            232.807 us   216.809 us   231.749 us
// 2.10376e+09
// ******* testing mTranspose() over 64x1024 wide matrix(65536 elements). **************
// OBl            857.306 us   735.385 us   780.678 us
// LXS            204.673 us   191.156 us   201.258 us
// SeqR            825.308 us   712.766 us   756.333 us
// SeqW            207.154 us   190.604 us   199.912 us  <<-------------------
// 5.31042e+09
// ******* testing mTranspose() over 1000x100 TALL matrix(100000 elements). **************
// OBl            208.258 us   171.020 us   188.650 us
// LXS            297.078 us   270.873 us   291.582 us
// SeqR            203.844 us   172.399 us   190.880 us  <<-------------------
// SeqW            239.979 us   213.223 us   232.795 us
// - 2.32183e+09
// ******* testing mTranspose() over 100x1000 wide matrix(100000 elements). **************
// OBl            224.257 us   192.811 us   203.709 us
// LXS            190.881 us   155.849 us   185.149 us  <<-------------------
// SeqR            218.188 us   192.535 us   206.421 us  <<----<<-------------------
// SeqW            236.394 us   190.605 us   225.144 us
// 2.37755e+09
// ******* testing mTranspose() over 10000x10 TALL matrix(100000 elements). **************
// OBl            317.214 us   293.216 us   305.878 us
// LXS            279.424 us   263.977 us   279.638 us
// SeqR            308.939 us   282.459 us   295.510 us
// SeqW            141.230 us   122.472 us   138.039 us  <<-----<<-------------------
// 9.69117e+08
// ******* testing mTranspose() over 10x10000 wide matrix(100000 elements). **************
// OBl            155.849 us   130.471 us   143.699 us
// LXS            227.291 us   219.291 us   233.107 us
// SeqR            150.056 us   138.747 us   149.577 us   <<----<<-------------------
// SeqW            230.601 us   223.153 us   235.151 us
// 1.62712e+09
// ******* testing mTranspose() over 10000x100 TALL matrix(1000000 elements). **************
// OBl              7.579 ms     7.377 ms     7.567 ms
// LXS              3.311 ms     3.246 ms     3.323 ms
// SeqR              7.617 ms     7.356 ms     7.579 ms
// SeqW              2.318 ms     2.255 ms     2.338 ms  <<-----<<-------------------
// 2.61649e+09
// ******* testing mTranspose() over 100x10000 wide matrix(1000000 elements). **************
// OBl              2.345 ms     2.233 ms     2.324 ms
// LXS              6.423 ms     6.199 ms     6.406 ms
// SeqR              2.353 ms     2.157 ms     2.346 ms  <<----<<-------------------
// SeqW              6.615 ms     6.382 ms     6.653 ms
// - 6.78666e+08

//basically same as previous, despite some minor differences


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// UNCLOGGED
// ******* testing mTranspose() over 128x16 TALL matrix(2048 elements). **************
// OBl              1.931 us     1.655 us     1.898 us
// LXS              1.655 us     1.655 us     1.820 us  <<-------------------
// SeqR              1.931 us     1.655 us     1.947 us  <<-------------------
// SeqW              2.483 us     2.206 us     2.675 us
// - 4898.8
// ******* testing mTranspose() over 16x128 wide matrix(2048 elements). **************
// OBl              2.207 us     2.206 us     2.344 us
// LXS              1.103 us   827.000 ns     1.018 us  <<----<<-------------------
// SeqR              2.483 us     2.206 us     2.422 us
// SeqW              1.931 us     1.655 us     1.901 us  <<-------------------
// - 4898.8
// ******* testing mTranspose() over 128x32 TALL matrix(4096 elements). **************
// OBl              3.586 us     3.585 us     3.663 us
// LXS              2.482 us     2.206 us     2.493 us  <<----<<-------------------
// SeqR              3.586 us     3.585 us     3.664 us  <<-------------------
// SeqW              4.138 us     3.861 us     4.387 us
// - 2365.29
// ******* testing mTranspose() over 32x128 wide matrix(4096 elements). **************
// OBl              3.861 us     3.861 us     3.970 us
// LXS              1.655 us     1.655 us     1.884 us  <<----<<-------------------
// SeqR              4.137 us     4.137 us     4.275 us
// SeqW              3.586 us     3.585 us     3.687 us  <<-------------------
// 7031.39
// ******* testing mTranspose() over 128x64 TALL matrix(8192 elements). **************
// OBl             17.102 us     9.378 us    10.002 us
// LXS             11.309 us     8.551 us     9.072 us  <<----<<-------------------
// SeqR             11.310 us     9.654 us    10.266 us  <<-------------------
// SeqW             12.689 us    11.309 us    11.751 us
// 13957.9
// ******* testing mTranspose() over 64x128 wide matrix(8192 elements). **************
// OBl             10.206 us     9.654 us    10.044 us
// LXS              8.276 us     7.999 us     8.499 us  <<----<<-------------------
// SeqR             10.481 us     9.930 us    10.554 us  <<------<<-------------------
// SeqW             11.861 us    11.309 us    11.646 us
// 13958.1
// ******* testing mTranspose() over 128x128 wide matrix(16384 elements). **************
// OBl             21.792 us    21.239 us    22.454 us
// LXS             19.033 us    18.205 us    18.728 us  <<----<<-------------------
// SeqR             22.067 us    21.515 us    22.565 us  <<----<<-------------------
// SeqW             23.171 us    22.618 us    23.264 us
// 5262.69
// ******* testing mTranspose() over 128x128 wide matrix(16384 elements). **************
// OBl             22.343 us    21.240 us    22.427 us
// LXS             18.757 us    18.205 us    18.835 us  <<----<<-------------------
// SeqR             22.342 us    21.515 us    22.647 us  <<----<<-------------------
// SeqW             23.171 us    22.618 us    23.340 us
// 5262.69
// ******* testing mTranspose() over 256x16 TALL matrix(4096 elements). **************
// OBl              3.586 us     3.310 us     3.579 us
// LXS              3.310 us     3.310 us     3.457 us  <<----<<-------------------
// SeqR              3.586 us     3.310 us     3.716 us  <<-------------------
// SeqW              4.690 us     4.413 us     5.013 us
// 7031.24
// ******* testing mTranspose() over 16x256 wide matrix(4096 elements). **************
// OBl              4.690 us     4.413 us     4.466 us
// LXS              1.655 us     1.379 us     1.670 us  <<----<<-------------------
// SeqR              5.241 us     4.413 us     4.988 us
// SeqW              3.586 us     3.310 us     3.590 us  <<-------------------
// 7031.33
// ******* testing mTranspose() over 256x32 TALL matrix(8192 elements). **************
// OBl              9.930 us     9.102 us     9.803 us
// LXS             10.482 us     9.378 us     9.749 us  <<-------------------
// SeqR             16.551 us     9.378 us     9.920 us  <<-------------------
// SeqW             12.964 us    12.412 us    12.786 us
// 13958.4
// ******* testing mTranspose() over 32x256 wide matrix(8192 elements). **************
// OBl             11.585 us     9.930 us    10.513 us
// LXS              9.103 us     7.723 us     8.283 us  <<----<<-------------------
// SeqR             20.136 us    10.481 us    11.008 us  <<-------------------
// SeqW             13.240 us     9.654 us    11.551 us  <<-------------------
// 13958.1
// ******* testing mTranspose() over 256x64 TALL matrix(16384 elements). **************
// OBl             22.895 us    21.515 us    22.100 us
// LXS             22.894 us    20.963 us    21.956 us  <<----<<-------------------
// SeqR             33.653 us    21.240 us    22.428 us  <<-------------------
// SeqW             30.894 us    24.273 us    24.919 us
// 5263.03
// ******* testing mTranspose() over 64x256 wide matrix(16384 elements). **************
// OBl             37.514 us    22.618 us    24.001 us
// LXS             19.033 us    17.929 us    18.487 us  <<----<<-------------------
// SeqR             25.377 us    23.446 us    24.892 us
// SeqW             23.722 us    23.170 us    23.866 us  <<-------------------
// 5263.29
// ******* testing mTranspose() over 256x128 TALL matrix(32768 elements). **************
// OBl            129.368 us   105.647 us   108.835 us
// LXS             92.958 us    91.302 us    92.032 us  <<-------------------
// SeqR            113.922 us   113.645 us   115.370 us
// SeqW             91.303 us    90.474 us    91.225 us  <<----<<-------------------
// 2992.29
// ******* testing mTranspose() over 128x256 wide matrix(32768 elements). **************
// OBl            113.921 us    97.095 us   115.021 us
// LXS             90.475 us    81.648 us    91.227 us  <<-------------------
// SeqR            116.955 us   110.611 us   117.628 us
// SeqW             90.200 us    89.647 us    90.665 us  <<-------------------
// 2992.09
// ******* testing mTranspose() over 512x16 TALL matrix(8192 elements). **************
// OBl             10.482 us     9.102 us     9.623 us
// LXS             23.170 us    20.412 us    20.735 us
// SeqR             15.999 us     9.102 us     9.569 us  <<-------------------
// SeqW             12.965 us    11.861 us    13.020 us
// 13647.6
// ******* testing mTranspose() over 16x512 wide matrix(8192 elements). **************
// OBl             11.310 us    10.481 us    10.893 us
// LXS              8.551 us     7.447 us     7.973 us  <<----<<-------------------
// SeqR             17.377 us    11.033 us    11.753 us
// SeqW             12.688 us    11.309 us    11.655 us  <<-------------------
// 727.307
// ******* testing mTranspose() over 512x32 TALL matrix(16384 elements). **************
// OBl             36.135 us    22.618 us    23.943 us
// LXS             52.409 us    44.134 us    45.121 us
// SeqR             24.550 us    22.067 us    23.438 us  <<-------------------
// SeqW             27.860 us    26.204 us    28.154 us
// - 6760.59
// ******* testing mTranspose() over 32x512 wide matrix(16384 elements). **************
// OBl             27.033 us    25.377 us    26.276 us
// LXS             20.136 us    18.757 us    19.438 us  <<----<<-------------------
// SeqR             45.789 us    27.308 us    29.539 us
// SeqW             25.102 us    24.273 us    24.797 us  <<-------------------
// - 6760.48
// ******* testing mTranspose() over 512x64 TALL matrix(32768 elements). **************
// OBl            127.161 us    86.337 us   107.651 us
// LXS            118.887 us   118.058 us   119.279 us
// SeqR            115.852 us   114.748 us   117.141 us
// SeqW             95.164 us    91.854 us    93.547 us  <<----<<-------------------
// - 31677.4
// ******* testing mTranspose() over 64x512 wide matrix(32768 elements). **************
// OBl            120.266 us   117.783 us   120.668 us
// LXS             90.751 us    90.475 us    91.347 us  <<----<<-------------------
// SeqR            119.162 us   118.334 us   120.366 us
// SeqW             90.751 us    89.923 us    91.145 us  <<-------------------
// - 31676.9
// ******* testing mTranspose() over 512x128 TALL matrix(65536 elements). **************
// OBl            187.570 us   183.156 us   187.645 us
// LXS            243.289 us   198.879 us   245.413 us
// SeqR            184.260 us   181.502 us   185.455 us  <<-------------------
// SeqW            209.086 us   186.466 us   188.518 us  <<-------------------
// - 9670.4
// ******* testing mTranspose() over 128x512 wide matrix(65536 elements). **************
// OBl            305.629 us   300.388 us   305.914 us
// LXS            180.122 us   172.399 us   180.549 us  <<-------------------
// SeqR            315.559 us   239.979 us   246.448 us
// SeqW            180.674 us   173.503 us   181.061 us  <<-------------------
// - 9671.63
// ******* testing mTranspose() over 1024x16 TALL matrix(16384 elements). **************
// OBl             36.135 us    24.825 us    26.286 us
// LXS             41.376 us    40.272 us    40.987 us
// SeqR             27.032 us    24.550 us    26.087 us
// SeqW             37.514 us    24.826 us    37.080 us
// - 14211.8
// ******* testing mTranspose() over 16x1024 wide matrix(16384 elements). **************
// OBl             44.410 us    31.446 us    32.687 us
// LXS             29.515 us    19.308 us    20.149 us  <<----<<-------------------
// SeqR             36.686 us    32.825 us    34.464 us
// SeqW             25.929 us    25.653 us    26.109 us  <<-------------------
// - 14211.5
// ******* testing mTranspose() over 1024x32 TALL matrix(32768 elements). **************
// OBl            125.507 us   103.991 us   107.386 us
// LXS             86.337 us    84.406 us    86.705 us
// SeqR            117.507 us   116.955 us   117.901 us
// SeqW             99.026 us    95.440 us    96.429 us
// - 44571.7
// ******* testing mTranspose() over 32x1024 wide matrix(32768 elements). **************
// OBl            204.396 us   198.879 us   208.107 us
// LXS             91.302 us    86.337 us    91.380 us  <<-------------------
// SeqR            149.780 us   147.849 us   152.547 us
// SeqW             91.027 us    83.303 us    91.273 us  <<-------------------
// - 44571.7
// ******* testing mTranspose() over 1024x64 TALL matrix(65536 elements). **************
// OBl            180.123 us   179.571 us   182.780 us
// LXS            235.566 us   235.290 us   239.417 us
// SeqR            191.156 us   188.673 us   191.491 us  <<-------------------
// SeqW            220.671 us   210.741 us   212.367 us
// - 39461.6
// ******* testing mTranspose() over 64x1024 wide matrix(65536 elements). **************
// OBl            738.971 us   667.804 us   675.039 us
// LXS            182.053 us   168.537 us   182.579 us  <<-------------------
// SeqR            659.253 us   639.393 us   658.824 us
// SeqW            184.812 us   180.674 us   182.404 us  <<-------------------
// - 33612
// * ****** testing mTranspose() over 1000x100 TALL matrix(100000 elements). **************
// OBl            145.643 us   145.643 us   149.821 us
// LXS            327.144 us   257.909 us   263.119 us
// SeqR            151.987 us   147.573 us   152.503 us  <<-------------------
// SeqW            203.569 us   194.190 us   197.166 us
// - 74410.7
// ******* testing mTranspose() over 100x1000 wide matrix(100000 elements). **************
// OBl            172.123 us   172.123 us   177.902 us
// LXS            143.436 us   117.507 us   145.160 us  <<----<<-------------------
// SeqR            177.364 us   171.020 us   177.277 us
// SeqW            214.603 us   153.918 us   184.439 us  <<-------------------
// - 21410.8
// ******* testing mTranspose() over 10000x10 TALL matrix(100000 elements). **************
// OBl            265.909 us   262.598 us   266.564 us
// LXS            302.043 us   255.426 us   263.430 us
// SeqR            256.530 us   253.219 us   257.813 us
// SeqW            114.197 us   110.335 us   123.154 us  <<---<<---<<----<<-------------------
// - 45929.6
// ******* testing mTranspose() over 10x10000 wide matrix(100000 elements). **************
// OBl            119.162 us   119.162 us   129.162 us
// LXS            219.568 us   216.533 us   220.743 us
// SeqR            133.781 us   125.783 us   134.162 us  <<---<<---<<----<<-------------------
// SeqW            232.255 us   186.191 us   224.827 us
// - 15429.9
// ******* testing mTranspose() over 10000x100 TALL matrix(1000000 elements). **************
// OBl              7.307 ms     7.079 ms     7.336 ms
// LXS              3.152 ms     2.915 ms     3.022 ms
// SeqR              7.327 ms     7.129 ms     7.361 ms
// SeqW              2.333 ms     2.168 ms     2.285 ms  <<---<<---<<----<<-------------------
// 31086.2
// ******* testing mTranspose() over 100x10000 wide matrix(1000000 elements). **************
// OBl              2.227 ms     2.061 ms     2.170 ms
// LXS              6.895 ms     6.124 ms     6.447 ms
// SeqR              2.191 ms     2.114 ms     2.209 ms  <<---<<---<<----<<-------------------
// SeqW              6.764 ms     6.278 ms     6.663 ms
// - 32313.6
//now libxsmm improves significantly with very wide/narrow matrices, but general pattern remains the same.
//on average I see no point in libxsmm


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////// 

template<typename base_t> struct mMul_BLAS_EPS {};
template<> struct mMul_BLAS_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct mMul_BLAS_EPS<float> { static constexpr float eps = 1e-6f; };
template<typename IntfT>
void mMul_prevAct_weights_2_act_corr(typename IntfT::iMath_t& iM, typename IntfT::iRng_t& iR
	, vec_len_t batchSiz, vec_len_t prevNc, vec_len_t thisNc, const bool bThisBiases, const bool prevBiR, const bool thisBiR)
{
	typedef typename IntfT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;
	typedef math::smatrix_deform<real_t> realmtxdef_t;
	
	constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT/5;

	realmtx_t weights(thisNc, prevNc + 1);
	realmtx_t prevAct(prevBiR, batchSiz, prevNc, true), Act(thisBiR, batchSiz, thisNc, bThisBiases);
	realmtx_t prevActET(false, batchSiz, prevNc, true), ActET(false, batchSiz, thisNc, bThisBiases), Act2(false, batchSiz, thisNc, bThisBiases);

	ASSERT_TRUE(!weights.isAllocationFailed() && !prevAct.isAllocationFailed() && !prevActET.isAllocationFailed()
		&& !Act.isAllocationFailed() && !ActET.isAllocationFailed() && !Act2.isAllocationFailed());
	
	constexpr unsigned _scopeMsgLen = 200;
	char _scopeMsg[_scopeMsgLen];
	sprintf_s(_scopeMsg, "mMul_prevAct_weights_2_act_corr: prevAct=[%d,%d, BiR=%d], thisAct=[%d,%d, bias=%d, BiR=%d]"
		, prevAct.batch_size(), prevAct.sample_size(), prevAct.bBatchInRow(), Act.batch_size(), Act.sample_size()
		, int(Act.emulatesBiases()), int(Act.bBatchInRow()));
	SCOPED_TRACE(_scopeMsg);

	const auto Eps = mMul_BLAS_EPS<real_t>::eps * prevNc;

	for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
		iR.gen_matrixAny(weights, real_t(1));
		ASSERT_TRUE(!weights.emulatesBiases() && weights.bBatchInColumn());
		iR.gen_matrixAny(prevActET, real_t(2));
		ASSERT_TRUE(prevActET.emulatesBiases() && prevActET.test_biases_strict());
		
		iM.mMulABt_Cnb(prevActET, weights, ActET);
		if (bThisBiases) ASSERT_TRUE(ActET.emulatesBiases() && ActET.test_biases_strict());

		if (prevBiR) {
			iM.mTranspose(prevActET, prevAct);
		} else prevActET.copy_to(prevAct);
		ASSERT_TRUE(prevAct.bBatchInRow() == prevBiR && prevAct.emulatesBiases());
		ASSERT_TRUE(prevAct.test_biases_strict());
		iM.mMul_prevAct_weights_2_act(prevAct, weights, Act);
		ASSERT_TRUE(thisBiR == Act.bBatchInRow());
		if (bThisBiases) ASSERT_TRUE(Act.emulatesBiases() && Act.test_biases_strict());

		if (thisBiR) {
			iM.mTranspose(Act, Act2);
			ASSERT_REALMTX_NEAR(Act2, ActET, "mMul_prevAct_weights_2_act() failed with thisBiR!", Eps);
		}else ASSERT_REALMTX_NEAR(Act, ActET, "mMul_prevAct_weights_2_act() failed with !thisBiR!", Eps);
	}
}

TEST(TestMathN, mMul_prevAct_weights_2_act) {
	//typedef float real_t;
	typedef dt_interfaces<real_t> myInterfaces_t;

	//typename myInterfaces_t::iMath_t iM;
	//const vec_len_t g_MinDataSizeDelta = 2 * iM.ithreads().workers_count() + 2;

	STDCOUTL("sizeof(real_t) = " << sizeof(real_t));
	typename myInterfaces_t::iRng_t iR;
	iR.init_ithreads(iM.ithreads());

	//my_libxsmm_init();

	for (vec_len_t bs = 1; bs < g_MinDataSizeDelta; ++bs) {
		for (vec_len_t prevNc = 1; prevNc < g_MinDataSizeDelta; ++prevNc) {
			for (vec_len_t thisNc = 1; thisNc < g_MinDataSizeDelta; ++thisNc) {
				for (int f = 0; f < 8; ++f) {
					ASSERT_NO_FATAL_FAILURE(mMul_prevAct_weights_2_act_corr<myInterfaces_t>(iM, iR, bs, prevNc, thisNc, !(f & 1), !(f & 2), !(f & 4)));
				}
			}
		}
	}

	constexpr unsigned rowsCnt = _baseRowsCnt;
	const vec_len_t maxCols = 3*g_MinDataSizeDelta, maxRows = rowsCnt + g_MinDataSizeDelta;
	for (vec_len_t bs = rowsCnt; bs < maxRows; ++bs) {
		for (vec_len_t prevNc = 3; prevNc < maxCols; prevNc += 3) {
			for (vec_len_t thisNc = 3; thisNc < maxCols; thisNc += 2) {
				for (int f = 0; f < 8; ++f) {
					ASSERT_NO_FATAL_FAILURE(mMul_prevAct_weights_2_act_corr<myInterfaces_t>(iM, iR, bs, prevNc, thisNc, !(f & 1), !(f & 2), !(f & 4)));
				}
			}
		}
	}

}

