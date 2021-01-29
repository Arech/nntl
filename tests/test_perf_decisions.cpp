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

#include "../nntl/utils/tictoc.h"

#include "../nntl/weights_init.h"
#include "../nntl/activation.h"

#include "imath_etalons.h"

using namespace nntl;
using namespace ::std::chrono;
using namespace nntl::utils;
using namespace nntl::math_etalons;

typedef d_interfaces::real_t real_t;
typedef math::smatrix<real_t> realmtx_t;
typedef math::smatrix_deform<real_t> realmtxdef_t;

//declaration of 'iM' hides global declaration
#pragma warning(disable:4459)

typedef d_interfaces::iThreads_t iThreads_t;
typedef math::MathN<real_t, iThreads_t> imath_basic_t;

static imath_basic_t iM;

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
//faster!
template<typename iMathT, typename iRngT>
void makeDropout_old(iMathT& iM, iRngT& iR, typename iMathT::real_t dpa
	, typename iMathT::realmtx_t& DM, typename iMathT::realmtx_t& Act)noexcept
{
	NNTL_ASSERT(Act.emulatesBiases() && Act.test_biases_strict());
	iR.gen_vector_norm_st(DM.data(), DM.numel());
	iM.make_dropout_st(Act, dpa, DM);
	NNTL_ASSERT(Act.emulatesBiases() && Act.test_biases_strict());
}
//slower!
template<typename iMathT, typename iRngT>
void makeDropout_new(iMathT& iM, iRngT& iR, typename iMathT::real_t dpa
	, typename iMathT::realmtx_t& DM, typename iMathT::realmtx_t& Act)noexcept
{
	NNTL_ASSERT(Act.emulatesBiases() && Act.test_biases_strict());
	iR.bernoulli_vector_st(DM.data(), DM.numel(), dpa, real_t(1) / dpa, real_t(0));
	iM.evMul_ip_Anb_st(Act, DM);
	NNTL_ASSERT(Act.emulatesBiases() && Act.test_biases_strict());
}

template<typename iMathT, typename iRngT>
void testperf_makeDropout(iMathT& iM, iRngT& iR, typename iMathT::real_t dpa, vec_len_t _r, vec_len_t _c)noexcept {
	typedef typename iMathT::real_t real_t;
	typedef typename iMathT::realmtx_t realmtx_t;

	realmtx_t DM(_r, _c), Act(_r, _c, true), DMe(_r, _c), Acte(_r, _c, true), Acts(_r, _c, true);

	STDCOUTL("******* testing make_dropout() over " << _r << "x" << _c << " matrix (" << DM.numel() << " elements) ***********");

	//checking coherence between implementations
	iR.gen_matrix_no_bias(Act, real_t(2));
	Act.clone_to(Acte);
	Act.clone_to(Acts);

	const size_t seedVal = ::std::time(0);
	iR.seed64(seedVal);
	iR.gen_vector_norm_st(DM.data(), DM.numel());
	DM.clone_to(DMe);
	make_dropout_ET(Acte, dpa, DMe);

	DM.zeros();
	iR.seed64(seedVal);
	makeDropout_old(iM, iR, dpa, DM, Act);
	ASSERT_MTX_EQ(Acte, Act, "Act mtx differs for makeDropout_old");
	ASSERT_MTX_EQ(DMe, DM, "DM mtx differs for makeDropout_old");
	
	Acts.clone_to(Act);
	DM.zeros();
	iR.seed64(seedVal);
	makeDropout_new(iM, iR, dpa, DM, Act);
	ASSERT_MTX_EQ(Acte, Act, "Act mtx differs for makeDropout_old");
	ASSERT_MTX_EQ(DMe, DM, "DM mtx differs for makeDropout_old");

	tictoc tO, tN;
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, typename iMathT::iThreads_t> pw(iR.ithreads());
	real_t v = real_t(0);
	for (unsigned i = 0; i < TEST_PERF_REPEATS_COUNT; ++i) {
		iR.gen_matrix_no_bias(Act, real_t(2));
		tO.tic();
		makeDropout_old(iM, iR, dpa, DM, Act);
		tO.toc();
		for (const auto& e : Act) v += e;
		v = ::std::log(::std::abs(v));

		iR.gen_matrix_no_bias(Act, real_t(2));
		tN.tic();
		makeDropout_new(iM, iR, dpa, DM, Act);
		tN.toc();
		for (const auto& e : Act) v += e;
		v = ::std::log(::std::abs(v));
	}

	tO.say("old");
	tN.say("new");
	STDCOUTL(v);
}

TEST(TestPerfDecisions, makeDropoutPerf) {
	typedef typename d_interfaces::real_t real_t;

	d_interfaces::iMath_t iM;
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

#ifdef NNTL_DEBUG
	testperf_makeDropout(iM, rg, real_t(.5), 100, 100);
#else
	STDCOUTL("=============== Small ===============");
	testperf_makeDropout(iM, rg, real_t(.5), 100, 50);
	testperf_makeDropout(iM, rg, real_t(.7), 100, 50);
	testperf_makeDropout(iM, rg, real_t(.85), 100, 50);

	STDCOUTL("=============== Big ===============");
	testperf_makeDropout(iM, rg, real_t(.5), 1000, 100);
	testperf_makeDropout(iM, rg, real_t(.7), 1000, 100);
	testperf_makeDropout(iM, rg, real_t(.85), 1000, 100);
#endif
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename RealT>
void makeDropoutMask_baseline(math::smatrix<RealT>& dropoutMask, const RealT dpa)noexcept {
	typedef RealT real_t;

	//iR.gen_vector_norm_st(dropoutMask.data(), dropoutMask.numel());

	const real_t dropPercActInv = real_t(1.) / dpa;
	auto pDM = dropoutMask.data();// +er.elmBegin;
	const auto pDME = pDM + dropoutMask.numel();// +er.totalElements();
	while (pDM != pDME) {
		const auto v = *pDM;
		NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
		*pDM++ = v < dpa ? dropPercActInv : real_t(0.);
	}
}

// this routine is a way slower
template<typename RealT>
void makeDropoutMask_vec(math::smatrix<RealT>& dropoutMask, const RealT dpa)noexcept {
	typedef RealT real_t;

	const real_t dropPercActInv = real_t(1.) / dpa;

	auto pDM = dropoutMask.data();// +er.elmBegin;
	const auto pDME = pDM + dropoutMask.numel();// +er.totalElements();
	while (pDM != pDME) {
		const auto v = *pDM;
		NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
		*pDM++ = dropPercActInv*::std::floor(dpa + v);
		//*pDM++ = ::std::floor(dpa + v);
	}
	/*pDM = dropoutMask.data();
	while (pDM != pDME) {
		*pDM++ *= dropPercActInv;
	}*/
}

template<typename iRngT>
void testperf_makeDropoutMask(iRngT& iR, const typename iRngT::real_t dpa, const vec_len_t _tr, const vec_len_t _tc = 10)noexcept {
	typedef typename iRngT::real_t real_t;
	typedef typename iRngT::realmtx_t realmtx_t;

	realmtx_t dm_b(_tr, _tc), dm_v(_tr, _tc);
	ASSERT_TRUE(!dm_b.isAllocationFailed() && !dm_v.isAllocationFailed());

	STDCOUTL("**** evaluating makeDropoutMask() variations over " << _tr << "x" << _tc << " matrix (" << dm_b.numel() << " elements) ****");

	const unsigned rc = 100;

	tictoc tB, tVec;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, d_interfaces::iThreads_t> pw(iR.ithreads());

	for (unsigned r = 0; r<rc; ++r)	{
		iR.gen_matrix_norm(dm_b);
		dm_b.clone_to(dm_v);
		for (auto&& e : dm_v) {
			e = real_t(1) - e;
		}

		tB.tic();
		makeDropoutMask_baseline(dm_b, dpa);
		tB.toc();

		tVec.tic();
		makeDropoutMask_vec(dm_v, dpa);
		tVec.toc();

		//STDCOUTL("b=" << vSumAbs_ET(dm_b) << " , v=" << vSumAbs_ET(dm_v));

		//they might be slightly different in a rare cases, that's ok.
		//ASSERT_MTX_EQ(dm_b, dm_v, "Algorithms yelded different results!");
	}

	tB.say("base");
	tVec.say("vec");
}

TEST(TestPerfDecisions, makeDropoutMask) {
	typedef typename d_interfaces::real_t real_t;

	d_interfaces::iThreads_t trd;
	d_interfaces::iRng_t rg;
	rg.init_ithreads(trd);

	testperf_makeDropoutMask(rg, real_t(.8), 1000, 100);
	testperf_makeDropoutMask(rg, real_t(.2), 1000, 100);
	testperf_makeDropoutMask(rg, real_t(.5), 1000, 100);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

real_t t_sum_stdacc(const real_t* pVec, const numel_cnt_t& n)noexcept {
	//current implementation of accumulate is similar to while{} statement, however it's also for() based
	return ::std::accumulate(pVec, pVec + n, real_t(0.));
}
//this seems a bit faster 
real_t t_sum_for(const real_t* pVec, const numel_cnt_t& n)noexcept {
	real_t r(0.);
	for (numel_cnt_t i = 0; i < n; ++i) {
		r += pVec[i];
	}
	return r;
}

void t_sum(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** checking sum variations over vector with " << dataSize << " elements ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	
	::std::vector<real_t> vec1(dataSize), vec2(dataSize), vec3(dataSize);

	typedef math::SMath<real_t, d_interfaces::iThreads_t> SMath_t;

	d_interfaces::iThreads_t trd;
	d_interfaces::iRng_t rg;
	rg.init_ithreads(trd);

	tictoc tAcc, tFor, tFunctor;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, d_interfaces::iThreads_t> pw(trd);

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	real_t s1(0), s2(0), s3(0);
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_vector_norm(&vec1[0], dataSize);
		tAcc.tic();
		s1 += t_sum_stdacc(&vec1[0], dataSize);
		tAcc.toc();

		rg.gen_vector_norm(&vec2[0], dataSize);
		tFor.tic();
		s2 += t_sum_for(&vec2[0], dataSize);
		tFor.toc();

// 		rg.gen_vector_norm(&vec3[0], dataSize);
// 		tFunctor.tic();
// 		SMath_t::func_SUM<real_t, false> funct;
// 		SMath_t::_vec_apply_func(&vec3[0], dataSize, funct);
// 		s3 += funct.result();
// 		tFunctor.toc();
	}
	tAcc.say("std");
	tFor.say("for");
	//tFunctor.say("FUNC");
	STDCOUTL("s1=" << s1 << "   s2=" << s2 << "   s3=" << s3);
}

TEST(TestPerfDecisions, vectorSum) {
	constexpr vec_len_t maxCol = vec_len_t(1e6/100);
	for (unsigned c = 1; c <= maxCol; c*=10) t_sum(100, c);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void softmax_parts_st_cw(const realmtx_t& act, const real_t* pMax, real_t* pDenominator, real_t* pNumerator)noexcept {
	NNTL_ASSERT(pMax && pDenominator && act.numel() > 0);
	const auto rm = act.rows(), cm = act.cols();
	auto pA = act.data();
	const auto pME = pMax + rm;
	::std::fill(pDenominator, pDenominator + rm, real_t(0.0));
	for (vec_len_t c = 0; c < cm; ++c) {
		auto pDen = pDenominator;
		auto pM = pMax;
		while (pM != pME) {
			const auto num = ::std::exp(*pA++ - *pM++);
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
			const auto num = ::std::exp(pA[ofs] - m);
			den += num;
			pNumerator[ofs] = num;
			ofs += rm;
		}
		pDenominator[r] = den;
	}
}
template<typename base_t> struct softmax_parts_EPS {};
template<> struct softmax_parts_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct softmax_parts_EPS<float> { static constexpr double eps = 2e-5; };
template<typename iMath>
void check_softmax_parts(iMath& iM, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** checking softmax_parts() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT, testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;
	constexpr numel_cnt_t maxDataSizeForSt = 50000;

	realmtx_t A(rowsCnt, colsCnt);
	ASSERT_TRUE(!A.isAllocationFailed());
	::std::vector<real_t> vec_max(rowsCnt), vec_den(rowsCnt), vec_num(dataSize);

	iM.preinit(dataSize);
	ASSERT_TRUE(iM.init());
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	{
		::std::vector<real_t> vec_den2(rowsCnt), vec_num2(dataSize);

		for (unsigned rr = 0; rr < testCorrRepCnt; ++rr) {
			rg.gen_matrix(A, 2);
			mrwMax_ET(A, &vec_max[0]);

			softmax_parts_ET(A, &vec_max[0], &vec_den[0], &vec_num[0]);

			::std::fill(vec_den2.begin(), vec_den2.end(), real_t(0));
			::std::fill(vec_num2.begin(), vec_num2.end(), real_t(0));
			softmax_parts_st_cw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
			ASSERT_VECTOR_NEAR(vec_den, vec_den2, "st_cw() failed denominator vector comparision", softmax_parts_EPS<real_t>::eps);
			ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st_cw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);

			::std::fill(vec_den2.begin(), vec_den2.end(), real_t(0));
			::std::fill(vec_num2.begin(), vec_num2.end(), real_t(0));
			softmax_parts_st_rw(A, &vec_max[0], &vec_den2[0], &vec_num2[0]);
			ASSERT_VECTOR_NEAR(vec_den, vec_den2, "st_rw() failed denominator vector comparision", softmax_parts_EPS<real_t>::eps);
			ASSERT_VECTOR_NEAR(vec_num, vec_num2, "st_rw() failed numerator matrix comparision", softmax_parts_EPS<real_t>::eps);
		}
	}

	tictoc tStCw, tStRw;
	//////////////////////////////////////////////////////////////////////////
	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	//FFFFfffffffff... don't ever think about removing rg. calls that randomizes data...
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(A, 2);
		mrwMax_ET(A, &vec_max[0]);
		::std::fill(vec_den.begin(), vec_den.end(), real_t(0));
		::std::fill(vec_num.begin(), vec_num.end(), real_t(0));
		tStCw.tic();
		softmax_parts_st_cw(A, &vec_max[0], &vec_den[0], &vec_num[0]);
		tStCw.toc();

		rg.gen_matrix(A, 2);
		mrwMax_ET(A, &vec_max[0]);
		::std::fill(vec_den.begin(), vec_den.end(), real_t(0));
		::std::fill(vec_num.begin(), vec_num.end(), real_t(0));
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

void mrwIdxsOfMax_st_rw(const realmtx_t& M, vec_len_t* pDest)noexcept {
	const auto rim = M.rows();
	const auto ne = M.numel();
	//NNTL_ASSERT(rows == dest.size());

	auto pD = M.data();
	for (vec_len_t ri = 0; ri < rim; ++ri) {
		auto pV = &pD[ri];
		const auto pVEnd = pV + ne;
		auto m = ::std::numeric_limits<real_t>::lowest();
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
	::std::vector<vec_len_t> idxs_st_naive(rowsCnt), idxs_st_memf(rowsCnt);
	::std::vector<real_t> max_st_memf(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	//vec_len_t* pDummy = nullptr;
	{
		::std::vector<vec_len_t> idxs_et(rowsCnt);
		::std::vector<real_t> max_et(rowsCnt);

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
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

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
	iM.evMulC_ip_st(vW, momentum,false);
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
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(vW2, 2);
	rg.gen_matrix(W2, 2);
	vW2.clone_to(vW);
	W2.clone_to(W);

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

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

/*
void mTranspose_seq_read(const realmtx_t& src, realmtx_t& dest) noexcept {
	NNTL_ASSERT(src.rows() == dest.cols() && src.cols() == dest.rows());
	const ptrdiff_t sRows = src.rows(), sCols = src.cols();
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
	const ptrdiff_t sRows = src.rows(), sCols = src.cols();
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
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(src, 10);

	mTranspose_ignore_bias_ET(src, destEt);

	dest.zeros();
	mTranspose_seq_read(src, dest);
	ASSERT_EQ(destEt, dest) << "mTranspose_seq_read failed";

	dest.zeros();
	mTranspose_seq_write(src, dest);
	ASSERT_EQ(destEt, dest) << "mTranspose_seq_write failed";

	dest.zeros();
	mTranspose_OpenBLAS(src, dest);
	ASSERT_EQ(destEt, dest) << "mTranspose_OpenBLAS failed";

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

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

	/ * Very funny (and consistent through runs) results
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

	 * /
}
TEST(TestPerfDecisions, mTranspose) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef math::MathN<real_t, def_threads_t> iMB;
	iMB iM;

	STDCOUTL("sizeof(real_t) = " << sizeof(real_t));

	//for (unsigned i = 100; i <= 10000; i*=10) check_mTranspose(iM, i,i/10);
	check_mTranspose(iM, 100, 100);
	//check_mTranspose(iM, 10000,1000);
#ifndef TESTS_SKIP_LONGRUNNING
	check_mTranspose(iM, 1000,10);
	check_mTranspose(iM, 10, 1000);

	check_mTranspose(iM, 1000, 100);
	check_mTranspose(iM, 100, 1000);

	//check_mTranspose(iM, 1000, 1000);

	check_mTranspose(iM, 10000,10);
	check_mTranspose(iM, 10, 10000);

// 	check_mTranspose(iM, 10000, 100);
// 	check_mTranspose(iM, 100, 10000);

	//check_mTranspose(iM, 100000);
#endif
}
*/

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
	real_t meanNorm = ::std::accumulate(pTmp, pTmp+mRows, 0.0) / mRows;

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
	::std::vector<real_t> tmp(rowsCnt);
	::std::vector<size_t> ofs(rowsCnt);

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

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
	::std::vector<real_t> normvecEt(rowsCnt), normvec(rowsCnt);
	
	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	rg.gen_matrix(W, scale);
	rowwise_normsq_ET(W, &normvecEt[0]);
	real_t meanNorm = ::std::accumulate(normvecEt.begin(), normvecEt.end(), real_t(0.0)) / rowsCnt;
	STDCOUTL("Mean norm value is "<< meanNorm);

	::std::fill(normvec.begin(), normvec.end(), real_t(10.0));
	rowwise_normsq_naive(W, &normvec[0]);
	ASSERT_TRUE(0 == memcmp(&normvec[0], &normvecEt[0], rowsCnt*sizeof(real_t))) << "rowwise_normsq_naive wrong implementation";

	::std::fill(normvec.begin(), normvec.end(), real_t(10.0));
	rowwise_normsq_clmnw(W, &normvec[0]);
	ASSERT_TRUE(0 == memcmp(&normvec[0], &normvecEt[0], rowsCnt*sizeof(real_t))) << "rowwise_normsq_clmnw wrong implementation";

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

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
		ql += y*(a == real_t(0.0) ? log_zero : ::std::log(a)) + (real_t(1.0) - y)*(a == real_t(1.0) ? log_zero : nntl::math::log1p(-a));
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
			ql += (a == real_t(0.0) ? log_zero : ::std::log(a));
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
	constexpr auto realmin = ::std::numeric_limits<real_t>::min();
	real_t ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto a = ptrA[i], oma = real_t(1.0) - a;
		p1[i] = (a == real_t(0.0) ? realmin : a);
		p2[i] = (oma == real_t(0.0) ? realmin : oma);
	}
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		ql += y*::std::log(p1[i]) + (real_t(1.0) - y)*::std::log(p2[i]);
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}
template<typename base_t> struct run_sigm_loss_xentropy_EPS {};
template<> struct run_sigm_loss_xentropy_EPS<double> { static constexpr double eps = 1e-8; };
template<> struct run_sigm_loss_xentropy_EPS<float> { static constexpr double eps = 2e-2; };
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
	rg.init_ithreads(iM.ithreads());

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

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
	rg.init_ithreads(iM.ithreads());
	rg.gen_matrix(dW, 2);
	rg.gen_matrix(vW, 2);

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

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
	return val == 0 ? T(0) : ::std::copysign(magn, val);
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
	rg.init_ithreads(iM.ithreads());

	//testing performance
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath::iThreads_t> pw(iM.ithreads());

	real_t lr = real_t(.1);

	real_t pz = real_t(+0.0), nz = real_t(-0.0), p1 = real_t(1), n1 = real_t(-1);

	//::boost::sign
	/*{
		auto c_pz = ::boost::math::sign(pz), c_nz = ::boost::math::sign(nz), c_p1 = ::boost::math::sign(p1), c_n1 = ::boost::math::sign(n1);
		STDCOUTL("::boost::sign\t: +0.0=" << c_pz << " -0.0=" << c_nz << " +1=" << c_p1 << " -1=" << c_n1);
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
			*p++ = lr*::boost::math::sign(*p);
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

	using namespace ::std::chrono;
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
	rg.init_ithreads(iM.ithreads());

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
	NNTL_UNREF(fnc);
	NNTL_ASSERT(!srcdest.empty());
	auto pV = srcdest.data();
	const auto pVE = pV + srcdest.numel();
	while (pV != pVE) {
		const auto v = *pV;
		//*pV++ = v < real_t(+0.0) ? FunctorT::f_neg(v) : FunctorT::f_pos(v);
		*pV++ = fnc.f( v );
	}
}
//slightly faster (177vs192)
template<typename RealT, size_t LeakKInv100 = 10000, typename WeightsInitScheme = weights_init::He_Zhang<>>
class exp_leaky_relu : public activation::_i_activation<RealT, WeightsInitScheme,true> {
	exp_leaky_relu() = delete;
	~exp_leaky_relu() = delete;
public:
	//typedef WeightsInitScheme weights_scheme;
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
class exp3_leaky_relu : public activation::_i_activation<RealT, WeightsInitScheme, true> {
	exp3_leaky_relu() = delete;
	~exp3_leaky_relu() = delete;
public:
	//typedef WeightsInitScheme weights_scheme;
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
	rg.init_ithreads(iM.ithreads());

	realmtx_t XSrc(rowsCnt, colsCnt), X(rowsCnt, colsCnt), TV(rowsCnt, colsCnt);
	ASSERT_TRUE(!XSrc.isAllocationFailed() && !X.isAllocationFailed() && !TV.isAllocationFailed());
	
	typedef exp_leaky_relu<real_t> BType;
	typedef exp3_leaky_relu<real_t> AType;

	tictoc tA1, tB1, tA2, tB2, tA3, tB3;
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, def_threads_t> pw(iM.ithreads());
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

//////////////////////////////////////////////////////////////////////////
void test_vCountNonZeros_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing vCountNonZeros() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t dat(rowsCnt, colsCnt);
	ASSERT_TRUE(!dat.isAllocationFailed());
	const numel_cnt_t ne = dat.numel();

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, imath_basic_t::iThreads_t> pw(iM.ithreads());

	utils::tictoc tN, tN2, tV, tV2, tS, tS2;

	size_t a = 0, v, s;

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(dat, 1);
		iM.ewBinarize_ip(dat, real_t(0.5), real_t(0), real_t(1));

		real_t* ptr = dat.data();
		ptr[0] = real_t(-0.0);
		ptr[4] = real_t(-0.0);
		ptr[1] = real_t(+0.0);

		tN.tic();
		v = vCountNonZeros_naive(ptr, ne);
		tN.toc();
		a += v;
		s = v;

		tV.tic();
		v = iM.vCountNonZeros(ptr, ne);
		tV.toc();
		a += v;
		ASSERT_EQ(s, v);
		s = v;

		ptr[0] = real_t(0.0); ptr[4] = real_t(0.0);
		tS.tic();
		v = iM.vCountNonZerosStrict(ptr, ne);
		tS.toc();
		a += v;
		ASSERT_EQ(s, v);
		s = v;

		ptr[0] = real_t(-0.0); ptr[4] = real_t(-0.0);
		tN2.tic();
		v = vCountNonZeros_naive(ptr, ne);
		tN2.toc();
		a += v;
		ASSERT_EQ(s, v);
		s = v;

		tV2.tic();
		v = iM.vCountNonZeros(ptr, ne);
		tV2.toc();
		a += v;
		ASSERT_EQ(s, v);
		s = v;

		ptr[0] = real_t(0.0); ptr[4] = real_t(0.0);
		tS2.tic();
		v = iM.vCountNonZerosStrict(ptr, ne);
		tS2.toc();
		a += v;
		ASSERT_EQ(s, v);
		s = v;
	}
	tN.say("n1");
	tN2.say("n2");
	tV.say("v1");
	tV2.say("v2");
	tS.say("s1");
	tS2.say("s2");
	STDCOUTL(a);
}

TEST(TestPerfDecisions, vCountNonZeros) {
	NNTL_RUN_TEST2(10000, 1) test_vCountNonZeros_perf(i, 1);
	NNTL_RUN_TEST2(100000, 1) test_vCountNonZeros_perf(i, 1);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename real_t>
static bool isBinaryVec_rt(const real_t* ptr, const numel_cnt_t ne) noexcept {
	static_assert(::std::is_floating_point<real_t>::value, "");

	const auto pE = ptr + ne;
	int cond = 1;
	while (ptr != pE) {//err 500, doesn't vectorize
		const auto v = *ptr++;
		//doesn't catch negative zero, but it's ok for the test
		const int c = ((v == real_t(1.0)) | (v == real_t(0)));
		//NNTL_ASSERT(c || !"Not a binary vector!");
		cond = cond & c;
	}
	return !!cond;
}

template<typename real_t>
static bool isBinaryVec_st(const typename math::real_t_limits<real_t>::similar_FWI_t* ptr, const numel_cnt_t ne) noexcept {
	static_assert(::std::is_floating_point<real_t>::value, "");
	typedef typename math::real_t_limits<real_t>::similar_FWI_t similar_FWI_t;

	const auto _one = math::similar_FWI_one<real_t>();
	const auto _zero = math::similar_FWI_pos_zero<real_t>();

	const auto pE = ptr + ne;
	int cond = 1;
	while (ptr != pE) {//vectorizes
		const auto v = *ptr++;
		//we must make sure that binary zero is an actual unsigned(positive) zero
		const int c = ((v == _one) | (v == _zero));
		//NNTL_ASSERT(c || !"Not a binary vector!");
		cond = cond & c;
	}
	return !!cond;
}

template<typename real_t>
static bool isBinaryVec_stst(const typename math::real_t_limits<real_t>::similar_FWI_t* ptr, const numel_cnt_t ne) noexcept {
	static_assert(::std::is_floating_point<real_t>::value, "");
	typedef typename math::real_t_limits<real_t>::similar_FWI_t similar_FWI_t;

	const auto _one = math::similar_FWI_one<real_t>();
	const auto _zero = math::similar_FWI_pos_zero<real_t>();

	const auto pE = ptr + ne;
	similar_FWI_t cond = 1;
	while (ptr != pE) {//vectorizes
		const auto v = *ptr++;
		//we must make sure that binary zero is an actual unsigned(positive) zero
		const similar_FWI_t c = ((v == _one) | (v == _zero));
		//NNTL_ASSERT(c || !"Not a binary vector!");
		cond = cond & c;
	}
	return !!cond;
}

template<typename real_t>
void IsBinary_diffDataTypes_perf(const vec_len_t rowsCnt, const vec_len_t colsCnt) {
	typedef math::smatrix<real_t> realmtx_t;
	typedef math::real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
	typedef math::smatrix<similar_FWI_t> similar_FWI_mtx_t;

	typedef typename d_int_nI<real_t>::iRng_t iRng_t;
	typedef typename d_int_nI<real_t>::iMath_t iMath_t;

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("**** testing IsBinary() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

	realmtx_t rM(rowsCnt, colsCnt);
	similar_FWI_mtx_t sM(rowsCnt, colsCnt), sM2(rowsCnt, colsCnt);
	ASSERT_EQ(rM.byte_size(), sM.byte_size());

	iMath_t iM;
	iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	utils::tictoc tR, tS, tS2;
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iMath_t::iThreads_t> pw(iM.ithreads());

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(rM, real_t(2.));

		tR.tic();
		const auto vr = isBinaryVec_rt(rM.data(), dataSize);
		tR.toc();

		::std::memcpy(sM.data(), rM.data(), rM.byte_size());
		tS.tic();
		const auto vs = isBinaryVec_st<real_t>(sM.data(), dataSize);
		tS.toc();

		::std::memcpy(sM2.data(), rM.data(), rM.byte_size());
		tS2.tic();
		const auto vs2 = isBinaryVec_stst<real_t>(sM2.data(), dataSize);
		tS2.toc();

		ASSERT_EQ(vr, vs);
		ASSERT_EQ(vr, vs2);
	}
	tR.say("real_t");
	tS.say("int");
	tS2.say("size_t");
}


TEST(TestPerfDecisions, IsBinary_diffDataTypes) {
	IsBinary_diffDataTypes_perf<float>(1000, 100);
	IsBinary_diffDataTypes_perf<float>(1000, 1000);
	IsBinary_diffDataTypes_perf<float>(1000, 2000);

	IsBinary_diffDataTypes_perf<double>(1000, 10);
	IsBinary_diffDataTypes_perf<double>(1000, 100);
	IsBinary_diffDataTypes_perf<double>(1000, 200);
}

//////////////////////////////////////////////////////////////////////////
/*
void mExtractRowsByMask_baseline(const realmtx_t& src, realmtxdef_t& dest, const realmtx_t& mask, size_t*const pTmpIdxs)noexcept {
	NNTL_ASSERT(src.rows() == mask.rows());

	typedef ::nntl::math::real_t_limits<real_t>::similar_FWI_t similar_FWI_t;

	//making indexing vector
	const auto pM = reinterpret_cast<const similar_FWI_t*>(mask.data());
	const auto _one = ::nntl::math::similar_FWI_one<real_t>();
	const auto _zero = ::nntl::math::similar_FWI_pos_zero<real_t>();
	NNTL_ASSERT(_zero == 0);
	const size_t nRows = mask.rows();
	size_t ci = 0;
	for (size_t i = 0; i < nRows; ++i) {
		const auto m = pM[i];
		NNTL_ASSERT(m == _one || m == _zero);
		if (m) {
			pTmpIdxs[ci++] = i;
		}
	}

	NNTL_ASSERT(dest.cols() == src.cols());
	dest.deform_rows(static_cast<vec_len_t>(ci));

	iM.mExtractRows(src, pTmpIdxs, dest);
}


void test_mExtractRowsByMask_perfdec(real_t offProb, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing mExtractRowsByMask() variations over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize
		<< " elements). offProb = " << offProb << " **************");

	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	realmtx_t src(rowsCnt, colsCnt), mask(rowsCnt, 1);
	realmtxdef_t dest(rowsCnt, colsCnt), dest2(rowsCnt, colsCnt);
	::nntl::math::smatrix<size_t> maskIdxs(rowsCnt, 1);
	ASSERT_TRUE(!src.isAllocationFailed() && !dest.isAllocationFailed() && !mask.isAllocationFailed() && !maskIdxs.isAllocationFailed());

	d_interfaces::iRng_t rg;
	rg.init_ithreads(iM.ithreads());

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, imath_basic_t::iThreads_t> pw(iM.ithreads());

	utils::tictoc tBL, tSpec;

	real_t v = real_t(0);

	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_matrix(src, real_t(5));
		rg.gen_matrix_norm(mask);
		iM.ewBinarize_ip(mask, offProb, real_t(0), real_t(1));
		tBL.tic();
		mExtractRowsByMask_baseline(src, dest, mask, maskIdxs.data());
		tBL.toc();
		for (const auto e : dest) v += e;

		iM.mExtractRowsByMask(src, mask.data(), dest2);
		ASSERT_EQ(dest, dest2) << "mismatched results";

		rg.gen_matrix(src, real_t(5));
		rg.gen_matrix_norm(mask);
		iM.ewBinarize_ip(mask, offProb, real_t(0), real_t(1));
		tSpec.tic();
		iM.mExtractRowsByMask(src, mask.data(), dest2);
		tSpec.toc();
		for (const auto e : dest) v += e;
	}

	tBL.say("bl");
	tSpec.say("cust");
	STDCOUTL(v);
}

TEST(TestPerfDecisions, mExtractRowsByMask) {
	::std::array<real_t, 3> aMaskProb = { real_t(.99), real_t(.9), real_t(.5), };
	::std::array<vec_len_t, 3> aRows = { 100, 300, 1000 };
	::std::array<vec_len_t, 3> aCols = { 20, 70, 200 };

	for (const auto prob : aMaskProb) {
		for (const auto c : aCols) {
			for (const auto r : aRows) {
				test_mExtractRowsByMask_perfdec(real_t(1.) - prob, r, c);
			}
		}
	}
}
*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////// 

template<typename base_t> struct mMul_BLAS_EPS {};
template<> struct mMul_BLAS_EPS<double> { static constexpr double eps = 1e-10; };
template<> struct mMul_BLAS_EPS<float> { static constexpr float eps = 1e-6f; };

template<typename IntfT>
void mMul_BLAS_perf(typename IntfT::iMath_t& iM, typename IntfT::iRng_t& iR, vec_len_t batchSize, vec_len_t plnCnt
	, vec_len_t nCnt, numel_cnt_t CACHE_SIZE)
{
	typedef typename IntfT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;
	typedef math::smatrix_deform<real_t> realmtxdef_t;
	typedef typename IntfT::iMath_t::b_BLAS_t b_BLAS_t;

	auto bClog = CACHE_SIZE > 0; // dest.byte_size() < CACHE_SIZE;
	constexpr auto bClogOnlyTheRest = true;
	STDCOUTL("******* testing matrix multiplication performance in use-case of batchSize=" << batchSize
		<< ", pn=" << plnCnt << ", n=" << nCnt 
		<< (bClog ? " WITH cache clogging" : " without cache clogging") << " **************");

	constexpr unsigned maxReps = 150;
	realmtx_t prevActCur(batchSize, plnCnt), prevActT(true, batchSize, plnCnt);
	realmtx_t weightsCur(nCnt, plnCnt), weightsT(true, nCnt, plnCnt);
	realmtx_t actCur(batchSize, nCnt), actT(true, batchSize, nCnt);
	realmtx_t actCur2(batchSize, nCnt), actT2(true, batchSize, nCnt);

	realmtxdef_t cClog;
	realmtx_t cClogDest;

	if (bClog) {
		numel_cnt_t clogNumel = CACHE_SIZE / sizeof(real_t);
		if (bClogOnlyTheRest) clogNumel -= prevActCur.numel();

		if (clogNumel > 0) {
			cClog.resize(clogNumel);
			cClog.deform(static_cast<vec_len_t>(clogNumel), 1);
			cClogDest.resize(cClog.size());
			iR.gen_matrix(cClog, real_t(10));
		} else bClog = false;
	}

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, imath_basic_t::iThreads_t> pw(iM.ithreads());

	utils::tictoc t1WcPc2T, t2PcWc2c, t3WcPT2T, t4PTWc2c, t5PcWT2c, t6WTPc2T, t7PTWT2c, t8WTPT2T;
	real_t v = real_t(0.);

	iR.gen_matrix(prevActCur, real_t(2));
	iM.mTranspose_ignore_bias(prevActCur, prevActT);
	iR.gen_matrix(weightsCur, real_t(1));
	iM.mTranspose_ignore_bias(weightsCur, weightsT);

	const auto Eps = mMul_BLAS_EPS<real_t>::eps * plnCnt;
	//code warmup & sanity checks
	// current setup, i.e. for Pc[m,p] & Wc[n,p]
	// 1.   AT[n,m] = Wc*Pc'
	b_BLAS_t::gemm(false, true, actT.rows(), actT.cols(), weightsCur.cols(), real_t(1.), weightsCur.data(), weightsCur.ldimAsVecLen()
		, prevActCur.data(), prevActCur.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
	// 2.   Ac[m,n] = Pc*Wc'
	b_BLAS_t::gemm(false, true, actCur.rows(), actCur.cols(), prevActCur.cols(), real_t(1.), prevActCur.data(), prevActCur.ldimAsVecLen()
		, weightsCur.data(), weightsCur.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
	iM.mTranspose_ignore_bias(actT, actCur2);
	ASSERT_TRUE(actT.copy_to(actT2));
	ASSERT_REALMTX_NEAR(actCur, actCur2, "WTF?! Scenario 1!=2", Eps);
	// PT[p,m] & Wc[n,p]
	// 3.   AT[n,m] = Wc*PT
	b_BLAS_t::gemm(false, false, actT.rows(), actT.cols(), weightsCur.cols(), real_t(1.), weightsCur.data(), weightsCur.ldimAsVecLen()
		, prevActT.data(), prevActT.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
	ASSERT_REALMTX_NEAR(actT, actT2, "WTF?! Wrong scenario 3", Eps);
	// 4.   Ac[m,n] = PT'*Wc'
	b_BLAS_t::gemm(true, true, actCur.rows(), actCur.cols(), prevActT.rows(), real_t(1.), prevActT.data(), prevActT.ldimAsVecLen()
		, weightsCur.data(), weightsCur.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
	ASSERT_REALMTX_NEAR(actCur, actCur2, "WTF?! Wrong scenario 4", Eps);
	// Pc[m,p] & WT[p,n]
	// 5.   Ac[m,n] = Pc*WT
	b_BLAS_t::gemm(false, false, actCur.rows(), actCur.cols(), prevActCur.cols(), real_t(1.), prevActCur.data(), prevActCur.ldimAsVecLen()
		, weightsT.data(), weightsT.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
	ASSERT_REALMTX_NEAR(actCur, actCur2, "WTF?! Wrong scenario 5", Eps);
	// 6.   AT[n,m] = WT'*Pc'
	b_BLAS_t::gemm(true, true, actT.rows(), actT.cols(), weightsT.rows(), real_t(1.), weightsT.data(), weightsT.ldimAsVecLen()
		, prevActCur.data(), prevActCur.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
	ASSERT_REALMTX_NEAR(actT, actT2, "WTF?! Wrong scenario 6", Eps);
	// PT[p,m] & WT[p,n]
	// 7.   Ac[m,n] = PT'*WT
	b_BLAS_t::gemm(true, false, actCur.rows(), actCur.cols(), prevActT.rows(), real_t(1.), prevActT.data(), prevActT.ldimAsVecLen()
		, weightsT.data(), weightsT.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
	ASSERT_REALMTX_NEAR(actCur, actCur2, "WTF?! Wrong scenario 7", Eps);
	// 8.   AT[n,m] = WT'*PT
	b_BLAS_t::gemm(true, false, actT.rows(), actT.cols(), weightsT.rows(), real_t(1.), weightsT.data(), weightsT.ldimAsVecLen()
		, prevActT.data(), prevActT.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
	ASSERT_REALMTX_NEAR(actT, actT2, "WTF?! Wrong scenario 8", Eps);
	
	actCur2.clear(); actT2.clear();

	for (unsigned r = 0; r < maxReps; ++r) {
		// current setup, i.e. for Pc[m,p] & Wc[n,p]
		// 1.   AT[n,m] = Wc*Pc'
		iR.gen_matrix(prevActCur, real_t(2)); 		iR.gen_matrix(weightsCur, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t1WcPc2T.tic();
		b_BLAS_t::gemm(false, true, actT.rows(), actT.cols(), weightsCur.cols(), real_t(1.), weightsCur.data(), weightsCur.ldimAsVecLen()
			, prevActCur.data(), prevActCur.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
		t1WcPc2T.toc();
		for (const auto e : actT) v += e;
		
		// 2.   Ac[m,n] = Pc*Wc'
		iR.gen_matrix(prevActCur, real_t(2)); 		iR.gen_matrix(weightsCur, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t2PcWc2c.tic();
		b_BLAS_t::gemm(false, true, actCur.rows(), actCur.cols(), prevActCur.cols(), real_t(1.), prevActCur.data(), prevActCur.ldimAsVecLen()
			, weightsCur.data(), weightsCur.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
		t2PcWc2c.toc();
		for (const auto e : actCur) v += e;

		// PT[p,m] & Wc[n,p]
		// 3.   AT[n,m] = Wc*PT
		iR.gen_matrix(prevActT, real_t(2)); 		iR.gen_matrix(weightsCur, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t3WcPT2T.tic();
		b_BLAS_t::gemm(false, false, actT.rows(), actT.cols(), weightsCur.cols(), real_t(1.), weightsCur.data(), weightsCur.ldimAsVecLen()
			, prevActT.data(), prevActT.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
		t3WcPT2T.toc();
		for (const auto e : actT) v += e;

		// 4.   Ac[m,n] = PT'*Wc'
		iR.gen_matrix(prevActT, real_t(2)); 		iR.gen_matrix(weightsCur, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t4PTWc2c.tic();
		b_BLAS_t::gemm(true, true, actCur.rows(), actCur.cols(), prevActT.rows(), real_t(1.), prevActT.data(), prevActT.ldimAsVecLen()
			, weightsCur.data(), weightsCur.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
		t4PTWc2c.toc();
		for (const auto e : actCur) v += e;
		
		// Pc[m,p] & WT[p,n]
		// 5.   Ac[m,n] = Pc*WT
		iR.gen_matrix(prevActCur, real_t(2)); 		iR.gen_matrix(weightsT, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t5PcWT2c.tic();
		b_BLAS_t::gemm(false, false, actCur.rows(), actCur.cols(), prevActCur.cols(), real_t(1.), prevActCur.data(), prevActCur.ldimAsVecLen()
			, weightsT.data(), weightsT.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
		t5PcWT2c.toc();
		for (const auto e : actCur) v += e;

		// 6.   AT[n,m] = WT'*Pc'
		iR.gen_matrix(prevActCur, real_t(2)); 		iR.gen_matrix(weightsT, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t6WTPc2T.tic();
		b_BLAS_t::gemm(true, true, actT.rows(), actT.cols(), weightsT.rows(), real_t(1.), weightsT.data(), weightsT.ldimAsVecLen()
			, prevActCur.data(), prevActCur.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
		t6WTPc2T.toc();
		for (const auto e : actT) v += e;

		// PT[p,m] & WT[p,n]
		// 7.   Ac[m,n] = PT'*WT
		iR.gen_matrix(prevActT, real_t(2)); 		iR.gen_matrix(weightsT, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t7PTWT2c.tic();
		b_BLAS_t::gemm(true, false, actCur.rows(), actCur.cols(), prevActT.rows(), real_t(1.), prevActT.data(), prevActT.ldimAsVecLen()
			, weightsT.data(), weightsT.ldimAsVecLen(), real_t(0), actCur.data(), actCur.ldimAsVecLen());
		t7PTWT2c.toc();
		for (const auto e : actCur) v += e;
		
		// 8.   AT[n,m] = WT'*PT
		iR.gen_matrix(prevActT, real_t(2)); 		iR.gen_matrix(weightsT, real_t(1));
		if (bClog) { ASSERT_TRUE(cClog.copy_to(cClogDest));			for (const auto e : cClogDest) v += r*e; }
		t8WTPT2T.tic();
		b_BLAS_t::gemm(true, false, actT.rows(), actT.cols(), weightsT.rows(), real_t(1.), weightsT.data(), weightsT.ldimAsVecLen()
			, prevActT.data(), prevActT.ldimAsVecLen(), real_t(0), actT.data(), actT.ldimAsVecLen());
		t8WTPT2T.toc();
		for (const auto e : actT) v += e;
	}

	STDCOUTL("current setup, i.e. for Pc[m,p] & Wc[n,p]");
	t1WcPc2T.say("1. AT[n,m] = Wc*Pc' ");
	t2PcWc2c.say("2. Ac[m,n] = Pc*Wc' ");
	
	STDCOUTL("PT[p,m] & Wc[n,p]");
	t3WcPT2T.say("3. AT[n,m] = Wc*PT  ");
	t4PTWc2c.say("4. Ac[m,n] = PT'*Wc'");
	
	STDCOUTL("Pc[m,p] & WT[p,n]");
	t5PcWT2c.say("5. Ac[m,n] = Pc*WT  ");
	t6WTPc2T.say("6. AT[n,m] = WT'*Pc'");
	
	STDCOUTL("PT[p,m] & WT[p,n]");
	t7PTWT2c.say("7. Ac[m,n] = PT'*WT ");
	t8WTPT2T.say("8. AT[n,m] = WT'*PT ");

	STDCOUTL(v);	
}

TEST(TestPerfDecisions, mMulBlasPerf) {
	//typedef float real_t;
	typedef dt_interfaces<real_t> myInterfaces_t;

	//typename myInterfaces_t::iMath_t iM;
	//const vec_len_t g_MinDataSizeDelta = 2 * iM.ithreads().workers_count() + 2;

	const numel_cnt_t CACHE_SIZE = 6 * 1024 * 1024;

	STDCOUTL("sizeof(real_t) = " << sizeof(real_t));

	typename myInterfaces_t::iRng_t iR;
	iR.init_ithreads(iM.ithreads());

#ifdef TESTS_SKIP_LONGRUNNING
	ASSERT_NO_FATAL_FAILURE(mMul_BLAS_perf<myInterfaces_t>(iM, iR, 128, 20, 10, 0));
	ASSERT_NO_FATAL_FAILURE(mMul_BLAS_perf<myInterfaces_t>(iM, iR, 128, 20, 10, CACHE_SIZE));
#else
	ASSERT_NO_FATAL_FAILURE(mMul_BLAS_perf<myInterfaces_t>(iM, iR, 4096, 256, 128, 0));
	ASSERT_NO_FATAL_FAILURE(mMul_BLAS_perf<myInterfaces_t>(iM, iR, 4096, 256, 128, CACHE_SIZE));

	/*::std::vector<vec_len_t> batchSizes = { 128, 256, 512, 4096, 4096*4, 4096 * 32 , 4096 * 1024 };
	::std::vector<vec_len_t> prevSize = { 32, 64, 128, 256, 512, 1024, 2048 };
	::std::vector<vec_len_t> thisSize = { 32, 64, 128, 256, 512, 1024, 2048 };
	constexpr numel_cnt_t maxSize = numel_cnt_t(32) * 4096 * 1024;
	for(const auto bs :batchSizes){
		for (const auto n : thisSize) {
			for (const auto pn : prevSize) {
				if (math::smatrix_td::sNumel(bs, ::std::max(pn, n)) <= maxSize) {
					ASSERT_NO_FATAL_FAILURE(mMul_BLAS_perf<myInterfaces_t>(iM, iR, bs, pn, n, 0));
					//ASSERT_NO_FATAL_FAILURE(mMul_BLAS_perf<myInterfaces_t>(iM, iR, bs, pn, n, CACHE_SIZE));
				} else {
					STDCOUTL("-- skipping batchSize=" << bs << ", n=" << n << ", pn=" << pn);
				}				
			}
		}
	}*/
#endif //TESTS_SKIP_LONGRUNNING
}