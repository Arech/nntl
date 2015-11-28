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

#include "../nntl/interface/math/i_yeppp_openblas.h"
#include "../nntl/nnet_def_interfaces.h"

#include <array>
#include "../nntl/utils/chrono.h"

#include "../nntl/utils/prioritize_workers.h"


using namespace nntl;
using namespace std::chrono;
using floatmtx_t = math_types::floatmtx_ty;
using float_t_ = floatmtx_t::value_type;
using vec_len_t = floatmtx_t::vec_len_t;
using numel_cnt_t = floatmtx_t::numel_cnt_t;

//////////////////////////////////////////////////////////////////////////
#ifdef _DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
#endif // _DEBUG

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
	typedef math::i_Yeppp_OpenBlas<def_threads_t> i_Y_OB;
	i_Y_OB iM;

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
	typedef math::i_Yeppp_OpenBlas<def_threads_t> i_Y_OB;
	i_Y_OB iM;

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
	typedef math::i_Yeppp_OpenBlas<def_threads_t> i_Y_OB;

	i_Y_OB iM;

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
	typedef math::i_Yeppp_OpenBlas<def_threads_t> i_Y_OB;

	i_Y_OB iM;

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
	typedef math::i_Yeppp_OpenBlas<def_threads_t> i_Y_OB;

	i_Y_OB iM;

	//for (unsigned i = 100; i <= 140; i+=1) test_dropout_perf(iM, i,100);

	test_dropout_perf(iM, 100, 10);
#ifndef TESTS_SKIP_LONGRUNNING
	test_dropout_perf(iM, 1000);
	test_dropout_perf(iM, 10000);
	test_dropout_perf(iM, 100000);
#endif
}
//////////////////////////////////////////////////////////////////////////