/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "../nntl/interface/rng/cstd.h"
#include "../nntl/interface/rng/afrand.h"
#include "../nntl/interface/rng/afrand_mt.h"

#include "../nntl/interfaces.h"

#include "../nntl/utils/chrono.h"
#include "../nntl/utils/tictoc.h"

#include "../nntl/interface/rng/distr_normal_naive.h"

#pragma warning(push,3)
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#pragma warning(pop)

using namespace nntl;
using namespace nntl::utils;

typedef d_interfaces::real_t real_t;
typedef math::smatrix<real_t> realmtx_t;

#ifdef NNTL_DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
#endif // NNTL_DEBUG

void test_rng_perf(vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	//typedef realmtx_t::numel_cnt_t numel_cnt_t;
	
	using namespace ::std::chrono;
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing rng performance over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");

	//double tstStd;
	double tstAFMersenne, tstAFSFMT0, tstAFSFMT1;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
	
	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());

	/*
	 *turning it off because it works more than 100 slower
	{
		rng::CStd rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("CStd:\t" << utils::duration_readable(diff, maxReps, &tstStd));*/

	{
		rng::AFRand<real_t, AFog::CRandomMersenne> rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("AFMersenne:\t" << utils::duration_readable(diff, maxReps, &tstAFMersenne));

	{
		rng::AFRand<real_t, AFog::CRandomSFMT0> rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("AFSFMT0:\t" << utils::duration_readable(diff, maxReps, &tstAFSFMT0));

	{
		rng::AFRand<real_t, AFog::CRandomSFMT1> rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("AFSFMT1:\t" << utils::duration_readable(diff, maxReps, &tstAFSFMT1));
}

TEST(TestRNG, RngPerf) {	
	//for (unsigned i = 100; i <= 140; i+=1) test_rng_perf(i,100);

	test_rng_perf(1000);
#ifndef TESTS_SKIP_LONGRUNNING
	test_rng_perf(10000);
	//test_rng_perf(100000);
	//test_rng_perf(1000000);
#endif
}

//////////////////////////////////////////////////////////////////////////
template<typename AFRng, typename iThreadsT>
void test_rngmt(iThreadsT&iT, realmtx_t& m) {
	constexpr unsigned maxReps = 5*TEST_PERF_REPEATS_COUNT;

	real_t g = real_t(0);
	auto ptr = m.data();
	auto dataCnt = m.numel();
	utils::tictoc tS, tM, tB;
	rng::AFRand_mt<real_t, AFRng, iThreadsT> rg(iT);

	for (unsigned r = 0; r < maxReps; ++r) {
		tS.tic();
		rg.gen_vector_norm_st(ptr, dataCnt);
		tS.toc();
		for (const auto& e : m) g += e;
		g = ::std::log(::std::abs(g));

		tM.tic();
		rg.gen_vector_norm_mt(ptr, dataCnt);
		tM.toc();
		for (const auto& e : m) g += e;
		g = ::std::log(::std::abs(g));

		tB.tic();
		rg.gen_vector_norm(ptr, dataCnt);
		tB.toc();
		for (const auto& e : m) g += e;
		g = ::std::log(::std::abs(g));
	}
	tS.say("st");
	tM.say("mt");
	tB.say("()");
	STDCOUTL(g);
}

template<typename iRng, typename iThreadsT>
void test_rng_mt_perf(iThreadsT& iT, char* pName, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing multithreaded "<< pName<<	" performance over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iThreadsT> pw(iT);
	test_rngmt<iRng, iThreadsT>(iT, m);
}

TEST(TestRNG, RngMtPerf) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	def_threads_t Thr;

	NNTL_RUN_TEST2( (rng::_impl::AFRAND_MT_THR<AFog::CRandomMersenne, real_t>::bnd_gen_vector_norm), 10)
		test_rng_mt_perf<AFog::CRandomMersenne>(Thr, "AFMersenne", i, 10);
	NNTL_RUN_TEST2( (rng::_impl::AFRAND_MT_THR<AFog::CRandomSFMT0, real_t>::bnd_gen_vector_norm), 10)
		test_rng_mt_perf<AFog::CRandomSFMT0>(Thr, "AFSFMT0", i, 10);
	NNTL_RUN_TEST2( (rng::_impl::AFRAND_MT_THR<AFog::CRandomSFMT1, real_t>::bnd_gen_vector_norm), 10)
		test_rng_mt_perf<AFog::CRandomSFMT1>(Thr, "AFSFMT1", i, 10);
}

//////////////////////////////////////////////////////////////////////////
template<typename base_t> struct NormDistrCompat_EPS {};
template<> struct NormDistrCompat_EPS<double> { static constexpr double eps = 1e-4; };
template<> struct NormDistrCompat_EPS<float> { static constexpr float eps = 5e-2f; };
TEST(TestRNG, NormDistrCompat) {
	typedef d_interfaces::iRng_t iRng_t;
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef ::std::vector<real_t> vec_t;

	static constexpr real_t targMean(real_t(.5)), targStddev(real_t(2.));
	static constexpr int maxReps = 5, totalElms = 1000000;

	vec_t dest(totalElms);

	def_threads_t Thr;
	iRng_t iR(Thr);
	rng::distr_normal_naive<iRng_t> d(iR, targMean, targStddev);

	for (int i = 0; i < maxReps; ++i) {
		::std::fill(dest.begin(), dest.end(), real_t(0.));
		d.gen_vector(&dest.front(), totalElms);

		//see http://www.boost.org/doc/libs/1_63_0/doc/html/accumulators/user_s_guide.html

		::boost::accumulators::accumulator_set<real_t, ::boost::accumulators::stats<
			::boost::accumulators::tag::mean
			, ::boost::accumulators::tag::lazy_variance >
		> acc;
		for (const auto& v : dest) {
			acc(v);
		}
		const real_t _mean = ::boost::accumulators::extract_result< ::boost::accumulators::tag::mean >(acc)
			, _std = ::std::sqrt(::boost::accumulators::extract_result< ::boost::accumulators::tag::lazy_variance >(acc));
		STDCOUTL("Mean = " << _mean << ", std = " << _std);
		ASSERT_NEAR(_mean, targMean, NormDistrCompat_EPS<real_t>::eps) << "Wrong mean!!!";
		ASSERT_NEAR(_std, targStddev, NormDistrCompat_EPS<real_t>::eps) << "Wrong StdDev!!!";
	}
}

TEST(TestRNG, normal_vector_Compat) {
	typedef d_interfaces::iRng_t iRng_t;
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	typedef ::std::vector<real_t> vec_t;

	static constexpr real_t targMean(real_t(.5)), targStddev(real_t(2.));
	static constexpr unsigned maxReps = 5, totalElms = 1000000;

	vec_t dest(totalElms);

	def_threads_t Thr;
	iRng_t iR(Thr);
	//rng::distr_normal_naive<iRng_t> d(iR, targMean, targStddev);

	for (unsigned i = 0; i < maxReps; ++i) {
		::std::fill(dest.begin(), dest.end(), real_t(0.));
		iR.normal_vector(&dest.front(), totalElms, targMean, targStddev);

		::boost::accumulators::accumulator_set<real_t, ::boost::accumulators::stats<
			::boost::accumulators::tag::mean
			, ::boost::accumulators::tag::lazy_variance >
		> acc;
		for (const auto& v : dest) {
			acc(v);
		}
		const real_t _mean = ::boost::accumulators::extract_result< ::boost::accumulators::tag::mean >(acc)
			, _std = ::std::sqrt(::boost::accumulators::extract_result< ::boost::accumulators::tag::lazy_variance >(acc));
		STDCOUTL("Mean = " << _mean << ", std = " << _std);
		ASSERT_NEAR(_mean, targMean, NormDistrCompat_EPS<real_t>::eps) << "Wrong mean!!!";
		ASSERT_NEAR(_std, targStddev, NormDistrCompat_EPS<real_t>::eps) << "Wrong StdDev!!!";
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename iRng, typename iThreadsT>
void test_bernoulli_perf(iThreadsT& iT, char* pName, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing " << pName << ".bernoulli_vector performance over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());	
	constexpr unsigned maxReps = 5 * TEST_PERF_REPEATS_COUNT;

	const auto p = real_t(.6), pv = real_t(1.), nv = real_t(0);
	real_t g = real_t(0);
	auto ptr = m.data();
	auto dataCnt = m.numel();
	utils::tictoc tS, tM, tB;
	rng::AFRand_mt<real_t, iRng, iThreadsT> rg(iT);

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iThreadsT> pw(iT);
	for (unsigned r = 0; r < maxReps; ++r) {
		tS.tic();
		rg.bernoulli_vector_st(ptr, dataCnt, p, pv, nv);
		tS.toc();
		for (const auto& e : m) g += e;
		g = ::std::log(g);

		tM.tic();
		rg.bernoulli_vector_mt(ptr, dataCnt, p, pv, nv);
		tM.toc();
		for (const auto& e : m) g += e;
		g = ::std::log(g);

		tB.tic();
		rg.bernoulli_vector(ptr, dataCnt, p, pv, nv);
		tB.toc();
		for (const auto& e : m) g += e;
		g = ::std::log(g);
	}
	tS.say("st");
	tM.say("mt");
	tB.say("()");
	STDCOUTL(g);
}

TEST(TestRNG, BernoulliVectorPerf) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	def_threads_t Thr;

	NNTL_RUN_TEST2((rng::_impl::AFRAND_MT_THR<AFog::CRandomMersenne, real_t>::bnd_bernoulli_vector), 10)
		test_bernoulli_perf<AFog::CRandomMersenne>(Thr, "AFMersenne", i, 10);
	NNTL_RUN_TEST2((rng::_impl::AFRAND_MT_THR<AFog::CRandomSFMT0, real_t>::bnd_bernoulli_vector), 10)
		test_bernoulli_perf<AFog::CRandomSFMT0>(Thr, "AFSFMT0", i, 10);
	NNTL_RUN_TEST2((rng::_impl::AFRAND_MT_THR<AFog::CRandomSFMT1, real_t>::bnd_bernoulli_vector), 10)
		test_bernoulli_perf<AFog::CRandomSFMT1>(Thr, "AFSFMT1", i, 10);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename iRng, typename iThreadsT>
void test_normal_perf(iThreadsT& iT, char* pName, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing " << pName << ".normal_vector performance over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	realmtx_t M(rowsCnt, colsCnt);
	ASSERT_TRUE(!M.isAllocationFailed());
	constexpr unsigned maxReps = 5 * TEST_PERF_REPEATS_COUNT;

	const auto m = real_t(.0), st = real_t(1.);
	real_t g = real_t(0);
	auto ptr = M.data();
	auto dataCnt = M.numel();
	utils::tictoc tS, tM, tB;
	rng::AFRand_mt<real_t, iRng, iThreadsT> rg(iT);

	threads::prioritize_workers<threads::PriorityClass::PerfTesting, iThreadsT> pw(iT);
	for (unsigned r = 0; r < maxReps; ++r) {
		tS.tic();
		rg.normal_vector_st(ptr, dataCnt, m, st);
		tS.toc();
		for (const auto& e : M) g += e;
		g = ::std::log(::std::abs(g));

		tM.tic();
		rg.normal_vector_mt(ptr, dataCnt, m, st);
		tM.toc();
		for (const auto& e : M) g += e;
		g = ::std::log(::std::abs(g));

		tB.tic();
		rg.normal_vector(ptr, dataCnt, m, st);
		tB.toc();
		for (const auto& e : M) g += e;
		g = ::std::log(::std::abs(g));
	}
	tS.say("st");
	tM.say("mt");
	tB.say("()");
	STDCOUTL(g);
}

TEST(TestRNG, normal_vector_perf) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	def_threads_t Thr;

	NNTL_RUN_TEST2((rng::_impl::AFRAND_MT_THR<AFog::CRandomMersenne, real_t>::bnd_normal_vector), 10)
		test_normal_perf<AFog::CRandomMersenne>(Thr, "AFMersenne", i, 10);
	NNTL_RUN_TEST2((rng::_impl::AFRAND_MT_THR<AFog::CRandomSFMT0, real_t>::bnd_normal_vector), 10)
		test_normal_perf<AFog::CRandomSFMT0>(Thr, "AFSFMT0", i, 10);
	NNTL_RUN_TEST2((rng::_impl::AFRAND_MT_THR<AFog::CRandomSFMT1, real_t>::bnd_normal_vector), 10)
		test_normal_perf<AFog::CRandomSFMT1>(Thr, "AFSFMT1", i, 10);
}