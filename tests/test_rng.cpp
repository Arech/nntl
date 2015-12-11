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

#include "../nntl/interface/rng/std.h"
#include "../nntl/interface/rng/AFRandom.h"
#include "../nntl/interface/rng/AFRandom_mt.h"

#include "../nntl/nnet_def_interfaces.h"

#include "../nntl/utils/chrono.h"
#include "../nntl/utils/prioritize_workers.h"

using namespace nntl;
using namespace nntl::utils;

#ifdef NNTL_DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
#endif // NNTL_DEBUG

void test_rng_perf(math_types::realmtx_ty::vec_len_t rowsCnt, math_types::realmtx_ty::vec_len_t colsCnt = 10) {
	typedef math_types::realmtx_ty realmtx_t;
	typedef realmtx_t::value_type real_t;
	typedef realmtx_t::numel_cnt_t numel_cnt_t;
	
	using namespace std::chrono;
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
		rng::Std rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("Std:\t" << utils::duration_readable(diff, maxReps, &tstStd));*/

	{
		rng::AFRandom<Agner_Fog::CRandomMersenne> rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("AFMersenne:\t" << utils::duration_readable(diff, maxReps, &tstAFMersenne));

	{
		rng::AFRandom<Agner_Fog::CRandomSFMT0> rg;
		bt = steady_clock::now();
		for (unsigned r = 0; r < maxReps; ++r) {
			rg.gen_matrix_norm(m);
		}
		diff = steady_clock::now() - bt;
	}
	STDCOUTL("AFSFMT0:\t" << utils::duration_readable(diff, maxReps, &tstAFSFMT0));

	{
		rng::AFRandom<Agner_Fog::CRandomSFMT1> rg;
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
template<typename AFRng, typename iThreads>
void test_rngmt(iThreads&iT, math_types::realmtx_ty& m) {
	using namespace std::chrono;
	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = 5*TEST_PERF_REPEATS_COUNT;

	auto ptr = m.dataAsVec();
	auto dataCnt = m.numel();

	rng::AFRandom_mt<AFRng, iThreads> rg(iT);

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_vector_norm_st(ptr,dataCnt);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("st:\t" << utils::duration_readable(diff, maxReps));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_vector_norm_mt(ptr, dataCnt);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("mt:\t" << utils::duration_readable(diff, maxReps));

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		rg.gen_vector_norm(ptr, dataCnt);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("best\t" << utils::duration_readable(diff, maxReps));
}

/*
template<typename iThreads>
void test_rng_mt_perf(iThreads& iT, math_types::realmtx_ty::vec_len_t rowsCnt, math_types::realmtx_ty::vec_len_t colsCnt = 10) {
	typedef math_types::realmtx_ty realmtx_t;
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing multithreaded rng performance over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iThreads> pw(iT);
	STDCOUTL("AFMersenne:");
	test_rngmt<Agner_Fog::CRandomMersenne, iThreads>(iT, m);
	STDCOUTL("AFSFMT0:");
	test_rngmt<Agner_Fog::CRandomSFMT0, iThreads>(iT, m);
	STDCOUTL("AFSFMT1:");
	test_rngmt<Agner_Fog::CRandomSFMT1, iThreads>(iT, m);
}*/
template<typename iRng, typename iThreads>
void test_rng_mt_perf(iThreads& iT, char* pName, math_types::realmtx_ty::vec_len_t rowsCnt, math_types::realmtx_ty::vec_len_t colsCnt = 10) {
	typedef math_types::realmtx_ty realmtx_t;
	const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
	STDCOUTL("******* testing multithreaded "<< pName<<	" performance over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) **************");
	realmtx_t m(rowsCnt, colsCnt);
	ASSERT_TRUE(!m.isAllocationFailed());
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, iThreads> pw(iT);
	test_rngmt<iRng, iThreads>(iT, m);
}

TEST(TestRNG, RngMtPerf) {
	typedef math_types::real_ty real_t;
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	def_threads_t Thr;

	NNTL_RUN_TEST2( (rng::_impl::AFRandom_mt_thresholds<Agner_Fog::CRandomMersenne, real_t>::bnd_gen_vector_norm), 10)
		test_rng_mt_perf<Agner_Fog::CRandomMersenne>(Thr, "AFMersenne", i, 10);
	NNTL_RUN_TEST2( (rng::_impl::AFRandom_mt_thresholds<Agner_Fog::CRandomSFMT0, real_t>::bnd_gen_vector_norm), 10)
		test_rng_mt_perf<Agner_Fog::CRandomSFMT0>(Thr, "AFSFMT0", i, 10);
	NNTL_RUN_TEST2( (rng::_impl::AFRandom_mt_thresholds<Agner_Fog::CRandomSFMT1, real_t>::bnd_gen_vector_norm), 10)
		test_rng_mt_perf<Agner_Fog::CRandomSFMT1>(Thr, "AFSFMT1", i, 10);
}