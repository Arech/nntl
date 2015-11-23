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
#include "../nntl/nntl.h"
#include "../nntl/interface/rng/std.h"
#include "../nntl/interface/threads/winqdu.h"
#include "../nntl/interface/threads/std.h"
// #include "../nntl/interface/threads/winqduc.h"
// #include "../nntl/interface/threads/stdc.h"
// #include "../nntl/interface/threads/winqduc2.h"
// #include "../nntl/interface/threads/stdc2.h"
//#include "../nntl/interface/threads/intel_tbb.h"
#include "../nntl/nnet_def_interfaces.h"
#include "../nntl/utils/chrono.h"

//#define BOOST_USE_WINDOWS_H
//#include <boost/thread/barrier.hpp>

using namespace nntl;

#ifdef TESTS_SKIP_LONGRUNNING
#define TESTS_SKIP_THREADING_DELAYS
#define TESTS_SKIP_THREADING_PERFS
#endif

#ifndef TESTS_SKIP_THREADING_DELAYS
#define TESTS_SKIP_THREADING_DELAYS
#endif

//////////////////////////////////////////////////////////////////////////
template<typename TT>
void threads_basics_test(TT& t) {
	typedef TT::range_t range_t;
	typedef TT::par_range_t par_range_t;
	typedef math_types::floatmtx_ty::vec_len_t vec_len_t;
	typedef math_types::floatmtx_ty::value_type float_t_;

	const auto workersCnt = t.workers_count();

	const vec_len_t maxCnt = 2 * workersCnt;
	math_types::floatmtx_ty m(1, maxCnt + 1);
	ASSERT_TRUE(!m.isAllocationFailed());
	auto ptr = m.dataAsVec();

	for (range_t mnumel = 1; mnumel <= maxCnt; ++mnumel) {
		const auto maxCntPerWorker = static_cast<range_t>(ceil(static_cast<double>(mnumel) / workersCnt));

		//////////////////////////////////////////////////////////////////////////
		// RUN
		for (range_t i = 0; i <= mnumel; ++i) ptr[i] = -5;

		t.run([=](const par_range_t& r) {
			const auto cnt = r.cnt(), ofs=r.offset();
			EXPECT_TRUE(cnt <= maxCntPerWorker) << "cnt==" << cnt << " maxCntPerWorker==" << maxCntPerWorker;
			for (range_t i = 0; i < cnt; ++i) ptr[ofs+i] = -ptr[ofs+i];
		}, mnumel);

		for (range_t i = 0; i < mnumel; ++i)
			ASSERT_DOUBLE_EQ(ptr[i], 5) << "** unexpected element at offset " << i;
		ASSERT_DOUBLE_EQ(ptr[mnumel], -5) << "** guarding element changed at offset " << mnumel;

		//////////////////////////////////////////////////////////////////////////
		// REDUCE
		for (range_t i = 0; i < mnumel; ++i) ptr[i] = 1;

		//compute sum of squares
		auto lRed = [maxCntPerWorker,ptr](const par_range_t& r)->float_t_
		{
			const auto cnt = r.cnt();
			EXPECT_TRUE(cnt <= maxCntPerWorker) << "cnt=="<<cnt<<" maxCntPerWorker=="<< maxCntPerWorker;
			float_t_ ret = 0;
			for (range_t i = 0; i < cnt; ++i) ret += ptr[i] * ptr[i];
			return ret;
		};
		auto lRedF = [workersCnt](const float_t_* p, const range_t cnt)->float_t_
		{
			EXPECT_TRUE(cnt <= workersCnt) << "cnt=="<<cnt<<" workersCnt=="<< workersCnt;
			float_t_ ret = 0;
			for (range_t i = 0; i < cnt; ++i) ret += p[i];
			return ret;
		};
		float_t_ redRes = t.reduce(lRed, lRedF, mnumel);
		ASSERT_DOUBLE_EQ(redRes, static_cast<float_t_>(mnumel));
	}
}

TEST(TestThreading, WinQDUBasics) {
	threads::WinQDU<math_types::floatmtx_ty::numel_cnt_t> t;
	threads_basics_test(t);
	threads_basics_test(t);
}

TEST(TestThreading, StdBasics) {
	threads::Std<math_types::floatmtx_ty::numel_cnt_t> t;
	threads_basics_test(t);
	threads_basics_test(t);
}

#ifndef TESTS_SKIP_THREADING_PERFS

TEST(TestThreading, PerfComparision) {
	using namespace std::chrono;

	typedef math_types::floatmtx_ty::numel_cnt_t numel_cnt_t;
	STDCOUTL("The test may require a few seconds to complete. Define TESTS_SKIP_THREADING_PERFS to skip.");

	constexpr uint64_t maxreps = 200000;
	EXPECT_TRUE(steady_clock::is_steady);
	double tWinQDU, tStd;

	{
		typedef threads::WinQDU<numel_cnt_t> thr;
		typedef thr::par_range_t par_range_t;

		thr wint;
		//v is used to make sure compiler doesn't optimize away all machinery. Also it almost doesn't change time ratio between
		//WinQDU and Std in debug and release and almost doesn't change absolute times in release.
		std::atomic_ptrdiff_t v = 0;

		auto bt = steady_clock::now();
		for (uint64_t i = 0; i < maxreps; ++i) {
			wint.run([&](const par_range_t&) {
				v++;
			}, par_range_t(100000000) );
		}
		auto diff = steady_clock::now() - bt;
		STDCOUTL("WinQDU:\t\t" << utils::duration_readable(diff, maxreps, &tWinQDU) << ",\t\t" << v << " incs)");
	}
	{
		typedef threads::Std<numel_cnt_t> thr;
		typedef thr::par_range_t par_range_t;
		thr stdt;
		std::atomic_ptrdiff_t v = 0;

		auto bt = steady_clock::now();
		for (uint64_t i = 0; i < maxreps; ++i) {
			stdt.run([&](const par_range_t&) {
				v++;
			}, par_range_t(100000000));
		}
		auto diff = steady_clock::now() - bt;
		STDCOUTL("Std:\t\t" << utils::duration_readable(diff, maxreps, &tStd) << ",\t\t" << v << " incs)");
	}
	

	STDCOUTL("threads::WinQDU  is " << std::setprecision(3) << tStd/tWinQDU << " times faster than threads::Std");
}

#endif // !TESTS_SKIP_THREADING_PERFS


#ifndef TESTS_SKIP_THREADING_DELAYS

template <typename TT>
void threading_delay_test(TT& t, double m=1) {
	//no need to make it significantly smaller because of resolution of Windows sleep timer
	const rng::Std::generated_scalar_t max_mks = 5000;
	//but this count should be large enough
	const uint64_t maxreps = 10000;

	STDCOUTL("The test would require a bit less than " << static_cast<uint64_t>(m*max_mks*maxreps / 1000000) << "s"
		<<(m>1?" (have no idea why is it longer)":"")
		<<".\nIf it lasts significantly longer, then the app hangs and test failed. Define TESTS_SKIP_THREADING_DELAYS to skip.");
	
	rng::Std r;
	for (uint64_t i = 0; i < maxreps; ++i) {
		t.run([=, &r](const TT::par_range_t&) {
			std::this_thread::sleep_for(std::chrono::microseconds(r.gen_i_scalar(max_mks)));
		}, TT::par_range_t(100000000));
	}
}

TEST(TestThreading, WinQDUDelays) {
	threads::WinQDU<math_types::floatmtx_ty::numel_cnt_t> t;
	threading_delay_test(t);
	ASSERT_TRUE(true) << "This tests if the execution reaches here or binary hangs";
}

TEST(TestThreading, StdDelays) {
	threads::Std<math_types::floatmtx_ty::numel_cnt_t> t;
	threading_delay_test(t,3);
	ASSERT_TRUE(true) << "This tests if the execution reaches here or binary hangs";
}

#endif // !TESTS_SKIP_THREADING_DELAYS

