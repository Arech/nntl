/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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
#include "../nntl/nntl.h"
#include "../nntl/interface/rng/cstd.h"
//#include "../nntl/interface/threads/winqdu.h"
//#include "../nntl/interface/threads/std.h"
#include "../nntl/interface/threads/workers.h"
#include "../nntl/interfaces.h"
#include "../nntl/utils/chrono.h"
#include "../nntl/interface/rng/cstd.h"

#include "../nntl/interface/threads/bgworkers.h"

//#define BOOST_USE_WINDOWS_H

using namespace nntl;

typedef d_interfaces::real_t real_t;

//#ifdef TESTS_SKIP_LONGRUNNING

#ifndef TESTS_SKIP_THREADING_PERFS
#define TESTS_SKIP_THREADING_PERFS 1
#endif // !TESTS_SKIP_THREADING_PERFS

#define TESTS_SKIP_THREADING_DELAYS

//#endif

// #ifndef TESTS_SKIP_THREADING_DELAYS
// #define TESTS_SKIP_THREADING_DELAYS
// #endif

//////////////////////////////////////////////////////////////////////////
template<typename TT>
void threads_basics_test(TT& t) {
	typedef TT::range_t range_t;
	typedef TT::par_range_t par_range_t;
	typedef math::smatrix_td::vec_len_t vec_len_t;

	const auto workersCnt = t.workers_count();

	const vec_len_t maxCnt = 2 * workersCnt;
	math::smatrix<real_t> m(1, maxCnt + 1);
	ASSERT_TRUE(!m.isAllocationFailed());
	auto ptr = m.data();
	
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
		auto lRed = [maxCntPerWorker,ptr](const par_range_t& r)->real_t
		{
			const auto cnt = r.cnt();
			EXPECT_TRUE(cnt <= maxCntPerWorker) << "cnt=="<<cnt<<" maxCntPerWorker=="<< maxCntPerWorker;
			real_t ret = 0;
			for (range_t i = 0; i < cnt; ++i) ret += ptr[i] * ptr[i];
			return ret;
		};
		auto lRedF = [workersCnt](const real_t* p, const range_t cnt)->real_t
		{
			EXPECT_TRUE(cnt <= workersCnt) << "cnt=="<<cnt<<" workersCnt=="<< workersCnt;
			real_t ret = 0;
			for (range_t i = 0; i < cnt; ++i) ret += p[i];
			return ret;
		};
		real_t redRes = t.reduce(lRed, lRedF, mnumel);
		ASSERT_DOUBLE_EQ(redRes, static_cast<real_t>(mnumel));
	}
}

// TEST(TestThreading, WinQDUBasics) {
// 	threads::WinQDU<real_t, math::smatrix_td::numel_cnt_t> t;
// 	threads_basics_test(t);
// 	threads_basics_test(t);
// }
// 
// TEST(TestThreading, StdBasics) {
// 	threads::Std<real_t, math::smatrix_td::numel_cnt_t> t;
// 	threads_basics_test(t);
// 	threads_basics_test(t);
// }

TEST(TestThreading, WorkersBasics) {
	threads::Workers<real_t, math::smatrix_td::numel_cnt_t> t;
	threads_basics_test(t);
	threads_basics_test(t);
}


#if !TESTS_SKIP_THREADING_PERFS

struct sp_win : public threads::winNativeSync {
	typedef mutex_t mutex_comp_t;
	typedef cond_var_t cond_var_comp_t;
};
struct sp_std : public threads::stdSync {
	typedef mutex_t mutex_comp_t;
	typedef cond_var_t cond_var_comp_t;
};
struct sp_stdShared : public threads::stdSync {
	typedef shared_mutex_t mutex_comp_t;
	typedef cond_var_any_t cond_var_comp_t;
};

TEST(TestThreading, PerfComparision) {

	typedef math::smatrix_td::numel_cnt_t numel_cnt_t;
	STDCOUTL("The test may require a few seconds to complete. Define TESTS_SKIP_THREADING_PERFS 1 to skip.");

#ifdef _DEBUG
	static constexpr uint64_t maxreps = 200, runCnt = 10;
#else
	static constexpr uint64_t maxreps = 100000, runCnt = 100;
#endif
	utils::tictoc tW, tWNE, tS;

	{
		typedef threads::Workers<real_t, numel_cnt_t, sp_win> thr;
		typedef thr::par_range_t par_range_t;

		thr wint;
		//v is used to make sure compiler doesn't optimize away all machinery. Also it almost doesn't change time ratio between
		//WinQDU and Std in debug and release and almost doesn't change absolute times in release.
		::std::atomic_ptrdiff_t v = 0, v2 = 0;

		for (uint64_t i = 0; i < maxreps; ++i) {
			tW.tic();
			wint.run([&](const par_range_t&) {
				v++;
			}, runCnt);
			tW.toc();
		}

		for (uint64_t i = 0; i < maxreps; ++i) {
			tWNE.tic();
			wint.run([&](const par_range_t&)noexcept {
				// we turned off exceptions at project level now, so this test probably won't show much.
				// but it might be helpful with other compiler settings, so leave it be
				v2++;
			}, runCnt);
			tWNE.toc();
		}
		tW.say("WinQDU");
		tWNE.say("WinQDU(noexcept)");
		tW.ratios(tWNE);
		STDCOUTL(v << v2);
	}
	{
		typedef threads::Workers<real_t, numel_cnt_t, sp_std> thr;
		typedef thr::par_range_t par_range_t;
		thr stdt;
		::std::atomic_ptrdiff_t v = 0;

		for (uint64_t i = 0; i < maxreps; ++i) {
			tS.tic();
			stdt.run([&](const par_range_t&) {
				v++;
			}, runCnt);
			tS.toc();
		}
		tS.say("Std");
		STDCOUTL(v);
	}

	STDCOUT("ratios to threads::WinQDU are: ");
	tS.ratios(tW);
}


#endif // !TESTS_SKIP_THREADING_PERFS


#ifndef TESTS_SKIP_THREADING_DELAYS

template <typename TT>
void threading_delay_test(TT& t) {
	//no need to make it significantly smaller because of resolution of Windows sleep timer
	const rng::CStd<real_t>::int_4_random_shuffle_t max_mks = 5000;
	//but this count should be large enough
	const uint64_t maxreps = 10000;

	STDCOUTL("The test would require a bit less than " << static_cast<uint64_t>(max_mks*maxreps / 1000000) << "s"
		<<".\nIf it lasts significantly longer, then the app hangs and test failed. Define TESTS_SKIP_THREADING_DELAYS to skip.");
	
	rng::CStd<real_t> r;
	for (uint64_t i = 0; i < maxreps; ++i) {
		t.run([=, &r](const TT::par_range_t&) {
			::std::this_thread::sleep_for(::std::chrono::microseconds(r.gen_i(max_mks)));
		}, 100000000);
	}
}

// TEST(TestThreading, WinQDUDelays) {
// 	threads::WinQDU<real_t, math::smatrix_td::numel_cnt_t> t;
// 	threading_delay_test(t);
// 	ASSERT_TRUE(true) << "This tests if the execution reaches here or binary hangs";
// }
// 
// TEST(TestThreading, StdDelays) {
// 	threads::Std<real_t, math::smatrix_td::numel_cnt_t> t;
// 	threading_delay_test(t);
// 	ASSERT_TRUE(true) << "This tests if the execution reaches here or binary hangs";
// }

TEST(TestThreading, WorkersDelays) {
	threads::Workers<real_t, math::smatrix_td::numel_cnt_t> t;
	threading_delay_test(t);
	ASSERT_TRUE(true) << "This tests if the execution reaches here or binary hangs";
}
// 
// TEST(TestThreading, StdDelays) {
// 	threads::Std<real_t, math::smatrix_td::numel_cnt_t> t;
// 	threading_delay_test(t);
// 	ASSERT_TRUE(true) << "This tests if the execution reaches here or binary hangs";
// }

#endif // !TESTS_SKIP_THREADING_DELAYS


template<typename SyncT>
void run_bgworkers_simpletest()noexcept {
	auto t1 = [](const thread_id_t tId)noexcept->bool {
		STDCOUTL(tId);
		static unsigned v = 0;
		::std::this_thread::sleep_for(::std::chrono::milliseconds(125));
		v += tId;
		return v < 7;
	};

	auto t2 = [](const thread_id_t tId)noexcept->bool {
		if (0 == tId) {
			STDCOUTL("In priority task");
		}
		return false;
	};

	STDCOUTL("There should be silence because there are no tasks...");
	threads::BgWorkers<SyncT> bgw(2);
	bgw.set_task_wait_timeout(::std::chrono::milliseconds(1500))
		.expect_tasks_count(2)
		.exec([](const thread_id_t tId)noexcept {
		STDCOUTL("Hello from " << tId);
	});

	::std::this_thread::sleep_for(::std::chrono::seconds(3));
	STDCOUTL("Now verbose");
	bgw.add_task(t1).add_task(t2, 2);
	::std::this_thread::sleep_for(::std::chrono::seconds(7));

	bgw.delete_tasks();
	STDCOUTL("Now there should be silence...");

	::std::this_thread::sleep_for(::std::chrono::seconds(5));
}

TEST(TestBgWorkers, Simple) {
#if NNTL_HAS_NATIVE_SRWLOCKS_AND_CODITIONALS
	STDCOUTL("Using Windows primitives");
	run_bgworkers_simpletest<threads::win_sync_primitives>();
#endif

	STDCOUTL("Using STL primitives");
	run_bgworkers_simpletest<threads::std_sync_primitives>();
}
