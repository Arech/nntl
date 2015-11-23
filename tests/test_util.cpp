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
//#include "../nntl/interface/math/i_open_blas.h"

#include "../nntl/nntl.h"

#include "../nntl/utils/clamp.h"
#include "../nntl/interface/rng/std.h"
#include "../nntl/utils/chrono.h"
#include "../nntl/utils/prioritize_workers.h"

using namespace nntl;
#ifdef _DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 100;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 50;
#endif // _DEBUG


TEST(TestUtil, PrioritizeWorkersPerf) {
	typedef nntl::nnet_def_interfaces::iThreads_t def_threads_t;
	using namespace std::chrono;

	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	def_threads_t iT;
	utils::prioritize_workers<utils::PriorityClass::PerfTesting, def_threads_t> pw(iT);

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		utils::prioritize_workers<utils::PriorityClass::Working, def_threads_t> pw(iT);
	}
	diff = steady_clock::now() - bt;
	STDCOUTL("prioritize_workers:\t" << utils::duration_readable(diff, maxReps));
}


TEST(TestUtil, OwnOrUsePtr) {
	int i = 1;
	auto pu = utils::make_own_or_use_ptr(&i);
	ASSERT_TRUE(!pu.bOwning);
	ASSERT_TRUE(!pu.empty());
	ASSERT_TRUE(1 == pu);
	pu.get() = 2;
	ASSERT_TRUE(2 == pu);
	
	pu.release();
	ASSERT_TRUE(pu.empty());

	auto po = utils::make_own_or_use_ptr<int>();
	ASSERT_TRUE(po.bOwning);
	ASSERT_TRUE(!po.empty());
	po.get() = 3;
	ASSERT_TRUE(3 == po);

	po.release();
	ASSERT_TRUE(po.empty());

	auto pu2 = utils::make_own_or_use_ptr(&i);
	ASSERT_TRUE(!pu2.bOwning);
	ASSERT_TRUE(!pu2.empty());
	ASSERT_TRUE(2 == pu2);

}

TEST(TestUtil, Clamp) {
	using float_t_ = math_types::float_ty;
	math_types::floatmtx_ty m(50, 50), d;

	rng::Std r;

	r.gen_matrix(m, 20);
	m.cloneTo(d);

	float_t_ lo = -10, hi = 10;

	bool bGotBig = false;
	auto p = m.dataAsVec();
	auto pd = d.dataAsVec();
	for (math_types::floatmtx_ty::numel_cnt_t i = 0, im = m.numel(); i < im; ++i) {
		auto v = p[i];
		if (v > hi || v < lo) {
			bGotBig = true;
			if (v > hi) {
				pd[i] = hi;
			}else if (v<lo) {
				pd[i] = lo;
			}
		}

	}
	ASSERT_TRUE(bGotBig);

	utils::boost::algorithm::clamp_range(p, p + m.numel(), p, lo, hi);

	bGotBig = false;
	for (math_types::floatmtx_ty::numel_cnt_t i = 0, im = m.numel(); i < im; ++i) {
		if (p[i] > hi || p[i] < lo) {
			bGotBig = true;
			break;
		}
	}
	ASSERT_TRUE(!bGotBig);

	ASSERT_EQ(m, d);
}