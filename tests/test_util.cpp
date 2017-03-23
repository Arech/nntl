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

#include "../nntl/math.h"
//#include "../nntl/interface/math/i_open_blas.h"

#include "../nntl/nntl.h"

//#include "../nntl/utils/clamp.h"
#include "../nntl/interface/rng/cstd.h"
#include "../nntl/utils/chrono.h"
#include "../nntl/utils/prioritize_workers.h"
#include "../nntl/utils/options.h"

#include "../nntl/utils/mixins.h"

using namespace nntl;
#ifdef NNTL_DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 100;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 50;
#endif // NNTL_DEBUG

TEST(TestUtils, PrioritizeWorkersPerf) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
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


TEST(TestUtils, OwnOrUsePtr) {
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

	//////////////////////////////////////////////////////////////////////////
	// move-related stuff
	/*utils::own_or_use_ptr<int*> pu3(std::move(pu2));
	ASSERT_TRUE(!pu3.bOwning);
	ASSERT_TRUE(!pu3.empty());
	ASSERT_TRUE(2 == pu3);
	ASSERT_TRUE(pu2.empty());*/
}

TEST(TestMatfile, Options) {
	struct NoOptions {};
	struct HasOptions : public utils::options<HasOptions> {};

	enum BinOpts {
		o1,
		o2,
		total_options
	};
	struct HasBinaryOptions : public utils::binary_options<BinOpts> {};

	NoOptions no;
	ASSERT_TRUE(!utils::binary_option(no, o2));
	ASSERT_TRUE(utils::binary_option<true>(no, o2));

	HasOptions ho;
	ASSERT_TRUE(!utils::binary_option(ho, o2));
	ASSERT_TRUE(utils::binary_option<true>(ho, o2));

	HasBinaryOptions hbo;
	ASSERT_TRUE(!utils::binary_option(hbo, o2));
	ASSERT_TRUE(!utils::binary_option<true>(hbo, o2));
	hbo.m_binary_options[o2] = true;
	ASSERT_TRUE(utils::binary_option(hbo, o2));
	ASSERT_TRUE(utils::binary_option<true>(hbo, o2));
	hbo.m_binary_options[o2] = false;
	ASSERT_TRUE(!utils::binary_option(hbo, o2));
	ASSERT_TRUE(!utils::binary_option<true>(hbo, o2));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

namespace _MixinsConcept {
	using namespace utils::mixins;
	using namespace indexed;

	typedef void mixinTestCfgT;

	//////////////////////////////////////////////////////////////////////////
	template<typename _FC, typename _CfgT, size_t MixinIdx>
	class Mixin1 {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

	public:
		enum OptsList {
			f_elFirst = 0,
			f_Opt2,
			f_elLast,

			opts_total
		};

		void f1() {
			STDCOUTL("Calling " << get_self()._hello() << " from f1. MixinIdx=" << MixinIdx);
		}

	private:
		void _checkInt() const {
			for (size_t i = f_elFirst + 1; i < f_elLast; ++i) {
				ASSERT_TRUE(!get_opt(i)) << "unexpected " << i;
			}
		}

	public:
		void M1_check_preAct() const {
			ASSERT_TRUE(!get_opt(f_elFirst) && !get_opt(f_elLast)) << "unexpected";
			_checkInt();
		}
		void M1_make_act() {
			set_opt(f_elFirst, true).set_opt(f_elLast, true);
		}
		void M1_make_redo() {
			set_opt(f_elFirst, false).set_opt(f_elLast, false);
		}
		void M1_check_postAct()const {
			ASSERT_TRUE(get_opt(f_elFirst) && get_opt(f_elLast)) << "unexpected";
			_checkInt();
		}
	};

	template<typename _FC, typename _CfgT, size_t MixinIdx>
	class Mixin2 {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

	public:
		enum OptsList {
			f_elFirst = 0,
			f_elLast,

			opts_total
		};

		void f2() {
			STDCOUTL("Calling " << get_self()._hello() << " from f2. MixinIdx=" << MixinIdx);
		}

	private:
		void _checkInt() const {
			for (size_t i = f_elFirst + 1; i < f_elLast; ++i) {
				ASSERT_TRUE(!get_opt(i)) << "unexpected " << i;
			}
		}

	public:
		void M2_check_preAct()const {
			ASSERT_TRUE(!get_opt(f_elFirst) && !get_opt(f_elLast)) << "unexpected";
			_checkInt();
		}
		void M2_make_act() {
			set_opt(f_elFirst, true).set_opt(f_elLast, true);
		}
		void M2_make_redo() {
			set_opt(f_elFirst, false).set_opt(f_elLast, false);
		}
		void M2_check_postAct()const {
			ASSERT_TRUE(get_opt(f_elFirst) && get_opt(f_elLast)) << "unexpected";
			_checkInt();
		}
	};

	template<typename prmT, template<typename, typename, size_t> class... MixinsT>
	class MainCl : public MixinsT< MainCl<prmT, MixinsT...>, mixinTestCfgT, ref_index<MixinsT, sizeof...(MixinsT), sizeof...(MixinsT), MixinsT...>::value >...
	{
	private:
		typedef MainCl<prmT, MixinsT...> self_t;
		NNTL_METHODS_SELF();

	public:
		enum OptsList {
			f_elFirst = 0,
			f_el2,
			f_el3,
			f_el4,
			f_elLast,
			opts_total
		};

		static constexpr size_t mixins_count = sizeof...(MixinsT);

		typedef make_mixin_vec<MainCl<prmT, MixinsT...>, mixinTestCfgT, MixinsT...> mixins_tvec;
		typedef make_mixin_options_count_vec_c<mixins_tvec, opts_total> mixin_opts_cnt;
		typedef make_cumsum_vec_c<mixin_opts_cnt> mixin_opts_ofs;

		static constexpr size_t TotalOpts = get_cumsum<mixin_opts_cnt>::value;

		binary_options_storage<mixin_opts_ofs, TotalOpts> m_opts;

	protected:
		NNTL_METHODS_MIXIN_ROOT_OPTIONS();

	protected:
		void _checkInt()const {
			for (size_t i = f_elFirst + 1; i < f_elLast; ++i) {
				ASSERT_TRUE(!get_opt(i)) << "unexpected " << i;
			}
		}

	public:
		void check_preAct() const {
			ASSERT_TRUE(!get_opt(f_elFirst) && !get_opt(f_elLast)) << "unexpected";
			_checkInt();
		}
		void make_act() {
			set_opt(f_elFirst, true).set_opt(f_elLast, true);
		}
		void make_redo() {
			set_opt(f_elFirst, false).set_opt(f_elLast, false);
		}
		void check_postAct()const {
			ASSERT_TRUE(get_opt(f_elFirst) && get_opt(f_elLast)) << "unexpected";
			_checkInt();
		}

		const char* _hello()const {
			return "hello()";
		}

		static void print() {
			STDCOUTL("There are " << TotalOpts << " opts:");

			STDCOUTL("root total=" << (boost::mpl::at_c<mixin_opts_cnt, 0>::type::value));
			STDCOUTL("M1 total=" << (boost::mpl::at_c<mixin_opts_cnt, 1>::type::value));
			STDCOUTL("M2 total=" << (boost::mpl::at_c<mixin_opts_cnt, 2>::type::value));

			STDCOUTL("\nOffsets are:");
			STDCOUTL("root =" << (boost::mpl::at_c<mixin_opts_ofs, 0>::type::value));
			STDCOUTL("M1 =" << (boost::mpl::at_c<mixin_opts_ofs, 1>::type::value));
			STDCOUTL("M2 =" << (boost::mpl::at_c<mixin_opts_ofs, 2>::type::value));
		}
	};
}

TEST(TestUtils, MixinsConcept) {
	_MixinsConcept::MainCl<int, _MixinsConcept::Mixin1, _MixinsConcept::Mixin2> obj;

	obj.f1();
	obj.f2();
	decltype(obj)::print();

	obj.check_preAct();
	obj.M1_check_preAct();
	obj.M2_check_preAct();

	obj.make_act();

	obj.check_postAct();
	obj.M1_check_preAct();
	obj.M2_check_preAct();

	obj.make_redo();

	obj.check_preAct();
	obj.M1_check_preAct();
	obj.M2_check_preAct();


	obj.M1_make_act();

	obj.check_preAct();
	obj.M1_check_postAct();
	obj.M2_check_preAct();


	obj.M1_make_redo();

	obj.check_preAct();
	obj.M1_check_preAct();
	obj.M2_check_preAct();


	obj.M2_make_act();

	obj.check_preAct();
	obj.M1_check_preAct();
	obj.M2_check_postAct();

	obj.M2_make_redo();

	obj.check_preAct();
	obj.M1_check_preAct();
	obj.M2_check_preAct();
}