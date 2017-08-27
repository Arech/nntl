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
#include "../nntl/interface/rng/cstd.h"

#include "../nntl/utils/options.h"

#include "../nntl/utils/mixins.h"

#include "../nntl/utils/tictoc.h"

#include "../nntl/utils/call_wrappers.h"

#include "../nntl/interface/rng/afrand_as.h"


using namespace nntl;
#ifdef NNTL_DEBUG
constexpr unsigned TEST_PERF_REPEATS_COUNT = 10;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 100;
#else
constexpr unsigned TEST_PERF_REPEATS_COUNT = 400;
constexpr unsigned TEST_CORRECTN_REPEATS_COUNT = 50;
#endif // NNTL_DEBUG


TEST(TestTupleUtils, TupleDisjunction) {
	typedef ::std::tuple<int, short, double, char*, void, double, unsigned&> tuple_t;
	//static_assert(tuple_utils::tuple_disjunction<::std::is_void, tuple_t>::value, "WTF?");
	static_assert(tuple_utils::aggregate<::std::disjunction,::std::is_void, tuple_t>::value, "WTF?");
	static_assert(!tuple_utils::aggregate<::std::conjunction, ::std::is_void, tuple_t>::value, "WTF?");

	typedef ::std::tuple<int, short, double, char*, double, unsigned&> tuple2_t;
	//static_assert(! tuple_utils::tuple_disjunction<::std::is_void, tuple2_t>::value, "WTF?");
	static_assert(!tuple_utils::aggregate<::std::disjunction,::std::is_void, tuple2_t>::value, "WTF?");
	static_assert(!tuple_utils::aggregate<::std::conjunction, ::std::is_void, tuple2_t>::value, "WTF?");
}

TEST(TestTupleUtils, TupleElementIdx) {
	typedef ::std::tuple<int, short, double, char*, void, double, unsigned&> tuple_t;
	constexpr auto s = ::std::tuple_size<tuple_t>::value;

	STDCOUTL("Tuple <int, short, double, char*, void, double, unsigned&> size = "<< s);
	STDCOUTL("idx of void = " << (tuple_utils::tuple_element_idx_safe<void, tuple_t>::value));
	static_assert(4 == tuple_utils::tuple_element_idx_safe<void, tuple_t>::value, "WTF");
	STDCOUTL("idx of double = " << (tuple_utils::tuple_element_idx_safe<double, tuple_t>::value));
	static_assert(2 == tuple_utils::tuple_element_idx_safe<double, tuple_t>::value, "WTF");
	STDCOUTL("idx of float = " << (tuple_utils::tuple_element_idx_safe<float, tuple_t>::value));
	static_assert(s == tuple_utils::tuple_element_idx_safe<float, tuple_t>::value, "WTF");
}

TEST(TestUtils, PrioritizeWorkersPerf) {
	typedef nntl::d_interfaces::iThreads_t def_threads_t;
	using namespace ::std::chrono;

	steady_clock::time_point bt;
	nanoseconds diff;
	constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;

	def_threads_t iT;
	threads::prioritize_workers<threads::PriorityClass::PerfTesting, def_threads_t> pw(iT);

	bt = steady_clock::now();
	for (unsigned r = 0; r < maxReps; ++r) {
		threads::prioritize_workers<threads::PriorityClass::Working, def_threads_t> pw(iT);
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
	/*utils::own_or_use_ptr<int*> pu3(::std::move(pu2));
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

			STDCOUTL("root total=" << (::boost::mpl::at_c<mixin_opts_cnt, 0>::type::value));
			STDCOUTL("M1 total=" << (::boost::mpl::at_c<mixin_opts_cnt, 1>::type::value));
			STDCOUTL("M2 total=" << (::boost::mpl::at_c<mixin_opts_cnt, 2>::type::value));

			STDCOUTL("\nOffsets are:");
			STDCOUTL("root =" << (::boost::mpl::at_c<mixin_opts_ofs, 0>::type::value));
			STDCOUTL("M1 =" << (::boost::mpl::at_c<mixin_opts_ofs, 1>::type::value));
			STDCOUTL("M2 =" << (::boost::mpl::at_c<mixin_opts_ofs, 2>::type::value));
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

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
// cdelegate<> == delegate<> from  https://codereview.stackexchange.com/questions/14730/impossibly-fast-delegate-in-c11

namespace utils_test {

	template <typename T> class cdelegate;

	template<class R, class ...A>
	class cdelegate<R(A...)>
	{
		using stub_ptr_type = R(*)(void*, A&&...);

		cdelegate(void* const o, stub_ptr_type const m) noexcept :
		object_ptr_(o),
			stub_ptr_(m)
		{
		}

	public:
		cdelegate() = default;

		cdelegate(cdelegate const&) = default;

		cdelegate(cdelegate&&) = default;

		cdelegate(::std::nullptr_t const) noexcept : cdelegate() { }

		template <class C, typename =
			typename ::std::enable_if < ::std::is_class<C>::value > ::type >
			explicit cdelegate(C const* const o) noexcept :
			object_ptr_(const_cast<C*>(o))
		{
		}

		template <class C, typename =
			typename ::std::enable_if < ::std::is_class<C>::value > ::type >
			explicit cdelegate(C const& o) noexcept :
		object_ptr_(const_cast<C*>(&o))
		{
		}

		template <class C>
		cdelegate(C* const object_ptr, R(C::* const method_ptr)(A...))
		{
			*this = from(object_ptr, method_ptr);
		}

		template <class C>
		cdelegate(C* const object_ptr, R(C::* const method_ptr)(A...) const)
		{
			*this = from(object_ptr, method_ptr);
		}

		template <class C>
		cdelegate(C& object, R(C::* const method_ptr)(A...))
		{
			*this = from(object, method_ptr);
		}

		template <class C>
		cdelegate(C const& object, R(C::* const method_ptr)(A...) const)
		{
			*this = from(object, method_ptr);
		}

		template <
			typename T,
			typename = typename ::std::enable_if <
			!::std::is_same<cdelegate, typename ::std::decay<T>::type>::value
			> ::type
		>
			cdelegate(T&& f) :
			store_(operator new(sizeof(typename ::std::decay<T>::type)),
				functor_deleter<typename ::std::decay<T>::type>),
			store_size_(sizeof(typename ::std::decay<T>::type))
		{
			using functor_type = typename ::std::decay<T>::type;

			new (store_.get()) functor_type(::std::forward<T>(f));

			object_ptr_ = store_.get();

			stub_ptr_ = functor_stub<functor_type>;

			deleter_ = deleter_stub<functor_type>;
		}

		cdelegate& operator=(cdelegate const&) = default;

		cdelegate& operator=(cdelegate&&) = default;

		template <class C>
		cdelegate& operator=(R(C::* const rhs)(A...))
		{
			return *this = from(static_cast<C*>(object_ptr_), rhs);
		}

		template <class C>
		cdelegate& operator=(R(C::* const rhs)(A...) const)
		{
			return *this = from(static_cast<C const*>(object_ptr_), rhs);
		}

		template <
			typename T,
			typename = typename ::std::enable_if <
			!::std::is_same<cdelegate, typename ::std::decay<T>::type>::value
			> ::type
		>
			cdelegate& operator=(T&& f)
		{
			using functor_type = typename ::std::decay<T>::type;

			if ((sizeof(functor_type) > store_size_) || !store_.unique())
			{
				store_.reset(operator new(sizeof(functor_type)),
					functor_deleter<functor_type>);

				store_size_ = sizeof(functor_type);
			} else
			{
				deleter_(store_.get());
			}

			new (store_.get()) functor_type(::std::forward<T>(f));

			object_ptr_ = store_.get();

			stub_ptr_ = functor_stub<functor_type>;

			deleter_ = deleter_stub<functor_type>;

			return *this;
		}

		template <R(*const function_ptr)(A...)>
		static cdelegate from() noexcept
		{
			return{ nullptr, function_stub<function_ptr> };
		}

		template <class C, R(C::* const method_ptr)(A...)>
		static cdelegate from(C* const object_ptr) noexcept
		{
			return{ object_ptr, method_stub<C, method_ptr> };
		}

		template <class C, R(C::* const method_ptr)(A...) const>
		static cdelegate from(C const* const object_ptr) noexcept
		{
			return{ const_cast<C*>(object_ptr), const_method_stub<C, method_ptr> };
		}

		template <class C, R(C::* const method_ptr)(A...)>
		static cdelegate from(C& object) noexcept
		{
			return{ &object, method_stub<C, method_ptr> };
		}

		template <class C, R(C::* const method_ptr)(A...) const>
		static cdelegate from(C const& object) noexcept
		{
			return{ const_cast<C*>(&object), const_method_stub<C, method_ptr> };
		}

		template <typename T>
		static cdelegate from(T&& f)
		{
			return ::std::forward<T>(f);
		}

		static cdelegate from(R(*const function_ptr)(A...))
		{
			return function_ptr;
		}

		template <class C>
		using member_pair =
			::std::pair<C* const, R(C::* const)(A...)>;

		template <class C>
		using const_member_pair =
			::std::pair<C const* const, R(C::* const)(A...) const>;

		template <class C>
		static cdelegate from(C* const object_ptr,
			R(C::* const method_ptr)(A...))
		{
			return member_pair<C>(object_ptr, method_ptr);
		}

		template <class C>
		static cdelegate from(C const* const object_ptr,
			R(C::* const method_ptr)(A...) const)
		{
			return const_member_pair<C>(object_ptr, method_ptr);
		}

		template <class C>
		static cdelegate from(C& object, R(C::* const method_ptr)(A...))
		{
			return member_pair<C>(&object, method_ptr);
		}

		template <class C>
		static cdelegate from(C const& object,
			R(C::* const method_ptr)(A...) const)
		{
			return const_member_pair<C>(&object, method_ptr);
		}

		void reset() { stub_ptr_ = nullptr; store_.reset(); }

		void reset_stub() noexcept { stub_ptr_ = nullptr; }

		void swap(cdelegate& other) noexcept { ::std::swap(*this, other); }

		bool operator==(cdelegate const& rhs) const noexcept
		{
			return (object_ptr_ == rhs.object_ptr_) && (stub_ptr_ == rhs.stub_ptr_);
		}

		bool operator!=(cdelegate const& rhs) const noexcept
		{
			return !operator==(rhs);
		}

		bool operator<(cdelegate const& rhs) const noexcept
		{
			return (object_ptr_ < rhs.object_ptr_) ||
				((object_ptr_ == rhs.object_ptr_) && (stub_ptr_ < rhs.stub_ptr_));
		}

		bool operator==(::std::nullptr_t const) const noexcept
		{
			return !stub_ptr_;
		}

		bool operator!=(::std::nullptr_t const) const noexcept
		{
			return stub_ptr_;
		}

		explicit operator bool() const noexcept { return stub_ptr_; }

		R operator()(A... args) const
		{
			//  assert(stub_ptr);
			return stub_ptr_(object_ptr_, ::std::forward<A>(args)...);
		}

	private:
		friend struct ::std::hash<cdelegate>;

		using deleter_type = void(*)(void*);

		void* object_ptr_;
		stub_ptr_type stub_ptr_{};

		deleter_type deleter_;

		::std::shared_ptr<void> store_;
		::std::size_t store_size_;

		template <class T>
		static void functor_deleter(void* const p)
		{
			static_cast<T*>(p)->~T();

			operator delete(p);
		}

		template <class T>
		static void deleter_stub(void* const p)
		{
			static_cast<T*>(p)->~T();
		}

		template <R(*function_ptr)(A...)>
		static R function_stub(void* const, A&&... args)
		{
			return function_ptr(::std::forward<A>(args)...);
		}

		template <class C, R(C::*method_ptr)(A...)>
		static R method_stub(void* const object_ptr, A&&... args)
		{
			return (static_cast<C*>(object_ptr)->*method_ptr)(
				::std::forward<A>(args)...);
		}

		template <class C, R(C::*method_ptr)(A...) const>
		static R const_method_stub(void* const object_ptr, A&&... args)
		{
			return (static_cast<C const*>(object_ptr)->*method_ptr)(
				::std::forward<A>(args)...);
		}

		template <typename>
		struct is_member_pair : std::false_type { };

		template <class C>
		struct is_member_pair< ::std::pair<C* const,
			R(C::* const)(A...)> > : std::true_type
		{
		};

		template <typename>
		struct is_const_member_pair : std::false_type { };

		template <class C>
		struct is_const_member_pair< ::std::pair<C const* const,
			R(C::* const)(A...) const> > : std::true_type
		{
		};

		template <typename T>
		static typename ::std::enable_if <!(is_member_pair<T>::value || is_const_member_pair<T>::value), R>::type
			functor_stub(void* const object_ptr, A&&... args)
		{
			return (*static_cast<T*>(object_ptr))(::std::forward<A>(args)...);
		}

		template <typename T>
		static typename ::std::enable_if <
			is_member_pair<T>::value ||
			is_const_member_pair<T>::value,
			R
		> ::type
			functor_stub(void* const object_ptr, A&&... args)
		{
			return (static_cast<T*>(object_ptr)->first->*
				static_cast<T*>(object_ptr)->second)(::std::forward<A>(args)...);
		}
	};

	/*namespace std {
	template <typename R, typename ...A>
	struct hash<::cdelegate<R(A...)> >
	{
	size_t operator()(::cdelegate<R(A...)> const& d) const noexcept
	{
	auto const seed(hash<void*>()(d.object_ptr_));

	return hash<typename ::cdelegate<R(A...)>::stub_ptr_type>()(
	d.stub_ptr_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	};
	}*/
}

template<typename HandlerT>
class caller_mock {
	typedef typename HandlerT::template call_tpl<void(int64_t)> call_through_t;

	call_through_t t;

	template<typename F2>
	void _proc_functor(F2&& f, int64_t i)noexcept {
		t = ::std::forward<F2>(f);
		t(i);
	}
public:

	template<typename F>
	void proc_functor(F&& f, int64_t i)noexcept {
		_proc_functor(HandlerT::wrap<F>(::std::forward<F>(f)), i);
	}
};

TEST(TestUtils, Forwarders) {
#ifdef _DEBUG
	static constexpr uint64_t maxreps = 200;
	static constexpr int64_t ccnt = 100;
#else
	static constexpr uint64_t maxreps = 1000;
	static constexpr int64_t ccnt = 50000;
#endif

	utils::tictoc tS, tF, tD, tCmcf, tCmcfnr;
	int64_t x(0), x2(0), x3(0), x4(0), x5(0);

// 	typedef utils::simpleWrapper<std::function<void(int64_t)>> stdWrap;
// 	typedef utils::simpleWrapper<utils_test::cdelegate<void(int64_t)>> delegWrap;

	typedef utils::simpleWrapper<::std::function> stdWrap;
	typedef utils::simpleWrapper<utils_test::cdelegate> delegWrap;

	caller_mock<stdWrap> callStd;
	caller_mock<delegWrap> callDeleg;
	caller_mock<utils::forwarderWrapper<2 * sizeof(void*)>> callFwd;
	caller_mock<utils::cmcforwarderWrapper<2 * sizeof(void*)>> callCmcfwd;

	for (uint64_t r = 0; r < maxreps; ++r)
	{
		auto ts2 = [&x](int64_t i) { return i + x; };
		tS.tic();
#pragma loop( no_vector )
		for (int64_t i = 0; i < ccnt; ++i) {
			callStd.proc_functor([&x, &ts2, &x2, &x3, &x4](int64_t i) { x = ts2(i); }, i);
		}
		tS.toc();

		auto t2 = ([&x2](int64_t i) { return i + x2; });
		tF.tic();
#pragma loop( no_vector )  
		for (int64_t i = 0; i < ccnt; ++i) {
			callFwd.proc_functor([&x2, &t2, &x, &x3, &x4](int64_t i) { x2 = t2(i); }, i);
		}
		tF.toc();
		ASSERT_EQ(x, x2);

		auto td2 = [&x3](int64_t i) { return i + x3; };
		tD.tic();
#pragma loop( no_vector )  
		for (int64_t i = 0; i < ccnt; ++i) {
			callDeleg.proc_functor([&x2, &x, &x3, &x4, &td2](int64_t i) { x3 = td2(i); }, i);
		}
		tD.toc();
		ASSERT_EQ(x, x3);

		auto tc2 = ([&x4](int64_t i) { return i + x4; });
		tCmcf.tic();
#pragma loop( no_vector )  
		for (int64_t i = 0; i < ccnt; ++i) {
			callCmcfwd.proc_functor([&x2, &tc2, &x, &x3, &x4](int64_t i) { x4 = tc2(i); }, i);
		}
		tCmcf.toc();
		ASSERT_EQ(x, x4);

		auto tc5 = ([&x5](int64_t i) { return i + x5; });
		tCmcfnr.tic();
#pragma loop( no_vector )  
		for (int64_t i = 0; i < ccnt; ++i) {
			callCmcfwd.proc_functor([&tc5, &x5](int64_t i) { x5 = tc5(i); }, i);
		}
		tCmcfnr.toc();
		ASSERT_EQ(x, x5);
	}
	tS.say("std");
	tF.say("fwd");
	tS.ratios(tF);
	tD.say("del");
	tS.ratios(tD);
	tF.ratios(tD);

	tCmcf.say("cmcfwd");
	tCmcf.ratios(tF);

	tCmcfnr.say("cmcfwdnr");
	tCmcfnr.ratios(tF);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

size_t _rng_less_than(const size_t m)noexcept {
	unsigned int r;
	if (rand_s(&r)) {
		NNTL_ASSERT(!"rand_s failed");
		STDCOUTL("rand_s failed");
		return false;
	}
	const auto v = static_cast<size_t>(::std::round((m*double(r)) / UINT_MAX));
	NNTL_ASSERT(v >= 0 && v <= m);
	return v;
}

//////////////////////////////////////////////////////////////////////////
// Several sets of tests must be implemented in order to make sure the AsynchRNG is correct.
// A first set consists of tests of DataBuffer structure:
// 1. plain state checks &&  buffer continuity test
// 2. same for atomics
// 3. atomic vs. mutex performance test
// 
template<bool bOnMutex = true, typename Rep, typename Dur>
void data_buffer_state_checks(const size_t bufSize = 71, const size_t maxThreadGenSize = 5
	, const size_t totalReads = TEST_CORRECTN_REPEATS_COUNT
	, const ::std::chrono::duration<Rep, Dur>& workersTO = ::std::chrono::milliseconds(43)
	, const ::std::chrono::duration<Rep, Dur>& checkerTO = ::std::chrono::milliseconds(7)
	, const ::std::chrono::duration<Rep, Dur>& readerSleep = ::std::chrono::milliseconds(47)
)noexcept
{
	//we'd need two sets of threads, the first one will be "filling" the buffer, the second - checking state.
	//main thread will repeatedly withdraw data from the buffer
	typedef uint32_t real_t;
	typedef threads::BgWorkers<> BgWorkers_t;
	typedef ::std::conditional_t<bOnMutex, typename BgWorkers_t::Sync_t, void> BufSync_t;
	typedef utils::DataBuffer<real_t, BufSync_t> BufferRange_t;

	// 	STDCOUTL("Working with bufSize = " << bufSize << ", maxThreadGenSize = " << maxThreadGenSize
	// 		<< ", totalReads = " << totalReads << ", workersTO = " << workersTO.count()
	// 		<< ", checkerTO = " << checkerTO.count() << ", readerSleep = " << readerSleep.count());

	BgWorkers_t bgFillers, bgChecker(1, threads::PriorityClass::threads_priority_no_change);

	const auto wc = bgFillers.workers_count();
	::std::vector<real_t> bufData(bufSize), threadsBufData(maxThreadGenSize*wc);
	::std::vector<real_t*> thrBuf(wc);
	::std::vector<size_t> thrRunCnt(wc);

	::std::fill(bufData.begin(), bufData.end(), real_t(0));
	::std::fill(thrRunCnt.begin(), thrRunCnt.end(), size_t(0));

	real_t* pBuf = &threadsBufData[0];
	for (size_t i = 0; i < wc; ++i) {
		thrBuf[i] = pBuf;
		pBuf += maxThreadGenSize;
	}

	BufferRange_t Buf(&bufData[0], bufSize);
	::std::atomic_size_t failsCnt(0);

	bgChecker.expect_tasks_count(1).set_task_wait_timeout(checkerTO);
	auto checkerFunc = [&Buf, &failsCnt](const thread_id_t tId)noexcept->bool {
		failsCnt += !Buf.TestState();
		return false;
	};
	bgChecker.add_task(checkerFunc);

	ASSERT_EQ(0, failsCnt);

	bgFillers.expect_tasks_count(1).set_task_wait_timeout(workersTO);
	auto workerFunc = [&Buf, &thrBuf, maxThreadGenSize, &thrRunCnt](const thread_id_t tId)noexcept->bool {
		++thrRunCnt[tId];
		size_t thisN = _rng_less_than(maxThreadGenSize);
		if (!thisN) ++thisN;
		const auto toWork = Buf._as_should_work(thisN);
		if (!toWork) return false;

		NNTL_ASSERT(toWork <= thisN);
		if (toWork > thisN) {
			STDCOUTL("Invalid _as_should_work!");
		}
		real_t*const pInt = thrBuf[tId];
		::std::fill(pInt, pInt + toWork, tId + 1);
		Buf._as_done<true>(pInt, toWork);
		return true;
	};
	bgFillers.add_task(workerFunc);

	ASSERT_EQ(0, failsCnt);

	::std::this_thread::sleep_for(readerSleep * 100);
	//STDCOUTL("Going to get data...");

	size_t ss(0), fails(0);

	for (size_t i = 0; i < totalReads; ++i) {
		::std::this_thread::sleep_for(readerSleep);
		ASSERT_EQ(0, failsCnt);
		const auto n = _rng_less_than(bufSize / 4);
		if (!n) continue;
		ASSERT_EQ(0, failsCnt);
		const auto a = Buf.acquire_data(n);
		ASSERT_EQ(0, failsCnt);
		if (a.acquired()) {
			::std::fill(const_cast<real_t*>(a.ptr1), const_cast<real_t*>(a.ptr1) + a.n1, real_t(0));
			if (a.isTwoArrays()) {
				::std::fill(const_cast<real_t*>(a.ptr2), const_cast<real_t*>(a.ptr2) + a.n2, real_t(0));
			}
			ASSERT_EQ(0, failsCnt);
			::std::atomic_thread_fence(::std::memory_order_release);
			Buf.release_data(a);
			++ss;
		} else {
			++fails;
			STDCOUTL("Failed to acquire buffer of size " << n);
		}
		ASSERT_EQ(0, failsCnt);
	}
	STDCOUTL("Succeeded " << ss << ", failed " << fails << ". Small amount of fails during startup is OK");
	ASSERT_TRUE(double(fails) / ss < .1) << "Too many fails here, it's suspicious";

	//MUST delete tasks now, or shuffled order of destructors may hurt.
	bgFillers.delete_tasks();
	bgChecker.delete_tasks();
}

TEST(TestAFRandASDataBuffer, StateChecks_Mutex) {
#ifdef NNTL_DEBUG
	for (unsigned i = 0; i < 5; ++i) {
		STDCOUT("it#" << i << " ");
		data_buffer_state_checks<true>(71, 7, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<true>(1000, 10, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<true>(1000, 100, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));
	}

#else
	for (unsigned i = 0; i < 25; ++i) {
		STDCOUT("it#" << i << " ");
		data_buffer_state_checks<true>(71, 7, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<true>(1000, 10, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<true>(1000, 100, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));
	}
#endif
}

TEST(TestAFRandASDataBuffer, StateChecks_SpinLock) {
#ifdef NNTL_DEBUG
	for (unsigned i = 0; i < 5; ++i) {
		STDCOUT("it#" << i << " ");
		data_buffer_state_checks<false>(71, 7, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<false>(1000, 10, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<false>(1000, 100, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));
	}
#else
	for (unsigned i = 0; i < 25; ++i) {
		STDCOUT("it#" << i << " ");
		data_buffer_state_checks<false>(71, 7, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<false>(1000, 10, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));

		data_buffer_state_checks<false>(1000, 100, 100, ::std::chrono::milliseconds(3)
			, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(5));
	}
#endif
}

template<bool bOnMutex = true, typename Rep, typename Dur>
void data_buffer_perf(::std::vector<utils::tictoc>& clocks//the first element is for the main (reader) thread.
	, const size_t bufSize = 71, const size_t maxThreadGenSize = 5
	, const size_t totalReads = TEST_CORRECTN_REPEATS_COUNT
	, const ::std::chrono::duration<Rep, Dur>& workersTO = ::std::chrono::milliseconds(43)
	, const ::std::chrono::duration<Rep, Dur>& readerSleep = ::std::chrono::milliseconds(47)
)noexcept
{
	//we'd need two sets of threads, the first one will be "filling" the buffer, the second - checking state.
	//main thread will repeatedly withdraw data from the buffer
	typedef uint32_t real_t;
	typedef threads::BgWorkers<> BgWorkers_t;
	typedef ::std::conditional_t<bOnMutex, typename BgWorkers_t::Sync_t, void> BufSync_t;
	typedef utils::DataBuffer<real_t, BufSync_t> BufferRange_t;

	// 	STDCOUTL("Working with bufSize = " << bufSize << ", maxThreadGenSize = " << maxThreadGenSize
	// 		<< ", totalReads = " << totalReads << ", workersTO = " << workersTO.count()
	// 		<< ", checkerTO = " << checkerTO.count() << ", readerSleep = " << readerSleep.count());

	BgWorkers_t bgFillers;

	const auto wc = bgFillers.workers_count();

	constexpr size_t totalClocksForEachBgthread = 3;
	ASSERT_EQ(clocks.size(), 2 + wc * totalClocksForEachBgthread);

	::std::vector<real_t> bufData(bufSize), threadsBufData(maxThreadGenSize*wc);
	::std::vector<real_t*> thrBuf(wc);
	::std::vector<size_t> thrRunCnt(wc);

	::std::fill(bufData.begin(), bufData.end(), real_t(0));
	::std::fill(thrRunCnt.begin(), thrRunCnt.end(), size_t(0));

	real_t* pBuf = &threadsBufData[0];
	for (size_t i = 0; i < wc; ++i) {
		thrBuf[i] = pBuf;
		pBuf += maxThreadGenSize;
	}

	BufferRange_t Buf(&bufData[0], bufSize);

	bgFillers.expect_tasks_count(1).set_task_wait_timeout(workersTO);
	auto workerFunc = [&Buf, &thrBuf, maxThreadGenSize, &thrRunCnt, &clocks, totalClocksForEachBgthread]
	(const thread_id_t tId)noexcept->bool {
		++thrRunCnt[tId];
		size_t thisN = _rng_less_than(maxThreadGenSize);
		if (!thisN) ++thisN;

		const size_t firstClockOfs = 2 + tId*totalClocksForEachBgthread;

		auto& cl1 = clocks[firstClockOfs];

		const auto t1 = cl1.now();
		const auto toWork = Buf._as_should_work(thisN);
		const auto dur = cl1.now() - t1;
		if (!toWork) {
			clocks[firstClockOfs].store(dur);
			return false;
		}
		clocks[firstClockOfs + 1].store(dur);

		NNTL_ASSERT(toWork <= thisN);
		if (toWork > thisN) {
			STDCOUTL("Invalid _as_should_work!");
		}
		real_t*const pInt = thrBuf[tId];
		::std::fill(pInt, pInt + toWork, tId + 1);

		auto& cl3 = clocks[firstClockOfs + 2];
		cl3.tic();
		Buf._as_done<true>(pInt, toWork);
		cl3.toc();
		return true;
	};
	bgFillers.add_task(workerFunc);

	::std::this_thread::sleep_for(readerSleep * 100);
	//STDCOUTL("Going to get data...");

	size_t ss(0), fails(0);

	auto& cl1 = clocks[0];
	auto& cl2 = clocks[1];
	for (size_t i = 0; i < totalReads; ++i) {
		::std::this_thread::sleep_for(readerSleep);

		const auto n = _rng_less_than(bufSize / 4);
		if (!n) continue;

		const auto t1 = cl1.now();
		const auto a = Buf.acquire_data(n);
		const auto dur = cl1.now() - t1;

		if (a.acquired()) {
			cl1.store(dur);

			::std::fill(const_cast<real_t*>(a.ptr1), const_cast<real_t*>(a.ptr1) + a.n1, real_t(0));
			if (a.isTwoArrays()) {
				::std::fill(const_cast<real_t*>(a.ptr2), const_cast<real_t*>(a.ptr2) + a.n2, real_t(0));
			}
			::std::atomic_thread_fence(::std::memory_order_release);

			cl2.tic();
			Buf.release_data(a);
			cl2.toc();
			++ss;
		} else {
			--i;
			++fails;
			STDCOUTL("Failed to acquire buffer of size " << n);
		}
	}
	STDCOUTL("Succeeded " << ss << ", failed " << fails << ". Small amount of fails during startup is OK");
	ASSERT_TRUE(double(fails) / ss < .1) << "Too many fails here, it's suspicious";

	//MUST delete tasks now, or shuffled order of destructors may hurt.
	bgFillers.delete_tasks();
}

void reportTime(::std::vector<utils::tictoc>& clocks, const size_t hwc) {
	const unsigned mw = 20;
	clocks[0].say("acquire_data", mw);
	clocks[1].say("release_data", mw);
	for (size_t i = 0; i < hwc; ++i) {
		const auto fo = 2 + i * 3;
		STDCOUTL("Thread #" << i);
		clocks[fo].say("_as_should_work(no)", mw);
		clocks[fo + 1].say("_as_should_work(yes)", mw);
		clocks[fo + 2].say("_as_done(yes)", mw);
	}

	for (auto& e : clocks) e.reset();
}

TEST(TestAFRandASDataBuffer, MutexVsSpinLockPerf) {
	const auto hwc = ::std::thread::hardware_concurrency() - 1;
	::std::vector<utils::tictoc> clocks(2 + 3 * hwc);

#ifdef NNTL_DEBUG
	STDCOUTL("----------------- Mutex");
	data_buffer_perf<true>(clocks, 100, 10, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);
	STDCOUTL("----------------- SpinLock");
	data_buffer_perf<false>(clocks, 100, 10, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);

#else
	STDCOUTL("Small data sizes (1000-100)");
	STDCOUTL("----------------- Mutex");
	data_buffer_perf<true>(clocks, 1000, 100, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);
	STDCOUTL("----------------- SpinLock");
	data_buffer_perf<false>(clocks, 1000, 100, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);

	STDCOUTL("Medium data sizes (20000-1000)");
	STDCOUTL("----------------- Mutex");
	data_buffer_perf<true>(clocks, 20000, 1000, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);
	STDCOUTL("----------------- SpinLock");
	data_buffer_perf<false>(clocks, 20000, 1000, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);

	STDCOUTL("Big data sizes (200000-4000)");
	STDCOUTL("----------------- Mutex");
	data_buffer_perf<true>(clocks, 200000, 4000, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);
	STDCOUTL("----------------- SpinLock");
	data_buffer_perf<false>(clocks, 200000, 4000, 100, ::std::chrono::milliseconds(1), ::std::chrono::milliseconds(3));
	reportTime(clocks, hwc);
#endif
}
