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

//to get rid of '... decorated name length exceeded, name was truncated'
#pragma warning( disable : 4503 )

#include "../nntl/math.h"
#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_test/test_weights_init.h"
#include "asserts.h"
#include "common_routines.h"
//#include "nn_base_arch.h"

using namespace nntl;

template<typename real_t>
void _maskStat(const math::smatrix<real_t> &m) {
	auto pD = m.data();
	size_t c = 0;
	const size_t ne = m.numel();
	for (size_t i = 0; i < ne; ++i) {
		if (pD[i] > 0) ++c;
	}
	STDCOUTL("Gating mask opens on average " << real_t(c) / m.cols() << " samples for each gate. Total gates count=" << m.cols());
}


/*
//copies srcMask element into mask element if corresponding column element of data_y is nonzero. Column to use is selected by c
void _allowMask(const realmtx_t& srcMask, realmtx_t& mask, const realmtx_t& data_y, const vec_len_t c) {
	auto pSrcM = srcMask.data();
	auto pM = mask.data();
	auto pY = data_y.colDataAsVec(c);
	const size_t _rm = data_y.rows();
	for (size_t r = 0; r < _rm; ++r) {
		if (pY[r]) pM[r] = pSrcM[r];
	}
}
*/

template<typename iMathT, typename iRngT>
void makeDataXForGatedSetup(iMathT& iM, iRngT& iR
	, const math::smatrix<typename iMathT::real_t>& data_x, math::smatrix<typename iMathT::real_t>& new_x
	, const vec_len_t gatesCnt, typename iMathT::real_t gateZeroProb)
{
	typedef typename iMathT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;

	SCOPED_TRACE("makeDataXForGatedSetup");

	realmtx_t realMask(data_x.rows(), gatesCnt, false);
	ASSERT_TRUE(!realMask.isAllocationFailed());
	iR.gen_matrix_norm(realMask);
	if (gateZeroProb < real_t(1)) {
		iM.ewBinarize_ip(realMask, gateZeroProb);
		_maskStat(realMask);
	} else STDCOUTL("Won't binarize the gating mask!");
	
	ASSERT_TRUE(new_x.empty());
	new_x.will_emulate_biases();
	ASSERT_TRUE(new_x.resize(data_x.rows(), data_x.cols_no_bias() + gatesCnt)) << "Failed to resize new_x";

	::std::memcpy(new_x.data(), realMask.data(), realMask.byte_size());
	::std::memcpy(new_x.colDataAsVec(gatesCnt), data_x.data(), data_x.byte_size_no_bias());
	ASSERT_TRUE(new_x.test_biases_strict());
}

template<typename iMathT, typename iRngT>
void makeTdForGatedSetup(iMathT& iM, iRngT& iR
	, const train_data<typename iMathT::real_t>& td, train_data<typename iMathT::real_t>& tdGated
	, const vec_len_t gatesCnt, typename iMathT::real_t gateZeroProb)
{
	typedef typename iMathT::real_t real_t;
	typedef math::smatrix<real_t> realmtx_t;

	SCOPED_TRACE("makeTdForGatedSetup");

	realmtx_t ntr, nt, ntry, nty;
	makeDataXForGatedSetup(iM, iR, td.train_x(), ntr, gatesCnt, gateZeroProb);
	makeDataXForGatedSetup(iM, iR, td.test_x(), nt, gatesCnt, gateZeroProb);

	td.train_y().clone_to(ntry);
	td.test_y().clone_to(nty);

	ASSERT_TRUE(tdGated.absorb(::std::move(ntr), ::std::move(ntry), ::std::move(nt), ::std::move(nty)));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename P>
struct TLPHO_simple_arch {
	typedef P params_t;
	typedef typename P::real_t real_t;
	
	typedef ::std::conditional_t<::std::is_void<typename P::custom_interfaces_t>::value
		, dt_interfaces<real_t>
		, typename P::custom_interfaces_t
	> myIntf;
	static_assert(::std::is_same<real_t, typename myIntf::real_t>::value, "real_t mismatch!");

	typedef grad_works_f<myIntf, GW::ILR_dummy, GW::Loss_Addendums_dummy> myGW;

	typedef activation::softsigm<real_t> act_hid;
	typedef activation::softsigm_quad_loss<real_t> act_outp;

	typedef LFC<act_hid, myGW> LH;

	layer_input<myIntf> lInp;

	LIGf<P::gateZeroProb1e6, P::bBinarizeGate, myIntf> lGate;

	struct lInner {
		typedef LPVt<::std::tuple<LH&, LH&>> LFinal_t;

		LH lh1;
		LH lh2;
		LFinal_t lV;

		lInner() noexcept 
			: lh1(P::simple_l1nc, P::learningRate)
			, lh2(P::simple_l2nc, P::learningRate)
			, lV(::std::tie(lh1,lh2))
		{}
	};
	::std::array<lInner, P::nGatesCnt> aInnerLyrs;

	static_assert(P::nGatesCnt == 3, "Change the code below");
	LPHOt<P::bAddFeatureNotPresent, ::std::tuple<PHL<decltype(lGate)>,
		PHL<typename lInner::LFinal_t>, PHL<typename lInner::LFinal_t>, PHL<typename lInner::LFinal_t>>> lLpho;

	LH lOvr;

	layer_output<act_outp, myGW> lOutp;

	layers<decltype(lInp), decltype(lLpho), decltype(lOvr), decltype(lOutp)> lp;


	//////////////////////////////////////////////////////////////////////////
	TLPHO_simple_arch(const train_data<real_t>& td) noexcept
		: lInp(td.train_x().cols_no_bias())
		, lLpho(::std::make_tuple(
			make_PHL(lGate, 0, P::nGatesCnt)
			, make_PHL(aInnerLyrs[0].lV, P::nGatesCnt + 0 * (td.train_x().cols_no_bias() / P::nGatesCnt), td.train_x().cols_no_bias() / P::nGatesCnt)
			, make_PHL(aInnerLyrs[1].lV, P::nGatesCnt + 1 * (td.train_x().cols_no_bias() / P::nGatesCnt), td.train_x().cols_no_bias() / P::nGatesCnt)
			, make_PHL(aInnerLyrs[2].lV, P::nGatesCnt + 2 * (td.train_x().cols_no_bias() / P::nGatesCnt), td.train_x().cols_no_bias() - P::nGatesCnt - 2 * (td.train_x().cols_no_bias() / P::nGatesCnt))
		))
		, lOvr(P::simple_lFd1nc, P::learningRate)
		, lOutp(td.train_y().cols(), P::learningRate)
		, lp(lInp, lLpho, lOvr, lOutp)
	{
		lp.for_each_layer([](auto& l)noexcept {
			LayerInit()(l);
		});
	}

	//////////////////////////////////////////////////////////////////////////
	struct LayerInit {
		template<typename _L> ::std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l)noexcept {
			l.m_gradientWorks
				.set_type(decltype(l.m_gradientWorks)::Adam)
				.beta1(real_t(.9))
				.beta2(real_t(.9))
				;
		}
		template<typename _L> ::std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L&)noexcept {}
	};
};

template<typename RealT, int iGateZeroProb1e6, bool bBinGate, bool bAddFeatureNotPres, typename CustIntfT = void>
struct TLPHO_simple_prms {
	typedef RealT real_t;
	typedef CustIntfT custom_interfaces_t;

	static_assert(iGateZeroProb1e6 >= 0, "");
	static constexpr bool bBinarizeGate = bBinGate;
	static constexpr int gateZeroProb1e6 = bBinarizeGate ? iGateZeroProb1e6 : 0;
	static constexpr real_t gateZeroProb = bBinarizeGate ? real_t(1.) : real_t(iGateZeroProb1e6) / real_t(1e6);

	static constexpr bool bAddFeatureNotPresent = bAddFeatureNotPres;

	static constexpr unsigned int nGatesCnt = 3;

#ifdef TESTS_SKIP_NNET_LONGRUNNING
	static constexpr size_t simple_l1nc = 40;
	static constexpr size_t simple_l2nc = 20;

	static constexpr size_t simple_lFd1nc = 15;
	
	static constexpr size_t epochs = 5;
#else
	static constexpr size_t simple_l1nc = 50;
	static constexpr size_t simple_l2nc = 30;

	static constexpr size_t simple_lFd1nc = 20;
	
	static constexpr size_t epochs = 7;
#endif

	static constexpr real_t learningRate = real_t(.001);
	static constexpr size_t batchSize = 100;

	static constexpr real_t nesterovMomentum = real_t(.9);
};

template<typename ParamsT>
void run_testLPHO_simple(const train_data<typename ParamsT::real_t>& baseTd, const size_t seedV
	, typename ParamsT::real_t& finErr)noexcept
{
	typedef typename ParamsT::real_t real_t;
	typedef TLPHO_simple_arch<ParamsT> Arch_t;

	STDCOUTL("gateZeroProb = " << (ParamsT::bBinarizeGate ? ParamsT::gateZeroProb1e6 / real_t(1e6) : ParamsT::gateZeroProb)
		<< ", bBinarizeGate = " << ParamsT::bBinarizeGate << ", bAddFeatureNotPresent = " << ParamsT::bAddFeatureNotPresent);

	train_data<real_t> td;
	{
		typedef typename Arch_t::myIntf::iMath_t iMath_t;
		typedef typename Arch_t::myIntf::iRng_t iRng_t;
		iMath_t iM;
		iRng_t iR;
		iR.init_ithreads(iM.ithreads(), iRng_t::s64to32(seedV));
		makeTdForGatedSetup(iM, iR, baseTd, td, ParamsT::nGatesCnt, ParamsT::gateZeroProb);
	}

	Arch_t Arch(td);
	nnet_train_opts<training_observer_stdcout<eval_classification_one_hot<real_t>>> opts(ParamsT::epochs);
	opts.batchSize(ParamsT::batchSize);

	auto nn = make_nnet(Arch.lp);
	nn.get_iRng().seed64(seedV + 1);

	auto ec = nn.train<false>(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

	finErr = opts.observer().m_lastTrainErr;
}


//The idea of this test is to assemble some simple architectures with LPHO and see whether it compiles and produces
//approximately right (expected) output. It doesn't make a full-scale testing of LPHO, but that should be enough to
// see whether it works or not.
TEST(TestLayerPackHorizontalOptional, Simple) {
	typedef float real_t;
// 	template<int iGateZeroProb1e6, bool bBinGate, bool bAddFeatureNotPres>
// 	using Prms_t = TLPHO_simple_prms<real_t, iGateZeroProb1e6, bBinGate, bAddFeatureNotPres>;

	train_data<real_t> baseTd;
	readTd(baseTd);

	size_t seedV = ::std::time(0);
	real_t e1, e2, e3, e4;

	STDCOUTL("************* Fully opened gate.");
	run_testLPHO_simple<TLPHO_simple_prms<real_t, 0, false, false>>(baseTd, seedV, e1);
	run_testLPHO_simple<TLPHO_simple_prms<real_t, 0, true, false>>(baseTd, seedV, e2);
	ASSERT_EQ(e1, e2) << "with/without gate binarization results mismatch!";

	run_testLPHO_simple<TLPHO_simple_prms<real_t, 0, false, true>>(baseTd, seedV, e3);
	run_testLPHO_simple<TLPHO_simple_prms<real_t, 0, true, true>>(baseTd, seedV, e4);
	ASSERT_EQ(e3, e4) << "with/without gate binarization results mismatch!";

	STDCOUTL("************* 1/3 closed gate.");
	static constexpr int g1 = 333333;
	run_testLPHO_simple<TLPHO_simple_prms<real_t, g1, false, false>>(baseTd, seedV, e1);
	run_testLPHO_simple<TLPHO_simple_prms<real_t, g1, true, false>>(baseTd, seedV, e2);
	ASSERT_EQ(e1, e2) << "with/without gate binarization results mismatch!";

	run_testLPHO_simple<TLPHO_simple_prms<real_t, g1, false, true>>(baseTd, seedV, e3);
	run_testLPHO_simple<TLPHO_simple_prms<real_t, g1, true, true>>(baseTd, seedV, e4);
	ASSERT_EQ(e3, e4) << "with/without gate binarization results mismatch!";

	STDCOUTL("************* 2/3 closed gate.");
	static constexpr int g2 = 666666;
	run_testLPHO_simple<TLPHO_simple_prms<real_t, g2, false, false>>(baseTd, seedV, e1);
	run_testLPHO_simple<TLPHO_simple_prms<real_t, g2, true, false>>(baseTd, seedV, e2);
	ASSERT_EQ(e1, e2) << "with/without gate binarization results mismatch!";

	run_testLPHO_simple<TLPHO_simple_prms<real_t, g2, false, true>>(baseTd, seedV, e3);
	run_testLPHO_simple<TLPHO_simple_prms<real_t, g2, true, true>>(baseTd, seedV, e4);
	ASSERT_EQ(e3, e4) << "with/without gate binarization results mismatch!";
}

#if NNTL_MATLAB_AVAILABLE

#include "../nntl/_supp/io/matfile.h"
#include "../nntl/interface/inspectors/dumper.h"

//to visually inspect what's inside
TEST(TestLayerPackHorizontalOptional, MakeDump) {
	typedef float real_t;
	struct my_interfaces : public d_int_nI<real_t> {
		typedef inspector::dumper<real_t, nntl_supp::omatfileEx<>, inspector::conds::EpochNum> iInspect_t;
	};
	static constexpr int g1 = 333333;
	typedef TLPHO_simple_prms<real_t, g1, false, true, my_interfaces> ParamsT;
	typedef TLPHO_simple_arch<ParamsT> Arch_t;

	train_data<real_t> baseTd, td;
	readTd(baseTd);
	size_t seedV = ::std::time(0);

	{
		typedef typename Arch_t::myIntf::iMath_t iMath_t;
		typedef typename Arch_t::myIntf::iRng_t iRng_t;
		iMath_t iM;
		iRng_t iR;
		iR.init_ithreads(iM.ithreads(), iRng_t::s64to32(seedV));
		makeTdForGatedSetup(iM, iR, baseTd, td, ParamsT::nGatesCnt, ParamsT::gateZeroProb);
	}

	Arch_t Arch(td);
	nnet_train_opts<training_observer_stdcout<eval_classification_one_hot<real_t>>> opts(ParamsT::epochs);
	opts.batchSize(ParamsT::batchSize);

	my_interfaces::iInspect_t Insp("./test_data");
	Insp.getCondDump().m_epochsToDump = { 9999999999999999999 };//the first batch of the last epoch

	auto nn = make_nnet(Arch.lp, Insp);
	nn.get_iRng().seed64(seedV + 1);

	auto ec = nn.train<false>(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}


#endif