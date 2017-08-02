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
#include "nn_base_arch.h"

using namespace nntl;
typedef nntl_supp::binfile reader_t;

template<typename ArchPrmsT>
struct GC_LPHG_NO : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	LIG<myInterfaces_t> lGate;
	myLFC l1;
	myLFC l2;
	LPHG<PHL<decltype(lGate)>,PHL<decltype(l1)>, PHL<decltype(l2)>> lFinal;

	~GC_LPHG_NO()noexcept {}
	GC_LPHG_NO(const ArchPrms_t& Prms)noexcept
		: lGate("lGate")
		, l1(70, Prms.learningRate, "l1")
		, l2(70, Prms.learningRate, "l2")
		, lFinal("lFinal"
			, make_PHL(lGate,0,2)
			, make_PHL(l1, 2, Prms.lUnderlay_nc / 2 - 2)
			, make_PHL(l2, Prms.lUnderlay_nc / 2, Prms.lUnderlay_nc - (Prms.lUnderlay_nc / 2))//to get rid of integer division rounding
		)
	{}
};
//This test should be run multiple times to test variuos gate "positions". gradcheck() routine could be updated to handle
//it automatically, but that require too much precious time I've already run out of.
TEST(TestLayerPackHorizontalGated, GradCheck_nonoverlapping) {
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	Prms.lUnderlay_nc = 400;
	nntl_tests::NN_arch<GC_LPHG_NO<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 10, 100);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.evalSetts.dLdA_setts.percOfZeros = 100;
	ngcSetts.evalSetts.dLdW_setts.percOfZeros = 100;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-3);//numeric errors due to dLdAPrev addition in LPH stacks up significantly
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 10, ngcSetts));
}


//////////////////////////////////////////////////////////////////////////
// most of tests below are extremely outdated

//////////////////////////////////////////////////////////////////////////
/*
template<bool b, typename _U, typename _G, class = ::std::void_t<> >
struct gate_type_obj {};
//specialization
template<bool b, typename _U, typename _G>
struct gate_type_obj<b, _U, _G, ::std::void_t<typename ::std::enable_if_t<!b> >> {
	typedef LPGFI<_U, _G> type;
	type value;
	gate_type_obj(_U&u, _G&g) : value(make_layer_pack_gated_from_input(u, g)) {}
};
template<bool b, typename _U, typename _G>
struct gate_type_obj<b, _U, _G, ::std::void_t<typename ::std::enable_if_t<b> >> {
	typedef LPG <_U, _G> type;
	type value;
	gate_type_obj(_U&u, _G&g) : value(make_layer_pack_gated(u, g)) {}
};

//NN implementation on layer_pack_gated
template<typename commonInfoT, bool bBinarize>
void comparative_gated(train_data<real_t>& td, const vec_len_t gateIdx, nnet_td_eval_results<real_t>& res, const uint64_t seedV = 0) {
	SCOPED_TRACE(::std::string("comparative_gated ") + (bBinarize ? "binarized" : "plain"));
	STDCOUTL("Working in comparative_gated, bBinarize is " << (bBinarize ? "TRUE" : "FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	layer_fully_connected<commonInfoT::act_hid> fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::act_hid> fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	LIG<> lid;

	layer_fully_connected<commonInfoT::act_hid> lFd1(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::act_hid> lFd2(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd = make_layer_pack_vertical(lFd1, lFd2);

	gate_type_obj<bBinarize, decltype(lFd), decltype(lid)> lGatedO(lFd, lid);

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, gateIdx),
		make_PHL(lid, gateIdx, 1),
		make_PHL(lGatedO.value, gateIdx + 1, td.train_x().cols_no_bias() - gateIdx - 1)
	);

	layer_output<commonInfoT::act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer_set_RMSProp_and_NM<commonInfoT> m;
		m(l);
	});

	nnet_train_opts<> opts(commonInfoT::epochs);
	opts.batchSize(commonInfoT::batchSize).NNEvalFinalResults(res); //ImmediatelyDeinit(false);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Training failed. Error code description: " << nn.get_last_error_string();

// 	ec = nn.td_eval(td, res);
// 	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Evaluation failed. Error code description: " << nn.get_last_error_string();
}*/
//////////////////////////////////////////////////////////////////////////

template<bool b, typename _U, typename _G, class = ::std::void_t<> >
struct horzgate_type_obj {};
//specialization
template<bool b, typename _U, typename _G>
struct horzgate_type_obj<b, _U, _G, ::std::void_t<typename ::std::enable_if_t<!b> >> {
	typedef LPHGFI< PHL<_G>, PHL<_U> > type;
	type value;
	horzgate_type_obj(_U&u, _G&g, const vec_len_t nc) : value(make_layer_pack_horizontal_gated_from_input(
		make_PHL(g,0,1),
		make_PHL(u,1,nc)
	)) {}
};
template<bool b, typename _U, typename _G>
struct horzgate_type_obj<b, _U, _G, ::std::void_t<typename ::std::enable_if_t<b> >> {
	typedef LPHG < PHL<_G>, PHL<_U> > type;
	type value;
	horzgate_type_obj(_U&u, _G&g, const vec_len_t nc) : value(make_layer_pack_horizontal_gated(
		make_PHL(g, 0, 1),
		make_PHL(u, 1, nc)
	)) {}
};

//NN implementation on layer_pack_horizontal_gated
template<typename commonInfoT, bool bBinarize>
void comparative_horzgated(train_data<real_t>& td, const vec_len_t gateIdx, nnet_td_eval_results<real_t>& res, const uint64_t seedV = 0) {
	SCOPED_TRACE(::std::string("comparative_horzgated ") + (bBinarize ? "binarized" : "plain"));
	STDCOUTL("Working in comparative_horzgated, bBinarize is " << (bBinarize ? "TRUE" : "FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	layer_fully_connected<commonInfoT::act_hid> fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::act_hid> fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	LIG<> lid;

	layer_fully_connected<commonInfoT::act_hid> lFd1(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::act_hid> lFd2(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd = make_layer_pack_vertical(lFd1, lFd2);

	horzgate_type_obj<bBinarize, decltype(lFd), decltype(lid)> lGatedO(lFd, lid, td.train_x().cols_no_bias() - gateIdx - 1);

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, gateIdx),
		make_PHL(lGatedO.value, gateIdx, td.train_x().cols_no_bias() - gateIdx)
	);

	layer_output<commonInfoT::act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer_set_RMSProp_and_NM<commonInfoT> m;
		m(l);
	});

	nnet_train_opts<> opts(commonInfoT::epochs);
	opts.batchSize(commonInfoT::batchSize).NNEvalFinalResults(res); //.ImmediatelyDeinit(false);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

// 	ec = nn.td_eval(td, res);
// 	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Evaluation failed. Error code description: " << nn.get_last_error_string();
}

struct TLPHG_simple {
	//typedef activation::sigm<> act_hid;
	//weights_init::XavierFour is suitable for very small layers
	typedef activation::sigm<real_t, weights_init::XavierFour> act_hid;
	//typedef activation::sigm_quad_loss<> act_outp;
	typedef activation::sigm_xentropy_loss<real_t, weights_init::XavierFour> act_outp;

#ifdef TESTS_SKIP_NNET_LONGRUNNING
	static constexpr size_t simple_l1nc = 40;
	static constexpr size_t simple_l2nc = 20;

	static constexpr size_t simple_lFd1nc = 15;
	static constexpr size_t simple_lFd2nc = 10;

	static constexpr size_t epochs = 5;
#else
	static constexpr size_t simple_l1nc = 50;
	static constexpr size_t simple_l2nc = 30;

	static constexpr size_t simple_lFd1nc = 20;
	static constexpr size_t simple_lFd2nc = 15;

	static constexpr size_t epochs = 7;
#endif

	static constexpr real_t learningRate = real_t(.001);
	static constexpr size_t batchSize = 100;

	static constexpr real_t nesterovMomentum = real_t(.9);
};

void run_comparativeSimple(train_data<real_t>& gatedTd, const vec_len_t gateIdx, const uint64_t seedV = 0) {
	SCOPED_TRACE("run_comparativeSimple");

	nnet_td_eval_results<real_t> gf, hf, gt, ht;
/*
	comparative_gated<TLPHG_simple, false>(gatedTd, gateIdx, gf, seedV);
	comparative_horzgated<TLPHG_simple, false>(gatedTd, gateIdx, hf, seedV);
	ASSERT_EQ(gf, hf) << "comparision between _gated and _horizontal_gated failed, binarization==false";

	comparative_gated<TLPHG_simple, true>(gatedTd, gateIdx, gt, seedV);
	ASSERT_EQ(gf, gt) << "comparision between _gated with and without binarization failed";*/

	comparative_horzgated<TLPHG_simple, true>(gatedTd, gateIdx, ht, seedV);
	//ASSERT_EQ(gt, ht) << "comparision between _gated and _horizontal_gated failed, binarization==true";
}

TEST(TestLayerPackHorizontalGated, Comparative) {
	const uint64_t seedV = 0;
	train_data<real_t> gatedTd, td;
	readTd(td);

	STDCOUTL("****** Single gate comparision *******");
	const auto gateIdx = td.train_x().cols_no_bias() / 2;
	makeTdForGatedSetup(td, gatedTd, seedV, true);

	STDCOUTL("With partially opened gate - should be exactly the same results");
	run_comparativeSimple(gatedTd, gateIdx, seedV);

	STDCOUTL("Gate completely closed - should be exactly the same results");
	gatedTd.train_x().fill_column_with(gateIdx, real_t(0.));
	gatedTd.test_x().fill_column_with(gateIdx, real_t(0.));
	run_comparativeSimple(gatedTd, gateIdx, seedV);

	STDCOUTL("Gate completely opened - should be exactly the same results");
	gatedTd.train_x().fill_column_with(gateIdx, real_t(1.));
	gatedTd.test_x().fill_column_with(gateIdx, real_t(1.));
	run_comparativeSimple(gatedTd, gateIdx, seedV);
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//NN implementation on layer_pack_gated
/*
template<typename commonInfoT, bool bBinarize>
void comparative_multi_gated(train_data<real_t>& td, const vec_len_t gateIdx, const vec_len_t gatesCnt
	, nnet_td_eval_results<real_t>& res, const uint64_t seedV = 0)
{
	SCOPED_TRACE(::std::string("comparative_multi_gated ") + (bBinarize ? "binarized" : "plain"));
	ASSERT_TRUE(gatesCnt == 3) << "The code expects only 3 gates here!";

	STDCOUTL("Working in comparative_multi_gated, bBinarize is " << (bBinarize ? "TRUE" : "FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	typedef layer_fully_connected<commonInfoT::act_hid> LH;
	//typedef LIG<> LIG;

	LH fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	LH fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	//1
	LIG lid1;
	LH lFd11(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	LH lFd21(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd1 = make_layer_pack_vertical(lFd11, lFd21);
	gate_type_obj<bBinarize, decltype(lFd1), decltype(lid1)> lGated1(lFd1, lid1);

	//2
	LIG lid2;
	LH lFd12(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	LH lFd22(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd2 = make_layer_pack_vertical(lFd12, lFd22);
	gate_type_obj<bBinarize, decltype(lFd2), decltype(lid2)> lGated2(lFd2, lid2);

	//3
	LIG lid3;
	LH lFd13(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	LH lFd23(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd3 = make_layer_pack_vertical(lFd13, lFd23);
	gate_type_obj<bBinarize, decltype(lFd3), decltype(lid3)> lGated3(lFd3, lid3);

	const vec_len_t totalXUnderGate = td.train_x().cols_no_bias() - gateIdx - gatesCnt;
	const vec_len_t usualWidth = totalXUnderGate / gatesCnt;

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, gateIdx),
		make_PHL(lid1, gateIdx, 1),
		make_PHL(lid2, gateIdx+1, 1),
		make_PHL(lid3, gateIdx+2, 1),
		make_PHL(lGated1.value, gateIdx + gatesCnt, usualWidth),
		make_PHL(lGated2.value, gateIdx + gatesCnt + usualWidth, usualWidth),
		make_PHL(lGated3.value, gateIdx + gatesCnt + 2 * usualWidth, totalXUnderGate - 2 * usualWidth)
	);

	layer_output<commonInfoT::act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer_set_RMSProp_and_NM<commonInfoT> m;
		m(l);
	});

	nnet_train_opts<> opts(commonInfoT::epochs);
	opts.batchSize(commonInfoT::batchSize).NNEvalFinalResults(res); //ImmediatelyDeinit(false);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Training failed. Error code description: " << nn.get_last_error_string();

	// 	ec = nn.td_eval(td, res);
	// 	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Evaluation failed. Error code description: " << nn.get_last_error_string();
}*/
//////////////////////////////////////////////////////////////////////////

template<bool b, typename U1,typename U2, typename U3, typename _G, class = ::std::void_t<> >
struct multihorzgate_type_obj {};
//specialization
template<bool b, typename U1, typename U2, typename U3, typename _G>
struct multihorzgate_type_obj<b, U1,U2,U3, _G, ::std::void_t<typename ::std::enable_if_t<!b> >> {
	typedef LPHGFI< PHL<_G>, PHL<U1>, PHL<U2>, PHL<U3> > type;
	type value;
	multihorzgate_type_obj(U1&u1, U2&u2, U3&u3, _G&g, vec_len_t nc1, vec_len_t nc2, vec_len_t nc3) : value(make_layer_pack_horizontal_gated_from_input(
		make_PHL(g, 0, 3),
		make_PHL(u1, 3, nc1),
		make_PHL(u2, 3 + nc1, nc2),
		make_PHL(u3, 3 + nc1 + nc2, nc3)
	)) {}
};
template<bool b, typename U1, typename U2, typename U3, typename _G>
struct multihorzgate_type_obj<b, U1, U2, U3, _G, ::std::void_t<typename ::std::enable_if_t<b> >> {
	typedef LPHG < PHL<_G>, PHL<U1>, PHL<U2>, PHL<U3> > type;
	type value;
	multihorzgate_type_obj(U1&u1, U2&u2, U3&u3, _G&g, vec_len_t nc1, vec_len_t nc2, vec_len_t nc3) : value(make_layer_pack_horizontal_gated(
		make_PHL(g, 0, 3),
		make_PHL(u1, 3, nc1),
		make_PHL(u2, 3 + nc1, nc2),
		make_PHL(u3, 3 + nc1 + nc2, nc3)
	)) {}
};

//NN implementation on layer_pack_horizontal_gated
template<typename commonInfoT, bool bBinarize>
void comparative_multi_horzgated(train_data<real_t>& td, const vec_len_t gateIdx, const vec_len_t gatesCnt
	, nnet_td_eval_results<real_t>& res, const uint64_t seedV = 0)
{
	SCOPED_TRACE(::std::string("comparative_multi_horzgated ") + (bBinarize ? "binarized" : "plain"));
	ASSERT_TRUE(gatesCnt == 3) << "The code expects only 3 gates here!";

	STDCOUTL("Working in comparative_horzgated, bBinarize is " << (bBinarize ? "TRUE" : "FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	typedef layer_fully_connected<commonInfoT::act_hid> LH;
	//typedef LIG<> LIG;

	LH fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	LH fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	LIG<> lid;

	LH lFd11(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	LH lFd21(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd1 = make_layer_pack_vertical(lFd11, lFd21);

	LH lFd12(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	LH lFd22(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd2 = make_layer_pack_vertical(lFd12, lFd22);

	LH lFd13(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	LH lFd23(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd3 = make_layer_pack_vertical(lFd13, lFd23);

	const vec_len_t totalXUnderGate = td.train_x().cols_no_bias() - gateIdx - gatesCnt;
	const vec_len_t usualWidth = totalXUnderGate / gatesCnt;

	multihorzgate_type_obj<bBinarize, decltype(lFd1), decltype(lFd2), decltype(lFd3),decltype(lid)> 
		lGatedO(lFd1, lFd2, lFd3, lid, usualWidth, usualWidth, totalXUnderGate - 2 * usualWidth);

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, gateIdx),
		make_PHL(lGatedO.value, gateIdx, td.train_x().cols_no_bias() - gateIdx)
	);

	layer_output<commonInfoT::act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer_set_RMSProp_and_NM<commonInfoT> m;
		m(l);
	});

	nnet_train_opts<> opts(commonInfoT::epochs);
	opts.batchSize(commonInfoT::batchSize).NNEvalFinalResults(res); //.ImmediatelyDeinit(false);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

	// 	ec = nn.td_eval(td, res);
	// 	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Evaluation failed. Error code description: " << nn.get_last_error_string();
}

struct TLPHG_multi {
	//typedef activation::sigm<> act_hid;
	//weights_init::XavierFour is suitable for very small layers
	typedef activation::sigm<real_t, weights_init::XavierFour> act_hid;
	//typedef activation::sigm_quad_loss<> act_outp;
	typedef activation::sigm_xentropy_loss<real_t, weights_init::XavierFour> act_outp;

#ifdef TESTS_SKIP_NNET_LONGRUNNING
	static constexpr size_t simple_l1nc = 40;
	static constexpr size_t simple_l2nc = 20;

	static constexpr size_t simple_lFd1nc = 15;
	static constexpr size_t simple_lFd2nc = 10;

	static constexpr size_t epochs = 5;
#else
	static constexpr size_t simple_l1nc = 50;
	static constexpr size_t simple_l2nc = 30;

	static constexpr size_t simple_lFd1nc = 20;
	static constexpr size_t simple_lFd2nc = 15;

	static constexpr size_t epochs = 7;
#endif

	static constexpr real_t learningRate = real_t(.001);
	static constexpr size_t batchSize = 100;

	static constexpr real_t nesterovMomentum = real_t(.9);
};

void run_comparativeMulti(train_data<real_t>& gatedTd, const vec_len_t gateIdx, const vec_len_t gatesCnt, const uint64_t seedV = 0) {
	SCOPED_TRACE("run_comparativeMulti");

	nnet_td_eval_results<real_t> gf, hf, gt, ht;
/*
	comparative_multi_gated<TLPHG_multi, false>(gatedTd, gateIdx, gatesCnt, gf, seedV);
	comparative_multi_horzgated<TLPHG_multi, false>(gatedTd, gateIdx, gatesCnt, hf, seedV);
	ASSERT_EQ(gf, hf) << "comparision between _gated and _horizontal_gated failed, binarization==false";

	comparative_multi_gated<TLPHG_multi, true>(gatedTd, gateIdx, gatesCnt, gt, seedV);
	ASSERT_EQ(gf, gt) << "comparision between _gated with and without binarization failed";*/

	comparative_multi_horzgated<TLPHG_multi, true>(gatedTd, gateIdx, gatesCnt, ht, seedV);
	//ASSERT_EQ(gt, ht) << "comparision between _gated and _horizontal_gated failed, binarization==true";
}

TEST(TestLayerPackHorizontalGated, ComparativeMultigate) {
	const uint64_t seedV = 0;
	const vec_len_t gatesCnt = 3;
	train_data<real_t> gatedTd, td;
	readTd(td);

	const auto gateIdx = td.train_x().cols_no_bias() / 4;
	makeTdForGatedSetup(td, gatedTd, seedV, true, gatesCnt);

	STDCOUTL("With partially opened gate - should be exactly the same results");
	run_comparativeMulti(gatedTd, gateIdx, gatesCnt, seedV);

	STDCOUTL("Gate completely closed - should be exactly the same results");
	for (unsigned i = 0; i < gatesCnt; ++i) {
		gatedTd.train_x().fill_column_with(gateIdx + i, real_t(0.));
		gatedTd.test_x().fill_column_with(gateIdx + i, real_t(0.));
	}
	run_comparativeMulti(gatedTd, gateIdx, gatesCnt, seedV);

	STDCOUTL("Gate completely opened - should be exactly the same results");
	for (unsigned i = 0; i < gatesCnt; ++i) {
		gatedTd.train_x().fill_column_with(gateIdx + i, real_t(1.));
		gatedTd.test_x().fill_column_with(gateIdx + i, real_t(1.));
	}
	run_comparativeMulti(gatedTd, gateIdx, gatesCnt, seedV);
}