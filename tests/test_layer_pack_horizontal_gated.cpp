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
#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_test/test_weights_init.h"
#include "asserts.h"
#include "common_routines.h"

using namespace nntl;
typedef nntl_supp::binfile reader_t;

//////////////////////////////////////////////////////////////////////////
//

//////////////////////////////////////////////////////////////////////////
template<bool b, typename _U, typename _G, class = std::void_t<> >
struct gate_type_obj {};
//specialization
template<bool b, typename _U, typename _G>
struct gate_type_obj<b, _U, _G, std::void_t<typename std::enable_if_t<!b> >> {
	typedef LPGFI<_U, _G> type;
	type value;
	gate_type_obj(_U&u, _G&g) : value(make_layer_pack_gated_from_input(u, g)) {}
};
template<bool b, typename _U, typename _G>
struct gate_type_obj<b, _U, _G, std::void_t<typename std::enable_if_t<b> >> {
	typedef LPG <_U, _G> type;
	type value;
	gate_type_obj(_U&u, _G&g) : value(make_layer_pack_gated(u, g)) {}
};

//NN implementation on layer_pack_gated
template<typename commonInfoT, bool bBinarize>
void comparative_gated(train_data<real_t>& td, const vec_len_t gateIdx, nnet_td_eval_results<real_t>& res, const uint64_t seedV = 0) {
	SCOPED_TRACE(std::string("comparative_gated ") + (bBinarize ? "binarized" : "plain"));
	STDCOUTL("Working in comparative_gated, bBinarize is " << (bBinarize ? "TRUE" : "FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	layer_fully_connected<commonInfoT::simple_act_hid> fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::simple_act_hid> fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	layer_identity_gate<> lid;

	layer_fully_connected<commonInfoT::simple_act_hid> lFd1(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::simple_act_hid> lFd2(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd = make_layer_pack_vertical(lFd1, lFd2);

	gate_type_obj<bBinarize, decltype(lFd), decltype(lid)> lGatedO(lFd, lid);

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, gateIdx),
		make_PHL(lid, gateIdx, 1),
		make_PHL(lGatedO.value, gateIdx + 1, td.train_x().cols_no_bias() - gateIdx - 1)
	);

	layer_output<commonInfoT::simple_act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer_set_RMSProp_and_NM<commonInfoT> m;
		m(l);
	});

	nnet_cond_epoch_eval cee(commonInfoT::epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));
	opts.batchSize(commonInfoT::batchSize).NNEvalFinalResults(res); //ImmediatelyDeinit(false);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Training failed. Error code description: " << nn.get_last_error_string();

// 	ec = nn.td_eval(td, res);
// 	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Evaluation failed. Error code description: " << nn.get_last_error_string();
}
//////////////////////////////////////////////////////////////////////////

template<bool b, typename _U, typename _G, class = std::void_t<> >
struct horzgate_type_obj {};
//specialization
template<bool b, typename _U, typename _G>
struct horzgate_type_obj<b, _U, _G, std::void_t<typename std::enable_if_t<!b> >> {
	typedef LPHGFI< PHL<_G>, PHL<_U> > type;
	type value;
	horzgate_type_obj(_U&u, _G&g, const vec_len_t nc) : value(make_layer_pack_horizontal_gated_from_input(
		make_PHL(g,0,1),
		make_PHL(u,1,nc)
	)) {}
};
template<bool b, typename _U, typename _G>
struct horzgate_type_obj<b, _U, _G, std::void_t<typename std::enable_if_t<b> >> {
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
	SCOPED_TRACE(std::string("comparative_horzgated ") + (bBinarize ? "binarized" : "plain"));
	STDCOUTL("Working in comparative_horzgated, bBinarize is " << (bBinarize ? "TRUE" : "FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	layer_fully_connected<commonInfoT::simple_act_hid> fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::simple_act_hid> fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	layer_identity_gate<> lid;

	layer_fully_connected<commonInfoT::simple_act_hid> lFd1(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::simple_act_hid> lFd2(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	auto lFd = make_layer_pack_vertical(lFd1, lFd2);

	horzgate_type_obj<bBinarize, decltype(lFd), decltype(lid)> lGatedO(lFd, lid, td.train_x().cols_no_bias() - gateIdx - 1);

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, gateIdx),
		make_PHL(lGatedO.value, gateIdx, td.train_x().cols_no_bias() - gateIdx)
	);

	layer_output<commonInfoT::simple_act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer_set_RMSProp_and_NM<commonInfoT> m;
		m(l);
	});

	nnet_cond_epoch_eval cee(commonInfoT::epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));
	opts.batchSize(commonInfoT::batchSize).NNEvalFinalResults(res); //.ImmediatelyDeinit(false);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

// 	ec = nn.td_eval(td, res);
// 	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Evaluation failed. Error code description: " << nn.get_last_error_string();
}

struct simple_case_common_info {
	//typedef activation::sigm<> simple_act_hid;
	//weights_init::XavierFour is suitable for very small layers
	typedef activation::sigm<weights_init::XavierFour> simple_act_hid;
	//typedef activation::sigm_quad_loss<> simple_act_outp;
	typedef activation::sigm_xentropy_loss<weights_init::XavierFour> simple_act_outp;

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

	static constexpr real_t learningRate = .001;
	static constexpr size_t batchSize = 100;

	static constexpr real_t nesterovMomentum = .9;
};

void run_comparativeSimple(train_data<real_t>& gatedTd, const vec_len_t gateIdx, const uint64_t seedV = 0) {
	SCOPED_TRACE("run_comparativeSimple");

	nnet_td_eval_results<real_t> gf, hf, gt, ht;
	comparative_gated<simple_case_common_info, false>(gatedTd, gateIdx, gf, seedV);
	comparative_horzgated<simple_case_common_info, false>(gatedTd, gateIdx, hf, seedV);
	ASSERT_EQ(gf, hf) << "comparision between _gated and _horizontal_gated failed, binarization==false";

	comparative_gated<simple_case_common_info, true>(gatedTd, gateIdx, gt, seedV);
	ASSERT_EQ(gf, gt) << "comparision between _gated with and without binarization failed";

	comparative_horzgated<simple_case_common_info, true>(gatedTd, gateIdx, ht, seedV);
	ASSERT_EQ(gt, ht) << "comparision between _gated and _horizontal_gated failed, binarization==true";
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