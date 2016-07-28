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

using namespace nntl;
typedef nntl_supp::binfile reader_t;

#define MNIST_FILE_DEBUG "../data/mnist200_100.bin"
#define MNIST_FILE_RELEASE  "../data/mnist60000.bin"

#if defined(TESTS_SKIP_NNET_LONGRUNNING)
//ALWAYS run debug build with similar relations of data sizes:
// if release will run in minibatches - make sure, there will be at least 2 minibatches)
// if release will use different data sizes for train/test - make sure, debug will also run on different datasizes
#define MNIST_FILE MNIST_FILE_DEBUG
#else
#define MNIST_FILE MNIST_FILE_RELEASE
#endif // _DEBUG


void readTd(train_data<real_t>& td) {
	SCOPED_TRACE("readTd");
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());
}

//////////////////////////////////////////////////////////////////////////
//

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

void _maskStat(const realmtx_t& m) {
	auto pD = m.data();
	size_t c = 0;
	const size_t ne = m.numel();
	for (size_t i = 0; i < ne; ++i) {
		if (pD[i] > 0) ++c;
	}
	STDCOUTL("Gating mask opens " << c << " samples. Total samples count is " << ne );
}

template<typename iRngT, typename iMathT>
void makeDataXForGatedSetup(const realmtx_t& data_x, const realmtx_t& data_y, iRngT& iR, iMathT& iM, const bool bBinarize, realmtx_t& new_x) {
	SCOPED_TRACE("makeDataXForGatedSetup");

	realmtx_t mask(data_x.rows(),1,false), realMask(data_x.rows(), 1, false);
	ASSERT_TRUE(!mask.isAllocationFailed() && !realMask.isAllocationFailed());
	iR.gen_matrix_norm(mask);
	if (bBinarize) iM.ewBinarize_ip(mask, real_t(.5));

	realMask.zeros();
	_allowMask(mask, realMask, data_y, 1);
	_allowMask(mask, realMask, data_y, 7);

	_maskStat(realMask);
	
	const auto origXWidth = data_x.cols_no_bias();

	new_x.will_emulate_biases();
	ASSERT_TRUE(new_x.resize(data_x.rows(), origXWidth + 1 + 2)) << "Failed to resize new_x";
	
	memcpy(new_x.data(), data_x.data(), data_x.byte_size_no_bias());
	memcpy(new_x.colDataAsVec(origXWidth), realMask.data(), realMask.byte_size());
	memcpy(new_x.colDataAsVec(origXWidth + 1), data_y.colDataAsVec(1), data_y.rows() * sizeof(real_t));
	memcpy(new_x.colDataAsVec(origXWidth + 2), data_y.colDataAsVec(7), data_y.rows() * sizeof(real_t));
}

void makeTdForGatedSetup(const train_data<real_t>& td, train_data<real_t>& tdGated, const bool bBinarize) {
	SCOPED_TRACE("makeTdForGatedSetup");

	nnet_def_interfaces::iMath_t iM;
	nnet_def_interfaces::iRng_t iR;
	iR.set_ithreads(iM.ithreads());

	realmtx_t ntr, nt, ntry,nty;
	makeDataXForGatedSetup(td.train_x(), td.train_y(), iR, iM, bBinarize, ntr);
	makeDataXForGatedSetup(td.test_x(), td.test_y(), iR, iM, bBinarize, nt);
	td.train_y().cloneTo(ntry);
	td.test_y().cloneTo(nty);

	ASSERT_TRUE(tdGated.absorb(std::move(ntr), std::move(ntry), std::move(nt), std::move(nty)));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename _I>
struct modify_layer {
	template<typename _L>
	std::enable_if_t<layer_has_gradworks<_L>::value> operator()(_L& l)noexcept {
		l.m_gradientWorks.set_type(decltype(l.m_gradientWorks)::RMSProp_Hinton).set_nesterov_momentum(_I::nesterovMomentum);
		//	.set_L2(l2).set_max_norm(0).set_ILR(.91, 1.05, .0001, 10000);
	}

	template<typename _L>
	std::enable_if_t<!layer_has_gradworks<_L>::value> operator()(_L& l)noexcept {}
};

//trains a baseline network
template <typename commonInfoT>
void simple_BaselineNN(train_data<real_t>& td, const uint64_t seedV=0) {
	SCOPED_TRACE("simple_BaselineNN");
	STDCOUTL("Working in simple_BaselineNN()");
	layer_input<> inp(td.train_x().cols_no_bias());
	layer_fully_connected<commonInfoT::simple_act_hid> fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::simple_act_hid> fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	layer_output<commonInfoT::simple_act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, fcl, fcl2, outp);
	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer<commonInfoT> m;
		m(l);
	});

	nnet_cond_epoch_eval cee(commonInfoT::epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));
	opts.batchSize(commonInfoT::batchSize);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}

//////////////////////////////////////////////////////////////////////////
template<bool b, typename _U, typename _G, class = std::void_t<> >
struct gate_type_obj {};
//specialization
template<bool b, typename _U, typename _G>
struct gate_type_obj<b,_U,_G,std::void_t<typename std::enable_if_t<!b> >> {
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


template<typename commonInfoT, bool bBinarize>
void simple_gatedNN(train_data<real_t>& td, const vec_len_t origXWidth, const uint64_t seedV=0) {
	SCOPED_TRACE(std::string("simple_gatedNN ")+(bBinarize? "binarized":"plain"));
	STDCOUTL("Working in simple_gatedNN, bBinarize is " << (bBinarize ?"TRUE":"FALSE"));

	layer_input<> inp(td.train_x().cols_no_bias());

	layer_fully_connected<commonInfoT::simple_act_hid> fcl(commonInfoT::simple_l1nc, commonInfoT::learningRate);
	layer_fully_connected<commonInfoT::simple_act_hid> fcl2(commonInfoT::simple_l2nc, commonInfoT::learningRate);
	auto lBaseline = make_layer_pack_vertical(fcl, fcl2);

	layer_identity_gate<> lid;

	//layer_fully_connected<commonInfoT::simple_act_hid> lFd1(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);
	//layer_fully_connected<commonInfoT::simple_act_hid> lFd2(commonInfoT::simple_lFd2nc, commonInfoT::learningRate);
	//auto lFd = make_layer_pack_vertical(lFd1, lFd2);
	
	layer_fully_connected<commonInfoT::simple_act_hid> lFd(commonInfoT::simple_lFd1nc, commonInfoT::learningRate);

	gate_type_obj<bBinarize, decltype(lFd), decltype(lid)> lGatedO(lFd,lid);

	auto lFirst = make_layer_pack_horizontal(
		make_PHL(lBaseline, 0, origXWidth),
		make_PHL(lid, origXWidth, 1),
		make_PHL(lGatedO.value, origXWidth + 1, td.train_x().cols_no_bias() - origXWidth - 1)
	);

	layer_output<commonInfoT::simple_act_outp> outp(td.train_y().cols(), commonInfoT::learningRate);

	auto lp = make_layers(inp, lFirst, outp);

	lp.for_each_layer_exc_input([](auto& l) {
		modify_layer<commonInfoT> m;
		m(l);
	});

	nnet_cond_epoch_eval cee(commonInfoT::epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));
	opts.batchSize(commonInfoT::batchSize);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedV);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

}

struct simple_case_common_info {
	//typedef activation::sigm<> simple_act_hid;
	//weights_init::XavierFour is suitable for very small layers
	typedef activation::sigm<weights_init::XavierFour> simple_act_hid;
	//typedef activation::sigm_quad_loss<> simple_act_outp;
	typedef activation::sigm_xentropy_loss<weights_init::XavierFour> simple_act_outp;

#ifdef TESTS_SKIP_NNET_LONGRUNNING
	static constexpr size_t simple_l1nc = 30;
	static constexpr size_t simple_l2nc = 10;

	static constexpr size_t simple_lFd1nc = 2;
	static constexpr size_t simple_lFd2nc = 10;

	static constexpr size_t epochs = 5;
#else
	static constexpr size_t simple_l1nc = 30;
	static constexpr size_t simple_l2nc = 10;

	static constexpr size_t simple_lFd1nc = 2;
	static constexpr size_t simple_lFd2nc = 20;

	static constexpr size_t epochs = 20;
#endif

	static constexpr real_t learningRate = .001;
	static constexpr size_t batchSize = 100;

	static constexpr real_t nesterovMomentum = .9;
};

//this is probably the worst test case ever (not a test case at all), but anyway... #todo !!!
TEST(TestLayerPackGated, Simple) {
	const uint64_t seedV = 0;
	train_data<real_t> td, gatedTd;
	readTd(td);

	simple_BaselineNN<simple_case_common_info>(td, seedV);

	const auto origXWidth = td.train_x().cols_no_bias();
	makeTdForGatedSetup(td, gatedTd, true);

	STDCOUTL("With partially opened gate");
	//simple_gatedNN<simple_case_common_info,true>(gatedTd, origXWidth, seedV);
	simple_gatedNN<simple_case_common_info, false>(gatedTd, origXWidth, seedV);

	STDCOUTL("Gate completely closed");
	gatedTd.train_x().fill_column_with(origXWidth, real_t(0.));
	gatedTd.test_x().fill_column_with(origXWidth, real_t(0.));
	//simple_gatedNN<simple_case_common_info, true>(gatedTd, origXWidth, seedV);
	simple_gatedNN<simple_case_common_info, false>(gatedTd, origXWidth, seedV);

	STDCOUTL("Gate completely opened");
	gatedTd.train_x().fill_column_with(origXWidth, real_t(1.));
	gatedTd.test_x().fill_column_with(origXWidth, real_t(1.));
	//simple_gatedNN<simple_case_common_info, true>(gatedTd, origXWidth, seedV);
	simple_gatedNN<simple_case_common_info, false>(gatedTd, origXWidth, seedV);
}