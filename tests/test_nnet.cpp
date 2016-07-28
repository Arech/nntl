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
// tests.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "../nntl/math.h"
#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_supp/io/matfile.h"

#include "asserts.h"

using namespace nntl;
typedef nntl_supp::binfile reader_t;
//using real_t = math_types::real_ty;

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

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


/*
TEST(TestNnet, ) {
	

}*/


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


void test_LayerPackVertical1(train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical1");
	size_t epochs = 5;
	const real_t learningRate = .01;

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Aifcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Aifcl1, Aifcl2, Aoutp);

	nnet_cond_epoch_eval Acee(epochs);
	nnet_train_opts<decltype(Acee)> Aopts(std::move(Acee));
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();


	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bifcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl1, Bifcl2);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, BlpVert, Boutp);

	nnet_cond_epoch_eval Bcee(epochs);
	nnet_train_opts<decltype(Bcee)> Bopts(std::move(Bcee));
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Bnn = make_nnet(Blp);
	Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();

	ASSERT_MTX_EQ(Aifcl1.get_weights(), Bifcl1.get_weights(),"First layer weights differ");
	ASSERT_MTX_EQ(Aifcl2.get_weights(), Bifcl2.get_weights(), "Second layer weights differ");
	ASSERT_MTX_EQ(Aoutp.get_weights(), Boutp.get_weights(), "Output layer weights differ");
}
void test_LayerPackVertical2(train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical2");
	size_t epochs = 5;
	const real_t learningRate = .01;

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Aifcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);
	layer_fully_connected<> Aifcl3(80, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Aifcl1, Aifcl2, Aifcl3, Aoutp);

	nnet_cond_epoch_eval Acee(epochs);
	nnet_train_opts<decltype(Acee)> Aopts(std::move(Acee));
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();


	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bifcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	layer_fully_connected<> Bifcl3(80, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl1, Bifcl2, Bifcl3);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, BlpVert, Boutp);

	nnet_cond_epoch_eval Bcee(epochs);
	nnet_train_opts<decltype(Bcee)> Bopts(std::move(Bcee));
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Bnn = make_nnet(Blp);
	Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();

	ASSERT_MTX_EQ(Aifcl1.get_weights(), Bifcl1.get_weights(), "First layer weights differs");
	ASSERT_MTX_EQ(Aifcl2.get_weights(), Bifcl2.get_weights(), "Second layer weights differs");
	ASSERT_MTX_EQ(Aifcl3.get_weights(), Bifcl3.get_weights(), "Third layer weights differs");
	ASSERT_MTX_EQ(Aoutp.get_weights(), Boutp.get_weights(), "Output layer weights differs");
}
void test_LayerPackVertical3(train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical3");
	size_t epochs = 5;
	const real_t learningRate = .01;

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Afcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);
	layer_fully_connected<> Aifcl3(80, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Afcl1, Aifcl2, Aifcl3, Aoutp);

	nnet_cond_epoch_eval Acee(epochs);
	nnet_train_opts<decltype(Acee)> Aopts(std::move(Acee));
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();


	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bfcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	layer_fully_connected<> Bifcl3(80, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl2, Bifcl3);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, Bfcl1, BlpVert, Boutp);

	nnet_cond_epoch_eval Bcee(epochs);
	nnet_train_opts<decltype(Bcee)> Bopts(std::move(Bcee));
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Bnn = make_nnet(Blp);
	Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();

	ASSERT_MTX_EQ(Afcl1.get_weights(), Bfcl1.get_weights(), "First layer weights differ");
	ASSERT_MTX_EQ(Aifcl2.get_weights(), Bifcl2.get_weights(), "Second layer weights differ");
	ASSERT_MTX_EQ(Aifcl3.get_weights(), Bifcl3.get_weights(), "Third layer weights differ");
	ASSERT_MTX_EQ(Aoutp.get_weights(), Boutp.get_weights(), "Output layer weights differ");
}
void test_LayerPackVertical4(train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical4");
	size_t epochs = 5;
	const real_t learningRate = .01;

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Afcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);
	layer_fully_connected<> Aifcl3(80, learningRate);
	layer_fully_connected<> Aifcl4(70, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Afcl1, Aifcl2, Aifcl3, Aifcl4, Aoutp);

	nnet_cond_epoch_eval Acee(epochs);
	nnet_train_opts<decltype(Acee)> Aopts(std::move(Acee));
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();


	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bfcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	layer_fully_connected<> Bifcl3(80, learningRate);
	layer_fully_connected<> Bifcl4(70, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl2, Bifcl3, Bifcl4);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, Bfcl1, BlpVert, Boutp);

	nnet_cond_epoch_eval Bcee(epochs);
	nnet_train_opts<decltype(Bcee)> Bopts(std::move(Bcee));
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Bnn = make_nnet(Blp);
	Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();

	ASSERT_MTX_EQ(Afcl1.get_weights(), Bfcl1.get_weights(), "First layer weights differs");
	ASSERT_MTX_EQ(Aifcl2.get_weights(), Bifcl2.get_weights(), "Second layer weights differs");
	ASSERT_MTX_EQ(Aifcl3.get_weights(), Bifcl3.get_weights(), "Third layer weights differs");
	ASSERT_MTX_EQ(Aifcl4.get_weights(), Bifcl4.get_weights(), "Fourth layer weights differs");
	ASSERT_MTX_EQ(Aoutp.get_weights(), Boutp.get_weights(), "Output layer weights differs");
}

TEST(TestNnet, LayerPackVertical) {
	train_data<real_t> td;
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());
	
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical1(td, std::time(0)));
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical2(td, std::time(0)));
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical3(td, std::time(0)));
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical4(td, std::time(0)));
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void testL2L1(const bool bL2, train_data<real_t>& td, const real_t coeff, uint64_t rngSeed, const size_t maxEpochs=3, const real_t LR=.02, const char* pDumpFileName=nullptr) noexcept {
	if (bL2) {
		STDCOUTL("Using l2coeff = " << coeff);
	} else STDCOUTL("Using l1coeff = " << coeff);


	//typedef activation::relu<> activ_func;
	typedef weights_init::XavierFour w_init_scheme;
	typedef activation::sigm<w_init_scheme> activ_func;

	layer_input<> inp(td.train_x().cols_no_bias());

	const size_t epochs = maxEpochs;
	const real_t learningRate = LR;

	layer_fully_connected<activ_func> fcl(100, learningRate);
	layer_fully_connected<activ_func> fcl2(100, learningRate);
	layer_output<activation::sigm_xentropy_loss<w_init_scheme>> outp(td.train_y().cols(), learningRate);

	if (bL2) {
		fcl.m_gradientWorks.set_L2(coeff);
		fcl2.m_gradientWorks.set_L2(coeff);
		outp.m_gradientWorks.set_L2(coeff);
	} else {
		fcl.m_gradientWorks.set_L1(coeff);
		fcl2.m_gradientWorks.set_L1(coeff);
		outp.m_gradientWorks.set_L1(coeff);
	}

	auto lp = make_layers(inp, fcl, fcl2, outp);

	nnet_cond_epoch_eval cee(epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));
	opts.calcFullLossValue(true).batchSize(100);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(rngSeed);

	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

#ifdef NNTL_MATLAB_AVAILABLE
	if (pDumpFileName) {
		nntl_supp::omatfile<> mf;
		mf.turn_on_all_options();
		mf.openForSave(std::string(pDumpFileName));
		mf << serialization::make_nvp("outpW", outp.get_weights());
		mf << serialization::make_nvp("fcl2W", fcl2.get_weights());
		mf << serialization::make_nvp("fclW", fcl.get_weights());
	}
#endif
}

TEST(TestNnet, L2L1) {
	train_data<math_types::real_ty> td;
	reader_t reader;

	const auto srcFile = MNIST_FILE_DEBUG;//intended to use small (debug) variation here

	STDCOUTL("Reading datafile '" << srcFile << "'...");
	reader_t::ErrorCode rec = reader.read(srcFile, td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	const unsigned im = 2;

	STDCOUTL("*************** Testing L2 regularizer ******************* ");
	for (unsigned i = 0; i < im; ++i) {
		auto sv = std::time(0);
		testL2L1(true,td, 0, sv);
		testL2L1(true,td, .1, sv);
	}

	STDCOUTL("*************** Testing L1 regularizer ******************* ");
	for (unsigned i = 0; i < im; ++i) {
		auto sv = std::time(0);
		testL2L1(false,td, 0, sv);
		testL2L1(false,td, .1, sv);
	}
}

TEST(TestNnet, L2Weights) {
	train_data<math_types::real_ty> td;
	reader_t reader;

	const auto srcFile = MNIST_FILE_DEBUG;//intended to use small (debug) variation here

	STDCOUTL("Reading datafile '" << srcFile << "'...");
	reader_t::ErrorCode rec = reader.read(srcFile, td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	/*testL2L1<true>(td, 0, 0, 5, .02, "d:/Docs/Math/play_matlab/NoL2.mat");
	testL2L1<true>(td, .1, 0, 5, .02, "d:/Docs/Math/play_matlab/L2.mat");
	testL2L1<false>(td, .1, 0, 5, .02, "d:/Docs/Math/play_matlab/L1.mat");*/
	testL2L1(true,td, 0, 0, 5, .02);
	testL2L1(true,td, .1, 0, 5, .02);
	testL2L1(false,td, .1, 0, 5, .02);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

TEST(TestNnet, Training) {
	train_data<real_t> td;
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	const real_t dropoutFrac = 0, momentum = 0;

	layer_input<> inp(td.train_x().cols_no_bias());

	typedef weights_init::Martens_SI_sigm<> w_init_scheme;
	typedef activation::sigm<w_init_scheme> activ_func;

#ifdef TESTS_SKIP_NNET_LONGRUNNING
	size_t epochs = 5;
	const real_t learningRate = .002;

	layer_fully_connected<activ_func> fcl(60, learningRate, dropoutFrac);
	layer_fully_connected<activ_func> fcl2(50, learningRate, dropoutFrac);
	//layer_fully_connected<activ_func> fcl3(15,learningRate,	 dropoutFrac);	

#else
	size_t epochs = 20;
	const real_t learningRate = 0.0005;

	layer_fully_connected<activ_func> fcl(500, learningRate, dropoutFrac);
	layer_fully_connected<activ_func> fcl2(300, learningRate, dropoutFrac);
#endif // TESTS_SKIP_LONGRUNNING

	auto optType = decltype(fcl)::grad_works_t::ClassicalConstant;

	fcl.m_gradientWorks.set_nesterov_momentum(momentum, false).set_type(optType);
	fcl2.m_gradientWorks.set_nesterov_momentum(momentum, false).set_type(optType);

	layer_output<activation::softmax_xentropy_loss<w_init_scheme>> outp(td.train_y().cols(), learningRate);
	outp.m_gradientWorks.set_nesterov_momentum(momentum, false).set_type(optType);


	auto lp = make_layers(inp, fcl, fcl2, outp);

	nnet_cond_epoch_eval cee(epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));

	opts.batchSize(100);

	auto nn = make_nnet(lp);

	auto ec = nn.train(td, opts);

	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}
