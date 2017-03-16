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

#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_supp/io/matfile.h"

#include "asserts.h"
#include "common_routines.h"

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

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void testL2L1(const bool bL2, train_data<real_t>& td, const real_t coeff, uint64_t rngSeed, const size_t maxEpochs=3, const real_t LR=.02, const char* pDumpFileName=nullptr) noexcept {
	if (bL2) {
		STDCOUTL("Using l2coeff = " << coeff);
	} else STDCOUTL("Using l1coeff = " << coeff);


	//typedef activation::relu<> activ_func;
	typedef weights_init::XavierFour w_init_scheme;
	typedef activation::sigm<real_t, w_init_scheme> activ_func;

	layer_input<> inp(td.train_x().cols_no_bias());

	const size_t epochs = maxEpochs;
	const real_t learningRate = LR;

	layer_fully_connected<activ_func> fcl(100, learningRate);
	layer_fully_connected<activ_func> fcl2(100, learningRate);
	layer_output<activation::sigm_xentropy_loss<real_t, w_init_scheme>> outp(td.train_y().cols(), learningRate);

	if (bL2) {
		fcl.m_gradientWorks.L2(coeff);
		fcl2.m_gradientWorks.L2(coeff);
		outp.m_gradientWorks.L2(coeff);
	} else {
		fcl.m_gradientWorks.L1(coeff);
		fcl2.m_gradientWorks.L1(coeff);
		outp.m_gradientWorks.L1(coeff);
	}

	auto lp = make_layers(inp, fcl, fcl2, outp);

	nnet_train_opts<> opts(epochs);
	opts.calcFullLossValue(true).batchSize(100);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(rngSeed);

	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

#if NNTL_MATLAB_AVAILABLE
	if (pDumpFileName) {
		nntl_supp::omatfile<> mf;
		mf.turn_on_all_options();
		mf.open(std::string(pDumpFileName));
		mf << serialization::make_nvp("outpW", outp.get_weights());
		mf << serialization::make_nvp("fcl2W", fcl2.get_weights());
		mf << serialization::make_nvp("fclW", fcl.get_weights());
	}
#endif
}

TEST(TestNnet, L2L1) {
	train_data<real_t> td;
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
		testL2L1(true,td, real_t(.1), sv);
	}

	STDCOUTL("*************** Testing L1 regularizer ******************* ");
	for (unsigned i = 0; i < im; ++i) {
		auto sv = std::time(0);
		testL2L1(false,td, 0, sv);
		testL2L1(false,td, real_t(.1), sv);
	}
}

TEST(TestNnet, L2Weights) {
	train_data<real_t> td;
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
	testL2L1(true,td, 0, 0, 5, real_t(.02));
	testL2L1(true,td, real_t(.1), 0, 5, real_t(.02));
	testL2L1(false,td, real_t(.1), 0, 5, real_t(.02));
}
