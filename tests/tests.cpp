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
// tests.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "../nntl/interface/math.h"
#include "../nntl/nntl.h"
#include "../nntl/_supp/binfile.h"

using namespace nntl;
typedef nntl_supp::binfile reader_t;
using real_t = math_types::real_ty;


#if defined(TESTS_SKIP_NNET_LONGRUNNING)
//ALWAYS run debug build with similar relations of data sizes:
// if release will run in minibatches - make sure, there will be at least 2 minibatches)
// if release will use different data sizes for train/test - make sure, debug will also run on different datasizes
#define MNIST_FILE "../data/mnist200_100.bin"
#else
#define MNIST_FILE "../data/mnist60000.bin"
#endif // _DEBUG

TEST(TestNntl, Training) {
	train_data td;
	reader_t reader;
	
	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	const real_t dropoutFrac = 0, momentum = 0.9;
	//const ILR ilr(.9, 1.1, .00000001, 1000);

	layer_input inp(td.train_x().cols_no_bias());

	//typedef activation::relu<> activ_func;
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

	layer_fully_connected<activ_func> fcl(500, learningRate,dropoutFrac);
	layer_fully_connected<activ_func> fcl2(300, learningRate, dropoutFrac);
#endif // TESTS_SKIP_LONGRUNNING

	const bool bSetMN = false;
	const real_t mul = 1.0 / 10000.0;
	//auto optType = decltype(fcl)::grad_works_t::ClassicalConstant;
	auto optType = decltype(fcl)::grad_works_t::RMSProp_Hinton;

	fcl.m_gradientWorks.set_nesterov_momentum(momentum, false).set_type(optType)
		.set_weight_vector_max_norm2(bSetMN ? mul * 768 * 768 : 0, true);
	fcl2.m_gradientWorks.set_nesterov_momentum(momentum, false).set_type(optType)
		.set_weight_vector_max_norm2(bSetMN ? mul * 500 * 500 : 0, true);
	
	//layer_output<activation::sigm_xentropy_loss<w_init_scheme>> outp(td.train_y().cols(), learningRate);
	layer_output<activation::sigm_quad_loss<w_init_scheme>> outp(td.train_y().cols(), learningRate);
	outp.m_gradientWorks.set_nesterov_momentum(momentum, false).set_type(optType)
		.set_weight_vector_max_norm2(bSetMN ? mul * 300 * 300 : 0, true);

	//uncomment to turn on derivative value restriction 
	//outp.restrict_dL_dZ(real_t(-10), real_t(10));

	auto lp = make_layers_pack(inp, fcl, fcl2, outp);
	//auto lp = make_layers_pack(inp, fcl, fcl2, fcl3, fcl4, outp);

	nnet_cond_epoch_eval cee(epochs);
	nnet_train_opts<decltype(cee)> opts(std::move(cee));

	opts.batchSize(100);
	
	auto nn = make_nnet(lp);

	auto ec = nn.train(td, opts);

	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}



int __cdecl main(int argc, char **argv) {
#ifdef NNTL_DEBUG
	STDCOUTL("\n******\n*** This is DEBUG binary! All performance reports are invalid!\n******\n");
#endif // NNTL_DEBUG

	::testing::InitGoogleTest(&argc, argv);

	int r = RUN_ALL_TESTS();
	//std::cin.ignore();
	return r;
}
