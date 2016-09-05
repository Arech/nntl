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
#include "../nntl/_supp/io/matfile.h"

#include "../nntl/interface/inspectors/stdcout.h"
#include "../nntl/interface/inspectors/dumper.h"

#include "asserts.h"
#include "common_routines.h"

using namespace nntl;

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


TEST(TestInspectors, StdcoutI) {
	train_data<real_t> td;
	readTd(td, MNIST_FILE_DEBUG);

	size_t epochs = 2, seedVal = 0;
	const real_t learningRate = real_t(.01), dropoutRate = real_t(1.);

	//redefining InterfacesT
	typedef inspector::stdcout<real_t> myInspector;
	//typedef inspector::dummy<real_t> myInspector;
	struct myIntf : public d_int_nI {
		typedef myInspector iInspect_t;
	};
	//and related layer's template params
	typedef grad_works<myIntf> myGW;
	typedef activation::sigm<real_t, weights_init::XavierFour> myAct;
	typedef activation::sigm_quad_loss<real_t, weights_init::XavierFour> myActO;

	//instantiating layer objects
	layer_input<myIntf> inp(td.train_x().cols_no_bias(), "Source");
	layer_fully_connected<myAct, myGW> ifcl1(20, learningRate, dropoutRate, "First");
	layer_fully_connected<myAct, myGW> ifcl2(15, learningRate, dropoutRate, "Second");
	layer_output<myActO, myGW> outp(td.train_y().cols(), learningRate, "Predictor");

	auto lp = make_layers(inp, ifcl1, ifcl2, outp);

	nnet_train_opts<> opts(epochs);
	opts.calcFullLossValue(false).batchSize(100);

	//instantiating inspector (though, could let nnet to spawn it by itself)
	//myInspector Insp(1);
	//auto Ann = make_nnet(Alp, Insp);
	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(seedVal);

	auto ec = nn.train(td, opts);

	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}

TEST(TestInspectors, DumperMat) {
#if NNTL_MATLAB_AVAILABLE
	train_data<real_t> td;
	readTd(td, MNIST_FILE_DEBUG);

	size_t epochs = 5, seedVal = 0;
	const real_t learningRate = real_t(.01), dropoutRate = real_t(1.);
	
	//redefining InterfacesT
	typedef inspector::dumper<real_t, nntl_supp::omatfileEx<>> myInspector;
	struct myIntf : public d_int_nI {
		typedef myInspector iInspect_t;
	};
	//and related layer's template params
	typedef grad_works<myIntf> myGW;
	typedef activation::sigm<real_t, weights_init::XavierFour> myAct;
	typedef activation::sigm_quad_loss<real_t, weights_init::XavierFour> myActO;

	//instantiating layer objects
	layer_input<myIntf> inp(td.train_x().cols_no_bias(), "Source");
	layer_fully_connected<myAct, myGW> ifcl1(20, learningRate, dropoutRate, "First");
	layer_fully_connected<myAct, myGW> ifcl2(15, learningRate, dropoutRate, "Second");
	layer_output<myActO, myGW> outp(td.train_y().cols(), learningRate, "Predictor");

	auto lp = make_layers(inp, ifcl1, ifcl2, outp);

	nnet_train_opts<> opts(epochs);
	opts.calcFullLossValue(false).batchSize(100);

	myInspector Insp("./test_data");
	auto nn = make_nnet(lp, Insp);

	nn.get_iRng().seed64(seedVal);

	auto ec = nn.train(td, opts);

	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
#else
	STDCOUTL("###To run the test compile a code with NNTL_MATLAB_AVAILABLE 1");
#endif
}