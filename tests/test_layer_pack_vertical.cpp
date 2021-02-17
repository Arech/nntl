/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
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

//to get rid of '... decorated name length exceeded, name was truncated'
#pragma warning( disable : 4503 )

#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_supp/io/matfile.h"

#include "asserts.h"
#include "common_routines.h"

#include "nn_base_arch.h"

using namespace nntl;
typedef nntl_supp::binfile reader_t;

typedef d_interfaces::real_t real_t;
typedef math::smatrix<real_t> realmtx_t;
typedef math::smatrix_deform<real_t> realmtxdef_t;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


void test_LayerPackVertical1(inmem_train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical1");
	size_t epochs = 5;
	const real_t learningRate = real_t(.01);

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Aifcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Aifcl1, Aifcl2, Aoutp);

	nnet_train_opts<real_t> Aopts(epochs);
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();

	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();

	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bifcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl1, Bifcl2);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, BlpVert, Boutp);

	nnet_train_opts<real_t> Bopts(epochs);
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Bnn = make_nnet(Blp);
	Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();

	ASSERT_MTX_EQ(Aifcl1.get_weights(), Bifcl1.get_weights(), "First layer weights differ");
	ASSERT_MTX_EQ(Aifcl2.get_weights(), Bifcl2.get_weights(), "Second layer weights differ");
	ASSERT_MTX_EQ(Aoutp.get_weights(), Boutp.get_weights(), "Output layer weights differ");
}
void test_LayerPackVertical2(inmem_train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical2");
	size_t epochs = 5;
	const real_t learningRate = real_t(.01);

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Aifcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);
	layer_fully_connected<> Aifcl3(80, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Aifcl1, Aifcl2, Aifcl3, Aoutp);

	nnet_train_opts<real_t> Aopts(epochs);
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();
	
	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();

	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bifcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	layer_fully_connected<> Bifcl3(80, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl1, Bifcl2, Bifcl3);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, BlpVert, Boutp);

	nnet_train_opts<real_t> Bopts(epochs);
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
void test_LayerPackVertical3(inmem_train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical3");
	size_t epochs = 5;
	const real_t learningRate = real_t(.01);

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Afcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);
	layer_fully_connected<> Aifcl3(80, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Afcl1, Aifcl2, Aifcl3, Aoutp);

	nnet_train_opts<real_t> Aopts(epochs);
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();
	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();

	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bfcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	layer_fully_connected<> Bifcl3(80, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl2, Bifcl3);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, Bfcl1, BlpVert, Boutp);

	nnet_train_opts<real_t> Bopts(epochs);
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
void test_LayerPackVertical4(inmem_train_data<real_t>& td, uint64_t rngSeed)noexcept {
	SCOPED_TRACE("test_LayerPackVertical4");
	size_t epochs = 5;
	const real_t learningRate = real_t(.01);

	layer_input<> Ainp(td.train_x().cols_no_bias());

	layer_fully_connected<> Afcl1(100, learningRate);
	layer_fully_connected<> Aifcl2(90, learningRate);
	layer_fully_connected<> Aifcl3(80, learningRate);
	layer_fully_connected<> Aifcl4(70, learningRate);

	layer_output<> Aoutp(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Afcl1, Aifcl2, Aifcl3, Aifcl4, Aoutp);

	nnet_train_opts<real_t> Aopts(epochs);
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto ec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, ec) << "Error code description: " << Ann.get_last_error_string();

	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();

	layer_input<> Binp(td.train_x().cols_no_bias());

	layer_fully_connected<> Bfcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);
	layer_fully_connected<> Bifcl3(80, learningRate);
	layer_fully_connected<> Bifcl4(70, learningRate);
	auto BlpVert = make_layer_pack_vertical(Bifcl2, Bifcl3, Bifcl4);

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, Bfcl1, BlpVert, Boutp);

	nnet_train_opts<real_t> Bopts(epochs);
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

TEST(TestLPV, LayerPackVertical) {
	inmem_train_data<real_t> td;
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical1(td, ::std::time(0)));
	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical2(td, ::std::time(0)));
	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical3(td, ::std::time(0)));
	//we must deinit td to make sure it'll be in the same state after reseeding RNG for B as it was for A when it was initialized first
	td.deinit4all();
	ASSERT_NO_FATAL_FAILURE(test_LayerPackVertical4(td, ::std::time(0)));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename ArchPrmsT>
struct GC_LPV : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	myLFC l1;
	myLFC l2;
	LPV<decltype(l1), decltype(l2)> lFinal;

	~GC_LPV()noexcept {}
	GC_LPV(const ArchPrms_t& Prms)noexcept
		: l1(50, Prms.learningRate, "l1")
		, l2(70, Prms.learningRate, "l2")
		, lFinal("lFinal", l1, l2)
	{}
};
TEST(TestLPV, GradCheck) {
#pragma warning(disable:4459)
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;
#pragma warning(default:4459)

	nntl::inmem_train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	nntl_tests::NN_arch<GC_LPV<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(1e-3);
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 3, ngcSetts));
}