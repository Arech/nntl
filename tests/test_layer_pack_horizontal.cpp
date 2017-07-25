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


//this is just to make sure it'll compile within nnet object
TEST(TestLayerPackHorizontal, Simple) {
	train_data<real_t> td;
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	size_t epochs = 5;
	const real_t learningRate = real_t(.01);

	const auto train_x_dim = td.train_x().cols_no_bias();
	layer_input<> Binp(train_x_dim);

	layer_fully_connected<> Bifcl1(100, learningRate);
	layer_fully_connected<> Bifcl2(90, learningRate);

	auto lpHor = make_layer_pack_horizontal(make_PHL(Bifcl1, 0, train_x_dim / 2), make_PHL(Bifcl2, train_x_dim / 2, train_x_dim / 2));

	layer_output<> Boutp(td.train_y().cols(), learningRate);

	auto Blp = make_layers(Binp, lpHor, Boutp);

	nnet_train_opts<> Bopts(epochs);
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);


	auto Bnn = make_nnet(Blp);
	//Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();
}


/*
void __test_same_layers(train_data<real_t>& td, uint64_t rngSeed) {
	size_t epochs = 5;
	const real_t learningRate = .01;
	const auto train_x_dim = td.train_x().cols_no_bias();
	constexpr unsigned undNeuronsCnt = 100, inrNeuronsCnt = 150;

	//////////////////////////////////////////////////////////////////////////
	// making etalon nnet
	layer_input<> Ainp(train_x_dim);
	//underlying layer to test the correctness of dLdA propagation
	layer_fully_connected< activation::sigm<test_weights_init_scheme<0>> > Aund(undNeuronsCnt, learningRate);
	//reference layer to test correctness of internal layer_pack_horizontal layers
	layer_fully_connected< activation::sigm<test_weights_init_scheme<10>> > Aint(inrNeuronsCnt, learningRate);

	layer_output< activation::sigm_quad_loss<test_weights_init_scheme<20>> > Aout(td.train_y().cols(), learningRate);

	auto Alp = make_layers(Ainp, Aund, Aint, Aout);

	vector_conditions Acee(epochs);
	nnet_train_opts<decltype(Acee)> Aopts(::std::move(Acee));
	Aopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Ann = make_nnet(Alp);
	Ann.get_iRng().seed64(rngSeed);
	auto Aec = Ann.train(td, Aopts);
	ASSERT_EQ(decltype(Ann)::ErrorCode::Success, Aec) << "Error code description: " << Ann.get_last_error_string();

	//////////////////////////////////////////////////////////////////////////
	// making test net

	layer_input<> Binp(train_x_dim);

	//underlying layer to test the correctness of dLdA propagation
	layer_fully_connected< activation::sigm<test_weights_init_scheme<0>> > Bund(undNeuronsCnt, learningRate/2);

	//layers to test correctness of internal layer_pack_horizontal layers
	layer_fully_connected< activation::sigm<test_weights_init_scheme<10>> > Bifcl1(inrNeuronsCnt, learningRate);
	layer_fully_connected< activation::sigm<test_weights_init_scheme<10>> > Bifcl2(inrNeuronsCnt, learningRate);
	auto lpHor = make_layer_pack_horizontal(make_PHL(Bifcl1, 0, undNeuronsCnt), make_PHL(Bifcl2, 0, undNeuronsCnt));

	layer_output< activation::sigm_quad_loss<test_weights_init_scheme<20>> > Bout(td.train_y().cols(), learningRate);
	
	auto Blp = make_layers(Binp, Bund, lpHor, Bout);

	vector_conditions Bcee(epochs);
	nnet_train_opts<decltype(Bcee)> Bopts(::std::move(Bcee));
	Bopts.calcFullLossValue(true).batchSize(100).ImmediatelyDeinit(false);

	auto Bnn = make_nnet(Blp);
	Bnn.get_iRng().seed64(rngSeed);
	auto Bec = Bnn.train(td, Bopts);
	ASSERT_EQ(decltype(Bnn)::ErrorCode::Success, Bec) << "Error code description: " << Bnn.get_last_error_string();

	ASSERT_MTX_EQ(Aout.get_weights(), Bout.get_weights(), "outpout layer weights differs");
}*/

void test_same_layers(train_data<real_t>& td, uint64_t rngSeed) {
	const real_t learningRate = 1;
	const auto train_x_dim = td.train_x().cols_no_bias();
	const bool bTrainIsBigger = td.train_x().rows() > td.test_x().rows();
	const realmtx_t& evalX = bTrainIsBigger ? td.train_x() : td.test_x();// , &evalY = bTrainIsBigger ? td.train_y() : td.test_y();
	const realmtx_t& trainX = bTrainIsBigger ? td.test_x() : td.train_x();// , &trainY = bTrainIsBigger ? td.test_y() : td.train_y();
	const auto evalSamplesCnt = evalX.rows(), trainSamplesCnt = trainX.rows();

	constexpr unsigned undNeuronsCnt = 100, inrNeuronsCnt = 150;

	d_interfaces::iMath_t iMath;
	d_interfaces::iRng_t iRng;
	d_interfaces::iInspect_t iInsp;
	if (d_interfaces::iRng_t::is_multithreaded) iRng.set_ithreads(iMath.ithreads());

	//////////////////////////////////////////////////////////////////////////
	//setting up etalon layers
	layer_input<> Ainp(train_x_dim);
	//underlying layer to test the correctness of dLdA propagation
	layer_fully_connected< activation::sigm<real_t> > Aund(undNeuronsCnt, learningRate);
	//reference layer to test correctness of internal layer_pack_horizontal layers
	layer_fully_connected< activation::sigm<real_t> > Aint(inrNeuronsCnt, learningRate);
	
	//assembling into list of layers
	auto AlayersTuple = ::std::make_tuple(::std::ref(Ainp), ::std::ref(Aund), ::std::ref(Aint));
	layer_index_t alc(0);
	tuple_utils::for_eachwp_up(AlayersTuple, _impl::_preinit_layers(alc));

	typedef _impl::common_nn_data<d_interfaces> common_data_t;
	bool bUNUSED = false;
	common_data_t CD(iMath, iRng, iInsp, bUNUSED);

	CD.init(evalSamplesCnt, trainSamplesCnt);
	_impl::_layer_init_data<common_data_t> lid(CD);
	_impl::layers_mem_requirements lmr;
	
	iRng.seed64(rngSeed-1);
	lid.clean_using();
	auto ec = Ainp.init(lid);
	ASSERT_EQ(ec, _nnet_errs::ErrorCode::Success) << "Failed to initialize Ainp";
	lmr.updateLayerReq(lid);
	
	iRng.seed64(rngSeed);
	lid.clean_using();
	ec = Aund.init(lid);
	ASSERT_EQ(ec, _nnet_errs::ErrorCode::Success) << "Failed to initialize Aund";
	lmr.updateLayerReq(lid);

	iRng.seed64(rngSeed+1);
	lid.clean_using();
	ec = Aint.init(lid);
	ASSERT_EQ(ec, _nnet_errs::ErrorCode::Success) << "Failed to initialize Aint";
	lmr.updateLayerReq(lid);
	
	ASSERT_TRUE(lmr.maxSingledLdANumel > 0 && lmr.maxMemLayerTrainingRequire > 0);
	const numel_cnt_t AtotalTempMemSize = lmr.maxMemLayerTrainingRequire + 2 * lmr.maxSingledLdANumel;
	::std::unique_ptr<real_t[]> AtempMemStorage(new(::std::nothrow)real_t[AtotalTempMemSize]);
	ASSERT_TRUE(nullptr != AtempMemStorage.get());
	realmtxdef_t AdLdA1, AdLdA2;
	{
		numel_cnt_t c = 0;
		real_t* ptr = AtempMemStorage.get();
		AdLdA1.useExternalStorage(ptr, lmr.maxSingledLdANumel);
		c += lmr.maxSingledLdANumel;
		AdLdA2.useExternalStorage(ptr + c, lmr.maxSingledLdANumel);
		c += lmr.maxSingledLdANumel;

		ptr += c;
		Aund.initMem(ptr, lmr.maxMemLayerTrainingRequire);
		Aint.initMem(ptr, lmr.maxMemLayerTrainingRequire);
	}

	//////////////////////////////////////////////////////////////////////////
	// setting up test layers

	layer_input<> Binp(train_x_dim);
	//underlying layer to test the correctness of dLdA propagation
	layer_fully_connected< activation::sigm<real_t> > Bund(undNeuronsCnt, learningRate);
	//layers to test correctness of internal layer_pack_horizontal layers
	layer_fully_connected< activation::sigm<real_t> > Bifcl1(inrNeuronsCnt, learningRate);
	layer_fully_connected< activation::sigm<real_t> > Bifcl2(inrNeuronsCnt, learningRate);
	auto lpHor = make_layer_pack_horizontal(make_PHL(Bifcl1, 0, undNeuronsCnt), make_PHL(Bifcl2, 0, undNeuronsCnt));

	//assembling into list of layers
	auto BlayersTuple = ::std::make_tuple(::std::ref(Binp), ::std::ref(Bund), ::std::ref(lpHor));
	layer_index_t blc(0);
	tuple_utils::for_eachwp_up(BlayersTuple, _impl::_preinit_layers(blc));

	lmr.zeros();

	iRng.seed64(rngSeed-1);
	lid.clean_using();
	ec = Binp.init(lid);
	ASSERT_EQ(ec, _nnet_errs::ErrorCode::Success) << "Failed to initialize Binp";
	lmr.updateLayerReq(lid);

	iRng.seed64(rngSeed);
	lid.clean_using();
	ec = Bund.init(lid);
	ASSERT_EQ(ec, _nnet_errs::ErrorCode::Success) << "Failed to initialize Bund";
	lmr.updateLayerReq(lid);

	iRng.seed64(rngSeed + 1);
	lid.clean_using();
	ec = lpHor.init(lid);
	ASSERT_EQ(ec, _nnet_errs::ErrorCode::Success) << "Failed to initialize lpHor";
	lmr.updateLayerReq(lid);

	ASSERT_TRUE(lmr.maxSingledLdANumel > 0 && lmr.maxMemLayerTrainingRequire > 0);
	const numel_cnt_t BtotalTempMemSize = lmr.maxMemLayerTrainingRequire + 2 * lmr.maxSingledLdANumel;
	::std::unique_ptr<real_t[]> BtempMemStorage(new(::std::nothrow)real_t[BtotalTempMemSize]);
	ASSERT_TRUE(nullptr != BtempMemStorage.get());

	realmtxdef_t BdLdA1, BdLdA2;
	{
		numel_cnt_t c = 0;
		real_t* ptr = BtempMemStorage.get();
		BdLdA1.useExternalStorage(ptr, lmr.maxSingledLdANumel);
		c += lmr.maxSingledLdANumel;
		BdLdA2.useExternalStorage(ptr + c, lmr.maxSingledLdANumel);
		c += lmr.maxSingledLdANumel;

		ptr += c;
		Bund.initMem(ptr, lmr.maxMemLayerTrainingRequire);
		lpHor.initMem(ptr, lmr.maxMemLayerTrainingRequire);
	}

	ASSERT_TRUE(iMath.init());

	//////////////////////////////////////////////////////////////////////////
	// checking weights

	ASSERT_EQ(Aund.get_weights(), Bund.get_weights());
	ASSERT_EQ(Aint.get_weights(), Bifcl1.get_weights());
	//only the first layer weighs will be the same as Aint, because we cant reinit rng during lpHor.init(). We'll set Bifcl2 weights by hand
	ASSERT_EQ(Aint.get_weights().size(), Bifcl2.get_weights().size());
	realmtx_t tmpMtx;
	Aint.get_weights().clone_to(tmpMtx);
	Bifcl2.set_weights(::std::move(tmpMtx));

	//////////////////////////////////////////////////////////////////////////
	// doing and checking forward pass
	CD.set_mode_and_batch_size(true, trainSamplesCnt);

	tuple_utils::for_each_up(AlayersTuple, [](auto& lyr)noexcept { lyr.on_batch_size_change(); });
	Ainp.fprop(trainX);
	Aund.fprop(Ainp);
	Aint.fprop(Aund);

	tuple_utils::for_each_up(BlayersTuple, [](auto& lyr)noexcept { lyr.on_batch_size_change(); });
	Binp.fprop(trainX);
	Bund.fprop(Binp);
	lpHor.fprop(Bund);

	ASSERT_EQ(Aund.get_activations(), Bund.get_activations());
	ASSERT_EQ(Aint.get_activations(), Bifcl2.get_activations());
	ASSERT_TRUE(Aint.get_activations() != Bifcl1.get_activations()); //this is because biases of Bifcl1.get_activations() are substituted by first row of activations of Bifcl2

	ASSERT_TRUE(0 == memcmp(Aint.get_activations().data(), lpHor.get_activations().data(), Aint.get_activations().byte_size_no_bias()));
	ASSERT_TRUE(0 == memcmp(Aint.get_activations().data(), lpHor.get_activations().colDataAsVec(Aint.get_activations().cols_no_bias()), Aint.get_activations().byte_size()));
	ASSERT_TRUE(0 == memcmp(Aint.get_activations().data(), Bifcl1.get_activations().data(), Aint.get_activations().byte_size_no_bias()));

	const real_t scaler = .5;
	//doing and checking backward pass
	AdLdA1.deform_like_no_bias(Aint.get_activations());
	AdLdA1.ones();
	iMath.evMulC_ip(AdLdA1, scaler);//just to get rid of 1.0, thought it's ok too
	AdLdA2.deform_like_no_bias(Aund.get_activations());
	ASSERT_EQ(1, Aint.bprop(AdLdA1, Aund, AdLdA2));

	BdLdA1.deform_like_no_bias(lpHor.get_activations());
	BdLdA1.ones();
	iMath.evMulC_ip(BdLdA1, scaler);//just to get rid of 1.0, thought it's ok too
	BdLdA2.deform_like_no_bias(Bund.get_activations());
	//not sure if we must check for 1 here, because it's implementation-dependent, but going to leave as is and fix it once/if needed
	ASSERT_EQ(1, lpHor.bprop(BdLdA1, Bund, BdLdA2));

	//testing weights
	ASSERT_EQ(Aint.get_weights(), Bifcl1.get_weights());
	ASSERT_EQ(Aint.get_weights(), Bifcl2.get_weights());

	//BdLdA2 must be 2*AdLdA2
	iMath.evMulC_ip(AdLdA2, real_t(2.0));
	ASSERT_EQ(BdLdA2, AdLdA2);

	AdLdA1.deform(0, 0);
	ASSERT_EQ(1, Aund.bprop(AdLdA2, Ainp, AdLdA1));
	BdLdA1.deform(0, 0);
	ASSERT_EQ(1, Bund.bprop(BdLdA2, Binp, BdLdA1));

	ASSERT_EQ(Aund.get_weights(), Bund.get_weights());
}


TEST(TestLayerPackHorizontal, SameLayers) {
	train_data<real_t> td;
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	ASSERT_NO_FATAL_FAILURE(test_same_layers(td, 0));
	ASSERT_NO_FATAL_FAILURE(test_same_layers(td, ::std::time(0)));
}


/*
TEST(TestLayerPackHorizontal, InnerLayersIntersectsTestCheck) {
#ifndef NNTL_DEBUG
	STDCOUTL("!!!! This test should primarily be ran in DEBUG mode to utilize nntl's internal asserts!");
#endif

	layer_input<> li1(10);
	LFC<> l11(2);
	LFC<> l12(2);
	auto lph1 = make_layer_pack_horizontal(
		make_PHL(l11, 0, 5),
		make_PHL(l12, 5, 5)
	);
	ASSERT_TRUE(!lph1.isInnerLayersIntersects());
	layer_output<> lo1(1);
	auto lp1 = make_layers(li1, lph1,lo1);//to test _preinit code

	layer_input<> li2(10);
	LFC<> l21(2);
	LFC<> l22(2);
	auto lph2 = make_layer_pack_horizontal(
		make_PHL(l21, 0, 6),
		make_PHL(l22, 5, 5)
	);
	ASSERT_TRUE(lph2.isInnerLayersIntersects());
	layer_output<> lo2(1);
	auto lp2 = make_layers(li2, lph2, lo2);
}*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//just a single LPH with 2 fully connected layers side-by-side (their receptive fields doesn't intersects)
template<typename ArchPrmsT>
struct GC_LPH_NO : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	myLFC l1;
	myLFC l2;
	LPH<PHL<decltype(l1)>, PHL<decltype(l2)>> lFinal;

	~GC_LPH_NO()noexcept {}
	GC_LPH_NO(const ArchPrms_t& Prms)noexcept
		: l1(50, Prms.learningRate, Prms.dropoutAlivePerc, "l1")
		, l2(70, Prms.learningRate, Prms.dropoutAlivePerc, "l2")
		, lFinal("lFinal"
			, make_PHL(l1, 0, Prms.lUnderlay_nc / 2)
			, make_PHL(l2, Prms.lUnderlay_nc / 2, Prms.lUnderlay_nc - (Prms.lUnderlay_nc / 2))//to get rid of integer division rounding
		)
	{}
};
TEST(TestLayerPackHorizontal, GradCheck_nonoverlapping) {
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	nntl_tests::NN_arch<GC_LPH_NO<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-3);//numeric errors due to dLdAPrev addition in LPH stacks up significantly
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 10, ngcSetts));
}

template<typename ArchPrmsT>
struct GC_LPH_OVR : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	myLFC l1;
	myLFC l2;
	myLFC l3;
	LPH<PHL<decltype(l1)>, PHL<decltype(l2)>, PHL<decltype(l3)>> lFinal;

	~GC_LPH_OVR()noexcept {}
	GC_LPH_OVR(const ArchPrms_t& Prms)noexcept
		: l1(50, Prms.learningRate, Prms.dropoutAlivePerc, "l1")
		, l2(70, Prms.learningRate, Prms.dropoutAlivePerc, "l2")
		, l3(90, Prms.learningRate, Prms.dropoutAlivePerc, "l3")
		, lFinal("lFinal"
			, make_PHL(l1, 0, Prms.lUnderlay_nc - 1)
			, make_PHL(l2, 1, Prms.lUnderlay_nc - 1)
			, make_PHL(l3, 0, Prms.lUnderlay_nc)
		)
	{}
};
TEST(TestLayerPackHorizontal, GradCheck_overlapping) {
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	nntl_tests::NN_arch<GC_LPH_OVR<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-3);//numeric errors due to dLdAPrev addition in LPH stacks up significantly
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 10, ngcSetts));
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/*

template<typename real_t, bool bNumStab>
struct GC_LPH_params_ovr_ns : public nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> {
private:
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> _base_class_t;
public:
	typedef nntl::activation::softsigm_quad_loss<real_t, 1000, weights_init::He_Zhang<>, bNumStab> myOutputActivation;
	GC_LPH_params_ovr_ns(const nntl::train_data<real_t>& td) noexcept : _base_class_t(td) {}
};

TEST(TestLayerPackHorizontal, GradCheck_overlapping_ns) {
	typedef double real_t;
	const size_t rngSeed = ::std::time(0);

	nntl::train_data<real_t> td;
	readTd(td);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.dLdW_setts.relErrWarnThrsh = real_t(5e-7);
	ngcSetts.evalSetts.dLdA_setts.relErrWarnThrsh = real_t(5e-8);
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-3);
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;

	{
		typedef GC_LPH_params_ovr_ns<real_t, false> ArchPrms_t;

		ArchPrms_t Prms(td);
		nntl_tests::NN_arch<GC_LPH_OVR<ArchPrms_t>> nnArch(Prms);

		nnArch.NN.get_iRng().seed64(rngSeed);

		auto ec = nnArch.warmup(td, 5, 200);
		ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);
		ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 10, ngcSetts));
	}

	{
		typedef GC_LPH_params_ovr_ns<real_t, true> ArchPrms_t;

		ArchPrms_t Prms(td);
		nntl_tests::NN_arch<GC_LPH_OVR<ArchPrms_t>> nnArch(Prms);

		nnArch.NN.get_iRng().seed64(rngSeed);

		auto ec = nnArch.warmup(td, 5, 200);
		ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);
		ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 10, ngcSetts));
	}
}*/
