/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "../nntl/weights_init/LsuvExt.h"

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



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename real_t>
struct testWeightsL1L2_res {
	::std::array<real_t, 3> w;
};
void testL2L1(const bool bL2, train_data<real_t>& td, const real_t coeff, uint64_t rngSeed, testWeightsL1L2_res<real_t>& res, const size_t maxEpochs=3, const real_t LR=.02, const char* pDumpFileName=nullptr) noexcept {
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

	auto& iM = nn.get_iMath();
	res.w[0] = iM.ewSumSquares_ns(fcl.get_weights());
	res.w[1] = iM.ewSumSquares_ns(fcl2.get_weights());
	res.w[2] = iM.ewSumSquares_ns(outp.get_weights());
	
#if NNTL_MATLAB_AVAILABLE
	if (pDumpFileName) {
		nntl_supp::omatfile<> mf;
		mf.turn_on_all_options();
		mf.open(::std::string(pDumpFileName));
		mf << serialization::make_nvp("outpW", outp.get_weights());
		mf << serialization::make_nvp("fcl2W", fcl2.get_weights());
		mf << serialization::make_nvp("fclW", fcl.get_weights());
	}
#endif
}

TEST(TestNnet, WeightsConstraintsL2L1) {
	train_data<real_t> td;
	reader_t reader;

	const auto srcFile = MNIST_FILE_DEBUG;//intended to use small (debug) variation here

	STDCOUTL("Reading datafile '" << srcFile << "'...");
	reader_t::ErrorCode rec = reader.read(srcFile, td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	const unsigned im = 2;
	testWeightsL1L2_res<real_t> noReg, wReg;

	STDCOUTL("*************** Testing L2 regularizer ******************* ");
	for (unsigned i = 0; i < im; ++i) {
		auto sv = ::std::time(0);
		testL2L1(true,td, 0, sv, noReg);
		testL2L1(true,td, real_t(.01), sv, wReg);
		STDCOUTL("Sum of squared weights without regularizer vs with regularizer:");
		for (size_t idx = 0; idx < noReg.w.size(); ++idx) {
			STDCOUT("Layer#" << idx << ": " << noReg.w[idx] << " - " << wReg.w[idx] << "    ");
			ASSERT_GT(noReg.w[idx], wReg.w[idx]) << "Comparison for the layer failed!";
		}
		STDCOUT(::std::endl);
	}

	STDCOUTL("*************** Testing L1 regularizer ******************* ");
	for (unsigned i = 0; i < im; ++i) {
		auto sv = ::std::time(0);
		testL2L1(false, td, 0, sv, noReg);
		testL2L1(false, td, real_t(.01), sv, wReg);
		STDCOUTL("Sum of squared weights without regularizer vs with regularizer:");
		for (size_t idx = 0; idx < noReg.w.size(); ++idx) {
			STDCOUT("Layer#" << idx << ": " << noReg.w[idx] << " - " << wReg.w[idx] << "    ");
			ASSERT_GT(noReg.w[idx], wReg.w[idx]) << "Comparison for the layer failed!";
		}
		STDCOUT(::std::endl);
	}
}

/*
TEST(TestNnet, L2Weights) {
	train_data<real_t> td;
	reader_t reader;

	const auto srcFile = MNIST_FILE_DEBUG;//intended to use small (debug) variation here

	STDCOUTL("Reading datafile '" << srcFile << "'...");
	reader_t::ErrorCode rec = reader.read(srcFile, td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	testL2L1(true,td, 0, 0, 5, real_t(.02));
	testL2L1(true,td, real_t(.1), 0, 5, real_t(.02));
	testL2L1(false,td, real_t(.1), 0, 5, real_t(.02));
}*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename RealT>
void test_LSUVExt(train_data<RealT>& td, bool bCentNorm, bool bScaleNorm, bool bIndNeurons,const size_t rngSeed)noexcept {
#pragma warning(disable:4459)
	typedef RealT real_t;
#pragma warning(default:4459)

	const bool bUseLSUVExt = bCentNorm || bScaleNorm;
	SCOPED_TRACE(bUseLSUVExt ? "test_LSUVExt, bUseLSUVExt=true" : "test_LSUVExt, bUseLSUVExt=false");
	STDCOUT(::std::endl << "Running with ");
	if (bUseLSUVExt) {
		STDCOUTL("LSUVExt bCentNorm=" << bCentNorm << ", bScaleNorm=" << bScaleNorm << ", bIndNeurons=" << bIndNeurons);
	} else {
		STDCOUTL("plain OrthoInit");
	}

	const real_t learningRate = real_t(.0005);
	const size_t epochs = 8;
	
	typedef dt_interfaces<real_t> myIntf;
	typedef grad_works<myIntf> myGW;
	typedef weights_init::OrthoInit<10000000> w_init_scheme;//10e6 used here to make plain nnet learn somehow
	typedef activation::softsign<real_t, 1000000, 1000000, w_init_scheme> activ_func;
	//typedef activation::leaky_relu_100<real_t, w_init_scheme> activ_func;

	layer_input<myIntf> inp(td.train_x().cols_no_bias());

	layer_fully_connected<activ_func, myGW> fcl(256, learningRate);
	layer_fully_connected<activ_func, myGW> fcl2(128, learningRate);
	layer_fully_connected<activ_func, myGW> fcl3(64, learningRate);
	layer_fully_connected<activ_func, myGW> fcl4(64, learningRate);
	layer_fully_connected<activ_func, myGW> fcl5(64, learningRate);
	layer_fully_connected<activ_func, myGW> fcl6(64, learningRate);
	layer_fully_connected<activ_func, myGW> fcl7(64, learningRate);
	layer_fully_connected<activ_func, myGW> fcl8(64, learningRate);
	layer_fully_connected<activ_func, myGW> fcl9(64, learningRate);
	layer_output<activation::softsigm_xentropy_loss<real_t, 1000, w_init_scheme>, myGW> outp(td.train_y().cols(), learningRate);

	auto lp = make_layers(inp, fcl, fcl2, fcl3, fcl4, fcl5, fcl6, fcl7, fcl8, fcl9, outp);

	nnet_train_opts<training_observer_stdcout<eval_classification_one_hot_cached<real_t>>> opts(epochs);
	opts.batchSize(200);

	auto nn = make_nnet(lp);

	nn.get_iRng().seed64(rngSeed);
	//we must init weights right now to make sure they are the same for bUseLSUVExt and !bUseLSUVExt cases
	nn.get_layer_pack().for_each_layer_exc_input([&iR = nn.get_iRng(), &iM = nn.get_iMath()](auto& lyr) {
		math::smatrix<real_t> W;
		ASSERT_TRUE(W.resize(lyr.get_neurons_cnt(), lyr.get_incoming_neurons_cnt() + 1));
		ASSERT_TRUE(::std::decay_t<decltype(lyr)>::Weights_Init_t::init(W, iR, iM));
		//note that we MUST not use lyr.reinit_weights() function here because layer is not initialized at this point
		ASSERT_TRUE(lyr.set_weights(::std::move(W)));
		lyr.m_gradientWorks.set_type(decltype(lyr.m_gradientWorks)::Adam);
	});

	if (bUseLSUVExt) {
		typedef weights_init::procedural::LSUVExt<decltype(nn)> winit_t;
		winit_t::LayerSetts_t def, outpS;
		
		//individual neuron stats requires a lot of data to be correctly evaluated
		const auto batchSize = ::std::min(vec_len_t(bIndNeurons ? td.train_x().rows() : 2000), td.train_x().rows());

		//just to illustrate separate settings for a layer (most commonly it'll be an output_layer)
		outpS.bOverPreActivations = true;
		outpS.bCentralNormalize = bCentNorm;
		outpS.bScaleNormalize = bScaleNorm;
		outpS.bNormalizeIndividualNeurons = bIndNeurons;
		outpS.batchSize = batchSize;

		def.bOverPreActivations = true;
		def.bCentralNormalize = bCentNorm;
		def.bScaleNormalize = bScaleNorm;
		def.bNormalizeIndividualNeurons = bIndNeurons;
		def.batchSize = batchSize;
		
		winit_t obj(nn,def);
		obj.setts().add(outp.get_layer_idx(), outpS);

		
		if (!obj.run(td.train_x())) {
			STDCOUTL("*** Layer with ID="<<obj.m_firstFailedLayerIdx<<" was the first to fail convergence. There might be more of them.");
		}
	}

	//setting common rng state
	nn.get_iRng().seed64(rngSeed+1);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}

TEST(TestNnet, LSUVExt) {
#pragma warning(disable:4459)
	typedef double real_t;
#pragma warning(default:4459)

	train_data<real_t> td;
	readTd(td);

	const size_t s = ::std::time(0);
	STDCOUTL("Seed = " << s);

	test_LSUVExt<real_t>(td, false, false, false, s);

	test_LSUVExt<real_t>(td, true, false, false, s);
	test_LSUVExt<real_t>(td, true, false, true, s);
	
	test_LSUVExt<real_t>(td, false, true, false, s);
	test_LSUVExt<real_t>(td, false, true, true, s);

	test_LSUVExt<real_t>(td, true, true, false, s);
	test_LSUVExt<real_t>(td, true, true, true, s);

}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename ArchPrmsT>
struct GC_DROPOUT : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	typedef nntl::LFC_DO<activation::softsign<real_t>, myGradWorks> testedLFC;

	testedLFC lFinal;

	~GC_DROPOUT()noexcept {}
	GC_DROPOUT(const ArchPrms_t& Prms)noexcept
		: lFinal(100, Prms.learningRate, "lFinal")
	{
		lFinal.dropoutPercentActive(Prms.specialDropoutAlivePerc);
	}
};
TEST(TestNnet, GradCheck_dropout) {
#pragma warning(disable:4459)
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;
#pragma warning(default:4459)

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	Prms.specialDropoutAlivePerc = real_t(.8);
	
	nntl_tests::NN_arch<GC_DROPOUT<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts(true, true, 1e-4);
	//ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	//ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-3);//numeric errors due to dLdAPrev addition in LPH stacks up significantly
	
	ngcSetts.evalSetts.dLdA_setts.percOfZeros = 70;
	ngcSetts.evalSetts.dLdW_setts.percOfZeros = 70;
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 5, ngcSetts));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
