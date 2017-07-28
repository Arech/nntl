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

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename real_t>
struct testActL1L2_res {
	::std::array<real_t, 2> w;
};
template<bool bRegularize, typename LayerT>
::std::enable_if_t<bRegularize> _testActivationsL2L1_setupAddendum(const real_t& coeff, LayerT& lyr)noexcept {
	auto& reg = lyr.addendum<0>();
	reg.scale(coeff);
	STDCOUTL("Regularizer " << reg.getName() << " is used for " << lyr.get_layer_name_str() << ". Scale coefficient = " << reg.scale());
}
template<bool bRegularize, typename LayerT>
::std::enable_if_t<!bRegularize> _testActivationsL2L1_setupAddendum(const real_t& coeff, LayerT& lyr)noexcept {
	STDCOUTL("No regularizer is used for " << lyr.get_layer_name_str());
}

template<typename LossAddT>
struct testActivationsL2L1_td {
	typedef activation::elu_ua<real_t> activ_func;
	//typedef weights_init::XavierFour w_init_scheme;
	//typedef activation::sigm<real_t, w_init_scheme> activ_func;

	typedef LFC<activ_func> _MyLFC;

	template<typename FpcT>
	using _MyLFC_tpl = _LFC<FpcT, activ_func, grad_works<d_interfaces>, default_dropout_for<activ_func>>;

	static constexpr bool bRegularize = loss_addendum::is_loss_addendum<LossAddT>::value;
	typedef ::std::conditional_t<bRegularize, LPA<_MyLFC_tpl, LossAddT>, _MyLFC> MyLFC;

	typedef layer_output<activation::sigm_xentropy_loss<real_t>> MyOutput;
};

template<typename LossAddT>
void testActivationsL2L1(train_data<real_t>& td, const real_t coeff, uint64_t rngSeed, testActL1L2_res<real_t>& res, const size_t maxEpochs = 3, const real_t LR = .02, const char* pDumpFileName = nullptr) noexcept {
	typedef testActivationsL2L1_td<LossAddT> my_td;

	layer_input<> inp(td.train_x().cols_no_bias());

	const size_t epochs = maxEpochs;
	const real_t learningRate = LR;

	my_td::MyLFC fcl("fcl1", 100, learningRate);
	my_td::MyLFC fcl2("fcl2", 100, learningRate);
	my_td::MyOutput outp(td.train_y().cols(), learningRate);

	auto lp = make_layers(inp, fcl, fcl2, outp);

	_testActivationsL2L1_setupAddendum<my_td::bRegularize>(coeff, fcl);
	_testActivationsL2L1_setupAddendum<my_td::bRegularize>(coeff, fcl2);

	nnet_train_opts<> opts(epochs);
	opts.calcFullLossValue(true).batchSize(100);

	auto nn = make_nnet(lp);
	nn.get_iRng().seed64(rngSeed);

	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "training failed, Error code description: " << nn.get_last_error_string();

	ec = nn.fprop(td.train_x());
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "fprop failed, Error code description: " << nn.get_last_error_string();

	auto& iM = nn.get_iMath();
	res.w[0] = iM.ewSumSquares_ns(fcl.get_activations());
	res.w[1] = iM.ewSumSquares_ns(fcl2.get_activations());

/*
#if NNTL_MATLAB_AVAILABLE
	if (pDumpFileName) {
		nntl_supp::omatfile<> mf;
		mf.turn_on_all_options();
		mf.open(::std::string(pDumpFileName));
		mf << serialization::make_nvp("outpW", outp.get_weights());
		mf << serialization::make_nvp("fcl2W", fcl2.get_weights());
		mf << serialization::make_nvp("fclW", fcl.get_weights());
	}
#endif*/
}

TEST(TestLPA, ActivationsConstraintsL2L1) {
	train_data<real_t> td;
	
	readTd(td);// , MNIST_FILE_DEBUG);//intended to use small (debug) variation here
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	const unsigned im = 2;
	testActL1L2_res<real_t> noReg, wReg;

	for (unsigned i = 0; i < im; ++i) {
		auto sv = ::std::time(0);
		testActivationsL2L1<void>(td, 0, sv, noReg);
		STDCOUT(::std::endl);

		testActivationsL2L1<loss_addendum::L1<real_t>>(td, real_t(.01), sv, wReg);
		STDCOUTL(::std::endl << "Sum of squared activations without regularizer vs with L1 regularizer:");
		for (size_t idx = 0; idx < noReg.w.size(); ++idx) {
			STDCOUT("Layer#" << idx << ": " << noReg.w[idx] << " - " << wReg.w[idx] << "    ");
			ASSERT_GT(noReg.w[idx], wReg.w[idx]) << "Comparison for the layer failed!";
		}
		STDCOUT(::std::endl << ::std::endl << ::std::endl);

		testActivationsL2L1<loss_addendum::L2<real_t>>(td, real_t(.01), sv, wReg);
		STDCOUTL(::std::endl<<"Sum of squared activations without regularizer vs with L2 regularizer:");
		for (size_t idx = 0; idx < noReg.w.size(); ++idx) {
			STDCOUT("Layer#" << idx << ": " << noReg.w[idx] << " - " << wReg.w[idx] << "    ");
			ASSERT_GT(noReg.w[idx], wReg.w[idx]) << "Comparison for the layer failed!";
		}
		STDCOUT(::std::endl << ::std::endl << ::std::endl);
	}
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename ArchPrmsT>
struct GC_LPA_deCov : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	template<typename FpcT>
	using _MyLFC_tpl = _LFC<FpcT, myActivation, myGradWorks, default_dropout_for<myActivation>>;

	typedef LPA<_MyLFC_tpl, loss_addendum::DeCov<real_t, true>> MyLPA;
	//typedef myLFC MyLPA;

	MyLPA lFinal;

	~GC_LPA_deCov()noexcept {}
	GC_LPA_deCov(const ArchPrms_t& Prms)noexcept
		: lFinal("lFinal", 70, Prms.learningRate, Prms.dropoutAlivePerc)
	{}
};

template<typename RealT>
struct GC_LPA_deCov_ArchPrms : public nntl_tests::NN_base_params<RealT, nntl::inspector::GradCheck<RealT>> {
private:
	typedef nntl_tests::NN_base_params<RealT, nntl::inspector::GradCheck<RealT>> _base_class_t;
public:
	typedef nntl::activation::softsigm_quad_loss <real_t, 1000, nntl::weights_init::He_Zhang<>, true> myOutputActivation;
	//typedef nntl::activation::debug_softsigm_zeros <real_t, 1000, nntl::weights_init::He_Zhang<>> myOutputActivation;

	~GC_LPA_deCov_ArchPrms()noexcept{}
	GC_LPA_deCov_ArchPrms(const nntl::train_data<real_t>& td)noexcept : _base_class_t(td) {}
};

TEST(TestLPA, deCovGradCheck) {
	typedef double real_t;
	//typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;
	typedef GC_LPA_deCov_ArchPrms<real_t> ArchPrms_t;

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	nntl_tests::NN_arch<GC_LPA_deCov<ArchPrms_t>> nnArch(Prms);

	nnArch.ArchObj.lFinal.addendum<0>().scale(real_t(10000));

	auto ec = nnArch.warmup(td, 5, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.onlineBatchSize = 4;//batch must be big enought to compute columnwise covariance
	
	//ngcSetts.evalSetts.dLdW_setts.percOfZeros = 100; //for debug_softsigm_zeros
	
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-4);
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 5, ngcSetts));
}