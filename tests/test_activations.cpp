/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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

TEST(TestActivations, Linear) {
	typedef math::smatrix<real_t> realmtx_t;
	typedef nntl_supp::binfile reader_t;

	train_data<real_t> td;
	reader_t reader;

	STDCOUTL("Reading datafile '" << MNIST_FILE << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(MNIST_FILE), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();

	size_t epochs = 5;
	const real_t learningRate(real_t(.001));

	layer_input<> inp(td.train_x().cols_no_bias());

	layer_fully_connected<activation::selu<real_t>> fcl(500, learningRate);
	//layer_fully_connected<activation::selu<real_t>> fcl2(300, learningRate);
	layer_fully_connected<activation::linear<real_t>> fcl2(300, learningRate);

	layer_output<activation::linear_quad_loss<real_t>> outp(td.train_y().cols(), learningRate);

	auto lp = make_layers(inp, fcl, fcl2, outp);

	nnet_train_opts<> opts(epochs);
	opts.batchSize(100);
	auto nn = make_nnet(lp);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();
}

template<typename ArchPrmsT>
struct GC_Activ_Linear : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	myLFC lFinal;

	~GC_Activ_Linear()noexcept {}
	GC_Activ_Linear(const ArchPrms_t& Prms)noexcept
		: lFinal(70, Prms.learningRate, "lFinal")
	{}
};

template <typename RealT, typename outputActivT>
struct GC_Activ_Linear_base_params : public nntl_tests::NN_base_params<RealT, nntl::inspector::GradCheck<RealT>> {
	typedef outputActivT myOutputActivation;

	~GC_Activ_Linear_base_params()noexcept {}
	GC_Activ_Linear_base_params(const nntl::train_data<real_t>& td)noexcept
		: nntl_tests::NN_base_params<RealT, nntl::inspector::GradCheck<RealT>>(td) {}
};

TEST(TestActivations, GradCheck_Linear_quadratic) {
#pragma warning(disable:4459)
	typedef double real_t;
	typedef GC_Activ_Linear_base_params<real_t, nntl::activation::linear_quad_loss<real_t, weights_init::SNNInit, true>> ArchPrms_t;
#pragma warning(default:4459)

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	nntl_tests::NN_arch<GC_Activ_Linear<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 1, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	//ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(1e-3);
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 3, ngcSetts));
}

TEST(TestActivations, GradCheck_Linear_quadWeighted) {
#pragma warning(disable:4459)
	typedef double real_t;
	typedef GC_Activ_Linear_base_params<real_t
		, nntl::activation::linear_output<nntl::activation::Linear_Loss_quadWeighted_FP<real_t>>> ArchPrms_t;
#pragma warning(default:4459)

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	nntl_tests::NN_arch<GC_Activ_Linear<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 1, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	//ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(1e-3);
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 3, ngcSetts));
}

