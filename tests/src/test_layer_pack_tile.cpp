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

#include "stdafx.h"

//to get rid of '... decorated name length exceeded, name was truncated'
#pragma warning( disable : 4503 )

#include <nntl/math_details.h>
#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_test/test_weights_init.h"
#include "asserts.h"
#include "common_routines.h"
#include "nn_base_arch.h"

using namespace nntl;

typedef d_interfaces::real_t real_t;
typedef math::smatrix<real_t> realmtx_t;
typedef math::smatrix_deform<real_t> realmtxdef_t;

template<typename ArchPrmsT>
struct GC_LPT : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	myLFC lBase;
	LPT<decltype(lBase)> lFinal;

	~GC_LPT()noexcept {}
	GC_LPT(const ArchPrms_t& Prms)noexcept
		: lBase(50, Prms.learningRate, "lBase")
		, lFinal(lBase, 3, "lFinal")
	{}
};
TEST(TestLayerPackTile, GradCheck) {
#pragma warning(disable:4459)
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;
#pragma warning(default:4459)

	nntl::inmem_train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	Prms.lUnderlay_nc = 300;
	nntl_tests::NN_arch<GC_LPT<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 100);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(1e-2);//numeric errors may stacks up significantly
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 3, ngcSetts));
}

template<typename ArchPrmsT>
struct GC_LPT_LPV : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	myLFC l1;
	myLFC l2;
	LPV<decltype(l1), decltype(l2)> lBase;
	LPT<decltype(lBase)> lFinal;

	~GC_LPT_LPV()noexcept {}
	GC_LPT_LPV(const ArchPrms_t& Prms)noexcept
		: l1(50, Prms.learningRate, "l1")
		, l2(70, Prms.learningRate, "l2")
		, lBase("lBase", l1, l2)
		, lFinal(lBase, 3, "lFinal")
	{}
};
TEST(TestLayerPackTile, GradCheck_LPV) {
#pragma warning(disable:4459)
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;
#pragma warning(default:4459)

	nntl::inmem_train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	Prms.lUnderlay_nc = 300;
	nntl_tests::NN_arch<GC_LPT_LPV<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 100);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts;
	ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(5e-3);//numeric errors stacks up significantly
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 10, ngcSetts));
}



//////////////////////////////////////////////////////////////////////////

template<typename base_t> struct TestLayerPackTile_EPS {};
template<> struct TestLayerPackTile_EPS <double> { static constexpr double eps = 1e-15; };
template<> struct TestLayerPackTile_EPS <float> { static constexpr float eps = 7e-7f; };

TEST(TestLayerPackTile, ComparativeNonSpecialX) {
	constexpr vec_len_t samplesCount = 109;
	realmtx_t _train_x(samplesCount, 31, true), _train_y(samplesCount, 1, false);

	const vec_len_t batchSize = _train_x.rows();

	constexpr neurons_count_t K = 3, tiledLayerNeurons = 37, tiledLayerIncomingNeurons = 43;
	const real_t lr = 1*K;

	typedef LFC<activation::sigm<real_t, weights_init::XavierFour>> FCL;
	typedef layer_output<activation::sigm_quad_loss<real_t, weights_init::XavierFour>> LO;

	//////////////////////////////////////////////////////////////////////////

	layer_input<> Ainp(_train_x.cols_no_bias());
	FCL Aund(tiledLayerIncomingNeurons * K, lr);//underlying layer to test dLdA correctness

	FCL Atlfc(tiledLayerNeurons, lr);//layer to tile
	LPT<decltype(Atlfc)> Alpt(Atlfc, K);

	LO Aoutp(_train_y.cols(), lr);

	auto Alp = make_layers(Ainp, Aund, Alpt, Aoutp);
	auto Ann = make_nnet(Alp);

	//Ann.get_iRng().seed64(0);
	Ann.get_iRng().gen_matrix_no_bias_norm(_train_x);
	Ann.get_iRng().gen_matrix_norm(_train_y);

	//////////////////////////////////////////////////////////////////////////
	// initializing layers with a help from nn object

	auto ec = Ann.___init(batchSize, batchSize, false);

	//saving layers weights to reuse in comparison
	realmtx_t AundW, AundOrigW, AtlfcW, AundAct, AlptAct, AoutpAct, AoutpW;
	Aund.get_weights().clone_to(AundW);
	AundW.clone_to(AundOrigW);
	Atlfc.get_weights().clone_to(AtlfcW);
	Aoutp.get_weights().clone_to(AoutpW);

	Ann.___get_common_data().set_mode_and_batch_size(true, batchSize);
	Alp.on_batch_size_change(batchSize);
	Alp.fprop(_train_x);

	//saving activations for comparison
	ASSERT_TRUE(Aund.get_activations().clone_to(AundAct));
	ASSERT_TRUE(Alpt.get_activations().clone_to(AlptAct));
	ASSERT_TRUE(Aoutp.get_activations().clone_to(AoutpAct));

	//doing bprop
	Alp.bprop(_train_y);

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// now assembling the same architecture from different components for comparison

	layer_input<> Binp(_train_x.cols_no_bias());
	FCL Bund(tiledLayerIncomingNeurons * K, lr);//underlying layer to test dLdA correctness

	FCL Blfc1(tiledLayerNeurons, lr), Blfc2(tiledLayerNeurons, lr), Blfc3(tiledLayerNeurons, lr);

	LPH<PHL<decltype(Blfc1)>, PHL<decltype(Blfc2)>, PHL<decltype(Blfc3)>> Blph(
		make_PHL(Blfc1, 0 * tiledLayerIncomingNeurons, tiledLayerIncomingNeurons),
		make_PHL(Blfc2, 1 * tiledLayerIncomingNeurons, tiledLayerIncomingNeurons),
		make_PHL(Blfc3, 2 * tiledLayerIncomingNeurons, tiledLayerIncomingNeurons)
	);

	LO Boutp(_train_y.cols(), lr);

	auto Blp = make_layers(Binp, Bund, Blph, Boutp);
	auto Bnn = make_nnet(Blp);
	//Bnn.get_iRng().seed64(1);

	//////////////////////////////////////////////////////////////////////////
	// initializing layers with a help from nn object

	ec = Bnn.___init(batchSize, batchSize, false);

	//setting the same layer weights
	ASSERT_TRUE(Bund.set_weights(::std::move(AundW)));

	AtlfcW.clone_to(AundW);
	ASSERT_TRUE(Blfc1.set_weights(::std::move(AundW)));
	AtlfcW.clone_to(AundW);
	ASSERT_TRUE(Blfc2.set_weights(::std::move(AundW)));
	ASSERT_TRUE(Blfc3.set_weights(::std::move(AtlfcW)));

	ASSERT_TRUE(Boutp.set_weights(::std::move(AoutpW)));

	// doing fprop
	Bnn.___get_common_data().set_mode_and_batch_size(true, batchSize);
	Blp.on_batch_size_change(batchSize);
	Blp.fprop(_train_x);

	//comparing activations
	ASSERT_MTX_EQ(AundAct, static_cast<const realmtx_t&>(Bund.get_activations()), "Underlying level post-fprop activations comparison failed!");
	ASSERT_REALMTX_NEAR(AlptAct,
		static_cast<const realmtx_t&>(Blph.get_activations()),
		"Tiled layer post-fprop activations comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);

	ASSERT_REALMTX_NEAR(AoutpAct,
		static_cast<const realmtx_t&>(Boutp.get_activations()),
		"Output layer post-fprop activations comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);

	//doing bprop
	Blp.bprop(_train_y);
	
	ASSERT_REALMTX_NEAR(Aund.get_weights(), Bund.get_weights(),
		"Underlying layer post-bprop weights comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);

	//weights of Blfc1/Blfc2/Blfc3 were updated based on batchSize samples, however, weights of Atlfc were updated
	// based on batchSize*K samples, therefore they just can't be the same
	//ASSERT_REALMTX_NEAR(Atlfc.get_weights(), Blfc1.get_weights(), "! must fail at the first element", TestLayerPackTile_EPS<real_t>::eps);

}
