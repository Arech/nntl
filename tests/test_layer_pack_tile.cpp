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

#include "../nntl/math.h"
#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_test/test_weights_init.h"
#include "asserts.h"
#include "common_routines.h"

using namespace nntl;

template<typename base_t> struct TestLayerPackTile_EPS {};
template<> struct TestLayerPackTile_EPS <double> { static constexpr double eps = 1e-15; };
template<> struct TestLayerPackTile_EPS <float> { static constexpr double eps = 2e-7; };

TEST(TestLayerPackTile, ComparativeNonSpecialX) {
	constexpr vec_len_t samplesCount = 109;
	realmtx_t _train_x(samplesCount, 31, true), _train_y(samplesCount, 1, false);

	const vec_len_t batchSize = _train_x.rows();

	constexpr neurons_count_t K = 3, tiledLayerNeurons = 37, tiledLayerIncomingNeurons = 43;
	const real_t lr = 1*K;

	typedef LFC<activation::sigm<weights_init::XavierFour>> FCL;
	typedef layer_output<activation::sigm_quad_loss<weights_init::XavierFour>> LO;

	//////////////////////////////////////////////////////////////////////////

	layer_input<> Ainp(_train_x.cols_no_bias());
	FCL Aund(tiledLayerIncomingNeurons * K, lr);//underlying layer to test dLdA correctness

	FCL Atlfc(tiledLayerNeurons, lr);//layer to tile
	auto Alpt = make_layer_pack_tile<K, false>(Atlfc);

	LO Aoutp(_train_y.cols(), lr);

	auto Alp = make_layers(Ainp, Aund, Alpt, Aoutp);
	auto Ann = make_nnet(Alp);

	//Ann.get_iRng().seed64(0);
	Ann.get_iRng().gen_matrix_no_bias_norm(_train_x);
	Ann.get_iRng().gen_matrix_norm(_train_y);

	//////////////////////////////////////////////////////////////////////////
	// initializing layers with a help from nn object

	_impl::_tmp_train_data<decltype(Ann)::layers_pack_t> Attd;
	auto ec = Ann.___init(batchSize, batchSize, false, _train_x.cols(), _train_y.cols(), &Attd);

	//saving layers weights to reuse in comparison
	realmtx_t AundW, AundOrigW, AtlfcW, AundAct, AlptAct, AoutpW;
	Aund.get_weights().cloneTo(AundW);
	AundW.cloneTo(AundOrigW);
	Atlfc.get_weights().cloneTo(AtlfcW);
	Aoutp.get_weights().cloneTo(AoutpW);

	Alp.set_mode(0);
	Alp.fprop(_train_x);

	//saving activations for comparison
	Aund.get_activations().cloneTo(AundAct);
	Alpt.get_activations().cloneTo(AlptAct);

	//doing bprop
	Alp.bprop(_train_y, Attd.a_dLdA);
	ASSERT_MTX_EQ(AundAct, static_cast<const realmtx_t&>(Aund.get_activations()), "bprop has modified the AundAct!");
	ASSERT_MTX_EQ(AlptAct, static_cast<const realmtx_t&>(Alpt.get_activations()), "bprop has modified the AlptAct!");

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// now assembling the same architecture from different components for comparison

	layer_input<> Binp(_train_x.cols_no_bias());
	FCL Bund(tiledLayerIncomingNeurons * K, lr);//underlying layer to test dLdA correctness

	FCL Blfc1(tiledLayerNeurons, lr / K), Blfc2(tiledLayerNeurons, lr / K), Blfc3(tiledLayerNeurons, lr / K);
	auto Blph = make_layer_pack_horizontal(
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

	_impl::_tmp_train_data<decltype(Bnn)::layers_pack_t> Bttd;
	ec = Bnn.___init(batchSize, batchSize, false, _train_x.cols(), _train_y.cols(), &Bttd);

	//setting the same layer weights
	ASSERT_TRUE(Bund.set_weights(std::move(AundW)));

	AtlfcW.cloneTo(AundW);
	ASSERT_TRUE(Blfc1.set_weights(std::move(AundW)));
	AtlfcW.cloneTo(AundW);
	ASSERT_TRUE(Blfc2.set_weights(std::move(AundW)));
	ASSERT_TRUE(Blfc3.set_weights(std::move(AtlfcW)));

	ASSERT_TRUE(Boutp.set_weights(std::move(AoutpW)));

	// doing fprop
	Blp.set_mode(0);
	Blp.fprop(_train_x);

	//comparing activations
	ASSERT_MTX_EQ(AundAct, static_cast<const realmtx_t&>(Bund.get_activations()), "Underlying level post-fprop activations comparison failed!");
	ASSERT_REALMTX_NEAR(AlptAct,
		static_cast<const realmtx_t&>(Blph.get_activations()),
		"Tiled layer post-fprop activations comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);

	ASSERT_REALMTX_NEAR(static_cast<const realmtx_t&>(Aoutp.get_activations()),
		static_cast<const realmtx_t&>(Boutp.get_activations()),
		"Output layer post-fprop activations comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);

	//doing bprop
	Blp.bprop(_train_y, Bttd.a_dLdA);

	//comparing activations and weights
	ASSERT_MTX_EQ(AundAct, static_cast<const realmtx_t&>(Bund.get_activations()), "Underlying level post-bprop activations comparison failed!");
	ASSERT_REALMTX_NEAR(AlptAct,
		static_cast<const realmtx_t&>(Blph.get_activations()),
		"Tiled layer post-bprop activations comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);


	ASSERT_REALMTX_NEAR(Aund.get_weights(), Bund.get_weights(),
		"Underlying layer post-bprop weights comparison failed!",
		TestLayerPackTile_EPS<real_t>::eps);

	//weights of Blfc1/Blfc2/Blfc3 were updated based on batchSize samples, however, weights of Atlfc were updated
	// based on batchSize*K samples, therefore they just can't be the same
	//ASSERT_REALMTX_NEAR(Atlfc.get_weights(), Blfc1.get_weights(), "! must fail at the first element", TestLayerPackTile_EPS<real_t>::eps);

}
