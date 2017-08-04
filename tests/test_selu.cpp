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


template<typename ArchPrmsT>
struct GC_ALPHADROPOUT : public nntl_tests::NN_base_arch_td<ArchPrmsT> {
	typedef nntl::LFC_DO<activation::selu<real_t>, myGradWorks> testedLFC;

	testedLFC lFinal;

	~GC_ALPHADROPOUT()noexcept {}
	GC_ALPHADROPOUT(const ArchPrms_t& Prms)noexcept
		: lFinal(100, Prms.learningRate, "lFinal")
	{
		lFinal.dropoutPercentActive(Prms.specialDropoutAlivePerc);
	}
};
TEST(TestSelu, GradCheck_alphaDropout) {
	typedef double real_t;
	typedef nntl_tests::NN_base_params<real_t, nntl::inspector::GradCheck<real_t>> ArchPrms_t;

	nntl::train_data<real_t> td;
	readTd(td);

	ArchPrms_t Prms(td);
	Prms.specialDropoutAlivePerc = real_t(.75);

	nntl_tests::NN_arch<GC_ALPHADROPOUT<ArchPrms_t>> nnArch(Prms);

	auto ec = nnArch.warmup(td, 5, 200);
	ASSERT_EQ(decltype(nnArch)::ErrorCode_t::Success, ec) << "Reason: " << nnArch.NN.get_error_str(ec);

	gradcheck_settings<real_t> ngcSetts(true, true, 1e-4);
	//ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer = true;
	ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh = real_t(1e-1);//big error is possible due to selu derivative kink :(
	//need some handling for it :(

	ngcSetts.evalSetts.dLdA_setts.percOfZeros = 70;
	ngcSetts.evalSetts.dLdW_setts.percOfZeros = 70;
	ASSERT_TRUE(nnArch.NN.gradcheck(td.train_x(), td.train_y(), 5, ngcSetts));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

template<typename RealT>
struct inspector_act_var_checker : public inspector::_impl::_base<RealT> {
protected:
	typedef utils::layer_idx_keeper<layer_index_t, _NoLayerIdxSpecified, 32> keeper_t;
	keeper_t m_curLayer;

	layer_index_t m_lastLayerIdxToCheck;

	typedef ::boost::accumulators::accumulator_set<ext_real_t
		, ::boost::accumulators::stats<
		::boost::accumulators::tag::mean
		, ::boost::accumulators::tag::lazy_variance
		>
	> stats_t;

	struct layers_stats_t {
		stats_t sMean;
		stats_t sVar;
	};

	typedef ::std::vector<layers_stats_t> a_layers_stat_t;
	a_layers_stat_t m_layersStats;

	void _calc_neuronwise_stats(const realmtx_t& act, const layer_index_t lidx)noexcept {
		auto& lStats = m_layersStats[lidx];

		const ptrdiff_t tr = act.rows();
		auto pA = act.data();
		const auto pAE = act.colDataAsVec(act.cols_no_bias());
		while (pA != pAE) {
			stats_t st;
			const auto pAEr = pA + tr;
			while (pA != pAEr) {
				st(*pA++);
			}

			lStats.sMean(::boost::accumulators::extract_result<::boost::accumulators::tag::mean>(st));
			lStats.sVar(::boost::accumulators::extract_result<::boost::accumulators::tag::lazy_variance>(st));
		}
	}

public:
	void init_nnet(const size_t totalLayers, const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
		m_lastLayerIdxToCheck = static_cast<layer_index_t>(totalLayers - 2);
		m_layersStats.resize(totalLayers - 1);
	}
	void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) noexcept {
		m_curLayer.push(lIdx);
	}
	void fprop_end(const realmtx_t& act) noexcept {
		if (m_curLayer <= m_lastLayerIdxToCheck) {
			_calc_neuronwise_stats(act, m_curLayer);
		}

		m_curLayer.pop();
	}

	template<typename base_t> struct stats_EPS {};
	template<> struct stats_EPS<double> {
		static constexpr double mean_eps = .07;
		static constexpr double var_eps = .17;
	};
	template<> struct stats_EPS<float> { 
		static constexpr float mean_eps = .07f;
		static constexpr float var_eps = .17f;
	};

	void report_stats(bool bDoAsserts = true)const noexcept {
		for (unsigned i = 0; i <= m_lastLayerIdxToCheck; ++i) {
			STDCOUTL("Reporting data distribution for layer#" << i << (i ? " -- SELU" : " -- input data"));

			const auto& lStats = m_layersStats[i];

			const auto mean_of_mean = ::boost::accumulators::extract_result<::boost::accumulators::tag::mean>(lStats.sMean);
			const auto var_of_mean = ::boost::accumulators::extract_result<::boost::accumulators::tag::lazy_variance>(lStats.sMean);
			const auto mean_of_var = ::boost::accumulators::extract_result<::boost::accumulators::tag::mean>(lStats.sVar);
			const auto var_of_var = ::boost::accumulators::extract_result<::boost::accumulators::tag::lazy_variance>(lStats.sVar);

			printf_s("mean = %05.3f +/- %06.4f, variance = %05.3f +/- %06.4f\n", mean_of_mean, var_of_mean, mean_of_var, var_of_var);

			if (bDoAsserts) {
				ASSERT_NEAR(mean_of_mean, real_t(0), stats_EPS<real_t>::mean_eps);
				ASSERT_NEAR(mean_of_var, real_t(1), stats_EPS<real_t>::var_eps);
			}
		}
	}

};

template<typename iRngT>
void _test_selu_make_td(train_data< typename iRngT::real_t >& td, const vec_len_t tr_cnt, const neurons_count_t xwidth, iRngT& iR)noexcept
{
	typedef typename iRngT::real_t real_t;
	typedef typename iRngT::realmtx_t realmtx_t;

	realmtx_t trX(tr_cnt, xwidth, true), trY(tr_cnt, 1), tX(1, xwidth, true), tY(1, 1);

	rng::distr_normal_naive<iRngT> rg(iR, real_t(0), real_t(1));
	rg.gen_matrix_no_bias(trX); rg.gen_matrix_no_bias(tX);

	iR.gen_matrix_norm(trY), iR.gen_matrix_norm(tY);

	ASSERT_TRUE(td.absorb(::std::move(trX), ::std::move(trY), ::std::move(tX), ::std::move(tY)));
}

template<typename RealT/*, bool bCorrectDO=false*/>
void test_selu_distr(const size_t seedVal, const RealT dpa, const neurons_count_t xwidth = 10, const neurons_count_t nc = 30
	, const bool bApplyWeightNorm = false, const bool bVerbose = true
	, const vec_len_t batchSize = 1000, const vec_len_t batchesCnt = 100)noexcept
{
	typedef RealT real_t;

	constexpr unsigned _scopeMsgLen = 128;
	char _scopeMsg[_scopeMsgLen];
	sprintf_s(_scopeMsg, "SELU_Distribution for X=%d/nc=%d with dropout dpa=%04.3f", xwidth, nc, dpa);
	SCOPED_TRACE(_scopeMsg);
	STDCOUTL(_scopeMsg);

	const real_t learningRate(real_t(0.));

	struct myIntf : public d_int_nI<real_t> {
		typedef inspector_act_var_checker<real_t> iInspect_t;
	};
	typedef grad_works_f<myIntf
		, GW::ILR_dummy
		, GW::Loss_Addendums_dummy
	> GrW;

	//typedef activation::selu<real_t, 0, 0, 0, 1000000, bCorrectDO> mySelu_t;
	typedef activation::selu<real_t, 0, 0, 0, 1000000, ADCorr::no> mySelu_t;

	layer_input<myIntf> inp(xwidth);
	LFC_DO<mySelu_t, GrW> fcl(nc, learningRate);
	fcl.dropoutPercentActive(dpa);
#ifndef TESTS_SKIP_LONGRUNNING
	LFC_DO<mySelu_t, GrW> fcl2(nc, learningRate);
	fcl2.dropoutPercentActive(dpa);
	LFC_DO<mySelu_t, GrW> fcl3(nc, learningRate);
	fcl3.dropoutPercentActive(dpa);
	LFC_DO<mySelu_t, GrW> fcl4(nc, learningRate);
	fcl4.dropoutPercentActive(dpa);
#endif

	layer_output<activation::softsigm_quad_loss<real_t>, GrW> outp(1, learningRate);

#ifdef TESTS_SKIP_LONGRUNNING
	auto lp = make_layers(inp, fcl, outp);
#else
	auto lp = make_layers(inp, fcl, fcl2, fcl3, fcl4, outp);
#endif

	nnet_train_opts<training_observer_stdcout<eval_classification_binary<real_t>>> opts(1);
	opts.batchSize(batchSize);

	auto nn = make_nnet(lp);

	nn.get_iRng().seed64(seedVal);

	train_data<real_t> td;
	_test_selu_make_td(td, batchesCnt*batchSize, xwidth, nn.get_iRng());


	if (bApplyWeightNorm) {
		typedef weights_init::procedural::LSUVExt<decltype(nn)> winit_t;
		winit_t::LayerSetts_t def, outpS;

		def.bOverPreActivations = false;
		def.bCentralNormalize = true;
		def.bScaleNormalize = true;
		def.bNormalizeIndividualNeurons = true;
		def.maxTries = 10;
		def.targetScale = real_t(1.);
		def.bVerbose = bVerbose;

		outpS.bCentralNormalize = false;
		outpS.bScaleNormalize = false;
		outpS.bVerbose = bVerbose;

		winit_t obj(nn, def);
		
		obj.setts().add(outp.get_layer_idx(), outpS);

		//individual neuron stats requires a lot of data to be correctly evaluated
		if (!obj.run(td.train_x())) {
			STDCOUTL("*** Layer with ID=" << obj.m_firstFailedLayerIdx << " was the first to fail convergence. There might be more of them.");
		}
	}

	nn.get_iRng().seed64(seedVal + 1);

	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

	ASSERT_NO_FATAL_FAILURE(nn.get_iInspect().report_stats(bVerbose));

}

TEST(TestSelu, SELU_Distribution) {
	typedef float real_t;

	const size_t t = ::std::time(0);
	const real_t dpa = real_t(.8);

	STDCOUTL("================ No weight renormalizing ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, real_t(1.), 10, 30, false));
	STDCOUTL("================ With weight renormalizing ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, real_t(1.), 10, 30, true));

	STDCOUTL("================ No weight renormalizing + AlphaDropout ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, dpa, 10, 30, false));
	STDCOUTL("================ With weight renormalizing + AlphaDropout ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, dpa, 10, 30, true));

#ifndef TESTS_SKIP_LONGRUNNING
	STDCOUTL("================ No weight renormalizing ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, real_t(1.), 100, 400, false));
	STDCOUTL("================ With weight renormalizing ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, real_t(1.), 100, 400, true));

	STDCOUTL("================ No weight renormalizing + AlphaDropout ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, dpa, 100, 400, false));
	STDCOUTL("================ With weight renormalizing + AlphaDropout ================");
	ASSERT_NO_FATAL_FAILURE(test_selu_distr(t, dpa, 100, 400, true));
#endif
}

/*
TEST(TestSelu, AlphaDropoutDistributionWithCorrection) {
	typedef float real_t;

	const size_t t = ::std::time(0);

	vec_len_t xW = 20;

	for (neurons_count_t nc = 5; nc <= 30; nc+=5) {
		real_t dpa = real_t(0.97);
		STDCOUTL(::std::endl<<"================ No correction ================");
		ASSERT_NO_FATAL_FAILURE((test_selu_distr<real_t, false>(t, dpa, xW, nc, true, false)));
		STDCOUTL("================ With correction ================");
		ASSERT_NO_FATAL_FAILURE((test_selu_distr<real_t, true>(t, dpa, xW, nc, true, false)));

		dpa = real_t(0.7);
		STDCOUTL("================ No correction ================");
		ASSERT_NO_FATAL_FAILURE((test_selu_distr<real_t, false>(t, dpa, xW, nc, true, false)));
		STDCOUTL("================ With correction ================");
		ASSERT_NO_FATAL_FAILURE((test_selu_distr<real_t, true>(t, dpa, xW, nc, true, false)));
	}

}*/