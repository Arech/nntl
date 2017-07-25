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
#pragma once

#include "../nntl/interface/inspectors/dummy.h"

namespace nntl_tests {

	template <typename RealT, typename InspectorT = nntl::inspector::dummy<RealT>>
	struct NN_base_params : public nntl::math::smatrix_td {
		typedef RealT real_t;
		typedef InspectorT Inspector_t;

		typedef nntl::activation::softsign<real_t> myActivation;
		//typedef nntl::activation::leaky_relu_100<real_t> myActivation;

		typedef nntl::activation::softsigm_quad_loss <real_t, 1000, nntl::weights_init::He_Zhang<>, true> myOutputActivation;
		//typedef nntl::activation::sigm_quad_loss<real_t> myOutputActivation;

		const vec_len_t xCols, yCols;
		nntl::neurons_count_t lUnderlay_nc;

		real_t learningRate;
		real_t outputLearningRate;

		real_t dropoutAlivePerc, specialDropoutAlivePerc;
		real_t nesterovMomentum;
		real_t l2regularizer, l1regularizer;
		
		real_t maxNormVal;
		bool bNormIncludesBias;

		~NN_base_params()noexcept {}
		NN_base_params(const nntl::train_data<real_t>& td)noexcept
			: xCols(td.train_x().cols_no_bias()), yCols(td.train_y().cols())
			, lUnderlay_nc(37)//any sufficiently big number suits
			, learningRate(real_t(.001)), outputLearningRate(learningRate)
			, dropoutAlivePerc(real_t(1.)) , nesterovMomentum(real_t(0.))
			, specialDropoutAlivePerc(real_t(1.))
			, l2regularizer(real_t(0.)), l1regularizer(real_t(0.))
			,maxNormVal(real_t(0.)), bNormIncludesBias(false)
		{}
	};


	template<typename ArchPrmsT>
	struct NN_base_arch_td : public nntl::math::smatrix_td {
		typedef ArchPrmsT ArchPrms_t;
		typedef typename ArchPrms_t::real_t real_t;

		struct myInterfaces_t : public nntl::d_int_nI<real_t> {
			typedef typename ArchPrms_t::Inspector_t iInspect_t;
		};		
		//typedef nntl::dt_interfaces<real_t, InspectorT> myInterfaces_t;

		typedef nntl::grad_works<myInterfaces_t> myGradWorks;

		typedef typename ArchPrms_t::myActivation myActivation;

		typedef nntl::LFC<myActivation, myGradWorks> myLFC;
	};

	template <typename NnInnerArchT>
	struct NN_arch : public NN_base_arch_td<typename NnInnerArchT::ArchPrms_t> {
		typedef NnInnerArchT nnInnerArch_t;
		typedef typename nnInnerArch_t::ArchPrms_t ArchPrms_t;

		typedef typename ArchPrms_t::myOutputActivation myOutputActivation;

		typedef nntl::layer_input<myInterfaces_t> myLayerInput;
		typedef nntl::layer_output<myOutputActivation, myGradWorks> myLayerOutput;

		//////////////////////////////////////////////////////////////////////////

		myLayerInput lInput;
		myLFC lUnderlay; //need it to check dLdA computed inside of ArchObj.lFinal
		nnInnerArch_t ArchObj;
		myLayerOutput lOutput;

		nntl::layers<decltype(lInput), decltype(lUnderlay), decltype(nnInnerArch_t::lFinal), decltype(lOutput)> lp;

		typedef nntl::nnet<decltype(lp)> myNnet_t;
		typedef typename myNnet_t::ErrorCode ErrorCode_t;

		myNnet_t NN;

		//////////////////////////////////////////////////////////////////////////

		template<typename ArchPrmsT>
		struct LayerInit {
			const ArchPrmsT& Prms;

			LayerInit(const ArchPrmsT& P) noexcept : Prms(P) {}

			template<typename _L> std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l)const noexcept {
				l.m_gradientWorks
					//.set_type(decltype(l.m_gradientWorks)::RProp)
					//.set_type(decltype(l.m_gradientWorks)::RMSProp_Hinton)
					//.set_type(decltype(l.m_gradientWorks)::RMSProp_Graves)
					//.set_type(decltype(l.m_gradientWorks)::ModProp)
					.set_type(decltype(l.m_gradientWorks)::Adam)
					//.set_type(decltype(l.m_gradientWorks)::AdaMax)
					//.numeric_stabilizer(real_t(.0001))
					//.beta1(real_t(.95))

					.nesterov_momentum(Prms.nesterovMomentum)
					.L2(Prms.l2regularizer)
					.L1(Prms.l1regularizer)
					.max_norm(Prms.maxNormVal, Prms.bNormIncludesBias)
					//.applyILRToMomentum(Prms.bILR2Momentum)
					//.set_ILR(Prms.ilrProps)
					;
			}
			template<typename _L> std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L& l)const noexcept {}
		};

		//////////////////////////////////////////////////////////////////////////

		~NN_arch()noexcept {}
		NN_arch(const ArchPrms_t& Prms) noexcept
			: lInput(Prms.xCols, "lInput")
			, lUnderlay(Prms.lUnderlay_nc, Prms.learningRate, Prms.dropoutAlivePerc, "lUnderlay")
			, ArchObj(Prms)
			, lOutput(Prms.yCols, Prms.outputLearningRate, "lOutput")
			, lp(lInput, lUnderlay, ArchObj.lFinal, lOutput)
			, NN(lp)
		{
			initLayers(LayerInit<ArchPrms_t>(Prms));
		}

		template<typename LayerInitializerT>
		void initLayers(const LayerInitializerT& liObj)noexcept {
			lp.for_each_layer_exc_input([&liObj](auto& l) {
				liObj(l);
			});
		}

		ErrorCode_t warmup(nntl::train_data<real_t>& td, const size_t epochs, const vec_len_t batch_size
			, const bool bCalcFullLoss=true)noexcept
		{
			STDCOUTL("Going to perform warmup for "<<epochs<<" epochs...");

			nntl::nnet_train_opts<
				nntl::training_observer_stdcout<nntl::eval_classification_one_hot<real_t>>//need this to pass correct real_t
			> opts(epochs, false);

			opts.batchSize(batch_size).ImmediatelyDeinit(false).calcFullLossValue(bCalcFullLoss);

			return NN.train<false>(td, opts);
		}
	};
};
