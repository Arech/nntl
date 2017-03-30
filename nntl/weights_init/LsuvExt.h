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

#include <type_traits>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "_procedural_base.h"
#include "../utils/layers_settings.h"

//LSUV algorithm (D.Mishkin, J.Matas, "All You Need is a Good Init", ArXiv:1511.06422) with some extensions.
// Default settings replicate LSUV as described in the paper (and in the supplemental code in
// https://github.com/ducha-aiki/LSUVinit/blob/master/tools/extra/lsuv_init.py), provided that one has used a correct 
// weights_init::* scheme (usually it is a weights_init::OrthoInit) to parametrize layers templates
// 
// set bNormalizeIndividualNeurons flag to calculate variance over individual neuron's batch values instead of a whole set of neurons
// set bCentralNormalize to make activations zero centered
// set bOverPreActivations to calculate batch statistics over pre-activation values instead of values obtained after non-linearity
// 
// Actually, the pure LSUV algorithm as described in the paper could be not the most effective one for a given data. Try to play with
// the settings to see which one is better for your data.

namespace nntl {
namespace weights_init {
	namespace procedural {

		template<typename RealT>
		struct WeightNormSetts {
			typedef RealT real_t;

			real_t targetScale;
			real_t ScaleTolerance, CentralTolerance;

			unsigned maxTries;

			bool bCentralNormalize;
			bool bScaleNormalize;
			bool bScaleChangeBiasesToo;//in general setting this flag to true works better, though in some odd cases
										 // the 'false' setting might have a little bit of advantage (possibly just because of numeric effects).
										 //Leave it here until findout the correct way

			bool bNormalizeIndividualNeurons;
			bool bOverPreActivations;
			bool bVerbose;

			WeightNormSetts()noexcept:targetScale(real_t(1.))
				, ScaleTolerance(real_t(.01)), CentralTolerance(real_t(.01))
				, maxTries(20), bCentralNormalize(false), bNormalizeIndividualNeurons(false)
				, bOverPreActivations(false), bVerbose(true), bScaleNormalize(true)
				, bScaleChangeBiasesToo(true)
			{}
		};

		//this code doesn't have to be very fast, so we can use boost and leave a lot of math code here instead of moving it into iMath

		template<typename NnetT
			, typename ScaleMeasureT = boost::accumulators::tag::lazy_variance
			, typename CentralMeasureT = boost::accumulators::tag::mean
		>
		struct LSUVExt : public _impl::_base<NnetT> {
		private:
			typedef _impl::_base<NnetT> _base_class_t;

		public:
			typedef WeightNormSetts<real_t> LayerSetts_t;
			typedef utils::layer_settings<LayerSetts_t> setts_keeper_t;

		public:
			setts_keeper_t m_setts;

			layer_index_t m_firstFailedLayerIdx;

		protected:


		public:
			LSUVExt(nnet_t& nn) noexcept : _base_class_t(nn), m_firstFailedLayerIdx(0) {}

			template<typename ST>
			LSUVExt(nnet_t& nn, ST&& defSetts) noexcept 
				: _base_class_t(nn), m_firstFailedLayerIdx(0), m_setts(std::forward<ST>(defSetts))
			{}/*
			LSUVExt(nnet_t& nn, const LayerSetts_t& defSetts) noexcept
				: _base_class_t(nn), m_firstFailedLayerIdx(0), m_setts(defSetts)
			{}*/

			setts_keeper_t& setts()noexcept { return m_setts; }

			bool run(const vec_len_t& batchSize, const realmtx_t& data_x)noexcept {
				_base_class_t::init(batchSize, data_x);
				m_firstFailedLayerIdx = 0;
				m_data.prepateToBatchSize(batchSize);

				//to fully initialize the nn and its layers
				_fprop();

				m_nn.get_layer_pack().for_each_packed_layer_exc_input(*this);

				_base_class_t::deinit();
				return m_firstFailedLayerIdx == 0;
			}

			template<typename LayerT> void operator()(LayerT& lyr) noexcept {
				//first processing internal layers
				_checkInnerLayers(lyr);
				//then adjusting own weights if applicable
				_processLayer(lyr);
			}

		protected:
			template<typename LayerT>
			std::enable_if_t<is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT& lyr)noexcept {
				lyr.for_each_packed_layer(*this);
			}
			template<typename LayerT>
			std::enable_if_t<!is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT& lyr)const noexcept {}

			template<typename LayerT>
			std::enable_if_t<!is_layer_learnable<LayerT>::value> _processLayer(LayerT& lyr) const noexcept {}

			template<typename LayerT>
			std::enable_if_t<is_layer_learnable<LayerT>::value> _processLayer(LayerT& lyr) noexcept {
				const LayerSetts_t& lSetts = m_setts[lyr.get_layer_idx()];

				if (!lSetts.bCentralNormalize && !lSetts.bScaleNormalize) {
					if (lSetts.bVerbose) STDCOUTL(lyr.get_layer_name_str() << ": --skipped");
					return;
				}

				if (lSetts.bVerbose) STDCOUTL(lyr.get_layer_name_str() << ":");

				lyr.setLayerLinear(lSetts.bOverPreActivations);

				if (lSetts.bNormalizeIndividualNeurons) {
					if (!_processWeights_individual(lyr.get_weights(), lyr.get_activations(), lSetts) && !m_firstFailedLayerIdx)
						m_firstFailedLayerIdx = lyr.get_layer_idx();
				} else {
					if (!_processWeights_shared(lyr.get_weights(), lyr.get_activations(), lSetts) && !m_firstFailedLayerIdx)
						m_firstFailedLayerIdx = lyr.get_layer_idx();
				}

				lyr.setLayerLinear(false);
			}
			
			bool _processWeights_shared(realmtx_t& weights, const realmtx_t& activations, const LayerSetts_t& lSetts)noexcept {
				NNTL_ASSERT(weights.rows() == activations.cols_no_bias());

				bool bScaleIsOk = true, bCentralIsOk = true;
				if (lSetts.bScaleNormalize) {
					for (unsigned i = 0; i < lSetts.maxTries; ++i) {
						_fprop();

						const auto stat = _calc<ScaleMeasureT>(activations.begin(), activations.end_no_bias());
						bScaleIsOk = (std::abs(stat - lSetts.targetScale) < lSetts.ScaleTolerance);

						if (lSetts.bVerbose) STDCOUTL("#" << i << " scale (\"variance\") = " << stat << (bScaleIsOk ? "(ok)" : "****"));
						
						if (!bScaleIsOk) {
							if (lSetts.bScaleChangeBiasesToo) {
								m_nn.get_iMath().evMulC_ip(weights, std::sqrt(lSetts.targetScale / stat));
							} else {
								realmtx_t nW;
								nW.useExternalStorage(weights.data(), weights.rows(), weights.cols() - 1);
								m_nn.get_iMath().evMulC_ip(nW, std::sqrt(lSetts.targetScale / stat));
							}
						} else break;
					}
					if (!bScaleIsOk) _say_failed();
				}
				//offsetting bias weights breaks orthogonality of weight matrix (when OrthoInit was used). Therefore to change
				//it little less we'll update biases after we've reached correct scale.
				if (lSetts.bCentralNormalize) {
					for (unsigned i = 0; i < lSetts.maxTries; ++i) {
						_fprop();

						const auto stat = _calc<CentralMeasureT>(activations.begin(), activations.end_no_bias());
						bCentralIsOk = (std::abs(stat - real_t(0.)) < lSetts.CentralTolerance);

						if (lSetts.bVerbose) STDCOUTL("#" << i << " central (\"mean\") = " << stat << (bCentralIsOk ? "(ok)" : "****"));

						if (!bCentralIsOk) {
							auto pB = weights.colDataAsVec(weights.cols() - 1);
							const auto pBE = weights.end();
							while (pB < pBE) *pB++ -= stat;
						}else break;
					}
					if (!bCentralIsOk) _say_failed();
				}
				return bScaleIsOk && bCentralIsOk;
			}

			bool _processWeights_individual(realmtx_t& weights, const realmtx_t& activations, const LayerSetts_t& lSetts)noexcept {
				NNTL_ASSERT(weights.rows() == activations.cols_no_bias());
				const neurons_count_t total_nc = activations.cols_no_bias();
				const ptrdiff_t batchSize = activations.rows(), _total_nc = total_nc;
				bool bScaleStatOK = true, bCentralStatOK = true;
				if (lSetts.bScaleNormalize) {
					for (unsigned i = 0; i < lSetts.maxTries; ++i) {
						_fprop();
						ext_real_t statSum=0;
						bScaleStatOK = true;

						for (neurons_count_t nrn = 0; nrn < total_nc; ++nrn) {
							const auto pD = activations.colDataAsVec(nrn);
							const auto stat = _calc<ScaleMeasureT>(pD, pD + batchSize);
							const bool bScaleIsOk = (std::abs(stat - lSetts.targetScale) < lSetts.ScaleTolerance);

							statSum += stat;

							if (!bScaleIsOk) {
								bScaleStatOK = false;
								const auto mulVal = std::sqrt(lSetts.targetScale / stat);
								auto pW = weights.data() + nrn;
								const auto pWE = weights.colDataAsVec(weights.cols() - !lSetts.bScaleChangeBiasesToo);
								while (pW < pWE) {
									*pW *= mulVal;
									pW += _total_nc;
								}
							}
						}

						if (lSetts.bVerbose) STDCOUTL("#" << i << " avg scale (\"variance\") = " << statSum / total_nc << (bScaleStatOK ? "(ok)" : "****"));
						if (bScaleStatOK) break;
					}
					if (!bScaleStatOK) _say_failed();
				}
				//offsetting bias weights breaks orthogonality of weight matrix (when OrthoInit is being used). Therefore to change
				//it little less we'll update biases after we've reached the correct scale.
				if (lSetts.bCentralNormalize) {
					for (unsigned i = 0; i < lSetts.maxTries; ++i) {
						_fprop();

						ext_real_t statSum = 0;
						bCentralStatOK = true;

						for (neurons_count_t nrn = 0; nrn < total_nc; ++nrn) {
							const auto pD = activations.colDataAsVec(nrn);
							const auto stat = _calc<CentralMeasureT>(pD, pD + batchSize);
							const bool bCentralIsOk = (std::abs(stat - real_t(0.)) < lSetts.CentralTolerance);
							statSum += stat;
							
							if (!bCentralIsOk) {
								bCentralStatOK = false;
								weights.get(nrn, weights.cols() - 1) -= stat;
							}
						}

						if (lSetts.bVerbose) STDCOUTL("#" << i << " avg central (\"variance\") = " << statSum / total_nc << (bCentralStatOK ? "(ok)" : "****"));
						if (bCentralStatOK) break;
					}
					if (!bCentralStatOK) _say_failed();
				}

				return bCentralStatOK && bScaleStatOK;
			}

			void _say_failed()noexcept {
				STDCOUTL("**** failed to converge");
			}

			void _fprop()noexcept {
				m_data.nextBatch(m_nn.get_iRng(), m_nn.get_iMath());
				m_nn.fprop(m_data.batchX());
			}

			template<typename TagStatT>
			real_t _calc(const real_t* pBegin, const real_t*const pEnd)noexcept {
				using namespace boost::accumulators;

				accumulator_set<real_t, stats<TagStatT>> acc;
				while (pBegin < pEnd) {
					acc(*pBegin++);
				}
				return extract_result<TagStatT>(acc);
			}

			/*void _calc(const real_t* pBegin, const real_t*const pEnd, _calcRes& res, const ptrdiff_t stride=1)noexcept {
				using namespace boost::accumulators;

				accumulator_set<real_t, stats<CentralMeasureT, ScaleMeasureT>> acc;
				while (pBegin < pEnd) {
					acc(*pBegin);
					pBegin += stride;
				}
				res.central = extract_result<CentralMeasureT>(acc);
				res.scale = extract_result<ScaleMeasureT>(acc);

				res.bCentralIsOk = !m_bCentralNormalize || (std::abs(res.central - real_t(0.)) < m_CentralTolerance);
				res.bScaleIsOk = !m_bScaleNormalize || (std::abs(res.scale - m_targetScale) < m_ScaleTolerance);
			}*/

		};

	}
}
}