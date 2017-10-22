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

#pragma warning(push, 3)
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#pragma warning(pop)

#include "_procedural_base.h"
#include "../utils/layers_settings.h"
//#include "../utils/layer_idx_keeper.h"

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
		struct WeightNormSetts : public math::smatrix_td {
			typedef RealT real_t;

			real_t targetScale;
			real_t ScaleTolerance, CentralTolerance;

			vec_len_t batchSize;//set to 0 for a full-batch mode (default)

			unsigned maxTries, maxReinitTries;

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
				, bScaleChangeBiasesToo(true), maxReinitTries(5), batchSize(0)
			{}
		};

		//this code doesn't have to be very fast, so we can use boost and leave a lot of math code here instead of moving it into iMath

		template<typename NnetT
			, typename ScaleMeasureT = ::boost::accumulators::tag::lazy_variance
			, typename CentralMeasureT = ::boost::accumulators::tag::mean
		>
		struct LSUVExt : public _impl::_base<NnetT> {
		private:
			typedef _impl::_base<NnetT> _base_class_t;

		public:
			typedef WeightNormSetts<real_t> LayerSetts_t;
			typedef utils::layer_settings<LayerSetts_t> setts_keeper_t;

		protected:
			struct gating_context_t : public nntl::_impl::GatingContext<real_t> {
				//vec_len_t curGatingMaskColumn;
				const real_t* pCurGatingColumn;//is has pGatingMask->rows()

				void updateCurColumn(const layer_index_t& idx)noexcept {
					NNTL_ASSERT(colsDescr.size() > 0);
					NNTL_ASSERT(pGatingMask);
					NNTL_ASSERT(colsDescr.size() <= pGatingMask->cols());
					const auto curGatingMaskColumn = colsDescr.at(idx);
					NNTL_ASSERT(curGatingMaskColumn < pGatingMask->cols());
					pCurGatingColumn = pGatingMask->colDataAsVec(curGatingMaskColumn);
				}
			};
			typedef ::std::vector<gating_context_t> gating_ctx_stack_t;

		public:
			setts_keeper_t m_setts;

		protected:
			gating_ctx_stack_t m_gatingStack;

		public:
			layer_index_t m_firstFailedLayerIdx;

		protected:
			vec_len_t m_fullBatchSize;
			bool m_bImmediatelyUnderGated;

		public:
			LSUVExt(nnet_t& nn) noexcept : _base_class_t(nn), m_firstFailedLayerIdx(0) {}

			template<typename ST>
			LSUVExt(nnet_t& nn, ST&& defSetts) noexcept 
				: _base_class_t(nn), m_firstFailedLayerIdx(0), m_setts(::std::forward<ST>(defSetts))
				, m_fullBatchSize(0)
			{}/*
			LSUVExt(nnet_t& nn, const LayerSetts_t& defSetts) noexcept
				: _base_class_t(nn), m_firstFailedLayerIdx(0), m_setts(defSetts)
			{}*/

			setts_keeper_t& setts()noexcept { return m_setts; }

			bool run(const realmtx_t& data_x)noexcept {
				vec_len_t secondBiggestBatch = 0, biggestBatch = 0;
				const auto fullBatch = data_x.rows();
				m_fullBatchSize = fullBatch;
				m_setts.for_each([&secondBiggestBatch, &biggestBatch, fullBatch](const LayerSetts_t& ls)noexcept
				{
					const auto bs = ::std::min(ls.batchSize, fullBatch);
					biggestBatch = ::std::max(biggestBatch, bs);
					if (bs < fullBatch) {
						secondBiggestBatch = ::std::max(secondBiggestBatch, ls.batchSize);
					}
				});

				_base_class_t::init(secondBiggestBatch, data_x);

				m_firstFailedLayerIdx = 0;
				m_gatingStack.clear();
				m_bImmediatelyUnderGated = false;

				//to fully initialize the nn and its layers
				prepareToBatchSize(biggestBatch);
				_fprop();

				m_nn.get_layer_pack().for_each_packed_layer_exc_input(*this);

				_base_class_t::deinit();
				NNTL_ASSERT(m_gatingStack.size() == 0);
				NNTL_ASSERT(!m_bImmediatelyUnderGated);
				
				return m_firstFailedLayerIdx == 0;
			}

			template<typename LayerT> void operator()(LayerT& lyr) noexcept {
				const bool bImmUnderGated = m_bImmediatelyUnderGated;
				if (m_bImmediatelyUnderGated) {
					NNTL_ASSERT(m_gatingStack.size() > 0);

					auto& ctx = m_gatingStack.back();
					const auto& lIdx = lyr.get_layer_idx();
					if (ctx.bShouldProcessLayer(lIdx)) {
						ctx.updateCurColumn(lIdx);
						if (m_setts[lIdx].bVerbose) {
							STDCOUTL("Layer \'" << lyr.get_layer_name_str()
								<< "\' is directly under a gated layer. Switched current gating mask column to column#"
								<< ctx.colsDescr[lIdx]);
						}
					} else {
						if (m_setts[lIdx].bVerbose) {
							STDCOUTL("Layer \'" << lyr.get_layer_name_str()
								<< "\' is directly under a gated layer, but the gate is non applicable to it.");
						}
					}
				}
				m_bImmediatelyUnderGated = false;

				_packGatingPrologue(lyr);
				_packTilingPrologue(lyr);
				
				//first processing internal layers
				_checkInnerLayers(lyr);

				_packTilingEpilogue(lyr);
				_packGatingEpilogue(lyr);

				m_bImmediatelyUnderGated = bImmUnderGated;

				//then adjusting own weights if applicable
				_processLayer(lyr);
			}

		protected:
			// Few special considirations:
			// In order to correctly rescale weights of layers that works under gates (such as parts of LPHG), we must keep a track of gating layers
			// and corresponding gates (they can be stacked on top of each other, BTW). Also, we must provide special handling for
			// tiled layers under gates, because they have different batchSizes and a gating mask must be applied appropriately.

			template<typename LayerT>
			::std::enable_if_t<is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT& lyr)noexcept {
				lyr.for_each_packed_layer(*this);
			}
			template<typename LayerT> ::std::enable_if_t<!is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT&)const noexcept {}


			template<typename LayerT>
			::std::enable_if_t<is_pack_gated<LayerT>::value> _packGatingPrologue(LayerT& lyr)noexcept {
				if (m_gatingStack.size() > 0) {
					//#todo we should make a mix of two masks - the new one and the previous
					STDCOUTL("*** attention, LSUVExt code for enclosed gated layers is not yet written. Proceed as there's only a single gated layer");
				}
				gating_context_t ctx;
				lyr.get_gating_info(ctx);
				ctx.pCurGatingColumn = nullptr;
				m_gatingStack.push_back(::std::move(ctx));

				m_bImmediatelyUnderGated = true;
			}
			template<typename LayerT> ::std::enable_if_t<!is_pack_gated<LayerT>::value> _packGatingPrologue(LayerT&)const noexcept {}

			template<typename LayerT>
			::std::enable_if_t<is_pack_gated<LayerT>::value> _packGatingEpilogue(LayerT&)noexcept {
				m_gatingStack.pop_back();
			}
			template<typename LayerT> ::std::enable_if_t<!is_pack_gated<LayerT>::value> _packGatingEpilogue(LayerT&)const noexcept {}


			template<typename LayerT>
			::std::enable_if_t<is_pack_tiled<LayerT>::value> _packTilingPrologue(LayerT&)noexcept {
				//#todo
				static_assert(false, "There must be code to synchronize current gating mask with a different batch size of tiled layer");
				//if you will not use a LPT under a LPHG, you may safely comment out the assert and leave everything as it is now.
			}
			template<typename LayerT> ::std::enable_if_t<!is_pack_tiled<LayerT>::value> _packTilingPrologue(LayerT&)const noexcept {}

			template<typename LayerT>
			::std::enable_if_t<is_pack_tiled<LayerT>::value> _packTilingEpilogue(LayerT&)noexcept {

			}
			template<typename LayerT> ::std::enable_if_t<!is_pack_tiled<LayerT>::value> _packTilingEpilogue(LayerT&)const noexcept {}


			template<typename LayerT>
			::std::enable_if_t<!is_layer_learnable<LayerT>::value> _processLayer(LayerT&) const noexcept {}

			template<typename LayerT>
			::std::enable_if_t<is_layer_learnable<LayerT>::value> _processLayer(LayerT& lyr) noexcept {
				const auto lIdx = lyr.get_layer_idx();
				const LayerSetts_t& lSetts = m_setts[lIdx];

				if (!lSetts.bCentralNormalize && !lSetts.bScaleNormalize) {
					if (lSetts.bVerbose) STDCOUTL(lyr.get_layer_name_str() << ": --skipped");
					return;
				}

				if (lSetts.bVerbose) STDCOUTL(lyr.get_layer_name_str() << ":");

				lyr.setLayerLinear(lSetts.bOverPreActivations);

				const real_t actScaling = lSetts.bOverPreActivations ? real_t(1.) : lyr.act_scaling_coeff();
				//const real_t actScaling = real_t(1.);

				prepareToBatchSize(lSetts.batchSize);

				unsigned rt;
				for (rt = 0; rt < lSetts.maxReinitTries; ++rt) {
					const bool bSuccess = lSetts.bNormalizeIndividualNeurons 
						? _processWeights_individual(lyr.get_weights(), lyr.get_activations_storage(), lSetts, actScaling)
						: _processWeights_shared(lyr.get_weights(), lyr.get_activations_storage(), lSetts, actScaling);

					if (bSuccess) {
						break;
					} else {
						STDCOUT("Reinitializing weights, try " << rt + 1);
						if (lyr.reinit_weights()) {
							STDCOUT(::std::endl);
						}else{
							STDCOUTL(" ***** failed!!! ****");
						}
					}
				}
				if (rt >= lSetts.maxReinitTries) {
					if (!m_firstFailedLayerIdx) m_firstFailedLayerIdx = lIdx;
					STDCOUTL("******* layer " << lyr.get_layer_name_str() << " failed to converge! *********");
				}

				lyr.setLayerLinear(false);
			}

			void prepareToBatchSize(const vec_len_t _bs)noexcept {
				NNTL_ASSERT(m_fullBatchSize);
				const auto bs = _bs ? ::std::min(_bs, m_fullBatchSize) : m_fullBatchSize;
				m_data.prepareToBatchSize(bs);
				m_nn.init4fixedBatchFprop(bs);
			}
			
			bool _processWeights_shared(realmtx_t& weights, const realmtx_t*const pAct, const LayerSetts_t& lSetts, const real_t actScaling)noexcept {
				NNTL_ASSERT(pAct);
				NNTL_ASSERT(weights.rows() == pAct->cols_no_bias());
				NNTL_ASSERT(actScaling > real_t(0));

				const real_t*const pCurGatingColumn = m_gatingStack.size() > 0 ? m_gatingStack.back().pCurGatingColumn : nullptr;
				bool bScaleIsOk = true, bCentralIsOk = true, bForceReinit = false;

				if (lSetts.bScaleNormalize) {
					for (unsigned i = 0; i < lSetts.maxTries; ++i) {
						_fprop();

						const auto stat = _calc<ScaleMeasureT>(pAct->begin(), pAct->end_no_bias(), pAct->rows(), pCurGatingColumn);

						if (::std::isnan(stat) || real_t(0) == stat || !::std::isfinite(stat)) {
							STDCOUTL("*** got invalid scale statistic, will try to reinit weights");
							bForceReinit = true;
							bScaleIsOk = false;
							break;
						}

						bScaleIsOk = (::std::abs(stat - lSetts.targetScale) < lSetts.ScaleTolerance);

						if (lSetts.bVerbose) STDCOUTL("#" << i << " scale (\"variance\") = " << stat << (bScaleIsOk ? "(ok)" : ""));
						
						if (!bScaleIsOk) {
							NNTL_ASSERT(stat != real_t(0));
							const auto mulVal = ::std::sqrt(lSetts.targetScale / stat);
							NNTL_ASSERT(!isnan(mulVal) && isfinite(mulVal));

							if (lSetts.bScaleChangeBiasesToo) {
								m_nn.get_iMath().evMulC_ip(weights, mulVal);
							} else {
								realmtx_t nW;
								nW.useExternalStorage(weights.data(), weights.rows(), weights.cols() - 1);
								m_nn.get_iMath().evMulC_ip(nW, mulVal);
							}
						} else break;
					}
					//if (!bScaleIsOk) _say_failed();
				} else {
					_fprop();
					const auto stat = _calc<ScaleMeasureT>(pAct->begin(), pAct->end_no_bias(), pAct->rows(), pCurGatingColumn);

					if (::std::isnan(stat) || real_t(0) == stat || !::std::isfinite(stat)) {
						STDCOUTL("*** got invalid scale statistic, will try to reinit weights");
						bForceReinit = true;
						bScaleIsOk = false;
					} else {
						bScaleIsOk = (::std::abs(stat - lSetts.targetScale) < lSetts.ScaleTolerance);

						if (lSetts.bVerbose) STDCOUTL("scale (\"variance\") = " << stat << (bScaleIsOk ? "(ok)" : "(*doesn't fit, but changing was forbidden*)"));
						bScaleIsOk = true;
					}
				}

				//offsetting bias weights breaks orthogonality of weight matrix (when OrthoInit was used). Therefore to change
				//it little less we'll update biases after we've reached correct scale.
				if (bScaleIsOk) {
					if (lSetts.bCentralNormalize) {
						for (unsigned i = 0; i < lSetts.maxTries; ++i) {
							_fprop();

							const auto stat = _calc<CentralMeasureT>(pAct->begin(), pAct->end_no_bias(), pAct->rows(), pCurGatingColumn);
							bCentralIsOk = (::std::abs(stat - real_t(0.)) < lSetts.CentralTolerance);

							if (lSetts.bVerbose) STDCOUTL("#" << i << " central (\"mean\") = " << stat << (bCentralIsOk ? "(ok)" : ""));

							if (!bCentralIsOk) {
								auto pB = weights.colDataAsVec(weights.cols() - 1);
								const auto pBE = weights.end();
								while (pB < pBE) *pB++ -= stat*actScaling;
							} else break;
						}
						//if (!bCentralIsOk) _say_failed();
					} else if (lSetts.bVerbose) {
						_fprop();

						const auto stat = _calc<CentralMeasureT>(pAct->begin(), pAct->end_no_bias(), pAct->rows(), pCurGatingColumn);
						bCentralIsOk = (::std::abs(stat - real_t(0.)) < lSetts.CentralTolerance);
						STDCOUTL("central (\"mean\") = " << stat << (bCentralIsOk ? "(ok)" : "(*doesn't fit, but changing was forbidden*)"));
						bCentralIsOk = true;
					}
				}
				return !bForceReinit && bScaleIsOk && bCentralIsOk;
			}

			bool _processWeights_individual(realmtx_t& weights, const realmtx_t*const pAct, const LayerSetts_t& lSetts, const real_t actScaling)noexcept {
				NNTL_ASSERT(pAct);
				NNTL_ASSERT(weights.rows() == pAct->cols_no_bias());
				NNTL_ASSERT(actScaling > real_t(0));

				const neurons_count_t total_nc = pAct->cols_no_bias();
				const ptrdiff_t batchSize = static_cast<ptrdiff_t>(pAct->rows());
				const ptrdiff_t _total_nc = static_cast<ptrdiff_t>(total_nc);
				const real_t*const pCurGatingColumn = m_gatingStack.size() > 0 ? m_gatingStack.back().pCurGatingColumn : nullptr;
				bool bScaleStatOK = true, bCentralStatOK = true, bForceReinit = false;
				if (lSetts.bScaleNormalize) {
					for (unsigned i = 0; i < lSetts.maxTries; ++i) {
						_fprop();
						ext_real_t statSum=0;
						bScaleStatOK = true;

						for (neurons_count_t nrn = 0; nrn < total_nc; ++nrn) {
							const auto pD = pAct->colDataAsVec(nrn);
							const auto stat = _calc<ScaleMeasureT>(pD, pD + batchSize, batchSize, pCurGatingColumn);

							if (::std::isnan(stat) || real_t(0)==stat || !::std::isfinite(stat)) {
								STDCOUTL("*** got invalid scale statistic, will try to reinit weights");
								bForceReinit = true;
								bScaleStatOK = false;
								break;
							}

							const bool bScaleIsOk = (::std::abs(stat - lSetts.targetScale) < lSetts.ScaleTolerance);

							statSum += stat;

							if (!bScaleIsOk) {
								bScaleStatOK = false;
								NNTL_ASSERT(stat != real_t(0.));
								const auto mulVal = ::std::sqrt(lSetts.targetScale / stat);
								NNTL_ASSERT(!isnan(mulVal) && isfinite(mulVal));
								auto pW = weights.data() + nrn;
								const auto pWE = weights.colDataAsVec(weights.cols() - !lSetts.bScaleChangeBiasesToo);
								while (pW < pWE) {
									*pW *= mulVal;
									pW += _total_nc;
								}
							}
						}

						if (!bForceReinit && lSetts.bVerbose) STDCOUTL("#" << i << " avg scale (\"variance\") = " << statSum / total_nc << (bScaleStatOK ? "(ok)" : ""));
						if (bScaleStatOK || bForceReinit) break;
					}
					//if (!bScaleStatOK) _say_failed();
				} else {
					_fprop();
					ext_real_t statSum = 0;

					for (neurons_count_t nrn = 0; nrn < total_nc; ++nrn) {
						const auto pD = pAct->colDataAsVec(nrn);
						const auto stat = _calc<ScaleMeasureT>(pD, pD + batchSize, batchSize, pCurGatingColumn);

						if (::std::isnan(stat) || real_t(0) == stat || !::std::isfinite(stat)) {
							STDCOUTL("*** got invalid scale statistic, will try to reinit weights");
							bForceReinit = true;
							bScaleStatOK = false;
							break;
						}

						const bool bScaleIsOk = (::std::abs(stat - lSetts.targetScale) < lSetts.ScaleTolerance);
						statSum += stat;

						if (!bScaleIsOk) {
							bScaleStatOK = false;
							NNTL_ASSERT(stat != real_t(0.));
						}
					}

					if (!bForceReinit) {
						if (lSetts.bVerbose) STDCOUTL("avg scale (\"variance\") = " << statSum / total_nc << (bScaleStatOK ? "(ok)" : "(*doesn't fit, but changing was forbidden*)"));
						bScaleStatOK = true;
					}
				}
				//offsetting bias weights breaks orthogonality of weight matrix (when OrthoInit is being used). Therefore to change
				//it little less we'll update biases after we've reached the correct scale.
				if (bScaleStatOK) {
					if (lSetts.bCentralNormalize) {
						for (unsigned i = 0; i < lSetts.maxTries; ++i) {
							_fprop();

							ext_real_t statSum = 0;
							bCentralStatOK = true;

							for (neurons_count_t nrn = 0; nrn < total_nc; ++nrn) {
								const auto pD = pAct->colDataAsVec(nrn);
								const auto stat = _calc<CentralMeasureT>(pD, pD + batchSize, batchSize, pCurGatingColumn);
								const bool bCentralIsOk = (::std::abs(stat - real_t(0.)) < lSetts.CentralTolerance);
								statSum += stat;

								if (!bCentralIsOk) {
									bCentralStatOK = false;
									weights.get(nrn, weights.cols() - 1) -= stat*actScaling;
								}
							}

							if (lSetts.bVerbose) STDCOUTL("#" << i << " avg central (\"mean\") = " << statSum / total_nc << (bCentralStatOK ? "(ok)" : ""));
							if (bCentralStatOK) break;
						}
						//if (!bCentralStatOK) _say_failed();
					} else {
						_fprop();
						ext_real_t statSum = 0;

						for (neurons_count_t nrn = 0; nrn < total_nc; ++nrn) {
							const auto pD = pAct->colDataAsVec(nrn);
							const auto stat = _calc<CentralMeasureT>(pD, pD + batchSize, batchSize, pCurGatingColumn);
							const bool bCentralIsOk = (::std::abs(stat - real_t(0.)) < lSetts.CentralTolerance);
							statSum += stat;

							if (!bCentralIsOk) {
								bCentralStatOK = false;
							}
						}

						if (lSetts.bVerbose) STDCOUTL("avg central (\"mean\") = " << statSum / total_nc << (bCentralStatOK ? "(ok)" : "(*doesn't fit, but changing was forbidden*)"));
						bCentralStatOK = true;
					}
				}

				return !bForceReinit && bCentralStatOK && bScaleStatOK;
			}

			/*void _say_failed()noexcept {
				STDCOUTL("**** failed to converge");
			}*/

			void _fprop()noexcept {
				m_data.nextBatch(m_nn.get_iRng(), m_nn.get_iMath());
				m_nn.doFixedBatchFprop(m_data.batchX());
			}

			template<typename TagStatT>
			real_t _calc_simple(const real_t* pBegin, const real_t*const pEnd)noexcept {
				using namespace ::boost::accumulators;

				accumulator_set<real_t, stats<TagStatT>> acc;
				while (pBegin < pEnd) {
					acc(*pBegin++);
				}
				return extract_result<TagStatT>(acc);
			}
			template<typename TagStatT>
			real_t _calc_with_mask(const real_t* pBegin, const real_t*const pEnd, const size_t rowsCnt, const real_t*const pMask)noexcept {
				NNTL_ASSERT(static_cast<double>(pEnd - pBegin) / rowsCnt == (pEnd - pBegin) / rowsCnt);

				using namespace ::boost::accumulators;

				accumulator_set<real_t, stats<TagStatT>> acc;
				while (pBegin < pEnd) {
					NNTL_ASSERT(pEnd >= pBegin + rowsCnt);
					for (size_t ri = 0; ri < rowsCnt; ++ri) {
						//floating-point positive zero is bitwise equal to fixed-point zero, but the latter works faster.
						//thought, it's probably doesn't make a big difference, because this loop can't be vectorized
						const math::real_t_limits<real_t>::similar_FWI_t*const pCond = reinterpret_cast<const math::real_t_limits<real_t>::similar_FWI_t*>(pMask + ri);
						if (*pCond != 0) {
							acc(*pBegin);
						}
						++pBegin;
					}
				}
				return extract_result<TagStatT>(acc);
			}

			template<typename TagStatT>
			real_t _calc(const real_t* pBegin, const real_t*const pEnd, const size_t rowsCnt, const real_t*const pMask)noexcept {
				NNTL_ASSERT(static_cast<double>(pEnd - pBegin) / rowsCnt == (pEnd - pBegin) / rowsCnt);

				const auto r= pMask ? _calc_with_mask<TagStatT>(pBegin, pEnd, rowsCnt, pMask) : _calc_simple<TagStatT>(pBegin, pEnd);
				NNTL_ASSERT(!isnan(r) && isfinite(r));
				return r;
			}

			/*void _calc(const real_t* pBegin, const real_t*const pEnd, _calcRes& res, const ptrdiff_t stride=1)noexcept {
				using namespace ::boost::accumulators;

				accumulator_set<real_t, stats<CentralMeasureT, ScaleMeasureT>> acc;
				while (pBegin < pEnd) {
					acc(*pBegin);
					pBegin += stride;
				}
				res.central = extract_result<CentralMeasureT>(acc);
				res.scale = extract_result<ScaleMeasureT>(acc);

				res.bCentralIsOk = !m_bCentralNormalize || (::std::abs(res.central - real_t(0.)) < m_CentralTolerance);
				res.bScaleIsOk = !m_bScaleNormalize || (::std::abs(res.scale - m_targetScale) < m_ScaleTolerance);
			}*/

		};

	}
}
}