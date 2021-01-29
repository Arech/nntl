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

#pragma once

#include <type_traits>

#include "../utils/layers_settings.h"
#include "../utils/mtx2Normal.h"

//LSUV algorithm (D.Mishkin, J.Matas, "All You Need is a Good Init", ArXiv:1511.06422) with some extensions.
// Default settings replicate LSUV as described in the paper (and in the supplemental code in
// https://github.com/ducha-aiki/LSUVinit/blob/master/tools/extra/lsuv_init.py), provided that one has used a correct 
// weights_init::* scheme (usually it is a weights_init::OrthoInit) to parametrize layers templates
// 
// set bOnInvidualNeurons flag to calculate variance over individual neuron's batch values instead of a whole set of neurons
// set bCentralNormalize to make activations zero centered
// set bOverPreActivations to calculate batch statistics over pre-activation values instead of values obtained after non-linearity
// 
// Actually, the pure LSUV algorithm as described in the paper could be not the most effective one for a given data. Try to play with
// the settings to see which one is better for your data.

namespace nntl {
namespace weights_init {
	namespace procedural {

		template<typename DataT, typename StatsT = DataT>
		struct WeightNormSetts : public utils::mtx2Normal::Settings<DataT, StatsT> {
			vec_len_t batchSize{0};//set to 0 for a full-batch mode (default)
			vec_len_t batchCount{0};//0 means gather stats over whole dataset, else this count of batches

			unsigned maxReinitTries{ 5 };

			bool bOnInvidualNeurons{ false };//columns instead of whole matrix
			//true value is slower, but may yield better weights.
			// Note that normalizing the central/mean only is the slowest to converge.

			bool bOverPreActivations{ false };
		};

		//this code doesn't have to be very fast, so we can use boost and leave a lot of math code here instead of moving it into iMath

		//note that actually we don't support anything except mean as central and var as scale measures as for now
		//(need a generic way to update weights for generic measures)

		template<typename NnetT, typename TDT, typename StatsT = typename NnetT::real_t>
		class LSUVExt {
		public:
			typedef LSUVExt<NnetT, TDT, StatsT> self_t;

			typedef NnetT nnet_t;
			typedef typename NnetT::real_t real_t;
			typedef typename NnetT::realmtx_t realmtx_t;
			typedef typename NnetT::realmtxdef_t realmtxdef_t;

			static_assert(is_train_data_intf<TDT>::value, "TDT template parameter must implement _i_train_data<> interface");
			typedef TDT td_t;

			typedef StatsT statsdata_t;
			typedef WeightNormSetts<real_t, statsdata_t> LayerSetts_t;
			typedef utils::layer_settings<LayerSetts_t> setts_keeper_t;

			//static constexpr bool bAdjustForSampleVar = true;
			typedef utils::mtx2Normal::_FNorm_stats_var_mean<true> StatsFunctor_t;

		protected:
			nnet_t& m_nn;

			//we'll use train_set::X only!
			td_t& m_td;

		public:
			setts_keeper_t m_setts;			
			layer_index_t m_firstFailedLayerIdx{ 0 };

		protected:
			vec_len_t m_largestBatchSize{ 0 };

		public:
			bool m_bDeinitNnetOnDestroy{ true };

		public:
			~LSUVExt()noexcept {
				if (m_bDeinitNnetOnDestroy) {
					m_nn.deinit();
				}
			}

			//Note that we'll only use train set's X of train data

			LSUVExt(nnet_t& nn, td_t& td) noexcept : m_nn(nn), m_td(td) {}

			template<typename ST>
			LSUVExt(nnet_t& nn, td_t& td, ST&& defSetts) noexcept
				: m_nn(nn), m_td(td), m_setts(::std::forward<ST>(defSetts))
			{}

			setts_keeper_t& setts()noexcept { return m_setts; }

			bool run()noexcept {
				m_firstFailedLayerIdx = 0;

				vec_len_t secondBiggestBatch = 0, biggestBatch = 0;
				const auto fullBatch = m_td.trainset_samples_count() > ::std::numeric_limits<vec_len_t>::max()
					? ::std::numeric_limits<vec_len_t>::max() : static_cast<vec_len_t>(m_td.trainset_samples_count());

				NNTL_ASSERT(fullBatch > 1);
				if (fullBatch < 2) {
					STDCOUTL("Too small data passed, variance normalization will fail");
					return false;
				}

				m_largestBatchSize = fullBatch;
				m_setts.for_each([&secondBiggestBatch, &biggestBatch, fullBatch](const LayerSetts_t& ls)noexcept
				{
					const auto bs = ::std::min(ls.batchSize, fullBatch);
					biggestBatch = ::std::max(biggestBatch, bs);
					if (bs < fullBatch) {
						secondBiggestBatch = ::std::max(secondBiggestBatch, ls.batchSize);
					}
				});
				
				m_nn.get_layer_pack().for_each_packed_layer_exc_input(*this);

				m_td.deinit4all();

				return m_firstFailedLayerIdx == 0;
			}

			template<typename LayerT> void operator()(LayerT& lyr) noexcept {
				_packGatingPrologue(lyr);
				_packTilingPrologue(lyr);
				
				//first processing internal layers
				_checkInnerLayers(lyr);

				_packTilingEpilogue(lyr);
				_packGatingEpilogue(lyr);

				//then adjusting own weights if applicable
				_processLayer(lyr);
			}

		protected:
			template<typename LayerT>
			::std::enable_if_t<is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT& lyr)noexcept {
				lyr.for_each_packed_layer(*this);
			}
			template<typename LayerT> ::std::enable_if_t<!is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT&)const noexcept {}

			//implementation was left empty by intent. Probably, will remove later
			template<typename LayerT> ::std::enable_if_t<is_pack_gated<LayerT>::value> _packGatingPrologue(LayerT&)noexcept {}
			template<typename LayerT> ::std::enable_if_t<!is_pack_gated<LayerT>::value> _packGatingPrologue(LayerT&)const noexcept {}
			template<typename LayerT> ::std::enable_if_t<is_pack_gated<LayerT>::value> _packGatingEpilogue(LayerT&)noexcept {}
			template<typename LayerT> ::std::enable_if_t<!is_pack_gated<LayerT>::value> _packGatingEpilogue(LayerT&)const noexcept {}

			template<typename LayerT> ::std::enable_if_t<is_pack_tiled<LayerT>::value> _packTilingPrologue(LayerT&)noexcept {}
			template<typename LayerT> ::std::enable_if_t<!is_pack_tiled<LayerT>::value> _packTilingPrologue(LayerT&)const noexcept {}
			template<typename LayerT> ::std::enable_if_t<is_pack_tiled<LayerT>::value> _packTilingEpilogue(LayerT&)noexcept {}
			template<typename LayerT> ::std::enable_if_t<!is_pack_tiled<LayerT>::value> _packTilingEpilogue(LayerT&)const noexcept {}

			template<typename LayerT> ::std::enable_if_t<!is_layer_learnable<LayerT>::value> _processLayer(LayerT&) const noexcept {}

			template<typename LayerT>
			::std::enable_if_t<is_layer_learnable<LayerT>::value> _processLayer(LayerT& lyr) noexcept {
				const auto lIdx = lyr.get_layer_idx();
				const LayerSetts_t& lSetts = m_setts[lIdx];

				if ((!lSetts.bCentralNormalize && !lSetts.bScaleNormalize) || lSetts.maxTries <= 0) {
					if (lSetts.bVerbose) STDCOUTL(lyr.get_layer_name_str() << ": --skipped");
					return;
				}

				NNTL_ASSERT(!lSetts.batchSize || lSetts.batchSize > 1);//variance calc need more than 1 sample
				if (lSetts.bVerbose) STDCOUTL(lyr.get_layer_name_str() << " (batchSize=" << lSetts.batchSize 
					<< ", batchCount=" << lSetts.batchCount << "):");

				lyr.setIgnoreActivation(lSetts.bOverPreActivations);

				const auto actScaling = lSetts.bOverPreActivations ? real_t(1.) : lyr.act_scaling_coeff();
				
				prepareToBatchSize(lSetts.batchSize);

				unsigned rt;
				for (rt = 0; rt < lSetts.maxReinitTries; ++rt) {
					const bool bSuccess = lSetts.bOnInvidualNeurons 
						? _processWeights_individual(lyr.get_weights(), *lyr.get_activations_storage(), lSetts, actScaling)
						: _processWeights_shared(lyr.get_weights(), *lyr.get_activations_storage(), lSetts, actScaling);

					if (bSuccess) {
						break;
					} else {
						if (rt + 1 < lSetts.maxReinitTries) {
							STDCOUT("Reinitializing weights of " << lyr.get_layer_name_str() << ", attempt #" << rt + 1);
							if (lyr.reinit_weights()) {
								STDCOUT(::std::endl);
							} else {
								STDCOUTL(" ***** failed!!! ****");
							}
						}
					}
				}
				if (rt >= lSetts.maxReinitTries) {
					if (!m_firstFailedLayerIdx) m_firstFailedLayerIdx = lIdx;
					STDCOUTL("******* layer " << lyr.get_layer_name_str() << " failed to converge! *********");
				}

				lyr.setIgnoreActivation(false);
			}

			//some fireworks
			static void die_die_die_my_darling(const char*const msg, const vec_len_t batchSize)noexcept {
				STDCOUTL("LSUVExt::die_die_die_my_darling: " << msg << ", batchSize=" << batchSize);
				NNTL_ASSERT(!"LSUVExt::die_die_die_my_darling!");
			#pragma warning(push)
			#pragma warning(disable:4297)//function assumed not to throw
				throw ::std::logic_error(msg);
			#pragma warning(pop)
			}

			vec_len_t nonZeroBatchSize(const vec_len_t _bs)const noexcept {
				NNTL_ASSERT(m_largestBatchSize && _bs >= 0);
				return _bs ? ::std::min(_bs, m_largestBatchSize) : m_largestBatchSize;
			}

			void prepareToBatchSize(const vec_len_t _bs)noexcept {
				auto bs = nonZeroBatchSize(_bs);
				const auto oldBs = m_td.get_maxFPropSize();
				if (bs > oldBs) {
					const auto ec = m_td.init4inference(bs);
					if (decltype(ec)::Success != ec) die_die_die_my_darling("m_td.init4inference failed", bs);
				}

				const auto ec = m_nn.init4fixedBatchFprop(bs);
				if (decltype(ec)::Success != ec) die_die_die_my_darling("m_nn.init4fixedBatchFprop failed", bs);
			}

			void _fprop(numel_cnt_t batchIdx)noexcept {
				m_td.next_subset(batchIdx, m_nn.get_const_common_data());
				m_nn.doFixedBatchFprop(m_td.batchX());
			}
			
			//////////////////////////////////////////////////////////////////////////

			struct _WeightsNormBase{
				const statsdata_t m_actScaling;
				realmtx_t& m_weights;
				const realmtx_t& m_Act;
				const LayerSetts_t& m_lSetts;

				self_t& m_thisHost;
				
				_WeightsNormBase(realmtx_t& w, const realmtx_t& A, const LayerSetts_t& lS, const statsdata_t asc, self_t& h) noexcept
					: m_actScaling(asc), m_weights(w), m_Act(A), m_lSetts(lS), m_thisHost(h) {}

				// Returns how many iterations (batch counts) must be done to satisfy preferred arguments
				numel_cnt_t prepareToWalk()noexcept {
					//const auto dataBatchCnt = m_thisHost.m_data.batchesCount();
					const auto dataBatchCnt = m_thisHost.m_td.walk_over_set(
						td_t::train_set_id
						, m_thisHost.m_nn.get_const_common_data(), m_thisHost.nonZeroBatchSize(m_lSetts.batchSize)
						, td_t::flag_exclude_dataY);

					NNTL_ASSERT(dataBatchCnt > 0);
					return ::std::min(dataBatchCnt, m_lSetts.batchCount ? numel_cnt_t(m_lSetts.batchCount) : dataBatchCnt);
				}
				//returns a pointer to matrix data. Matrix must have at least 1 element (and more than 1 over all batches)
				//walk() is not required to obey batchIdx, it's just a convenience argument. The only requirement is that
				// the whole matrix must be walked over with all batches.
				__declspec(restrict) const realmtx_t* __restrict walk(numel_cnt_t batchIdx)noexcept {
					m_thisHost._fprop(batchIdx);
					return &m_Act;
				}
			};

			//////////////////////////////////////////////////////////////////////////

			template<typename FinalT>
			class _WeightsNorm_whole 
				: public utils::mtx2Normal::_FNorm_whole_base<FinalT, real_t, StatsFunctor_t, statsdata_t>
				, protected _WeightsNormBase
			{
				typedef utils::mtx2Normal::_FNorm_whole_base<FinalT, real_t, StatsFunctor_t, statsdata_t> _base_t;
				typedef _WeightsNormBase _wnbase_t;

			public:
				using _wnbase_t::walk;
				using _wnbase_t::prepareToWalk;

				template<typename ...ArgsT>
				_WeightsNorm_whole(ArgsT&&... args)noexcept : _wnbase_t(::std::forward<ArgsT>(args)...){}
				
				void normalize_whole(statsdata_t scaleVal, const statsdata_t centralVal, const bool bScale, const bool bCentral)noexcept {
					if (bScale) {
						NNTL_ASSERT(scaleVal != statsdata_t(0.));
						scaleVal = m_lSetts.targetScale / scaleVal;
						m_thisHost.m_nn.get_iMath().evMulC_ip(m_weights, static_cast<real_t>(scaleVal));
					} else scaleVal = statsdata_t(1.0);

					if (bCentral) {
						const real_t centrOffs = static_cast<real_t>((m_lSetts.targetCentral - centralVal)*m_actScaling*scaleVal);
						auto pB = m_weights.colDataAsVec(m_weights.cols() - 1);
						const auto pBE = m_weights.end();
						while (pB < pBE) *pB++ += centrOffs;
					}
				}
			};

			class WeightsNorm_whole final : public _WeightsNorm_whole<WeightsNorm_whole> {
				typedef _WeightsNorm_whole<WeightsNorm_whole> _base_t;
			public:
				template<typename ...ArgsT>
				WeightsNorm_whole(ArgsT&&... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
			};

			bool _processWeights_shared(realmtx_t& weights, const realmtx_t& Act, const LayerSetts_t& lSetts, const real_t actScaling)noexcept {
				NNTL_ASSERT(Act.bBatchInColumn());//utils::mtx2Normal::normalize_whole requirement
				WeightsNorm_whole fn(weights, Act, lSetts, actScaling, *this);
				return utils::mtx2Normal::normalize_whole(lSetts, fn);
			}

			//////////////////////////////////////////////////////////////////////////
			template<typename FinalT>
			class _WeightsNorm_cw
				: public utils::mtx2Normal::_FNorm_cw_base<FinalT, real_t, StatsFunctor_t, statsdata_t>
				, protected _WeightsNormBase
			{
				typedef utils::mtx2Normal::_FNorm_cw_base<FinalT, real_t, StatsFunctor_t, statsdata_t> _base_t;
				typedef _WeightsNormBase _wnbase_t;

			public:
				using _wnbase_t::walk;
				using _wnbase_t::prepareToWalk;

				template<typename ...ArgsT>
				_WeightsNorm_cw(ArgsT&&... args)noexcept : _wnbase_t(::std::forward<ArgsT>(args)...) {}

				vec_len_t total_cols()const noexcept { return m_Act.cols_no_bias(); }

				//forward args to iThreads.run();
				template<typename ...ArgsT>
				void iThreads_run(ArgsT&&... args)noexcept {
					return m_thisHost.m_nn.get_iThreads().run(::std::forward<ArgsT>(args)...);
				}

				void normalize_cw(const vec_len_t colIdx, statsdata_t scaleVal, const statsdata_t centralVal
					, const bool bScale, const bool bCentral)noexcept
				{
					if (bScale) {
						NNTL_ASSERT(statsdata_t(0.) != scaleVal);
						scaleVal = m_lSetts.targetScale / scaleVal;

						//column==batch in activation matrix corresponds to a row in weight matrix.
						real_t*__restrict pW = m_weights.data() + colIdx;
						const auto pWE = m_weights.end();
						const ptrdiff_t ldw = m_weights.ldim();
						const auto mv = static_cast<real_t>(scaleVal);
						while (pW < pWE) {
							*pW *= mv;
							pW += ldw;
						}
					} else scaleVal = statsdata_t(1.);

					if (bCentral) {
						const real_t ofs = static_cast<real_t>((m_lSetts.targetCentral - centralVal)*m_actScaling*scaleVal);
						m_weights.get(colIdx, m_weights.cols() - 1) += ofs;
					}
				}
			};

			class WeightsNorm_cw final : public _WeightsNorm_cw<WeightsNorm_cw> {
				typedef _WeightsNorm_cw<WeightsNorm_cw> _base_t;
			public:
				template<typename ...ArgsT>
				WeightsNorm_cw(ArgsT&&... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
			};

			bool _processWeights_individual(realmtx_t& weights, const realmtx_t& Act, const LayerSetts_t& lSetts, const real_t actScaling)noexcept {
				NNTL_ASSERT(Act.bBatchInColumn());//utils::mtx2Normal::normalize_cw requirement
				WeightsNorm_cw fn(weights, Act, lSetts, actScaling, *this);
				return utils::mtx2Normal::normalize_cw(lSetts, fn);
			}

			template<typename AccT>
			static nntl_probably_force_inline void _calc(const real_t* __restrict pBegin, const real_t*const pEnd, AccT& acc)noexcept {
				while (pBegin < pEnd) {
					acc(*pBegin++);
				}
			}
		};

		template<typename LSUVExtT, template<class>typename ShouldApplyToLayer>
		struct LSUVExt_customInit {
			typedef typename LSUVExtT::LayerSetts_t LayerSetts_t;

			LSUVExtT& obj;
			LayerSetts_t setts; //set it externally
			::std::vector<layer_index_t> excludeIdList; //set it externally
			const char* pszSettingsName = nullptr;
			

			~LSUVExt_customInit()noexcept {}
			LSUVExt_customInit(LSUVExtT& o, const char* n = nullptr) noexcept : obj(o), pszSettingsName(n) {}
			
			template<typename LayerT>
			::std::enable_if_t<ShouldApplyToLayer<LayerT>::value> operator()(const LayerT& lyr)noexcept {
				const auto lyrIdx = lyr.get_layer_idx();
				if (::std::none_of(excludeIdList.cbegin(), excludeIdList.cend(), [lyrIdx](const auto e)noexcept {return e == lyrIdx; })) {
					if (pszSettingsName) {
						STDCOUTL("Will use custom '" << pszSettingsName << "' LSUVExt settings for layer '" << lyr.get_layer_name_str());
					}

					obj.setts().add(lyr.get_layer_idx(), setts);
				} else {
					if (pszSettingsName) {
						STDCOUTL("Custom '" << pszSettingsName << "' LSUVExt settings was explicitly forbidden for layer '" << lyr.get_layer_name_str());
					}
				}
			}

			template<typename LayerT>
			::std::enable_if_t<!ShouldApplyToLayer<LayerT>::value> operator()(const LayerT&)noexcept {}
		};
	}
}
}