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

#include <algorithm>
#include <functional>

#include "../utils/variant.h"
#include "../_nnet_errs.h"
#include "../utils/scope_exit.h"

#include "_td_base.h"

namespace nntl {

	template <typename XT, typename YT>
	class _i_td_transformer : public virtual DataSetsId {
	public:
		typedef XT x_t;
		typedef YT y_t;
		typedef inmem_train_data_stor<XT, XT> TD_stor_t;
		typedef const TD_stor_t const_TD_stor_t;

		typedef math::smatrix<XT> x_mtx_t;
		typedef math::smatrix<YT> y_mtx_t;
		typedef math::smatrix_deform<XT> x_mtxdef_t;
		typedef math::smatrix_deform<YT> y_mtxdef_t;

		//how many samples generate each base const_TD_stor_t sample?
		nntl_interface vec_len_t samplesInBaseSample()const noexcept;

		nntl_interface vec_len_t xWidth(const_TD_stor_t& tds)const noexcept;
		nntl_interface vec_len_t yWidth(const_TD_stor_t& tds)const noexcept;
		
		nntl_interface numel_cnt_t base_dataset_samples_count(const_TD_stor_t& tds, data_set_id_t dataSetId)const noexcept;

		//note that all batch sizes for _i_td_transformer<> are expressed in terms of const_TD_stor_t batch size == baseBatchSize
		//real batch size of data produced by transformer == baseBatchSize*samplesInBaseSample()

		nntl_interface void deinit()noexcept;

		template<typename iMathT>
		nntl_interface bool init(iMathT& iM, const_TD_stor_t& tds, vec_len_t baseBatchSize)noexcept;

		template<typename iMathT>
		static void preinit_iMath(iMathT& iM)noexcept {};

		template<typename CommonDataT>
		nntl_interface void next_epoch(const_TD_stor_t& tds, const numel_cnt_t epochIdx, const CommonDataT& cd
			, const vec_len_t baseBatchSize, x_mtx_t** ppBatchX, y_mtx_t** ppBatchY) noexcept;

		//real batch size (same as passed to next_epoch) here is in cd.get_cur_batch_size()
		template<typename CommonDataT>
		nntl_interface void next_batch(const_TD_stor_t& tds, const numel_cnt_t batchIdx, const CommonDataT& cd
			, const vec_len_t* pSampleIdxs, const vec_len_t baseBatchSize)noexcept;

		template<typename CommonDataT>
		nntl_interface void walk(const_TD_stor_t& tds, const data_set_id_t dataSetId, const CommonDataT& cd
			, const vec_len_t baseBatchSize, const unsigned excludeDataFlag, x_mtx_t** ppBatchX, y_mtx_t** ppBatchY)noexcept;

		//never use cd.get_cur_batch_size(), required batch size is given in baseBatchSize.
		// It's always less(! possible for the last batch) or equal to batchSize requested to walk()
		template<typename CommonDataT>
		nntl_interface void walk_next(const_TD_stor_t& tds, const data_set_id_t dataSetId, const numel_cnt_t batchIdx
			, const CommonDataT& cd, const vec_len_t baseBatchSize, const vec_len_t ofs, const unsigned excludeDataFlag)noexcept;
	};

	namespace _impl {

		// The class allows to make a run-time transformation of underlying train_data while requiring derived class to provide an
		// interface to train_data only (like the one exposed by inmem_train_data_stor)
		//
		// in addition to some not implemented in class _transf_train_data functions of _i_train_data interface,
		// _transf_train_data require a derived class to provide const realmtx_t& X(data_set_id_t dataSetId) and Y() functions
		// to return corresponding matrices with complete dataset data (that implies that the class can not work with datasets
		// with greater than max(vec_len_t) samples).
		// Note that batch sizes passed to _transf_train_data<> MUST be a multiple of TFunctT::samplesInBaseSample()
		// If not mentioned explicitly in a function comment, every member function of the class #supportsBatchInRow (at least it should)
		template<typename FinalPolymorphChild, typename TFuncT, typename StatsFuncT>
		class _transf_train_data : public _td_base <FinalPolymorphChild, typename TFuncT::x_t, typename TFuncT::y_t> {
			typedef _td_base<FinalPolymorphChild, typename TFuncT::x_t, typename TFuncT::y_t> _base_class_t;
		public:
			//resolving definitions clash
			using _base_class_t::x_t;
			using _base_class_t::y_t;
			using _base_class_t::x_mtx_t;
			using _base_class_t::y_mtx_t;
			using _base_class_t::x_mtxdef_t;
			using _base_class_t::y_mtxdef_t;
		
			static_assert(::std::is_base_of<_i_td_transformer<x_t, y_t>, TFuncT>::value, "TFuncT must implement _i_td_transformer");
			typedef TFuncT TransFunctor_t;
			typedef typename TransFunctor_t::const_TD_stor_t const_TD_stor_t;

			//overriding old definition, we may use only this functor for now for normalization
			typedef StatsFuncT default_StatsFuncT;
			typedef StatsFuncT StatsFunctor_t;

			//and fetching definitions necessary to setup run-time data normalization
			typedef typename StatsFunctor_t::mtx_update_t Norm_mtx_update_t;
			typedef typename Norm_mtx_update_t::template ScaleCentralData_tpl<x_t> ScaleCentralData_t;
			typedef typename Norm_mtx_update_t::template ScaleCentralVector_tpl<x_t> ScaleCentralVector_t;

			// if it is safe to cache training or testing set elsewhere, however the total size could be too big so disabling possible caching
			static constexpr bool allowExternalCachingOfSets = false;

		protected:
			//vector to hold randomized training set row numbers
			::std::vector<vec_len_t> m_vSampleIdxs;
			//::std::vector<vec_len_t>::iterator m_curSampleIdx;

			TransFunctor_t& m_TFunct;
			const_TD_stor_t& m_tdStor;

			//vec_len_t m_curWalkingBatchIdx{ 0 };

			unsigned m_excludeDataFlag;

			//monostate corresponds to 'no normalization', ScaleCentralData_t handles all the data necessary for `whole` type
			//normalization, while ScaleCentralVector_t is for batchwise/colwise normalization (though it's probably not supported by
			// normalization algo implementation yet)
			typedef NNTL_VARIANT_NS::variant<NNTL_VARIANT_NS::monostate, ScaleCentralData_t, ScaleCentralVector_t> NormPrms_t;

			//note that normalization doesn't reset with deinit(), an explicit call to reset_normalization() is required
			NormPrms_t m_NormalizationPrms;

		public:
			_transf_train_data(TransFunctor_t& tf, const_TD_stor_t& tds) noexcept : _base_class_t(), m_TFunct(tf), m_tdStor(tds)
				//, m_samplesInBaseSample(m_TFunct.samplesInBaseSample())
			{
				NNTL_ASSERT(m_TFunct.samplesInBaseSample() > 0);
			}

			TransFunctor_t& get_Functor()noexcept { return m_TFunct; }
			const TransFunctor_t& get_Functor()const noexcept { return m_TFunct; }
			const_TD_stor_t& get_TDStor()noexcept { return m_tdStor; }
			const const_TD_stor_t& get_TDStor()const noexcept { return m_tdStor; }

			const x_t* getOriginalSamplePtr(data_set_id_t dataSetId, const numel_cnt_t batchIdx
				, const vec_len_t SampleIdx, vec_len_t& len)const noexcept
			{
				NNTL_ASSERT(m_curBatchSize > 0);
				const auto ofs = batchIdx*m_curBatchSize + SampleIdx;
				if (m_curBatchSize <= 0 || batchIdx < 0 || SampleIdx < 0 || ofs<0
					|| ofs >= get_self().base_dataset_samples_count(dataSetId))
					return nullptr;
				return m_TFunct.getOriginalSamplePtr(m_tdStor, dataSetId, ofs, len);
			}

			//////////////////////////////////////////////////////////////////////////
			bool empty()const noexcept { return m_tdStor.empty(); }
			
			numel_cnt_t dataset_samples_count(data_set_id_t dataSetId)const noexcept {
				NNTL_ASSERT(!get_self().empty());
				return static_cast<numel_cnt_t>(m_TFunct.samplesInBaseSample())*get_self().base_dataset_samples_count(dataSetId);
			}

			numel_cnt_t base_dataset_samples_count(data_set_id_t dataSetId)const noexcept {
				NNTL_ASSERT(!get_self().empty());
				//return m_tdStor.X(dataSetId).batch_size();
				return m_TFunct.base_dataset_samples_count(m_tdStor, dataSetId);
			}
			numel_cnt_t base_trainset_samples_count()const noexcept { return get_self().base_dataset_samples_count(train_set_id); }
			numel_cnt_t base_testset_samples_count()const noexcept { return get_self().base_dataset_samples_count(test_set_id); }

			numel_cnt_t base_biggest_samples_count()const noexcept {
				const auto tds = get_self().datasets_count();
				numel_cnt_t r = 0;
				for (data_set_id_t i = 0; i < tds; ++i) {
					r = ::std::max(r, get_self().base_dataset_samples_count(i));
				}
				return r;
			}

			vec_len_t baseBatchSize2realBatchSize(const vec_len_t bs)const noexcept {
				return bs* m_TFunct.samplesInBaseSample();
			}

			template<typename iMathT>
			void preinit_iMath(iMathT& iM)noexcept { m_TFunct.preinit_iMath(iM); };
			
			//////////////////////////////////////////////////////////////////////////

			vec_len_t xWidth()const noexcept { NNTL_ASSERT(!get_self().empty()); return m_TFunct.xWidth(m_tdStor); }
			vec_len_t yWidth()const noexcept { NNTL_ASSERT(!get_self().empty()); return m_TFunct.yWidth(m_tdStor); }

			void deinit4all()noexcept {
				m_TFunct.deinit();
				_base_class_t::deinit4all();
				//m_curWalkingBatchIdx = 0;
				m_vSampleIdxs.clear();
				m_vSampleIdxs.shrink_to_fit();
				//m_curSampleIdx = m_vSampleIdxs.end();
			}

		protected:
			nnet_errors_t _base_init_checks(const vec_len_t maxFPropSiz, const vec_len_t maxTrainSiz = 0)noexcept {
				if (get_self().empty() || !(m_tdStor.samplesYStorageCoherent() && m_tdStor.samplesXStorageCoherent())) {
					STDCOUTL("Invalid m_tdStor state!");
					NNTL_ASSERT(!"Invalid m_tdStor state!");
					return nnet_errors_t::OtherTdInitError;
				}
				const auto sibs = m_TFunct.samplesInBaseSample();
				if (maxFPropSiz < 0 || (0 != maxFPropSiz%sibs)) {
					STDCOUTL("_base_init_checks: Invalid maxFPropSize!");
					NNTL_ASSERT(!"Invalid maxFPropSize!");
					return nnet_errors_t::OtherTdInitError;
				}
				if (maxTrainSiz < 0 || (maxTrainSiz > 0 && (0 != maxTrainSiz%sibs))) {
					STDCOUTL("_base_init_checks: Invalid maxBatchSize!");
					NNTL_ASSERT(!"Invalid maxBatchSize!");
					return nnet_errors_t::OtherTdInitError;
				}

				//always start from a known state
				get_self().deinit4all();
				return nnet_errors_t::Success;
			}

		public:
			bool is_initialized4inference(vec_len_t& bs)const noexcept {
				const auto tbs = bs ? bs : m_maxFPropSize;
				if (get_self().empty() || 0 == m_maxFPropSize || tbs > m_maxFPropSize || 0 != tbs%m_TFunct.samplesInBaseSample()) return false;
				bs = tbs;
				return true;
			}

			// Note that batch sizes passed to _transf_train_data<> MUST be a multiple of TFunctT::samplesInBaseSample()
			template<typename iMathT>
			nnet_errors_t init4inference(iMathT& iM, IN OUT vec_len_t& maxFPropSize)noexcept {
				const auto ec = _base_init_checks(maxFPropSize);
				if (nnet_errors_t::Success != ec) return ec;

				const auto sibs = m_TFunct.samplesInBaseSample();
				auto base_maxFPropSize = maxFPropSize / sibs;
				const auto base_biggestSetSize = get_self().base_biggest_samples_count();
				if (base_maxFPropSize <= 0 || base_maxFPropSize > base_biggestSetSize) {
					base_maxFPropSize = static_cast<vec_len_t>(base_biggestSetSize);
				}
				const numel_cnt_t realFPropSize = base_maxFPropSize*static_cast<numel_cnt_t>(sibs);
				
				if (realFPropSize > ::std::numeric_limits<vec_len_t>::max())//batch must fit into a matrix
					return nnet_errors_t::TooBigTrainTestSet;

				maxFPropSize = static_cast<vec_len_t>(realFPropSize);
				m_maxFPropSize = maxFPropSize;

				if (!m_TFunct.init(iM, m_tdStor, base_maxFPropSize)) return nnet_errors_t::OtherTdInitError;

				return nnet_errors_t::Success;
			}

			bool is_initialized4train(vec_len_t& fpropBs, vec_len_t& trainBs, bool& bMiniBatch)const noexcept {
				if (!get_self().is_initialized4inference(fpropBs)) return false;
				const auto tbs = trainBs ? trainBs : m_maxTrainBatchSize;
				//fullbatch mode must support only exactly specified batch sizes
				const auto dr = ::std::div(tbs, m_TFunct.samplesInBaseSample());
				if (0 == m_maxTrainBatchSize || tbs > m_maxTrainBatchSize 
					|| 0 != dr.rem || tbs > fpropBs) return false;
				trainBs = tbs;
				bMiniBatch = dr.quot < get_self().base_trainset_samples_count();
				return true;
			}

			//takes desired batch sizes, updates them if necessary to any suitable value and initializes internal state for training
			// Note that batch sizes passed to _transf_train_data<> MUST be a multiple of TFunctT::samplesInBaseSample()
			template<typename iMathT>
			nnet_errors_t init4train(iMathT& iM, IN OUT vec_len_t& maxFPropSize, IN OUT vec_len_t& maxBatchSize, OUT bool& bMiniBatch) noexcept {
				const auto ec = _base_init_checks(maxFPropSize, maxBatchSize);
				if (nnet_errors_t::Success != ec) return ec;

				bool bSuccess = false;
				utils::scope_exit deinit_if_error([this, &bSuccess]()noexcept {
					if (!bSuccess) get_self().deinit4all();
				});

				const auto sibs = m_TFunct.samplesInBaseSample();
				auto base_maxFPropSize = maxFPropSize / sibs;
				auto base_maxBatchSize = maxBatchSize / sibs;

				const auto base_trainCnt = get_self().base_trainset_samples_count();
				const auto base_biggestSetSize = get_self().base_biggest_samples_count();
				
				if (base_maxFPropSize <= 0 || base_maxFPropSize > base_biggestSetSize)
					base_maxFPropSize = static_cast<vec_len_t>(base_biggestSetSize);

				if (base_maxBatchSize <= 0 || base_maxBatchSize > base_trainCnt)
					base_maxBatchSize = static_cast<vec_len_t>(base_trainCnt);
				
				if (base_maxBatchSize > base_maxFPropSize) return nnet_errors_t::InvalidBatchSize2MaxFPropSizeRelation;

				//m_bMiniBatchTraining = 
				bMiniBatch = (base_maxBatchSize < base_trainCnt);

				const numel_cnt_t realmaxFPropSize = base_maxFPropSize*static_cast<numel_cnt_t>(sibs);
				const numel_cnt_t realmaxBatchSize = base_maxBatchSize*static_cast<numel_cnt_t>(sibs);

				//batch must fit in a matrix
				if (realmaxFPropSize > ::std::numeric_limits<vec_len_t>::max())
					return nnet_errors_t::TooBigTrainTestSet;
				if (realmaxBatchSize > ::std::numeric_limits<vec_len_t>::max())
					return nnet_errors_t::TooBigTrainSet;
				
				maxFPropSize = static_cast<vec_len_t>(realmaxFPropSize);
				m_maxFPropSize = maxFPropSize;
				maxBatchSize = static_cast<vec_len_t>(realmaxBatchSize);
				m_maxTrainBatchSize = maxBatchSize;

				//if (m_bMiniBatchTraining) {
					try {
						m_vSampleIdxs.resize(base_trainCnt);
					} catch (const ::std::exception&) {
						return nnet_errors_t::TdInitNoMemory;
					}
					//m_curSampleIdx = m_vSampleIdxs.end();
					::std::iota(m_vSampleIdxs.begin(), m_vSampleIdxs.end(), 0);
				//}

				if (!m_TFunct.init(iM, m_tdStor, ::std::max(base_maxFPropSize, base_maxBatchSize))) return nnet_errors_t::OtherTdInitError;

				bSuccess = true;
				return nnet_errors_t::Success;
			}

// 			vec_len_t get_orig_maxFPropSize()const noexcept { return m_maxFPropSize / m_TFunct.samplesInBaseSample(); }
// 			vec_len_t get_orig_maxTrainBatchSize()const noexcept { return m_maxTrainBatchSize / m_TFunct.samplesInBaseSample(); }

			//notifies object about new training epoch start and must return total count of batches to execute over the training set
			//batchSize MUST always be less or equal to maxBatchSize, returned by init4train()
			template<typename CommonDataT>
			numel_cnt_t on_next_epoch(const numel_cnt_t epochIdx, const CommonDataT& cd, vec_len_t batchSize = 0) noexcept {
				NNTL_ASSERT(epochIdx >= 0);
				NNTL_ASSERT(m_tdStor.samplesYStorageCoherent() && m_tdStor.samplesXStorageCoherent());

				m_curDataset2Walk = invalid_set_id;

				if (batchSize < 0) {
					batchSize = m_maxTrainBatchSize;
				} else if (0 == batchSize) {
					batchSize = cd.input_batch_size();
				}
				NNTL_ASSERT(batchSize > 0 && batchSize <= m_maxTrainBatchSize && batchSize <= get_self().trainset_samples_count());
				NNTL_DEBUG_DECLARE(auto _fbs = batchSize; auto _tbs = batchSize; bool bB;);
				NNTL_ASSERT(get_self().is_initialized4train(_fbs, _tbs, bB) && _fbs == batchSize && _tbs == batchSize);

				m_curBatchSize = batchSize / m_TFunct.samplesInBaseSample();
				//m_curSampleIdx = m_vSampleIdxs.begin();
				//making random permutations to define which data rows will be used as batch data
				::std::random_shuffle(m_vSampleIdxs.begin(), m_vSampleIdxs.end(), cd.iRng());

				m_TFunct.next_epoch(m_tdStor, epochIdx, cd, m_curBatchSize, &m_pCurBatchX, &m_pCurBatchY);

				const auto numBatches = get_self().trainset_samples_count() / batchSize;
				NNTL_ASSERT(numBatches > 0);
				return numBatches;
			}

			template<typename CommonDataT>
			void on_next_batch(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept {
				NNTL_ASSERT(batchIdx >= 0);
				NNTL_ASSERT(m_curDataset2Walk == invalid_set_id);

// 				NNTL_ASSERT(m_vSampleIdxs.size()*m_TFunct.samplesInBaseSample()
// 					>= static_cast<size_t>(m_curSampleIdx - m_vSampleIdxs.begin())*m_TFunct.samplesInBaseSample() + cd.get_cur_batch_size());
// 				NNTL_ASSERT(cd.get_cur_batch_size() == m_curBatchSize*m_TFunct.samplesInBaseSample());

				//using m_curBatchSize only. Don't use cd.get_cur_batch_size()!
				NNTL_ASSERT(batchIdx*m_curBatchSize <= ::std::numeric_limits<vec_len_t>::max());
				const auto batchOffs = static_cast<vec_len_t>(batchIdx*m_curBatchSize);
				NNTL_ASSERT(m_vSampleIdxs.size() >= batchOffs + m_curBatchSize);

				m_TFunct.next_batch(m_tdStor, batchIdx, cd, &(m_vSampleIdxs[batchOffs]), m_curBatchSize);
				//m_curSampleIdx += m_curBatchSize;

				if (0 != m_NormalizationPrms.index())
					_apply_normalization2batchX(cd.get_iMath());
			}
			//for testing only
			auto _get_curSampleIdxIt(const numel_cnt_t batchIdx)const noexcept {
				NNTL_ASSERT(batchIdx*m_curBatchSize <= ::std::numeric_limits<vec_len_t>::max());
				const auto batchOffs = static_cast<vec_len_t>(batchIdx*m_curBatchSize);
				NNTL_ASSERT(m_vSampleIdxs.size() >= batchOffs + m_curBatchSize);
				return m_vSampleIdxs.begin() + batchOffs;
			}

			//////////////////////////////////////////////////////////////////////////
			template<typename CommonDataT>
			numel_cnt_t walk_over_set(const data_set_id_t dataSetId, const CommonDataT& cd
				, vec_len_t batchSize = -1, const unsigned excludeDataFlag = flag_exclude_nothing)noexcept
			{
				NNTL_ASSERT(dataSetId >= 0 && dataSetId < get_self().datasets_count());
				NNTL_ASSERT(m_tdStor.samplesYStorageCoherent() && m_tdStor.samplesXStorageCoherent());

				m_curDataset2Walk = dataSetId;
				//m_curWalkingBatchIdx = 0;

				if (batchSize < 0) {
					batchSize = m_maxFPropSize;
				} else if (0 == batchSize) {
					batchSize = cd.input_batch_size();
				}
				NNTL_ASSERT(batchSize > 0 && batchSize <= m_maxFPropSize);
				NNTL_DEBUG_DECLARE(auto _bs = batchSize);
				NNTL_ASSERT(get_self().is_initialized4inference(_bs) && _bs == batchSize);

				const auto dsNumel = get_self().dataset_samples_count(dataSetId);
				NNTL_ASSERT(dsNumel > 0);
				if (batchSize > dsNumel)
					batchSize = static_cast<vec_len_t>(dsNumel);

				m_curBatchSize = batchSize / m_TFunct.samplesInBaseSample();
				m_excludeDataFlag = excludeDataFlag;
				m_TFunct.walk(m_tdStor, dataSetId, cd, m_curBatchSize, excludeDataFlag, &m_pCurBatchX, &m_pCurBatchY);

				const auto _dr = ::std::div(dsNumel, static_cast<numel_cnt_t>(batchSize));
				const auto numBatches = _dr.quot + (_dr.rem > 1);
				NNTL_ASSERT(numBatches > 0);
				return numBatches;
			}

			template<typename CommonDataT>
			void next_subset(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept {
				NNTL_ASSERT(batchIdx >= 0);
				NNTL_ASSERT(m_curDataset2Walk != invalid_set_id);

				//we MUST NOT use cd.get_cur_batch_size() here because it is not required to be set to proper value for just
				//walking over any dataset (what if you are just inspecting its content and not going to do inferencing at all?)

				const auto base_dsSize = m_tdStor.X(m_curDataset2Walk).batch_size();//OK to query directly, it's class design
				// _transf_train_data currently works only with matrix-sized datasets
				NNTL_ASSERT(base_dsSize == m_tdStor.Y(m_curDataset2Walk).batch_size());


// 				if (m_curWalkingBatchIdx >= base_dsSize) m_curWalkingBatchIdx = 0;
// 				const auto base_maxBs = base_dsSize - m_curWalkingBatchIdx;

				auto base_batchSize = m_curBatchSize;
				NNTL_ASSERT(batchIdx*base_batchSize <= ::std::numeric_limits<vec_len_t>::max());
				vec_len_t walkingBatchIdx = static_cast<vec_len_t>(batchIdx*base_batchSize);
				const auto base_maxBs = base_dsSize - walkingBatchIdx;
				NNTL_ASSERT(base_maxBs > 0);
				if (base_maxBs <= 0) {
					STDCOUTL("WTF? Invalid batchIdx passed?");
					::std::abort();
				}
				//checking batch size
				if (base_maxBs < m_curBatchSize) base_batchSize = base_maxBs;

				m_TFunct.walk_next(m_tdStor, m_curDataset2Walk, batchIdx, cd, base_batchSize, walkingBatchIdx, m_excludeDataFlag);
				//m_curWalkingBatchIdx += base_batchSize;

				if (!exclude_dataX(m_excludeDataFlag) && 0 != m_NormalizationPrms.index())
					_apply_normalization2batchX(cd.get_iMath());
			}

		protected:

			template<typename iMathT>
			struct _norm_visitor {
				x_mtx_t* pMtx;
				iMathT* piM;

				_norm_visitor(x_mtx_t& m, iMathT& imath) noexcept : pMtx(&m), piM(&imath) {}

				void operator()(const NNTL_VARIANT_NS::monostate &)const noexcept {}
				
				void operator()(const ScaleCentralData_t& st)const noexcept {
					Norm_mtx_update_t::whole(*piM, *pMtx, st);
				}

				void operator()(const ScaleCentralVector_t& allSt)const noexcept {
					Norm_mtx_update_t::batchwise(*piM, *pMtx, allSt);
				}
			};

			template<typename iMathT>
			void _apply_normalization2batchX(iMathT& iM) noexcept {
				NNTL_ASSERT(0 != m_NormalizationPrms.index());
				NNTL_ASSERT(m_pCurBatchX);
				NNTL_VARIANT_NS::visit(_norm_visitor<iMathT>(*m_pCurBatchX, iM), m_NormalizationPrms);
			}

		public:
			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			//
			// Normalization support functions.			
			void reset_normalization()noexcept {
				m_NormalizationPrms = NNTL_VARIANT_NS::monostate{};
			}
			// redefining normalize_data<>() without StatsFuncT parameter. We now have the only global one.
			// Note that base normalization algo works only in bBatchInColumn() mode, so TransFunctor_t must return
			// batch matrices in that mode.
			template<typename StatsT, typename CommonDataT, typename NormF = self_t::template NormalizerF_tpl<StatsT, StatsFunctor_t>>
			bool normalize_data(const CommonDataT& cd, const typename NormF::NormalizationSettings_t& Setts, const NormF& n = NormF())noexcept {
				static_assert(::std::is_same<StatsFunctor_t, typename NormF::StatsFunctor_t>::value, "NormF class must have the same StatsFunctor_t");
				get_self().reset_normalization();
				return _base_class_t::normalize_data<StatsT>(cd, Setts, n);
			}

			// after _fix_trainX_whole()/*_cw() any subset of train X returned by the td must be scaled with given values.
			// If it was called more than 1 time, effect must be cumulative
			template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			void _fix_trainX_whole(const CommonDataT& /*cd*/, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st)noexcept {
				static_assert(::std::is_same<MtxUpdT, Norm_mtx_update_t>::value, "Unexpected mtx update functor passed");
				NNTL_ASSERT(st.bScale || st.bOffset);
				
				const auto pOldVal = NNTL_VARIANT_NS::get_if<ScaleCentralData_t>(&m_NormalizationPrms);
				if (pOldVal) {
					pOldVal->_upd_all_typed(st);
				} else {
					m_NormalizationPrms = st.as_type<x_t>();
				}
			}

			// after _fix_trainX_whole()/*_cw() any subset of train X returned by the td must be scaled
			template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			void _fix_trainX_cw(const CommonDataT& /*cd*/, const typename MtxUpdT::template ScaleCentralVector_tpl<StatsT>& allSt)noexcept {
				static_assert(::std::is_same<MtxUpdT, Norm_mtx_update_t>::value, "Unexpected mtx update functor passed");
				NNTL_ASSERT(allSt.size() == xWidth());
				
				const auto pOldVec = NNTL_VARIANT_NS::get_if<ScaleCentralVector_t>(&m_NormalizationPrms);
				if (pOldVec) {
					NNTL_ASSERT(pOldVec->size() == allSt.size());
					const auto vs = conform_sign(allSt.size());
					for (numel_cnt_t i = 0; i < vs; ++i) {
						const auto& newSCD = allSt[i];
						auto& oldSCD = (*pOldVec)[i];
						oldSCD._upd_all_typed(newSCD);
					}
				} else {
					m_NormalizationPrms = _convert_SCVec(allSt);
				}
			}

			//This function gives us the final values
			template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			void _do_fix_dataX_whole(const CommonDataT& /*cd*/, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st)noexcept{
				m_NormalizationPrms = st.as_type<x_t>();
			}

			template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			void _do_fix_dataX_cw(const CommonDataT& /*cd*/, const typename MtxUpdT::template ScaleCentralVector_tpl<StatsT>& allSt)noexcept {
				NNTL_ASSERT(allSt.size() == xWidth());
				m_NormalizationPrms = _convert_SCVec(allSt);
			}

		protected:
			template<typename StatsT, bool c = ::std::is_same<x_t, StatsT>::value>
			static ::std::enable_if_t<c, const ScaleCentralVector_t&> _convert_SCVec(const typename Norm_mtx_update_t::template ScaleCentralVector_tpl<StatsT>& allSt)noexcept {
				return allSt;
			}

			template<typename StatsT, bool c = ::std::is_same<x_t, StatsT>::value>
			static ::std::enable_if_t<!c, ScaleCentralVector_t> _convert_SCVec(const typename Norm_mtx_update_t::template ScaleCentralVector_tpl<StatsT>& allSt)noexcept {
				ScaleCentralVector_t vec;
				const auto vs = conform_sign(allSt.size());
				vec.reserve(vs);
				for (numel_cnt_t i = 0; i < vs; ++i) {
					const auto& v = allSt[i];
					vec.emplace_back(static_cast<x_t>(v.sc), static_cast<x_t>(v.ofs), v.bScale, v.bOffset);
				}
				return vec;
			}
		};
	}


	// If not mentioned explicitly in a function comment, any member function of the class #supportsBatchInRow (at least it should)
	template<typename TFuncT, typename StatsFuncT = utils::mtx2Normal::_FNorm_stats_var_mean<true>>
	class transf_train_data final : public _impl::_transf_train_data<transf_train_data<TFuncT, StatsFuncT>, TFuncT, StatsFuncT> {
		typedef _impl::_transf_train_data<transf_train_data<TFuncT, StatsFuncT>, TFuncT, StatsFuncT> _base_class_t;
	public:
		template<typename ... ArgsT>
		transf_train_data(ArgsT&&... args)noexcept : _base_class_t(::std::forward<ArgsT>(args)...) {}
	};

}
