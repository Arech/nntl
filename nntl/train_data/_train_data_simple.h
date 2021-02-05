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

#include "../_nnet_errs.h"
#include "../utils/scope_exit.h"

#include "_td_base.h"

namespace nntl {

	namespace _impl {

		// in addition to some not implemented in class _train_data_simple functions of _i_train_data interface,
		// _train_data_simple require a derived class to provide const realmtx_t& X(data_set_id_t dataSetId) and Y() functions
		// to return corresponding matrices with complete dataset data (that implies that the class can not work with datasets
		// with greater than max(vec_len_t) samples).
		// Also bool samplesXStorageCoherent()/samplesYStorageCoherent() should be implemented
		// If not mentioned explicitly in a function comment, every member function of the class #supportsBatchInRow (at least it should)
		template<typename FinalPolymorphChild, typename XT, typename YT>
		class _train_data_simple : public _td_base<FinalPolymorphChild, XT, YT> {
			typedef _td_base<FinalPolymorphChild, XT, YT> _base_class_t;

		protected:
			x_mtxdef_t m_batch_x, m_walkX;
			y_mtxdef_t m_batch_y, m_walkY;
			// we will use corresponding m_walk* for inference instead of m_batch* if bBatchInRow was set

			//vector to hold randomized training set row numbers
			::std::vector<vec_len_t> m_vSampleIdxs;
			::std::vector<vec_len_t>::iterator m_curSampleIdx;

			vec_len_t m_curWalkingBatchIdx{ 0 };

			bool m_bMiniBatchTraining{ false };
			
		protected:
			bool _initBatches(const bool bMiniBatchInferencing)noexcept {
				NNTL_ASSERT(m_batch_x.empty() && m_batch_y.empty() && m_walkX.empty() && m_walkY.empty());
				NNTL_ASSERT(get_self().samplesYStorageCoherent() && get_self().samplesXStorageCoherent());

				//initializing additional data members
				if (m_bMiniBatchTraining || bMiniBatchInferencing) {
					const auto xW = get_self().xWidth(), yW = get_self().yWidth();
					const auto mbs = ::std::max(m_maxTrainBatchSize, m_maxFPropSize);
					NNTL_ASSERT(mbs > 0 && xW > 0 && yW > 0);

					const bool bUseWalkX = get_self().X(train_set_id).bBatchInRow() && bMiniBatchInferencing;
					if (m_bMiniBatchTraining || !bUseWalkX) {
						m_batch_x.will_emulate_biases();
						m_batch_x.set_batchInRow(get_self().X(train_set_id).bBatchInRow());

						if (!m_batch_x.resize_as_dataset(bUseWalkX ? m_maxTrainBatchSize : mbs, xW))
							return false;
					}
					
					const bool bUseWalkY = get_self().Y(train_set_id).bBatchInRow() && bMiniBatchInferencing;
					if (m_bMiniBatchTraining || !bUseWalkY) {
						m_batch_y.dont_emulate_biases();
						m_batch_y.set_batchInRow(get_self().Y(train_set_id).bBatchInRow());

						if (!m_batch_y.resize_as_dataset(bUseWalkY ? m_maxTrainBatchSize : mbs, yW)) {
							m_batch_x.clear();
							return false;
						}
					}
				}
				return true;
			}

		public:
			void deinit4all()noexcept {
				_base_class_t::deinit4all();
				m_curWalkingBatchIdx = 0;
				m_bMiniBatchTraining = false;
				m_batch_x.clear();
				m_batch_y.clear();
				m_walkX.clear();
				m_walkY.clear();
				m_vSampleIdxs.clear();
				m_vSampleIdxs.shrink_to_fit();
				m_curSampleIdx = m_vSampleIdxs.end();
			}

			numel_cnt_t biggest_samples_count()const noexcept {
				const auto bsc = _base_class_t::biggest_samples_count();
				if (bsc > ::std::numeric_limits<vec_len_t>::max()) {
					NNTL_ASSERT(!"_train_data_simple can't work with datasets greater than ::std::numeric_limits<vec_len_t>::max() elements!");
					//the reason is that it works with data matrices directly and addresses rows/cols with vec_len_t
				#pragma warning(push)
				#pragma warning(disable:4297)//function assumed not to throw
					throw ::std::logic_error("WTF?! Too long dataset for _train_data_simple class!");
				#pragma warning(pop)
				}
				return bsc;
			}

			template<typename iMathT>
			nnet_errors_t init4inference(iMathT& iM, IN OUT vec_len_t& maxFPropSize)noexcept {
				NNTL_UNREF(iM);
				NNTL_ASSERT(maxFPropSize >= 0);
				if (get_self().is_initialized4inference() || get_self().is_initialized4train()) {
					NNTL_ASSERT(!"already initialized!");
					return nnet_errors_t::TDDeinitializationRequired;
				}
				get_self().deinit4all();//it's always better to start from the known state

				const auto biggestSetSize = get_self().biggest_samples_count();

				//zero values are special case. Checking for numeric type overflow
				if (maxFPropSize == 0 && biggestSetSize > ::std::numeric_limits<vec_len_t>::max())
					return nnet_errors_t::TooBigTrainTestSet;
				if (maxFPropSize <= 0 || maxFPropSize > biggestSetSize) {
					maxFPropSize = static_cast<vec_len_t>(biggestSetSize);
				}
				m_maxFPropSize = maxFPropSize;

				if (!get_self()._initBatches(maxFPropSize < biggestSetSize)) return nnet_errors_t::TdInitNoMemory;

				return nnet_errors_t::Success;
			}

			//takes desired batch sizes, updates them if necessary to any suitable value and initializes internal state for training
			//implementation for simplest TD that completely fits into the memory
			template<typename iMathT>
			nnet_errors_t init4train(iMathT& iM, IN OUT vec_len_t& maxFPropSize, IN OUT vec_len_t& maxBatchSize, OUT bool& bMiniBatch) noexcept {
				NNTL_UNREF(iM);
				NNTL_ASSERT(maxBatchSize >= 0 && maxFPropSize >= 0);

				if (get_self().is_initialized4inference() || get_self().is_initialized4train()) {
					NNTL_ASSERT(!"already initialized!");
					return nnet_errors_t::TDDeinitializationRequired;
				}
				get_self().deinit4all();//it's always better to start from the known state

				bool bSuccess = false;
				utils::scope_exit deinit_if_error([this, &bSuccess]()noexcept {
					if (!bSuccess) get_self().deinit4all();
				});

				const auto trainCnt = get_self().trainset_samples_count();
				const auto biggestSetSize = get_self().biggest_samples_count();

				//zero values are special case. Checking for numeric type overflow
				if (maxFPropSize == 0 && biggestSetSize > ::std::numeric_limits<vec_len_t>::max())
					return nnet_errors_t::TooBigTrainTestSet;
				if (maxBatchSize == 0 && trainCnt > ::std::numeric_limits<vec_len_t>::max())
					return nnet_errors_t::TooBigTrainSet;

				if (maxFPropSize <= 0 || maxFPropSize > biggestSetSize) {
					maxFPropSize = static_cast<vec_len_t>(biggestSetSize);
				}

				if (maxBatchSize <= 0 || maxBatchSize > trainCnt) {
					maxBatchSize = static_cast<vec_len_t>(trainCnt);
				}

				if (maxBatchSize > maxFPropSize) return nnet_errors_t::InvalidBatchSize2MaxFPropSizeRelation;

				m_maxFPropSize = maxFPropSize;
				m_maxTrainBatchSize = maxBatchSize;

				m_bMiniBatchTraining = bMiniBatch = (maxBatchSize < trainCnt);

				if (m_bMiniBatchTraining) {
					try {
						m_vSampleIdxs.resize(trainCnt);
					} catch (const ::std::exception&) {
						return nnet_errors_t::TdInitNoMemory;
					}
					m_curSampleIdx = m_vSampleIdxs.end();
					::std::iota(m_vSampleIdxs.begin(), m_vSampleIdxs.end(), 0);
				}

				if (!get_self()._initBatches(maxFPropSize < biggestSetSize)) return nnet_errors_t::TdInitNoMemory;

				bSuccess = true;
				return nnet_errors_t::Success;
			}

			//notifies object about new training epoch start and must return total count of batches to execute over the training set
			//batchSize MUST always be less or equal to maxBatchSize, returned by init4train()
			template<typename CommonDataT>
			numel_cnt_t on_next_epoch(const numel_cnt_t epochIdx, const CommonDataT& cd, vec_len_t batchSize = 0) noexcept {
				NNTL_UNREF(epochIdx);
				NNTL_ASSERT(epochIdx >= 0);
				NNTL_ASSERT(get_self().is_initialized4train());
				
				NNTL_ASSERT(get_self().samplesYStorageCoherent() && get_self().samplesXStorageCoherent());

				m_curDataset2Walk = invalid_set_id;

				if (batchSize < 0) {
					batchSize = m_maxTrainBatchSize;
				} else if (0 == batchSize) {
					batchSize = cd.input_batch_size();
				}
				NNTL_ASSERT(batchSize > 0 && batchSize <= m_maxTrainBatchSize && batchSize <= get_self().trainset_samples_count());
				m_curBatchSize = batchSize;

				if (m_bMiniBatchTraining) {
					//fixing the size of m_batch* matrices
					NNTL_ASSERT(!m_batch_x.empty() && !m_batch_y.empty());
					NNTL_ASSERT(m_batch_x.emulatesBiases() && !m_batch_y.emulatesBiases());
					NNTL_ASSERT(m_batch_x.bBatchInRow() == get_self().X(train_set_id).bBatchInRow());
					NNTL_ASSERT(m_batch_y.bBatchInRow() == get_self().Y(train_set_id).bBatchInRow());

					m_batch_x.deform_batch_size_with_biases(batchSize);
					m_pCurBatchX = &m_batch_x;

					m_batch_y.deform_batch_size(batchSize);
					m_pCurBatchY = &m_batch_y;

					m_curSampleIdx = m_vSampleIdxs.begin();
					//making random permutations to define which data rows will be used as batch data
					::std::random_shuffle(m_curSampleIdx, m_vSampleIdxs.end(), cd.iRng());
				} else {
					m_pCurBatchX = &get_self().X_mutable(train_set_id);
					NNTL_ASSERT(batchSize == m_pCurBatchX->batch_size());
					m_pCurBatchY = &get_self().Y_mutable(train_set_id);
					NNTL_ASSERT(batchSize == m_pCurBatchY->batch_size());
				}

				const auto numBatches = get_self().trainset_samples_count() / batchSize;
				NNTL_ASSERT(numBatches > 0);
				return numBatches;
			}

			template<typename CommonDataT>
			void on_next_batch(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept {
				NNTL_UNREF(batchIdx);
				NNTL_ASSERT(batchIdx >= 0);
				NNTL_ASSERT(get_self().is_initialized4train());
				NNTL_ASSERT(m_curDataset2Walk == invalid_set_id);
				NNTL_ASSERT(m_curBatchSize > 0);

				if (m_bMiniBatchTraining) {
					auto& iM = cd.iMath();

					NNTL_ASSERT(m_batch_x.batch_size() == m_curBatchSize && m_batch_y.batch_size() == m_curBatchSize);
					NNTL_ASSERT(m_pCurBatchX == &m_batch_x && m_pCurBatchY == &m_batch_y);

					NNTL_ASSERT(m_curSampleIdx + m_curBatchSize <= m_vSampleIdxs.end());

					//X should be processed the last to leave it in cache
					iM.mExtractBatches(get_self().Y(train_set_id), m_curSampleIdx, m_batch_y);
					iM.mExtractBatches(get_self().X(train_set_id), m_curSampleIdx, m_batch_x);					
					m_curSampleIdx += m_curBatchSize;
				} else {
					NNTL_ASSERT(0 == batchIdx);//only one call is expected
				}
			}
			//////////////////////////////////////////////////////////////////////////
			template<typename CommonDataT>
			numel_cnt_t walk_over_set(const data_set_id_t dataSetId, const CommonDataT& cd
				, vec_len_t batchSize = -1, const unsigned excludeDataFlag = flag_exclude_nothing)noexcept
			{
				NNTL_ASSERT(dataSetId >= 0 && dataSetId < get_self().datasets_count());
				NNTL_ASSERT(get_self().is_initialized4inference());
				NNTL_ASSERT(get_self().samplesYStorageCoherent() && get_self().samplesXStorageCoherent());

				m_curDataset2Walk = dataSetId;
				m_curWalkingBatchIdx = 0;

				if (batchSize < 0) {
					batchSize = m_maxFPropSize;
				} else if (0 == batchSize) {
					batchSize = cd.input_batch_size();
				}

				NNTL_ASSERT(batchSize > 0 && batchSize <= m_maxFPropSize);
				const auto dsNumel = get_self().dataset_samples_count(dataSetId);
				NNTL_ASSERT(dsNumel > 0);
				NNTL_ASSERT(dsNumel <= ::std::numeric_limits<vec_len_t>::max());//requirement of the class
				if (batchSize > dsNumel)
					batchSize = static_cast<vec_len_t>(dsNumel);

				if (batchSize < dsNumel) {
					m_curBatchSize = batchSize;
					if (exclude_dataX(excludeDataFlag)) {
						m_pCurBatchX = nullptr;
					} else {
						if (get_self().X(train_set_id).bBatchInRow()) {
							//we can use m_walkX just as a view inside X(train_set_id)
							m_pCurBatchX = &m_walkX;
						} else {
							NNTL_ASSERT(!m_batch_x.empty() && m_batch_x.emulatesBiases());
							m_batch_x.deform_batch_size_with_biases(batchSize);
							m_pCurBatchX = &m_batch_x;
						}
					}

					if (exclude_dataY(excludeDataFlag)) {
						m_pCurBatchY = nullptr;
					} else {
						if (get_self().Y(train_set_id).bBatchInRow()) {
							m_pCurBatchY = &m_walkY;
						} else {
							NNTL_ASSERT(!m_batch_y.empty() && !m_batch_y.emulatesBiases());
							m_batch_y.deform_batch_size(batchSize);
							m_pCurBatchY = &m_batch_y;
						}
					}
				} else {
					m_curBatchSize = -1;
					m_pCurBatchX = exclude_dataX(excludeDataFlag) ? nullptr : &get_self().X_mutable(dataSetId);
					NNTL_ASSERT(batchSize == get_self().X(dataSetId).batch_size());
					m_pCurBatchY = exclude_dataY(excludeDataFlag) ? nullptr : &get_self().Y_mutable(dataSetId);
					NNTL_ASSERT(batchSize == get_self().Y(dataSetId).batch_size());
				}

				const auto _dr = ::std::div(dsNumel, static_cast<numel_cnt_t>(batchSize));
				const auto numBatches = _dr.quot + (_dr.rem > 1);
				NNTL_ASSERT(numBatches > 0);
				return numBatches;
			}

			template<typename CommonDataT>
			void next_subset(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept {
				NNTL_UNREF(batchIdx);
				NNTL_ASSERT(batchIdx >= 0);
				NNTL_ASSERT(get_self().is_initialized4inference());
				NNTL_ASSERT(m_curDataset2Walk != invalid_set_id);

				//we MUST NOT use cd.input_batch_size() here because it is not required to be set to proper value for just
				//walking over any dataset (what if you are just inspecting its content and not going to do inferencing at all?)

				if (m_curBatchSize > 0) {//it's a minibatch mode!		
					const bool bBatchX = (m_pCurBatchX == &m_batch_x), bWalkX = (m_pCurBatchX == &m_walkX);
					const bool bBatchY = (m_pCurBatchY == &m_batch_y), bWalkY = (m_pCurBatchY == &m_walkY);
					//it's in batch var if the source in bBatchInColumn() mode and in walk var if bBatchInRow() mode

					NNTL_ASSERT(!m_pCurBatchX || bBatchX || bWalkX);
					NNTL_ASSERT(!m_pCurBatchY || bBatchY || bWalkY);

					const auto dsSize = get_self().X(m_curDataset2Walk).batch_size();//OK to query directly, it's class design
									// _train_data_simple only works with matrix-sized datasets
					NNTL_ASSERT(dsSize == get_self().Y(m_curDataset2Walk).batch_size());

					if (m_curWalkingBatchIdx >= dsSize) m_curWalkingBatchIdx = 0;
					const auto maxBs = dsSize - m_curWalkingBatchIdx;

					//checking batch size
					auto batchSize = m_curBatchSize;
					if (maxBs < batchSize) {
						batchSize = maxBs;
						if (bBatchY) m_batch_y.deform_batch_size(maxBs);
						if (bBatchX) m_batch_x.deform_batch_size_with_biases(maxBs);
					} else {
						NNTL_ASSERT(!bBatchX || batchSize == m_batch_x.batch_size());
						NNTL_ASSERT(!bBatchY || batchSize == m_batch_y.batch_size());
					}

					auto& iM = cd.iMath();
					//X should be processed the last to leave it in cache
					//fetching data subset into m_batch* vars
					if (bBatchY) {
						NNTL_ASSERT(get_self().Y(m_curDataset2Walk).bBatchInColumn());
						iM.mExtractRowsSeq(get_self().Y(m_curDataset2Walk), m_curWalkingBatchIdx, m_batch_y);
					} else if (bWalkY) {
						const auto& dsMtx = get_self().Y(m_curDataset2Walk);//it's const anyway
						NNTL_ASSERT(dsMtx.bBatchInRow());
						//we guarantee that no modification should occur (at least under correct use), so const_cast here is fine
						auto pBegin = const_cast<y_t*>(dsMtx.colDataAsVec(m_curWalkingBatchIdx));
						NNTL_ASSERT(dsMtx.cols() >= (m_curWalkingBatchIdx + batchSize));
						NNTL_ASSERT(!dsMtx.emulatesBiases());
						m_walkY.useExternalStorage(pBegin, dsMtx.sample_size(), batchSize, false, false, true);
					}
					
					if (bBatchX) {
						NNTL_ASSERT(get_self().X(m_curDataset2Walk).bBatchInColumn());
						iM.mExtractRowsSeq(get_self().X(m_curDataset2Walk), m_curWalkingBatchIdx, m_batch_x);
					} else if (bWalkX) {
						const auto& dsMtx = get_self().X(m_curDataset2Walk);//it's const anyway
						NNTL_ASSERT(dsMtx.bBatchInRow());
						//we guarantee that no modification should occur (at least under correct use), so const_cast here is fine
						auto pBegin = const_cast<x_t*>(dsMtx.colDataAsVec(m_curWalkingBatchIdx));
						NNTL_ASSERT(dsMtx.cols() >= (m_curWalkingBatchIdx + batchSize));
						NNTL_ASSERT(dsMtx.emulatesBiases());
						//+1 to account for mandatory biases!
						m_walkX.useExternalStorage(pBegin, dsMtx.sample_size() + 1, batchSize, true, false, true);
					}

					m_curWalkingBatchIdx += batchSize;
				} else {
					NNTL_ASSERT(-1 == m_curBatchSize);
					//NNTL_ASSERT(0 == batchIdx);//only one call permitted
				}
			}
		};
	}
}


