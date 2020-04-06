/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include <vector>
#include <algorithm>
#include <functional>

#include "_i_train_data.h"

#include "../_nnet_errs.h"
#include "../utils/scope_exit.h"

namespace nntl {

	namespace _impl {

		// in addition to some not implemented in class _train_data_simple functions of _i_train_data interface,
		// _train_data_simple require a derived class to provide const realmtx_t& X(data_set_id_t dataSetId) and Y() functions
		// to return corresponding matrices with complete dataset data (that implies that the class can not work with datasets
		// with greater than max(vec_len_t) samples)
		template<typename FinalPolymorphChild, typename BaseT>
		class _train_data_simple : public _i_train_data<BaseT> {
		protected:
			realmtxdef_t m_batch_x, m_batch_y;

			//vector to hold randomized training set row numbers
			::std::vector<vec_len_t> m_vRowIdxs;
			::std::vector<vec_len_t>::iterator m_curRowIdx;

			const realmtx_t* m_pCurBatchX{ nullptr };
			const realmtx_t* m_pCurBatchY{ nullptr };

			vec_len_t m_maxFPropSize{ 0 }, m_maxTrainBatchSize{ 0 }, m_curWalkingRow{ 0 };
			data_set_id_t m_curDataset2Walk{ invalid_set_id };

			bool m_bMiniBatchTraining{ false };

		public:
			//////////////////////////////////////////////////////////////////////////
			//typedefs
			typedef FinalPolymorphChild self_t;
			NNTL_METHODS_SELF_CHECKED((::std::is_base_of<_train_data_simple, FinalPolymorphChild>::value)
				, "FinalPolymorphChild must derive from _train_data_simple<FinalPolymorphChild>");

			//////////////////////////////////////////////////////////////////////////
			// default implementation, redefine in derived class if necessary
			data_set_id_t datasets_count()const noexcept { return 2; }

			static DatasetNamingFunc_t get_dataset_naming_function() noexcept {
				return [](data_set_id_t dataSetId, char* pName, unsigned siz) {
					switch (dataSetId) {
					case train_set_id:
						::strcpy_s(pName, siz, "train");
						break;

					case test_set_id:
						::strcpy_s(pName, siz, "test");
						break;

					default:
						::sprintf_s(pName, siz, "!Unknown!%d", dataSetId);
						break;
					}
				};
			}

			//////////////////////////////////////////////////////////////////////////
			//convenience wrappers around dataset_samples_count()
			numel_cnt_t trainset_samples_count()const noexcept { return get_self().dataset_samples_count(train_set_id); }
			numel_cnt_t testset_samples_count()const noexcept { return get_self().dataset_samples_count(test_set_id); }

			//////////////////////////////////////////////////////////////////////////

			//convenience wrappers around walk_over_set(). Always uses maxFPropSize returned from init*() as a batch size
			template<typename CommonDataT>
			numel_cnt_t walk_over_train_set(const CommonDataT& cd)noexcept {
				NNTL_ASSERT(get_self().is_initialized4inference());
				return get_self().walk_over_set(train_set_id, cd);
			}
			template<typename CommonDataT>
			numel_cnt_t walk_over_test_set(const CommonDataT& cd)noexcept {
				NNTL_ASSERT(get_self().is_initialized4inference());
				return get_self().walk_over_set(test_set_id, cd);
			}

			//////////////////////////////////////////////////////////////////////////

			bool is_initialized4inference()const noexcept {
				NNTL_ASSERT(!get_self().empty());
				return m_maxFPropSize > 0;
			}

			bool is_initialized4train()const noexcept {
				NNTL_ASSERT(!get_self().empty());
				return m_maxTrainBatchSize > 0 && m_maxFPropSize > 0;
			}

			void deinit4all()noexcept {
				m_maxTrainBatchSize = m_maxFPropSize = m_curWalkingRow = 0;
				m_curDataset2Walk = invalid_set_id;
				m_bMiniBatchTraining = false;
				m_vRowIdxs.clear();
				m_vRowIdxs.shrink_to_fit();
				m_curRowIdx = m_vRowIdxs.end();
				m_batch_x.clear();
				m_batch_y.clear();
				m_pCurBatchX = m_pCurBatchY = nullptr;
			}

		protected:
			bool _initBatches(const bool bMiniBatchInferencing)noexcept {
				NNTL_ASSERT(m_batch_x.empty() && m_batch_y.empty());

				//initializing additional data members
				if (m_bMiniBatchTraining || bMiniBatchInferencing) {
					const auto xW = get_self().xWidth(), yW = get_self().yWidth();
					const auto mbs = ::std::max(m_maxTrainBatchSize, m_maxFPropSize);
					NNTL_ASSERT(mbs > 0 && xW > 0 && yW > 0);

					m_batch_x.will_emulate_biases();
					m_batch_y.dont_emulate_biases();

					if (!m_batch_x.resize(mbs, xW) || !m_batch_y.resize(mbs, yW)) {
						return false;
					}
				}
				return true;
			}

		public:
			numel_cnt_t biggest_samples_count()const noexcept {
				const auto tds = get_self().datasets_count();
				numel_cnt_t r = 0;
				for (data_set_id_t i = 0; i < tds; ++i) {
					const auto sc = get_self().dataset_samples_count(i);
					if (sc > ::std::numeric_limits<vec_len_t>::max()) {
						NNTL_ASSERT(!"_train_data_simple can't work with datasets greater than ::std::numeric_limits<vec_len_t>::max() elements!");
					#pragma warning(disable:4297)//function assumed not to throw
						throw ::std::logic_error("WTF?! Too long dataset for _train_data_simple class!");
					#pragma warning(default:4297)
					}
					r = ::std::max(r, sc);
				}
				return r;
			}

			nnet_errors_t init4inference(IN OUT vec_len_t& maxFPropSize)noexcept {
				NNTL_ASSERT(maxFPropSize >= 0);
				if (m_maxFPropSize != 0) {
					NNTL_ASSERT(!"already initialized!");
					return nnet_errors_t::InvalidTD;
				}
				NNTL_ASSERT(!get_self().is_initialized4inference());//in case it was overriden
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
			nnet_errors_t init4train(IN OUT vec_len_t& maxFPropSize, IN OUT vec_len_t& maxBatchSize, OUT bool& bMiniBatch) noexcept {
				NNTL_ASSERT(maxBatchSize >= 0 && maxFPropSize >= 0);

				if (m_maxTrainBatchSize != 0 || m_maxFPropSize != 0) {
					NNTL_ASSERT(!"already initialized!");
					return nnet_errors_t::InvalidTD;
				}
				NNTL_ASSERT(!get_self().is_initialized4train());//in case it was overriden
				get_self().deinit4all();//it's always better to start from the known state

				bool bSuccess = false;
				utils::scope_exit deinit_if_error([this, &bSuccess]() {
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

				m_bMiniBatchTraining = bMiniBatch = maxBatchSize < trainCnt;

				if (m_bMiniBatchTraining) {
					try {
						m_vRowIdxs.resize(trainCnt);
					} catch (const ::std::exception&) {
						return nnet_errors_t::TdInitNoMemory;
					}
					m_curRowIdx = m_vRowIdxs.end();
					::std::iota(m_vRowIdxs.begin(), m_vRowIdxs.end(), 0);
				}

				if (!get_self()._initBatches(maxFPropSize < biggestSetSize)) return nnet_errors_t::TdInitNoMemory;

				bSuccess = true;
				return nnet_errors_t::Success;
			}

			//for training & inference mode to return
			// - corresponding training set batches for given on_next_epoch(epochIdx) and on_next_batch(batchIdx) (if they were called just before)
			//		Each batch always contain the same amount of rows equal to the one was set to cd.get_cur_batch_size() during on_next_epoch()
			// - corresponding batches of data if walk_over_set()/next_subset() were called before
			//		Each batch contain at max maxFPropSize rows (set in init4inference())
			// Note that once the mode was changed, the caller MUST assume that underlying data, returned by these functions, became a junk
			const realmtx_t& batchX()const noexcept { NNTL_ASSERT(m_pCurBatchX); return *m_pCurBatchX; }
			const realmtx_t& batchY()const noexcept { NNTL_ASSERT(m_pCurBatchY); return *m_pCurBatchY; }

			//notifies object about new training epoch start and must return total count of batches to execute over the training set
			//batchSize MUST always be less or equal to maxBatchSize, returned by init4train()
			template<typename CommonDataT>
			numel_cnt_t on_next_epoch(const numel_cnt_t epochIdx, const CommonDataT& cd, vec_len_t batchSize = 0) noexcept {
				NNTL_UNREF(epochIdx);
				NNTL_ASSERT(epochIdx >= 0);
				NNTL_ASSERT(get_self().is_initialized4train());

				m_curDataset2Walk = invalid_set_id;

				if (batchSize<0) {
					batchSize = m_maxTrainBatchSize;
				} else if (0 == batchSize) {
					batchSize = cd.get_cur_batch_size();
				}
				NNTL_ASSERT(batchSize > 0 && batchSize <= m_maxTrainBatchSize && batchSize <= get_self().trainset_samples_count());

				if (m_bMiniBatchTraining) {
					//fixing the size of m_batch* matrices
					NNTL_ASSERT(!m_batch_x.empty() && !m_batch_y.empty());
					NNTL_ASSERT(m_batch_x.emulatesBiases() && !m_batch_y.emulatesBiases());

					const auto oldRows = m_batch_x.deform_rows(batchSize);
					if (oldRows != batchSize) {
						m_batch_x.set_biases();
					} else {
						NNTL_ASSERT(m_batch_x.test_biases_strict());
					}
					m_pCurBatchX = &m_batch_x;

					m_batch_y.deform_rows(batchSize);
					m_pCurBatchY = &m_batch_y;

					m_curRowIdx = m_vRowIdxs.begin();
					//making random permutations to define which data rows will be used as batch data
					::std::random_shuffle(m_curRowIdx, m_vRowIdxs.end(), cd.iRng());
				} else {
					m_pCurBatchX = &get_self().X(train_set_id);
					NNTL_ASSERT(batchSize == m_pCurBatchX->rows());
					m_pCurBatchY = &get_self().Y(train_set_id);
					NNTL_ASSERT(batchSize == m_pCurBatchY->rows());
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

				if (m_bMiniBatchTraining) {
					auto& iM = cd.iMath();

					NNTL_ASSERT(m_batch_x.rows() == cd.get_cur_batch_size() && m_batch_y.rows() == cd.get_cur_batch_size());
					NNTL_ASSERT(m_pCurBatchX == &m_batch_x && m_pCurBatchY == &m_batch_y);

					iM.mExtractRows(get_self().X(train_set_id), m_curRowIdx, m_batch_x);
					iM.mExtractRows(get_self().Y(train_set_id), m_curRowIdx, m_batch_y);
					m_curRowIdx += cd.get_cur_batch_size();
				} else {
					NNTL_ASSERT(0 == batchIdx);//only one call permitted
				}
			}

			template<typename CommonDataT>
			numel_cnt_t walk_over_set(const data_set_id_t dataSetId, const CommonDataT& cd, vec_len_t batchSize = -1)noexcept {
				NNTL_ASSERT(dataSetId >= 0 && dataSetId < get_self().datasets_count());
				NNTL_ASSERT(get_self().is_initialized4inference());

				m_curDataset2Walk = dataSetId;
				m_curWalkingRow = 0;

				if (batchSize < 0) {
					batchSize = m_maxFPropSize;
				} else if (0 == batchSize) {
					batchSize = cd.get_cur_batch_size();
				}

				NNTL_ASSERT(batchSize > 0 && batchSize <= m_maxFPropSize);
				const auto dsNumel = get_self().dataset_samples_count(dataSetId);
				NNTL_ASSERT(dsNumel <= ::std::numeric_limits<vec_len_t>::max());//requirement of the class
				if (batchSize > dsNumel) {
					batchSize = static_cast<vec_len_t>(dsNumel);
				}
				const auto _dr = ::std::div(dsNumel, static_cast<numel_cnt_t>(batchSize));
				const auto numBatches = _dr.quot + (_dr.rem > 1);
				NNTL_ASSERT(numBatches > 0);

				if (batchSize < dsNumel) {
					//fixing the size of m_batch* matrices
					NNTL_ASSERT(!m_batch_x.empty() && !m_batch_y.empty());
					NNTL_ASSERT(m_batch_x.emulatesBiases() && !m_batch_y.emulatesBiases());

					const auto oldRows = m_batch_x.deform_rows(batchSize);
					if (oldRows != batchSize) {
						m_batch_x.set_biases();
					} else {
						NNTL_ASSERT(m_batch_x.test_biases_strict());
					}
					m_pCurBatchX = &m_batch_x;

					m_batch_y.deform_rows(batchSize);
					m_pCurBatchY = &m_batch_y;
				} else {
					m_pCurBatchX = &get_self().X(dataSetId);
					NNTL_ASSERT(batchSize == m_pCurBatchX->rows());
					m_pCurBatchY = &get_self().Y(dataSetId);
					NNTL_ASSERT(batchSize == m_pCurBatchY->rows());
				}

				return numBatches;
			}

			template<typename CommonDataT>
			void next_subset(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept {
				NNTL_UNREF(batchIdx);
				NNTL_ASSERT(batchIdx >= 0);
				NNTL_ASSERT(get_self().is_initialized4inference());
				NNTL_ASSERT(m_curDataset2Walk != invalid_set_id);

				//we MUST NOT use cd.get_cur_batch_size() here because it is not required to be set to proper value for infencing

				/*const auto batchSize = m_pCurBatchX->rows();
				NNTL_ASSERT(batchSize == m_pCurBatchY->rows());
				const auto dsNumel = get_self().dataset_samples_count(m_curDataset2Walk);*/

				if (m_pCurBatchX == &m_batch_x) {//it's a minibatch mode!
					auto& iM = cd.iMath();

					NNTL_ASSERT(m_pCurBatchY == &m_batch_y);

					//checking batch size
					const auto batchSize = m_batch_x.rows();
					const auto dsSize = get_self().X(m_curDataset2Walk).rows();//OK to query directly, because it's
					// assumed _train_data_simple only works with matrix-sized datasets
					NNTL_ASSERT(dsSize == get_self().Y(m_curDataset2Walk).rows());
					const auto maxBs = dsSize - m_curWalkingRow;
					NNTL_ASSERT(maxBs > 0);
					if (maxBs < batchSize) {
						m_batch_y.deform_rows(maxBs);
						m_batch_x.deform_rows(maxBs);
						m_batch_x.set_biases();
					}

					//fetching data sunset into m_batch* vars

					iM.mExtractRowsSeq(get_self().X(m_curDataset2Walk), m_curWalkingRow, m_batch_x);
					iM.mExtractRowsSeq(get_self().Y(m_curDataset2Walk), m_curWalkingRow, m_batch_y);
					m_curWalkingRow += m_batch_x.rows();
				} else {
					NNTL_ASSERT(0 == batchIdx);//only one call permitted
				}
			}

		};
	}
}


