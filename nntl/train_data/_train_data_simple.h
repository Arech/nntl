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

#include "../utils/mtx2Normal.h"

namespace nntl {

	template<typename DataT, typename StatsT = DataT>
	struct NormTDBaseSettings : public utils::mtx2Normal::Settings<DataT, StatsT> {
		vec_len_t batchSize{ -1 };//see walk_over_set() batchSize argument
		vec_len_t batchCount{ 0 };//0 means gather stats over whole dataset, else this count of batches
	};

	template<typename DataT, typename StatsT = DataT>
	struct NormTDSettings : public NormTDBaseSettings<DataT, StatsT> {
		bool bColumnwise{ false };//columns instead of whole matrix
	};

	namespace _impl {

		// in addition to some not implemented in class _train_data_simple functions of _i_train_data interface,
		// _train_data_simple require a derived class to provide const realmtx_t& X(data_set_id_t dataSetId) and Y() functions
		// to return corresponding matrices with complete dataset data (that implies that the class can not work with datasets
		// with greater than max(vec_len_t) samples)
		template<typename FinalPolymorphChild, typename XT, typename YT>
		class _train_data_simple : public _i_train_data<XT, YT> {
		protected:
			x_mtxdef_t m_batch_x;
			y_mtxdef_t m_batch_y;

			//vector to hold randomized training set row numbers
			::std::vector<vec_len_t> m_vRowIdxs;
			::std::vector<vec_len_t>::iterator m_curRowIdx;

			const x_mtx_t* m_pCurBatchX{ nullptr };
			const y_mtx_t* m_pCurBatchY{ nullptr };

			vec_len_t m_maxFPropSize{ 0 }, m_maxTrainBatchSize{ 0 }, m_curWalkingRow{ 0 };
			data_set_id_t m_curDataset2Walk{ invalid_set_id }; // set to non invalid_set_id value only for inferencing
			
			bool m_bMiniBatchTraining{ false };

		public:
			//////////////////////////////////////////////////////////////////////////
			//typedefs
			typedef FinalPolymorphChild self_t;
			typedef FinalPolymorphChild tds_self_t;
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

			bool isSuitableForOutputOf(neurons_count_t n)const noexcept { return n == get_self().yWidth(); }

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
				m_pCurBatchX = nullptr;
				m_pCurBatchY = nullptr;
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
						//the reason is that it works with data matrices directly and addresses rows/cols with vec_len_t
					#pragma warning(push)
					#pragma warning(disable:4297)//function assumed not to throw
						throw ::std::logic_error("WTF?! Too long dataset for _train_data_simple class!");
					#pragma warning(pop)
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
			const x_mtx_t& batchX()const noexcept { NNTL_ASSERT(m_pCurBatchX); return *m_pCurBatchX; }
			const y_mtx_t& batchY()const noexcept { NNTL_ASSERT(m_pCurBatchY); return *m_pCurBatchY; }

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
					} else NNTL_ASSERT(m_batch_x.test_biases_strict());

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

					NNTL_ASSERT(m_curRowIdx + cd.get_cur_batch_size() <= m_vRowIdxs.end());

					iM.mExtractRows(get_self().X(train_set_id), m_curRowIdx, m_batch_x);
					iM.mExtractRows(get_self().Y(train_set_id), m_curRowIdx, m_batch_y);
					m_curRowIdx += cd.get_cur_batch_size();
				} else {
					NNTL_ASSERT(0 == batchIdx);//only one call permitted
				}
			}
			//////////////////////////////////////////////////////////////////////////
			template<typename CommonDataT>
			numel_cnt_t walk_over_set(const data_set_id_t dataSetId, const CommonDataT& cd
				, vec_len_t batchSize = -1, const unsigned excludeDataFlag = flag_exclude_nothing)noexcept
			{
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
				NNTL_ASSERT(dsNumel > 0);
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
					} else NNTL_ASSERT(m_batch_x.test_biases_strict());

					m_pCurBatchX = exclude_dataX(excludeDataFlag) ? nullptr : &m_batch_x;

					m_batch_y.deform_rows(batchSize);
					m_pCurBatchY = exclude_dataY(excludeDataFlag) ? nullptr : &m_batch_y;
				} else {
					m_pCurBatchX = exclude_dataX(excludeDataFlag) ? nullptr : &get_self().X(dataSetId);
					NNTL_ASSERT(batchSize == get_self().X(dataSetId).rows());
					m_pCurBatchY = exclude_dataY(excludeDataFlag) ? nullptr : &get_self().Y(dataSetId);
					NNTL_ASSERT(batchSize == get_self().Y(dataSetId).rows());
				}

				return numBatches;
			}

			template<typename CommonDataT>
			void next_subset(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept {
				NNTL_UNREF(batchIdx);
				NNTL_ASSERT(batchIdx >= 0);
				NNTL_ASSERT(get_self().is_initialized4inference());
				NNTL_ASSERT(m_curDataset2Walk != invalid_set_id);

				//we MUST NOT use cd.get_cur_batch_size() here because it is not required to be set to proper value for inferencing

				/*const auto batchSize = m_pCurBatchX->rows();
				NNTL_ASSERT(batchSize == m_pCurBatchY->rows());
				const auto dsNumel = get_self().dataset_samples_count(m_curDataset2Walk);*/

				if (m_pCurBatchX == &m_batch_x || m_pCurBatchY == &m_batch_y) {//it's a minibatch mode!
					NNTL_ASSERT(!m_pCurBatchX || m_pCurBatchX == &m_batch_x);
					NNTL_ASSERT(!m_pCurBatchY || m_pCurBatchY == &m_batch_y);

					auto& iM = cd.iMath();

					//checking batch size
					const auto batchSize = m_batch_x.rows();
					const auto dsSize = get_self().X(m_curDataset2Walk).rows();//OK to query directly, because it's
					if (m_curWalkingRow >= dsSize) m_curWalkingRow = 0;

					// assumed _train_data_simple only works with matrix-sized datasets
					NNTL_ASSERT(dsSize == get_self().Y(m_curDataset2Walk).rows());
					const auto maxBs = dsSize - m_curWalkingRow;
					NNTL_ASSERT(maxBs > 0);
					if (maxBs < batchSize) {
						m_batch_y.deform_rows(maxBs);
						const auto oldRows = m_batch_x.deform_rows(maxBs);
						if (oldRows != maxBs) {
							m_batch_x.set_biases();
						} else NNTL_ASSERT(m_batch_x.test_biases_strict());	
					}

					//fetching data sunset into m_batch* vars
					if (m_pCurBatchX) iM.mExtractRowsSeq(get_self().X(m_curDataset2Walk), m_curWalkingRow, m_batch_x);
					if (m_pCurBatchY) iM.mExtractRowsSeq(get_self().Y(m_curDataset2Walk), m_curWalkingRow, m_batch_y);
					m_curWalkingRow += m_batch_x.rows();
				} else {
					//NNTL_ASSERT(0 == batchIdx);//only one call permitted
				}
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			// data normalization stuff
		public:
			template<typename StatsT = x_t>
			using NormalizationSettings_tpl = NormTDSettings<x_t, StatsT>;

			template<typename StatsT = x_t>
			using NormBaseSettings_tpl = NormTDBaseSettings<x_t, StatsT>;

			static constexpr bool bAdjustForSampleVar = true;

		protected:
			template<typename StatsT>
			struct ScaleCentralData {
				StatsT sc{ StatsT(1.) };
				StatsT ofs{ StatsT(0.) };
				bool bSc{ false }, bOfs{ false };

				// Note that scale & offset is expected in form x1=sc*x0 + ofs
				// For the form x1 = sc*(x0+ofs) we'd need different aggregation formula

				void _upd_scale(const StatsT v)noexcept {
					sc *= v;
					if (bOfs) ofs *= v;
					bSc = true;
				}
				void _upd_ofs(const StatsT v)noexcept {
					ofs += v;
					bOfs = true;
				}
				void _upd_both(const StatsT vSc, const StatsT vOfs)noexcept {
					sc *= vSc;
					ofs = vSc*ofs + vOfs;
					bSc = bOfs = true;
				}

				void reset()noexcept {
					sc = StatsT(1.);
					ofs = StatsT(0.);
					bSc = bOfs = false;
				}
			};

			template<typename StatsT>
			using fullstats_tpl = ::std::vector<ScaleCentralData<StatsT>>;

		protected:
			template<typename iMathT>
			static void _apply_norm_scale(iMathT& iM, x_mtx_t& m, const x_t multVal)noexcept {
				iM.evMulC_ip_nb(m, multVal);
			}
			template<typename iMathT>
			static void _apply_norm_offs(iMathT& iM, x_mtx_t& m, const x_t ofs)noexcept {
				iM.evAddC_ip_nb(m, ofs);
			}
			template<typename iMathT>
			static void _apply_norm_both(iMathT& iM, x_mtx_t& m, const x_t multVal, const x_t ofs)noexcept {
				iM.evMulCAddC_ip_nb(m, multVal, ofs);
			}

			//this implementation assumes the training data is modifiable. At the cost of storing statistics in
			//the parent object that assumption could be dropped and data will be updated/normalized on the fly

			template<typename CommonDataT, typename StatsT>
			struct _NormBase {
				typedef NormBaseSettings_tpl<StatsT> Setts_t;
				typedef CommonDataT common_data_t;

				tds_self_t& m_thisHost;
				const Setts_t& m_Setts;
				const common_data_t& m_CD;

				_NormBase(tds_self_t& h, const Setts_t& S, const common_data_t& cd) noexcept
					: m_thisHost(h), m_Setts(S), m_CD(cd) {}

				// Returns how many iterations (batch counts) must be done to satisfy preferred arguments
				numel_cnt_t prepareToWalk()noexcept {
					const auto dataBatchCnt = m_thisHost.walk_over_set(train_set_id, m_CD, m_Setts.batchSize, flag_exclude_dataY);
					NNTL_ASSERT(dataBatchCnt > 0);
					return ::std::min(dataBatchCnt, m_Setts.batchCount ? numel_cnt_t(m_Setts.batchCount) : dataBatchCnt);
				}
				//returns a pointer to matrix data. Matrix must have at least 1 element (and more than 1 over all batches)
				//walk() is not required to obey batchIdx, it's just a convenience argument. The only requirement is that
				// the whole matrix must be walked over with all batches.
				__declspec(restrict) const x_mtx_t* __restrict walk(numel_cnt_t batchIdx)noexcept {
					m_thisHost.next_subset(batchIdx, m_CD);
					return &m_thisHost.batchX();
				}
			};

			template<typename FinalT, typename CommonDataT, typename StatsT>
			class _Norm_whole
				: public utils::mtx2Normal::_FNorm_whole_base<FinalT, x_t, bAdjustForSampleVar, StatsT>
				, protected _NormBase<CommonDataT, StatsT>
			{
				typedef utils::mtx2Normal::_FNorm_whole_base<FinalT, x_t, bAdjustForSampleVar, StatsT> _base_t;
				typedef _NormBase<CommonDataT, StatsT> _nbase_t;

				typedef ScaleCentralData<StatsT> indstat_t;

			public:
				using _nbase_t::walk;
				using _nbase_t::prepareToWalk;

				indstat_t m_Stat;

			public:
				void _upd_scale(const statsdata_t v)noexcept {
					die_check_fpvar(v);
					m_Stat._upd_scale(v);
				}
				void _upd_ofs(const statsdata_t v)noexcept {
					die_check_fpvar(v);
					m_Stat._upd_ofs(v);
				}
				void _upd_both(const statsdata_t vSc, const statsdata_t vOfs)noexcept {
					die_check_fpvar(vSc); die_check_fpvar(vOfs);
					m_Stat._upd_both(vSc, vOfs);
				}

				template<typename ...ArgsT>
				_Norm_whole(ArgsT&&... args)noexcept : _nbase_t(::std::forward<ArgsT>(args)...) {}

				void change_scale(const statsdata_t scaleVal)noexcept {
					NNTL_ASSERT(scaleVal != statsdata_t(0));
					const auto multVal = m_Setts.targetScale / scaleVal;
					NNTL_ASSERT(!::std::isnan(multVal) && ::std::isfinite(multVal));

					_upd_scale(multVal);

					tds_self_t::_apply_norm_scale(m_CD.get_iMath(), m_thisHost.X_mutable(train_set_id), static_cast<x_t>(multVal));
				}
				void change_central(const statsdata_t centralVal)noexcept {
					const auto centrOffs = (m_Setts.targetCentral - centralVal);
					_upd_ofs(centrOffs);
					tds_self_t::_apply_norm_offs(m_CD.get_iMath(), m_thisHost.X_mutable(train_set_id), static_cast<x_t>(centrOffs));
				}
				void change_both(const statsdata_t scaleVal, const statsdata_t centralVal)noexcept {
					NNTL_ASSERT(scaleVal != statsdata_t(0));
					const auto multVal = m_Setts.targetScale / scaleVal;
					NNTL_ASSERT(!::std::isnan(multVal) && ::std::isfinite(multVal));

					const auto centrOffs = (m_Setts.targetCentral - centralVal)*multVal;

					_upd_both(multVal, centrOffs);

					tds_self_t::_apply_norm_both(m_CD.get_iMath(), m_thisHost.X_mutable(train_set_id)
						, static_cast<x_t>(multVal), static_cast<x_t>(centrOffs));
				}
			};

			template<typename CommonDataT, typename StatsT>
			class Norm_whole final : public _Norm_whole<Norm_whole<CommonDataT, StatsT>, CommonDataT, StatsT> {
				typedef _Norm_whole<Norm_whole<CommonDataT, StatsT>, CommonDataT, StatsT> _base_t;
			public:
				template<typename ...ArgsT>
				Norm_whole(ArgsT&&... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
			};

			//////////////////////////////////////////////////////////////////////////

			template<typename FinalT, typename CommonDataT, typename StatsT>
			class _Norm_cw
				: public utils::mtx2Normal::_FNorm_cw_base<FinalT, x_t, bAdjustForSampleVar, StatsT>
				, protected _NormBase<CommonDataT, StatsT>
			{
				typedef utils::mtx2Normal::_FNorm_cw_base<FinalT, x_t, bAdjustForSampleVar, StatsT> _base_t;
				typedef _NormBase<CommonDataT, StatsT> _nbase_t;

			public:
				typedef fullstats_tpl<statsdata_t> fullstats_t;

			public:
				fullstats_t m_Stats;

			protected:
				fullstats_t m_iterStats;
				int iterations{ 0 };

			public:
				using _nbase_t::walk;
				using _nbase_t::prepareToWalk;

				template<typename ...ArgsT>
				_Norm_cw(ArgsT&&... args)noexcept : _nbase_t(::std::forward<ArgsT>(args)...) {
					const auto wid = total_cols();
					m_Stats.resize(wid);
					m_iterStats.resize(wid);
				}

			protected:
				void _upd_scale(const vec_len_t c, const statsdata_t v)noexcept {
					die_check_fpvar(v);
					m_Stats[c]._upd_scale(v);
					m_iterStats[c]._upd_scale(v);
				}
				void _upd_ofs(const vec_len_t c, const statsdata_t v)noexcept {
					die_check_fpvar(v);
					m_Stats[c]._upd_ofs(v);
					m_iterStats[c]._upd_ofs(v);
				}
				void _upd_both(const vec_len_t c, const statsdata_t vSc, const statsdata_t vOfs)noexcept {
					die_check_fpvar(vSc); die_check_fpvar(vOfs);
					m_Stats[c]._upd_both(vSc, vOfs);
					m_iterStats[c]._upd_both(vSc, vOfs);
				}

			public:
				vec_len_t total_cols()const noexcept { return m_thisHost.xWidth(); }

				//forward args to iThreads.run();
				template<typename ...ArgsT>
				void iThreads_run(ArgsT&&... args)noexcept {
					return m_CD.get_iThreads().run(::std::forward<ArgsT>(args)...);
				}

				//resetting iteration statistics
				void cw_begin(const unsigned /*tryIdx*/) noexcept {
					for (auto& e : m_iterStats) e.reset();
				}
				//actually applying updates on iteration end
				void cw_end() noexcept {
					auto& mtx = m_thisHost.X_mutable(train_set_id);
					NNTL_ASSERT(mtx.cols_no_bias() == m_Stats.size() && m_Stats.size() == m_iterStats.size());

					m_CD.get_iThreads().run([pSt = &m_iterStats, &mtx, pIm = &m_CD.get_iMath()](const auto& pr)noexcept {
						tds_self_t::_apply_norm_cw_st(*pIm, mtx, math::s_vec_range(pr), *pSt);
					}, mtx.cols_no_bias());

					if (++iterations > 2) STDCOUTL("There were " << iterations << " iterations, please double check the normalization works correctly and delete that notice then!");
				}

				//just collecting stats for colIdx. Will update all at once in cw_end()
				void cw_change_scale(const vec_len_t colIdx, const statsdata_t scaleVal)noexcept {
					NNTL_ASSERT(scaleVal != statsdata_t(0));
					const auto multVal = m_Setts.targetScale / scaleVal;
					NNTL_ASSERT(!::std::isnan(multVal) && ::std::isfinite(multVal));
					die_check_fpvar(multVal);
					_upd_scale(colIdx, multVal);
				}
				void cw_change_central(const vec_len_t colIdx, const statsdata_t centralVal)noexcept {
					const auto ofs = (m_Setts.targetCentral - centralVal);
					die_check_fpvar(ofs);
					_upd_ofs(colIdx, ofs);
				}
				void cw_change_both(const vec_len_t colIdx, const statsdata_t scaleVal, const statsdata_t centralVal)noexcept {
					NNTL_ASSERT(scaleVal != statsdata_t(0));
					const auto multVal = m_Setts.targetScale / scaleVal;
					NNTL_ASSERT(!::std::isnan(multVal) && ::std::isfinite(multVal));
					die_check_fpvar(multVal);

					const auto ofs = (m_Setts.targetCentral - centralVal)*multVal;
					die_check_fpvar(ofs);
					_upd_both(colIdx, multVal, ofs);
				}
			};

			template<typename iMathT, typename StatsT>
			static void _apply_norm_cw_st(iMathT& iM, x_mtx_t& m, const math::s_vec_range& colR, const fullstats_tpl<StatsT>& st)noexcept {
				const auto colEnd = colR.elmEnd;
				NNTL_ASSERT(colEnd <= st.size());
				NNTL_ASSERT(st.size() == m.cols_no_bias());
				vec_len_t colIdx = colR.elmBegin;
				auto pM = m.colDataAsVec(colIdx);
				const auto ldm = static_cast<ptrdiff_t>(m.ldim());
				const auto mRows = static_cast<ptrdiff_t>(m.rows());

				for (; colIdx < colEnd; ++colIdx) {
					const auto& colSt = st[colIdx];

					const x_t vSc = static_cast<x_t>(colSt.sc), vOfs = static_cast<x_t>(colSt.ofs);

					//performing actual data normalization
					const bool bScale = colSt.bSc && !::std::isnan(vSc) && ::std::isfinite(vSc);
					const bool bOfs = colSt.bOfs && !::std::isnan(vOfs) && ::std::isfinite(vOfs);

					if (bScale || bOfs) {
						const auto pME = pM + mRows;
						if (bScale) {
							if (bOfs) {
								iM.vMulCAddC_ip_st(pM, pME, vSc, vOfs);
							} else {
								iM.vMulC_ip_st(pM, pME, vSc);
							}
						} else {
							iM.vAddC_ip_st(pM, pME, vOfs);
						}
					}

					pM += ldm;
				}
			}

			template<typename CommonDataT, typename StatsT>
			class Norm_cw final : public _Norm_cw<Norm_cw<CommonDataT, StatsT>, CommonDataT, StatsT> {
				typedef _Norm_cw<Norm_cw<CommonDataT, StatsT>, CommonDataT, StatsT> _base_t;
			public:
				template<typename ...ArgsT>
				Norm_cw(ArgsT&&... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
			};



		public:

			template<typename StatsT>
			static bool bShouldNormalize(const NormBaseSettings_tpl<StatsT>& Setts)noexcept {
				return (Setts.bCentralNormalize || Setts.bScaleNormalize) && Setts.maxTries > 0;
			}

			//performs X data normalization using given settings. Statistics are gathered using train_x dataset and then applied
			//to every other X dataset.
			// Dataset MUST be properly initialized for inference
			template<typename CommonDataT, typename StatsT>
			bool normalize_data(const CommonDataT& cd, const NormalizationSettings_tpl<StatsT>& Setts)noexcept {
				NNTL_ASSERT(get_self().is_initialized4inference());
				return Setts.bColumnwise
					? get_self().normalize_data_cw(cd, Setts)
					: get_self().normalize_data_whole(cd, Setts);
			}

			template<typename CommonDataT, typename StatsT>
			bool normalize_data_whole(const CommonDataT& cd, const NormBaseSettings_tpl<StatsT>& Setts)noexcept {
				if (!get_self().is_initialized4inference()) {
					NNTL_ASSERT(!"Initialize with at least init4inference() first");
					STDCOUTL("Initialize with at least init4inference() first");
					return false;
				}

				if (!(Setts.bCentralNormalize || Setts.bScaleNormalize) || Setts.maxTries <= 0) {
					//STDCOUTL("Hey, set proper flags, nothing to do now!");
					return true;
				}
				STDCOUTL("Doing train_x normalization over the whole data...");

				Norm_whole<CommonDataT, StatsT> fn(get_self(), Setts, cd);
				bool r = utils::mtx2Normal::normalize_whole(Setts, fn);
				if (r) {
					const auto& st = fn.m_Stat;
					const x_t vSc = static_cast<x_t>(st.sc), vOfs = static_cast<x_t>(st.ofs);

					//performing actual data normalization
					STDCOUT("TrainX dataset stats: ");
					if (st.bOfs) {
						STDCOUT(vOfs);
					} else STDCOUT("n/a");
					STDCOUT(", ");
					if (st.bSc) {
						STDCOUT(vSc);
					} else STDCOUT("n/a");
					STDCOUTL(". Going to update other datasets...");

					const bool bScale = st.bSc && !::std::isnan(vSc) && ::std::isfinite(vSc);
					const bool bOfs = st.bOfs && !::std::isnan(vOfs) && ::std::isfinite(vOfs);

					if (!bScale && !bOfs) {
						STDCOUTL("Failed to obtain statistics on train_x, cancelling normalization");
						return false;
					}

					const auto dc = get_self().datasets_count();
					auto& iM = cd.get_iMath();

					for (data_set_id_t dsi = test_set_id; dsi < dc; ++dsi) {
						if (bScale || bOfs) {
							auto& mtx = get_self().X_mutable(dsi);
							if (bScale) {
								if (bOfs) {
									self_t::_apply_norm_both(iM, mtx, vSc, vOfs);
								} else {
									self_t::_apply_norm_scale(iM, mtx, vSc);
								}
							} else {
								self_t::_apply_norm_offs(iM, mtx, vOfs);
							}
						}
					}
				}
				return r;
			}

			template<typename CommonDataT, typename StatsT>
			bool normalize_data_cw(const CommonDataT& cd, const NormBaseSettings_tpl<StatsT>& Setts)noexcept {
				if (!get_self().is_initialized4inference()) {
					NNTL_ASSERT(!"Initialize with at least init4inference() first");
					STDCOUTL("Initialize with at least init4inference() first");
					return false;
				}

				if (!(Setts.bCentralNormalize || Setts.bScaleNormalize) || Setts.maxTries <= 0) {
					//STDCOUTL("Hey, set proper flags, nothing to do now!");
					return true;
				}
				STDCOUTL("Doing train_x normalization columnwise...");

				Norm_cw<CommonDataT, StatsT> fn(get_self(), Setts, cd);
				bool r = utils::mtx2Normal::normalize_cw(Setts, fn);

				if (r) {
					const auto& allSt = fn.m_Stats;
					const auto siz = allSt.size();
					StatsT sumSc{ StatsT(0.) }, sumCentr{ StatsT(0.) };
					int nSc{ 0 }, nOfs{ 0 }, nSkip{ 0 };

					for (size_t idx = 0; idx < siz; ++idx) {
						const auto& ist = allSt[idx];
						const auto vSc = ist.sc, vOfs = ist.ofs;

						const bool bScale = ist.bSc && !::std::isnan(vSc) && ::std::isfinite(vSc);
						const bool bOfs = ist.bOfs && !::std::isnan(vOfs) && ::std::isfinite(vOfs);

						if (bScale || bOfs) {
							if (bScale) {
								++nSc;
								sumSc += vSc;
							}
							if (bOfs) {
								++nOfs;
								sumCentr += vOfs;	
							}
						} else {
							STDCOUTL("** Note, row " << idx << " doesn't have normalization data at all! Double check it's ok");
							++nSkip;
						}
					}

					STDCOUTL("Average train_x stats: " << (sumCentr / nOfs) << ", " << (sumSc / nSc) << ". " << nSkip << " rows skipped." \
						" Applying to the rest datasets...");

					const auto dc = get_self().datasets_count();
					auto& iM = cd.get_iMath();
					x_mtx_t* pM = nullptr;
					auto f = [&pM, &iM, &allSt](const auto& pr)noexcept {
						tds_self_t::_apply_norm_cw_st(iM, *pM, math::s_vec_range(pr), allSt);
					};

					for (data_set_id_t dsi = test_set_id; dsi < dc; ++dsi) {
						pM = &get_self().X_mutable(dsi);
						NNTL_ASSERT(pM->cols_no_bias() == siz);
						iM.ithreads().run(f, siz);
					}
				}
				return r;
			}


		};
	}
}


