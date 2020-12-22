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

#include <array>
#include <vector>

#include "utils/chrono.h"

#include "nnet_evaluator.h"

namespace nntl {

	//Observer is a code that evaluates (with a help from nnet_evaluator.h classes) and reports how well nnet performs on data

	// i_training_observer and derived classes must be default constructible
	template<typename RealT>
	struct i_training_observer : public virtual DataSetsId {
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;		
		typedef ::std::chrono::nanoseconds nanoseconds;

		template<typename TrainDataT, typename CommonDataT>
		nntl_interface bool init(numel_cnt_t epochs, TrainDataT& td, const CommonDataT& cd)noexcept;
		nntl_interface void deinit()noexcept;

		nntl_interface void report_results_begin(const data_set_id_t dataSetId, const numel_cnt_t totalBatches)noexcept;

		template<typename YT, typename CommonDataT>
		nntl_interface void report_results(const numel_cnt_t batchIdx, const realmtx_t& activations
			, const math::smatrix<YT>& data_y, const CommonDataT& cd)noexcept;

		//the main purpose of report_results_end() is to report performance on non-standard data sets.
		nntl_interface void report_results_end(const real_t lossVal)noexcept;


		template<typename TrainDataT>
		nntl_interface void on_training_start(const TrainDataT& td, vec_len_t batchSize, vec_len_t maxFpropSize, numel_cnt_t numParams)noexcept;
		
		//epochEnded is zero based, so the first epoch is 0. Will be ==-1 on initial report (before training begins)
		nntl_interface void on_training_fragment_end(const numel_cnt_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept;
		nntl_interface void on_training_end(const nanoseconds& trainTime)noexcept;
	};

	template<typename RealT>
	struct training_observer_silent : public i_training_observer<RealT> {
		
		template<typename TrainDataT, typename CommonDataT>
		static constexpr bool init(numel_cnt_t epochs, TrainDataT& td, const CommonDataT& cd) noexcept { return true; }
		static constexpr void deinit()noexcept {}

		static constexpr void report_results_begin(const data_set_id_t dataSetId, const numel_cnt_t totalBatches)noexcept{}

		template<typename YT, typename CommonDataT>
		static constexpr void report_results(const numel_cnt_t batchIdx, const realmtx_t& activations
			, const math::smatrix<YT>& data_y, const CommonDataT& cd)noexcept {}

		static constexpr void report_results_end(const real_t lossVal)noexcept {}

		template<typename TrainDataT>
		static constexpr void on_training_start(const TrainDataT& td, vec_len_t batchSize, vec_len_t maxFpropSize, numel_cnt_t numParams)noexcept {}
		static constexpr void on_training_fragment_end(const numel_cnt_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept{}
		static constexpr void on_training_end(const nanoseconds& trainTime) noexcept {}
	};

	//////////////////////////////////////////////////////////////////////////
	namespace _impl {
		template<typename RealT>
		struct training_observer_stdcout_base : public i_training_observer<RealT> {
			void deinit()noexcept { }

			template<typename TrainDataT>
			void on_training_start(const TrainDataT& td, vec_len_t batchSize, vec_len_t maxFpropSize, numel_cnt_t numParams)noexcept
			{
				static constexpr char* szReportFmt = "Model f:%d->%d has %zd params. There are %zd training samples and %zd for testing."
					" BatchSize=%d, batchCnt=%zd, maxFpropSize=%d";
				static constexpr unsigned uBufSize = 256;

				char szRep[uBufSize];
				sprintf_s(szRep, szReportFmt, td.xWidth(), td.yWidth(), numParams, td.trainset_samples_count()
					, td.testset_samples_count(), batchSize, td.trainset_samples_count() / batchSize, maxFpropSize);

				::std::cout << szRep << ::std::endl;
			}
		};
	}

	//////////////////////////////////////////////////////////////////////////
	//simple observer, that outputs only loss function value to ::std::cout
	template<typename RealT>
	class training_observer_simple_stdcout : public _impl::training_observer_stdcout_base<RealT> {
		typedef _impl::training_observer_stdcout_base<RealT> _base_class_t;
	protected:
		numel_cnt_t m_epochs;

	public:

		template<typename TrainDataT, typename CommonDataT>
		bool init(numel_cnt_t epochs, TrainDataT& td, const CommonDataT& cd)noexcept {
			NNTL_UNREF(td); NNTL_UNREF(cd);
			m_epochs = epochs;
			return true;
		}
		
		static constexpr void report_results_begin(const data_set_id_t /*dataSetId*/, const numel_cnt_t /*totalBatches*/)noexcept {}

		template<typename YT, typename CommonDataT>
		static constexpr void report_results(const numel_cnt_t /*batchIdx*/, const realmtx_t& /*activations*/
			, const math::smatrix<YT>& /*data_y*/, const CommonDataT& /*cd*/)noexcept {}

		static constexpr void report_results_end(const real_t /*lossVal*/)noexcept {}

		void on_training_fragment_end(const numel_cnt_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept {
			static constexpr char* szReportFmt = "% 3zd/%-3zd %3.1fs trL=%8.5f, vL=%8.5f";
			static constexpr unsigned uBufSize = 128;
			
			char szRep[uBufSize];
			const real_t secs = real_t(elapsedSincePrevFragment.count()) / real_t(1e9);

			sprintf_s(szRep, uBufSize, szReportFmt, epochEnded+1, m_epochs, secs, trainLoss, testLoss);
			::std::cout << szRep << ::std::endl;
		}

		static constexpr void on_training_end(const nanoseconds& trainTime)noexcept {
			::std::cout << "Training time: " << utils::duration_readable(trainTime) << ::std::endl;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//training_observer, that output results (including number of correct/incorrect classifications) to ::std::cout
	template<typename RealT, typename Evaluator = eval_classification_one_hot_cached<RealT>
		, ::std::enable_if_t<::std::is_arithmetic<RealT>::value, int> = 0//just a guardian for correct instantiation
	>
	class training_observer_stdcout : public _impl::training_observer_stdcout_base<typename Evaluator::real_t>{
		typedef _impl::training_observer_stdcout_base<typename Evaluator::real_t> _base_class_t;

	protected:
		struct CLASSIFICATION_RESULTS {
			numel_cnt_t totalElements;
			numel_cnt_t correctlyClassif;//in fact, it's kind of a metric... Will refactor in future
		};

		//typedef ::std::array<CLASSIFICATION_RESULTS, 2> classif_results_t;
		typedef ::std::vector<CLASSIFICATION_RESULTS> classif_results_t;

		//data sets id used as indexes in classif_results_t
		static_assert(0 <= train_set_id && train_set_id <= 1, "");
		static_assert(0 <= test_set_id && test_set_id <= 1, "");
		static_assert(train_set_id != test_set_id, "");

	public:
		typedef Evaluator evaluator_t;
		static_assert(::std::is_base_of<i_nnet_evaluator<real_t>, Evaluator>::value, "Evaluator class must be derived from i_nnet_evaluator");
		static_assert(::std::is_default_constructible<evaluator_t>::value, "Evaluator class must be default constructible");

	protected:
		classif_results_t m_classifRes;
	public:
		evaluator_t m_evaluator;
	protected:
		DatasetNamingFunc_t m_dsNaming;
		
		numel_cnt_t m_epochs;
		data_set_id_t m_curDataSetId;

	public:
		//these vars are used to query latest loss value from outside
		real_t m_lastTrainErr, m_lastTestErr;

	public:
		template<typename TrainDataT, typename CommonDataT>
		bool init(numel_cnt_t epochs, TrainDataT& td, const CommonDataT& cd)noexcept {
			m_epochs = epochs;
			
			m_curDataSetId = invalid_set_id;
			const auto totalDatasets = td.datasets_count();
			NNTL_ASSERT(totalDatasets >= 2);

			//#exceptions low memory handler here
			m_classifRes.resize(totalDatasets);

			if (totalDatasets > 2) m_dsNaming = td.get_dataset_naming_function();

			return m_evaluator.init(td, cd);
		}
		void deinit()noexcept {
			m_classifRes.clear();

			m_dsNaming = nullptr;

			m_evaluator.deinit();
			_base_class_t::deinit();
		}

		void report_results_begin(const data_set_id_t dataSetId, const numel_cnt_t totalBatches)noexcept {
			NNTL_ASSERT((dataSetId >= 0 && dataSetId < m_classifRes.size()) || !"Unknown dataSetId passed!");

			m_curDataSetId = dataSetId;
			m_classifRes[dataSetId].totalElements = 0;
			m_classifRes[dataSetId].correctlyClassif = 0;
			m_evaluator.prepare_to_dataset(dataSetId, totalBatches);
		}

		template<typename YT, typename CommonDataT>
		void report_results(const numel_cnt_t batchIdx, const realmtx_t& activations, const math::smatrix<YT>& data_y, const CommonDataT& cd)noexcept
		{
			NNTL_UNREF(batchIdx);
			NNTL_ASSERT((m_curDataSetId >= 0 && m_curDataSetId < m_classifRes.size()) || !"WTF?! Invalid m_curDataSetId!");
			NNTL_ASSERT(!data_y.emulatesBiases() && !activations.emulatesBiases());
			NNTL_ASSERT(data_y.rows() == activations.rows()); //to permit supplementary columns in data_y we don't check cols() here

			m_classifRes[m_curDataSetId].totalElements += m_evaluator.totalSamples(data_y);
			m_classifRes[m_curDataSetId].correctlyClassif += m_evaluator.correctlyClassified(m_curDataSetId, data_y, activations, cd.iMath());
		}

		void report_results_end(const real_t lossVal)const noexcept {
			if (m_curDataSetId != train_set_id && m_curDataSetId != test_set_id) {
				NNTL_ASSERT(m_curDataSetId > 1 && m_classifRes.size() > m_curDataSetId && m_classifRes.size() > 2);
				//outputting dataset stats here
				static constexpr char* szReportFmt = "%s: Loss=%05.3f, Err=%.2f%% (%zd)";
				static constexpr unsigned uBufSize = 128;

				char szRep[uBufSize], dsName[LongestDatasetNameWNull];
				m_dsNaming(m_curDataSetId, dsName, LongestDatasetNameWNull);
				
				const auto trainTE = m_classifRes[m_curDataSetId].totalElements, trainW = trainTE - m_classifRes[m_curDataSetId].correctlyClassif;
				const real_t trErr = real_t(trainW * 100) / trainTE;

				sprintf_s(szRep, uBufSize, szReportFmt, dsName, lossVal, trErr, trainW);

				STDCOUTL(szRep);
			}
		}

		/*template<typename NnetT>
		void report_results(const numel_cnt_t epochEnded, const realmtx_t& data_y, const bool bOnTestData, const NnetT& nn)noexcept {
			NNTL_UNREF(epochEnded);

			const auto& activations = nn.get_layer_pack().output_layer().get_activations();
			NNTL_ASSERT(!data_y.emulatesBiases() && !activations.emulatesBiases());
			NNTL_ASSERT(data_y.rows() == activations.rows()); //to permit supplementary columns in data_y we don't check cols() here
			
			m_classifRes[bOnTestData].totalElements = m_evaluator.totalSamples(data_y);
			m_classifRes[bOnTestData].correctlyClassif = m_evaluator.correctlyClassified(data_y, activations, bOnTestData, nn.get_iMath());
		}*/

		void on_training_fragment_end(const numel_cnt_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept {
			static constexpr char* szReportFmt = "% 3zd/%-3zd %3.1fs trL=%05.3f, trErr=%.2f%% (%zd), vL=%05.3f, vErr=%.2f%% (%zd)";
			static constexpr unsigned uBufSize = 128;
			
			m_lastTrainErr = trainLoss;
			m_lastTestErr = testLoss;

			char szRep[uBufSize];
			const real_t secs = real_t(elapsedSincePrevFragment.count()) / real_t(1e9);

			const auto trainTE = m_classifRes[train_set_id].totalElements, trainW = trainTE - m_classifRes[train_set_id].correctlyClassif;
			const real_t trErr = real_t(trainW * 100) / trainTE;
			
			const auto testTE = m_classifRes[test_set_id].totalElements, testW = testTE - m_classifRes[test_set_id].correctlyClassif;
			const real_t tErr = real_t(testW * 100) / testTE;
			
			sprintf_s(szRep, uBufSize, szReportFmt, epochEnded+1, m_epochs, secs, trainLoss, 
				trErr, trainW, testLoss,tErr, testW);
			
			STDCOUTL(szRep);
		}

		void on_training_end(const nanoseconds& trainTime)noexcept {
			::std::cout << "Training time: " << utils::duration_readable(trainTime) << ::std::endl;
		}
	};
}