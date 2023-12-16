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

//this file defines an interface to classes, that processes ground truth information and nnet prediction about it and
//produces some kind of goodness-of-fit metric 

#include <nntl/_defs.h>

namespace nntl {

	//Attention: regarding the evaluator classes naming - always use '_cached' suffix when evaluator caches train_y or test_y!

	template<typename RealT>
	struct i_nnet_evaluator : public virtual DataSetsId {
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;

		template<typename TrainDataT, typename CommonDataT>
		nntl_interface bool init(TrainDataT& td, const CommonDataT& cd)noexcept;
		nntl_interface void deinit()noexcept;

		nntl_interface void prepare_to_dataset(const data_set_id_t dataSetId, const numel_cnt_t totalBatches)noexcept;

		//Note that correctlyClassified() is expected to be called sequentially for each data batch in a set.
		template<typename iMath>
		nntl_interface numel_cnt_t correctlyClassified(const data_set_id_t dataSetId, const realmtx_t& data_y
			, const realmtx_t& activations, iMath& iM)noexcept;

		nntl_interface numel_cnt_t totalSamples(const realmtx_t& data_y)noexcept;
	};


	//////////////////////////////////////////////////////////////////////////
	// simple evaluator to be used with binary classification tasks, when data_y may contain more than one "turned on" element
	// Note that data_y as well as activations to test are binarized using the same threshold
	// NOTE: evaluator performs train_y and test_y caching, so you can't safely change them during training
	template<typename RealT>
	struct eval_classification_binary_cached : public i_nnet_evaluator<RealT> {
	protected:
		typedef math::smatrix_deform<char> binmtx_t;
		//typedef ::std::vector<char> binvec_t;//char instead of bool should be a bit faster (not tested though)
		
		//note that we're preprocessing data_y to reserve less memory for storing postprocessed activations and to compare
		//them with postprocessed data_y faster

		//for each element of Y data (training/testing) contains binary flag if it's turned on or off
		typedef ::std::array<binmtx_t, 2> y_data_class_idx_t;
		//note that we're caching only two main datasets (train & test), other datasets will be processed without caching

	protected:
		//storage for
		y_data_class_idx_t m_ydataPP;//preprocessed ground truth labels
		y_data_class_idx_t m_predictionsPP_orYData;//NN predictions and not preprocessed ground truth
		
		//data sets id used as indexes in m_ydataPP
		static_assert(0 <= train_set_id && train_set_id <= 1, "");
		static_assert(0 <= test_set_id && test_set_id <= 1, "");
		static_assert(train_set_id != test_set_id, "");

		real_t m_binarizeThreshold;
		vec_len_t m_curYOfs;

		NNTL_DEBUG_DECLARE(vec_len_t m_maxOutpBS);

	public:
		~eval_classification_binary_cached()noexcept {}
		eval_classification_binary_cached()noexcept : m_binarizeThreshold(.5) {}

		void set_threshold(real_t t)noexcept { m_binarizeThreshold = t; }
		real_t threshold()const noexcept { return m_binarizeThreshold; }

		template<typename TrainDataT, typename CommonDataT>
		bool init(TrainDataT& td, const CommonDataT& cd)noexcept {
			static_assert(is_train_data_intf<TrainDataT>::value, "td object MUST be derived from _i_train_data interface");
			static_assert(TrainDataT::allowExternalCachingOfSets, "This evaluator needs to cache Y-data, but TrainDataT prohibits that");

			//we may rely on dataset_samples_count() to return true size of dataset, because TrainDataT::allowExternalCachingOfSets==true
			NNTL_ASSERT(!td.empty());

			const auto trainSamples = td.trainset_samples_count();
			const auto testSamples = td.testset_samples_count();

			if (::std::max(trainSamples, testSamples) > ::std::numeric_limits<vec_len_t>::max()) {
				NNTL_ASSERT(!"Too big dataset to work with _cached nnet_evaluator!");
				return false;
			}
			
			const auto maxOutpBS = cd.get_outBatchSizes().biggest();
			const auto ySize = td.yWidth();
			
			if (!m_ydataPP[train_set_id].resize(static_cast<vec_len_t>(trainSamples), ySize)//will contain preprocessed Y for training set
				|| !m_ydataPP[test_set_id].resize(static_cast<vec_len_t>(testSamples), ySize)//and for test set
				|| !m_predictionsPP_orYData[0].resize(maxOutpBS, ySize)//will binarize activations in here
				|| (td.datasets_count() > 2 && !m_predictionsPP_orYData[1].resize(maxOutpBS, ySize)))//will binarize dataY for supplementary datasets in here
			{
				deinit();
				return false;
			}

			NNTL_DEBUG_DECLARE(m_maxOutpBS = maxOutpBS);//just to make sure it won't change since init()
			
			binarizeYOfDataset(train_set_id, td, m_ydataPP[train_set_id], cd);
			binarizeYOfDataset(test_set_id, td, m_ydataPP[test_set_id], cd);
			return true;
		}

		template<typename TrainDataT, typename CommonDataT>
		void binarizeYOfDataset(const data_set_id_t dataSetId, TrainDataT& td, binmtx_t& dest, const CommonDataT& cd)const noexcept {
			NNTL_ASSERT(dest.rows() == td.dataset_samples_count(dataSetId));

			auto& iM = cd.iMath();
			const auto maxBatches = td.walk_over_set(dataSetId, cd, -1, td.flag_exclude_dataX);
			vec_len_t rOfs = 0;
			for (numel_cnt_t bi = 0; bi < maxBatches; ++bi) {
				td.next_subset(bi, cd);

				iM.ewBinarizeBatch(dest, rOfs, td.batchY(), m_binarizeThreshold);
				rOfs += td.batchY().rows();
			}
			NNTL_ASSERT(rOfs == td.dataset_samples_count(dataSetId));
			NNTL_ASSERT(dest._isBinaryStrictNoBias());
		}

		void deinit()noexcept {
			for (int i = 0; i <= 1; ++i) {
				m_ydataPP[i].clear();
				m_predictionsPP_orYData[i].clear();
			}
		}

		void prepare_to_dataset(const data_set_id_t dataSetId, const numel_cnt_t totalBatches)noexcept {
			NNTL_UNREF(dataSetId); NNTL_UNREF(totalBatches);
			m_curYOfs = 0;
		}

		//returns a count of correct predictions, i.e. (True Positive + True Negative)'s
		//note that no matrix may have more rows, than were defined by cd.max_fprop_batch_size() during init()
		template<typename iMath>
		numel_cnt_t correctlyClassified(const data_set_id_t dataSetId, const realmtx_t& data_y, const realmtx_t& activations, iMath& iM)noexcept {
			NNTL_ASSERT(data_y.size() == activations.size());
			NNTL_ASSERT(data_y.rows() <= m_maxOutpBS);
			NNTL_ASSERT(dataSetId > 1 || data_y.cols() == m_ydataPP[dataSetId].cols());

			m_predictionsPP_orYData[0].deform_like(activations);
			iM.ewBinarize(m_predictionsPP_orYData[0], activations, m_binarizeThreshold);

			numel_cnt_t ret;
			if (dataSetId <= 1) {
				ret = iM.vCountSameBatch(m_ydataPP[dataSetId], m_curYOfs, m_predictionsPP_orYData[0]);
				m_curYOfs += activations.rows();
				NNTL_ASSERT(m_curYOfs <= m_ydataPP[dataSetId].rows());
			} else {
				//processing data_y into m_predictionsPP_orYData[1]
				m_predictionsPP_orYData[1].deform_like(data_y);
				iM.ewBinarize(m_predictionsPP_orYData[1], data_y, m_binarizeThreshold);
				ret = iM.vCountSame(m_predictionsPP_orYData[1], m_predictionsPP_orYData[0]);
			}			
			return ret;
		}

		static numel_cnt_t totalSamples(const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(!data_y.emulatesBiases());
			return data_y.numel();
		}
	};



	//////////////////////////////////////////////////////////////////////////
	//evaluator to be used with classification tasks when data_y class is specified by a column with a greatest value
	// (one-hot generalization)
	// NOTE: evaluator performs train_y and test_y caching, so you can't safely change them during training
	template<typename RealT>
	struct eval_classification_one_hot_cached : public i_nnet_evaluator<RealT> {
	protected:
		//for each element of Y data (training/testing) contains index of true element class (column number of biggest element in a row)
		typedef ::std::vector<vec_len_t> idxVec_t;
		//typedef math::smatrix_deform<vec_len_t> idxVec_t;
		typedef ::std::array<idxVec_t, 2> y_data_class_idx_t;

		//data sets id used as indexes in y_data_class_idx_t
		static_assert(0 <= train_set_id && train_set_id <= 1, "");
		static_assert(0 <= test_set_id && test_set_id <= 1, "");
		static_assert(train_set_id != test_set_id, "");

	protected:
		y_data_class_idx_t m_ydataClassIdxs;//preprocessed ground truth for train&test sets
		y_data_class_idx_t m_predictionClassOrYDataIdxs;//storage for NN predictions and GT of other sets

		vec_len_t m_curYOfs;

		NNTL_DEBUG_DECLARE(vec_len_t m_biggestBatch);

	public:
		~eval_classification_one_hot_cached()noexcept {}
		eval_classification_one_hot_cached()noexcept {}

		template<typename TrainDataT, typename CommonDataT>
		bool init(TrainDataT& td, const CommonDataT& cd)noexcept {
			static_assert(is_train_data_intf<TrainDataT>::value, "td object MUST be derived from _i_train_data interface");
			static_assert(TrainDataT::allowExternalCachingOfSets, "This evaluator needs to cache Y-data, but TrainDataT prohibits that");
			
			//we may rely on dataset_samples_count() to return true size of dataset, because TrainDataT::allowExternalCachingOfSets==true
			NNTL_ASSERT(!td.empty());

			const auto trainSamples = td.trainset_samples_count();
			const auto testSamples = td.testset_samples_count();

			if (::std::max(trainSamples, testSamples) > ::std::numeric_limits<vec_len_t>::max()) {
				NNTL_ASSERT(!"Too big dataset to work with _cached nnet_evaluator!");
				return false;
			}

			const auto biggestOutpBatch = cd.get_outBatchSizes().biggest();
			NNTL_DEBUG_DECLARE(m_biggestBatch = biggestOutpBatch);

			auto& iM = cd.iMath();
			iM.preinit(iM.mrwIdxsOfMax_needTempMem<real_t>(biggestOutpBatch));
			iM.init();

			try {
				m_ydataClassIdxs[train_set_id].resize(trainSamples);
				m_ydataClassIdxs[test_set_id].resize(testSamples);
				m_predictionClassOrYDataIdxs[0].resize(biggestOutpBatch);
				if (td.datasets_count() > 2) m_predictionClassOrYDataIdxs[1].resize(biggestOutpBatch);
			}catch(const ::std::exception&){
				NNTL_ASSERT(!"Exception caught while resizing vectors in eval_classification_one_hot_cached::init");
				deinit();
				return false;
			}
			
			maxYOfDataset(train_set_id, td, m_ydataClassIdxs[train_set_id], cd);
			maxYOfDataset(test_set_id, td, m_ydataClassIdxs[test_set_id], cd);
			return true;
		}

		template<typename TrainDataT, typename CommonDataT>
		void maxYOfDataset(const data_set_id_t dataSetId, TrainDataT& td, idxVec_t& dest, const CommonDataT& cd)const noexcept {
			NNTL_ASSERT(conform_sign(dest.size()) == td.dataset_samples_count(dataSetId));

			auto& iM = cd.iMath();
			const auto maxBatches = td.walk_over_set(dataSetId, cd, -1, td.flag_exclude_dataX);
			vec_len_t rOfs = 0;
			for (numel_cnt_t bi = 0; bi < maxBatches; ++bi) {
				td.next_subset(bi, cd);

				iM.mrwIdxsOfMax(td.batchY(), &dest[rOfs]);
				rOfs += td.batchY().rows();
			}
			NNTL_ASSERT(rOfs == td.dataset_samples_count(dataSetId));
		}

		void deinit()noexcept {
			for (int i = 0; i <= 1; ++i) {
				m_ydataClassIdxs[i].clear();
				m_ydataClassIdxs[i].shrink_to_fit();
				m_predictionClassOrYDataIdxs[i].clear();
				m_predictionClassOrYDataIdxs[i].shrink_to_fit();
			}
		}

		void prepare_to_dataset(const data_set_id_t dataSetId, const numel_cnt_t totalBatches)noexcept {
			NNTL_UNREF(dataSetId); NNTL_UNREF(totalBatches);
			m_curYOfs = 0;
		}

		template<typename iMath>
		numel_cnt_t correctlyClassified(const data_set_id_t dataSetId, const realmtx_t& data_y, const realmtx_t& activations, iMath& iM)noexcept {
			NNTL_ASSERT(data_y.size() == activations.size());
			NNTL_ASSERT(data_y.batch_size() <= m_biggestBatch);

			NNTL_ASSERT(conform_sign(m_predictionClassOrYDataIdxs[0].capacity()) >= activations.batch_size());
			m_predictionClassOrYDataIdxs[0].resize(activations.rows());
			iM.mrwIdxsOfMax(activations, &m_predictionClassOrYDataIdxs[0][0]);

			numel_cnt_t ret;
			if (dataSetId <= 1) {
				ret = iM.vCountSame(m_ydataClassIdxs[dataSetId], m_predictionClassOrYDataIdxs[0], m_curYOfs);
				m_curYOfs += activations.rows();
				NNTL_ASSERT(m_curYOfs <= m_ydataClassIdxs[dataSetId].size());
			} else {
				//processing data_y into m_predictionsPP_orYData[1]
				NNTL_ASSERT(conform_sign(m_predictionClassOrYDataIdxs[1].capacity()) >= data_y.rows());
				m_predictionClassOrYDataIdxs[1].resize(data_y.rows());
				iM.mrwIdxsOfMax(data_y, &m_predictionClassOrYDataIdxs[1][0]);
				ret = iM.vCountSame(m_predictionClassOrYDataIdxs[1], m_predictionClassOrYDataIdxs[0]);
			}
			return ret;
		}
		
		static numel_cnt_t totalSamples(const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(!data_y.emulatesBiases());
			return data_y.rows();
		}
	};

}