/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015, Arech (aradvert@gmail.com; https://github.com/Arech)
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

namespace nntl {

	// i_training_observer and derived classes must be default constructible
	struct i_training_observer {
		typedef math_types::realmtx_ty realmtx_t;
		typedef realmtx_t::value_type real_t;
		typedef realmtx_t::vec_len_t vec_len_t;
		typedef realmtx_t::numel_cnt_t numel_cnt_t;
		
		typedef std::chrono::nanoseconds nanoseconds;

		//may preprocess train_y/test_y here
		template<typename iMath>
		nntl_interface bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept;
		nntl_interface void deinit()noexcept;

		//always called before on_training_fragment_end() twice: on training and on testing/validation data
		//data_y must be the same as init(train_y)|bOnTestData==false or init(test_y)|bOnTestData==true
		template<typename iMath>
		nntl_interface void inspect_results(const realmtx_t& data_y, const realmtx_t& activations, const bool bOnTestData, iMath& iM)noexcept;

		nntl_interface void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nL)noexcept;
		//epochEnded is zero based, so the first epoch is 0. Will be ==-1 on initial report (before training begins)
		nntl_interface void on_training_fragment_end(size_t epochEnded, real_t trainLoss, real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept;
		nntl_interface void on_training_end(const nanoseconds& trainTime)noexcept;
	};

	struct training_observer_silent : public i_training_observer {
		template<typename iMath>
		bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept { return true; }
		void deinit()noexcept {}

		//always called before on_training_fragment_end() twice: on training and on testing/validation data
		template<typename iMath>
		void inspect_results(const realmtx_t& data_y, const realmtx_t& activations, const bool bOnTestData, iMath& iM)noexcept {}

		void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nL)noexcept {}
		void on_training_fragment_end(size_t epochEnded, real_t trainLoss, real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept{}
		void on_training_end(const nanoseconds& trainTime)noexcept {}
	};


	//training_observer, that output results to std::cout
	class training_observer_stdcout : public i_training_observer {
	protected:
		struct CLASSIFICATION_RESULTS {
			size_t totalElements;
			size_t correctlyClassified;
		};

		typedef std::array<CLASSIFICATION_RESULTS, 2> classif_results_t;
		//for each element of Y data (training/testing) contains index of true element class (column number of biggest element in a row)
		typedef std::array<std::vector<vec_len_t>, 2> y_data_class_idx_t;

	public:
		template<typename iMath>
		bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept {
			m_epochs = epochs;
			m_prevEpoch = 0;
			
			//TODO:exception handling!
			m_ydataClassIdxs[0].resize(train_y.rows());
			m_nnClassIdxs[0].resize(train_y.rows());
			iM.mFindIdxsOfMaxRowwise(train_y, m_ydataClassIdxs[0]);

			m_ydataClassIdxs[1].resize(test_y.rows());
			m_nnClassIdxs[1].resize(test_y.rows());
			iM.mFindIdxsOfMaxRowwise(test_y, m_ydataClassIdxs[1]);
			return true;
		}
		void deinit()noexcept {
			for (unsigned i = 0; i <= 1; ++i) {
				m_ydataClassIdxs[i].clear();
				m_nnClassIdxs[i].clear();
			}
		}

		void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nLP)noexcept {
			static constexpr strchar_t* szReportFmt = "Going to model f:%d->%d with %zd params on %d training samples (%d validation). BatchSize=%d";
			static constexpr unsigned uBufSize = 192;

			strchar_t szRep[uBufSize];
			sprintf_s(szRep, uBufSize, szReportFmt, inDim, outDim, nLP, trainElements, testElements, batchSize);
			std::cout << szRep << std::endl;
		}

		//always called before on_training_fragment_end() twice: on training and on testing/validation data
		//data_y must be the same as init(train_y)|bOnTestData==false or init(test_y)|bOnTestData==true
		template<typename iMath>
		void inspect_results(const realmtx_t& data_y, const realmtx_t& activations, const bool bOnTestData, iMath& iM)noexcept {
			NNTL_ASSERT(data_y.size() == activations.size());
			NNTL_ASSERT(data_y.rows() == m_ydataClassIdxs[bOnTestData].size());

			iM.mFindIdxsOfMaxRowwise(activations, m_nnClassIdxs[bOnTestData]);

			m_classifRes[bOnTestData].totalElements = data_y.rows();
			m_classifRes[bOnTestData].correctlyClassified = iM.vCountSame(m_ydataClassIdxs[bOnTestData],m_nnClassIdxs[bOnTestData]);
		}

		void on_training_fragment_end(size_t epochEnded, real_t trainLoss, real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept {
			static constexpr strchar_t* szReportFmt = "% 3zd/%3zd %3.1fs trL=%05.3f, trErr=%.2f%% (%zd), vL=%05.3f, vErr=%.2f%% (%zd)";
			static constexpr unsigned uBufSize = 128;

			++epochEnded;

			strchar_t szRep[uBufSize];
			const real_t secs = real_t(elapsedSincePrevFragment.count()) / real_t(1e9);

			const auto trainTE = m_classifRes[0].totalElements, trainW=trainTE- m_classifRes[0].correctlyClassified;
			//trainCC = m_classifRes[0].correctlyClassified,
			const real_t trErr = static_cast<real_t>(100 * trainW) / trainTE;
			
			const auto testTE = m_classifRes[1].totalElements, testW = testTE - m_classifRes[1].correctlyClassified;
			//testCC = m_classifRes[1].correctlyClassified, 
			const real_t tErr = static_cast<real_t>(100 * testW) / testTE;
			
			sprintf_s(szRep, uBufSize, szReportFmt, epochEnded, m_epochs, secs, trainLoss, 
				trErr, trainW, testLoss,tErr, testW);
			
			std::cout << szRep << std::endl;

			m_prevEpoch = epochEnded;
		}

		void on_training_end(const nanoseconds& trainTime)noexcept {
			std::cout << "Training time: " << utils::duration_readable(trainTime) << std::endl;
		}

	protected:
		classif_results_t m_classifRes;
		y_data_class_idx_t m_ydataClassIdxs;//preprocessed ground truth
		y_data_class_idx_t m_nnClassIdxs;//storage for NN predictions

		size_t m_epochs, m_prevEpoch;
	};
}