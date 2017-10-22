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

#include <array>
#include <vector>

#include "utils/chrono.h"

#include "nnet_evaluator.h"

namespace nntl {

	// i_training_observer and derived classes must be default constructible
	template<typename RealT>
	struct i_training_observer {
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_td::vec_len_t vec_len_t;
		typedef math::smatrix_td::numel_cnt_t numel_cnt_t;
		
		typedef ::std::chrono::nanoseconds nanoseconds;

		//may preprocess train_y/test_y here
		template<typename iMath>
		nntl_interface bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept;
		nntl_interface void deinit()noexcept;

		//always called before on_training_fragment_end() twice: on the training and on the testing data
		// Call sequence <inspect_results(bOnTestData==false) 
		// + inspect_results(bOnTestData==true) + on_training_fragment_end()> is guaranteed for every epoch to be evaluated
		// data_y is the same as init(train_y)|bOnTestData==false or init(test_y)|bOnTestData==true
		template<typename NnetT>
		nntl_interface void inspect_results(const size_t epochEnded, const realmtx_t& data_y, const bool bOnTestData, const NnetT& nn)noexcept;

		nntl_interface void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nL)noexcept;
		//epochEnded is zero based, so the first epoch is 0. Will be ==-1 on initial report (before training begins)
		nntl_interface void on_training_fragment_end(const size_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept;
		nntl_interface void on_training_end(const nanoseconds& trainTime)noexcept;
	};

	template<typename RealT=d_interfaces::real_t>
	struct training_observer_silent : public i_training_observer<RealT> {
		template<typename iMath>
		constexpr bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)const noexcept { return true; }
		void deinit()const noexcept {}

		//always called before on_training_fragment_end() twice: on the training and on the testing data
		template<typename NnetT>
		void inspect_results(const size_t epochEnded, const realmtx_t& data_y, const bool bOnTestData, const NnetT& nn)const noexcept {}

		void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nL)const noexcept {}
		void on_training_fragment_end(const size_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)const noexcept{}
		void on_training_end(const nanoseconds& trainTime)const noexcept {}
	};

	//////////////////////////////////////////////////////////////////////////
	//simple observer, that outputs only loss function value to ::std::cout
	template<typename RealT = d_interfaces::real_t>
	class training_observer_simple_stdcout : public i_training_observer<RealT> {
	protected:
		size_t m_epochs;

	public:
		template<typename iMath>
		bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept {
			m_epochs = epochs;
			return true;
		}
		void deinit()noexcept { }

		void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nLP)noexcept {
			static constexpr strchar_t* szReportFmt = "Going to model f:%d->%d with %zd params on %d training samples (%d validation). BatchSize=%d, batchCnt=%d";
			static constexpr unsigned uBufSize = 192;

			strchar_t szRep[uBufSize];
			sprintf_s(szRep, uBufSize, szReportFmt, inDim, outDim, nLP, trainElements, testElements, batchSize, trainElements / batchSize);
			::std::cout << szRep << ::std::endl;
		}

		//always called before on_training_fragment_end() twice: on the training and on the testing data
		//data_y must be the same as init(train_y)|bOnTestData==false or init(test_y)|bOnTestData==true
		template<typename NnetT>
		void inspect_results(const size_t epochEnded, const realmtx_t& data_y, const bool bOnTestData, const NnetT& nn)noexcept { }

		void on_training_fragment_end(const size_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept {
			static constexpr strchar_t* szReportFmt = "% 3zd/%-3zd %3.1fs trL=%8.5f, vL=%8.5f";
			static constexpr unsigned uBufSize = 128;
			
			strchar_t szRep[uBufSize];
			const real_t secs = real_t(elapsedSincePrevFragment.count()) / real_t(1e9);

			sprintf_s(szRep, uBufSize, szReportFmt, epochEnded+1, m_epochs, secs, trainLoss, testLoss);
			::std::cout << szRep << ::std::endl;
		}

		void on_training_end(const nanoseconds& trainTime)noexcept {
			::std::cout << "Training time: " << utils::duration_readable(trainTime) << ::std::endl;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//training_observer, that output results to ::std::cout
	template<typename Evaluator = eval_classification_one_hot<d_interfaces::real_t>>
	class training_observer_stdcout : public i_training_observer<typename Evaluator::real_t> {
	protected:
		struct CLASSIFICATION_RESULTS {
			size_t totalElements;
			size_t correctlyClassified;//in fact, it's kind of a metric... Will refactor in future
		};

		typedef ::std::array<CLASSIFICATION_RESULTS, 2> classif_results_t;
	public:
		typedef Evaluator evaluator_t;
		static_assert(::std::is_base_of<i_nnet_evaluator<real_t>, Evaluator>::value, "Evaluator class must be derived from i_nnet_evaluator");
		static_assert(::std::is_default_constructible<evaluator_t>::value, "Evaluator class must be default constructible");

	protected:
		classif_results_t m_classifRes;
	public:
		evaluator_t m_evaluator;
	protected:
		size_t m_epochs;

	public:
		real_t m_lastTrainErr, m_lastTestErr;

	public:
		template<typename iMath>
		bool init(size_t epochs, const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept {
			m_epochs = epochs;
			return m_evaluator.init(train_y, test_y, iM);
		}
		void deinit()noexcept {
			m_evaluator.deinit();
		}

		void on_training_start(vec_len_t trainElements, vec_len_t testElements, vec_len_t inDim, vec_len_t outDim, vec_len_t batchSize, numel_cnt_t nLP)noexcept {
			static constexpr strchar_t* szReportFmt = "Going to model f:%d->%d with %zd params on %d training samples (%d validation). BatchSize=%d";
			static constexpr unsigned uBufSize = 192;

			strchar_t szRep[uBufSize];
			sprintf_s(szRep, uBufSize, szReportFmt, inDim, outDim, nLP, trainElements, testElements, batchSize);
			::std::cout << szRep << ::std::endl;
		}

		//always called before on_training_fragment_end() twice: on the training and on the testing data
		//data_y is the same as init(train_y)|bOnTestData==false or init(test_y)|bOnTestData==true
		template<typename NnetT>
		void inspect_results(const size_t epochEnded, const realmtx_t& data_y, const bool bOnTestData, const NnetT& nn)noexcept {
			NNTL_UNREF(epochEnded);

			const auto& activations = nn.get_layer_pack().output_layer().get_activations();
			NNTL_ASSERT(data_y.size() == activations.size());
			
			m_classifRes[bOnTestData].totalElements = data_y.rows();
			m_classifRes[bOnTestData].correctlyClassified = m_evaluator.correctlyClassified(data_y, activations, bOnTestData, nn.get_iMath());
		}

		void on_training_fragment_end(const size_t epochEnded, const real_t trainLoss, const real_t testLoss, const nanoseconds& elapsedSincePrevFragment)noexcept {
			static constexpr strchar_t* szReportFmt = "% 3zd/%-3zd %3.1fs trL=%05.3f, trErr=%.2f%% (%zd), vL=%05.3f, vErr=%.2f%% (%zd)";
			static constexpr unsigned uBufSize = 128;
			
			m_lastTrainErr = trainLoss;
			m_lastTestErr = testLoss;

			strchar_t szRep[uBufSize];
			const real_t secs = real_t(elapsedSincePrevFragment.count()) / real_t(1e9);

			const auto trainTE = m_classifRes[0].totalElements, trainW = trainTE - m_classifRes[0].correctlyClassified;
			const real_t trErr = real_t(trainW * 100) / trainTE;
			
			const auto testTE = m_classifRes[1].totalElements, testW = testTE - m_classifRes[1].correctlyClassified;
			const real_t tErr = real_t(testW * 100) / testTE;
			
			sprintf_s(szRep, uBufSize, szReportFmt, epochEnded+1, m_epochs, secs, trainLoss, 
				trErr, trainW, testLoss,tErr, testW);
			
			::std::cout << szRep << ::std::endl;
		}

		void on_training_end(const nanoseconds& trainTime)noexcept {
			::std::cout << "Training time: " << utils::duration_readable(trainTime) << ::std::endl;
		}
	};
}