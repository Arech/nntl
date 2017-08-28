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

//this file defines an interface to classes, that processes ground truth information and nnet prediction about it and
//produces some kind of goodness-of-fit metric 

#include "_defs.h"

namespace nntl {

	template<typename RealT>
	struct i_nnet_evaluator {
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_td::vec_len_t vec_len_t;
		typedef math::smatrix_td::numel_cnt_t numel_cnt_t;

		//may preprocess train_y/test_y here
		template<typename iMath>
		nntl_interface bool init(const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept;
		nntl_interface void deinit()noexcept;

		//define correct returning type
		template<typename iMath>
		nntl_interface size_t correctlyClassified(const realmtx_t& data_y, const realmtx_t& activations, const bool bOnTestData, iMath& iM)noexcept;

	};


	//////////////////////////////////////////////////////////////////////////
	//simple evaluator to be used with binary classification tasks, when data_y may contain more than one "turned on" element
	//data_y is binarized according to threshold
	// #TODO: it's not really suitable for more than 1D data (need correct metrics). So this code as well as observer's code will be heavily
	// refactored
	template<typename RealT>
	struct eval_classification_binary : public i_nnet_evaluator<RealT> {
	protected:
		//typedef math::smatrix<char> binmtx_t;
		typedef ::std::vector<char> binvec_t;

		//for each element of Y data (training/testing) contains binary flag if it's turned on or off
		typedef ::std::array<binvec_t, 2> y_data_class_idx_t; //char instead of bool should be a bit faster (not tested)

	protected:
		y_data_class_idx_t m_ydataPP;//preprocessed ground truth
		y_data_class_idx_t m_predictionsPP;//storage for NN predictions

		real_t m_binarizeThreshold;

	public:
		~eval_classification_binary()noexcept {}
		eval_classification_binary()noexcept : m_binarizeThreshold(.5) {}

		void set_threshold(real_t t)noexcept { m_binarizeThreshold = t; }
		real_t threshold()const noexcept { return m_binarizeThreshold; }

		template<typename iMath>
		bool init(const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept {
			NNTL_ASSERT(train_y.cols() == test_y.cols() && test_y.cols() == 1);

			//if (!m_ydataPP[0].resize(train_y.size()) || !m_ydataPP[1].resize(test_y.size())
			//	|| !m_predictionsPP[0].resize(train_y.size()) || !m_predictionsPP[1].resize(test_y.size()) ) return false;
			m_ydataPP[0].resize(train_y.rows());
			m_ydataPP[1].resize(test_y.rows());
			m_predictionsPP[0].resize(train_y.rows());
			m_predictionsPP[1].resize(test_y.rows());
			
			iM.ewBinarize(m_ydataPP[0], train_y, m_binarizeThreshold);
			iM.ewBinarize(m_ydataPP[1], test_y, m_binarizeThreshold);			
			return true;
		}

		void deinit()noexcept {
			for (unsigned i = 0; i <= 1; ++i) {
				m_ydataPP[i].clear();
				m_predictionsPP[i].clear();
			}
		}

		//returns a count of correct predictions, i.e. (True Positive + True Negative)'s
		template<typename iMath>
		size_t correctlyClassified(const realmtx_t& data_y, const realmtx_t& activations, const bool bOnTestData, iMath& iM)noexcept {
			NNTL_UNREF(data_y);
			NNTL_ASSERT(data_y.size() == activations.size());
			NNTL_ASSERT(data_y.rows() == m_ydataPP[bOnTestData].size());

			iM.ewBinarize(m_predictionsPP[bOnTestData], activations, m_binarizeThreshold);
			return iM.vCountSame(m_ydataPP[bOnTestData], m_predictionsPP[bOnTestData]);
		}
	};



	//////////////////////////////////////////////////////////////////////////
	//evaluator to be used with classification tasks when data_y class is specified by a column with a greatest value
	// (one-hot generalization)
	template<typename RealT>
	struct eval_classification_one_hot : public i_nnet_evaluator<RealT> {
	protected:
		//for each element of Y data (training/testing) contains index of true element class (column number of biggest element in a row)
		typedef ::std::array<::std::vector<vec_len_t>, 2> y_data_class_idx_t;

	protected:
		y_data_class_idx_t m_ydataClassIdxs;//preprocessed ground truth
		y_data_class_idx_t m_predictionClassIdxs;//storage for NN predictions

	public:
		~eval_classification_one_hot()noexcept {}
		eval_classification_one_hot()noexcept {}

		template<typename iMath>
		bool init(const realmtx_t& train_y, const realmtx_t& test_y, iMath& iM)noexcept {
			NNTL_ASSERT(train_y.cols() == test_y.cols());

			//#TODO:exception handling!
			m_ydataClassIdxs[0].resize(train_y.rows());
			m_predictionClassIdxs[0].resize(train_y.rows());
			iM.mrwIdxsOfMax(train_y, &m_ydataClassIdxs[0][0]);

			m_ydataClassIdxs[1].resize(test_y.rows());
			m_predictionClassIdxs[1].resize(test_y.rows());
			iM.mrwIdxsOfMax(test_y, &m_ydataClassIdxs[1][0]);
			return true;
		}

		void deinit()noexcept {
			for (unsigned i = 0; i <= 1; ++i) {
				m_ydataClassIdxs[i].clear();
				m_predictionClassIdxs[i].clear();
			}
		}

		//returns a count of correct predictions, i.e. (True Positive + True Negative)'s
		template<typename iMath>
		size_t correctlyClassified(const realmtx_t& data_y, const realmtx_t& activations, const bool bOnTestData, iMath& iM)noexcept {
			NNTL_UNREF(data_y);
			NNTL_ASSERT(data_y.size() == activations.size());
			NNTL_ASSERT(data_y.rows() == m_ydataClassIdxs[bOnTestData].size());

			iM.mrwIdxsOfMax(activations, &m_predictionClassIdxs[bOnTestData][0]);
			return iM.vCountSame(m_ydataClassIdxs[bOnTestData], m_predictionClassIdxs[bOnTestData]);
		}
	};

}