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

#include "training_observer.h"
#include "./utils/vector_conditions.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	//structure to hold the results of NNet evaluation
	template<typename RealT>
	struct nnet_eval_results {
		typedef RealT real_t;

		math::smatrix<real_t> output_activations;
		real_t lossValue;

		nnet_eval_results()noexcept:lossValue(0) {}

		void reset()noexcept {
			output_activations.clear();
			lossValue = 0;
		}

		const bool operator==(const nnet_eval_results& rhs)const noexcept {
			return lossValue == rhs.lossValue && output_activations == rhs.output_activations;
		}
		const bool operator!=(const nnet_eval_results& rhs)const noexcept {
			return !operator==(rhs);
		}
	};

	template<typename RealT>
	struct nnet_td_eval_results {
		typedef RealT real_t;

		nnet_eval_results<real_t> trainSet;
		nnet_eval_results<real_t> testSet;

		void reset()noexcept {
			trainSet.reset();
			testSet.reset();
		}

		const bool operator==(const nnet_td_eval_results& rhs)const noexcept {
			return trainSet == rhs.trainSet && testSet == rhs.testSet;
		}
		const bool operator!=(const nnet_td_eval_results& rhs)const noexcept {
			return !operator==(rhs);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// options of training algo
	template <typename TrainingObserver = training_observer_stdcout<>>
	class nnet_train_opts : public math::smatrix_td {
		static_assert(::std::is_base_of<i_training_observer<typename TrainingObserver::real_t>, TrainingObserver>::value,
			"TrainingObserver template parameter must be derived from i_training_observer");

	public:
		typedef TrainingObserver training_observer_t;
		typedef nnet_train_opts<training_observer_t> self_t;
		
		typedef typename training_observer_t::real_t real_t;
		typedef typename training_observer_t::realmtx_t realmtx_t;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		training_observer_t m_trainingObserver;

		nnet_td_eval_results<real_t>* m_pNNEvalFinalRes;

		vector_conditions m_vbEpochEval;

		//DivergenceCheck* vars describes how to check for nn algo divergence
		real_t m_DivergenceCheckThreshold;

		vec_len_t m_BatchSize;

		int16_t m_DivergenceCheckLastEpoch;//set to zero to turn off divergence check

		bool m_bCalcFullLossValue;//if set to false, then only the main part of loss function will be calculated 
		// (i.e. additional summands such as L2 weight penalty value will be stripped.
		// This is done by skipping _layer_base::lossAddendum() calls for all layers)
		
		//setting to false will prevent nnet and all its components from deinitialization (except for some temporary memory, that 
		// don't store anything valuable) on .train() exit
		bool m_bImmediatelyDeinit;

		//set this flag to true to skip forward pass during training set error calculation in full-batch mode
		//This will make error value report slightly wrong (errVal corresponds to the previous pass), but will make a significant speedup
		bool m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch;

		void _ctor()noexcept {
			m_BatchSize = 0;
			m_DivergenceCheckLastEpoch = 5;
			m_DivergenceCheckThreshold = real_t(1e5);
			m_bCalcFullLossValue = true;
			m_bImmediatelyDeinit = false;
			m_pNNEvalFinalRes = nullptr;
			m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch = false;
		}

	public:
		~nnet_train_opts()noexcept {}
		nnet_train_opts(size_t _maxEpoch, const bool& defVal = true)noexcept : m_vbEpochEval(_maxEpoch, defVal)
		{_ctor();}
		nnet_train_opts(size_t _maxEpoch, size_t stride)noexcept : m_vbEpochEval(_maxEpoch, stride)
		{		_ctor();	}
		nnet_train_opts(size_t _maxEpoch, size_t startsAt, size_t stride)noexcept : m_vbEpochEval(_maxEpoch, startsAt, stride)
		{		_ctor();	}


		size_t maxEpoch()const noexcept { return m_vbEpochEval.maxEpoch(); }
		vector_conditions& getCondEpochEval() noexcept { return m_vbEpochEval; }
		const vector_conditions& getCondEpochEval()const noexcept { return m_vbEpochEval; }

		real_t divergenceCheckThreshold() const noexcept { return m_DivergenceCheckThreshold; }
		self_t& divergenceCheckThreshold(real_t val) noexcept { m_DivergenceCheckThreshold = val; return *this; }

		size_t divergenceCheckLastEpoch() const noexcept { return m_DivergenceCheckLastEpoch; }
		self_t& divergenceCheckLastEpoch(int16_t val) noexcept { m_DivergenceCheckLastEpoch = val; return *this; }

		vec_len_t batchSize() const noexcept { return m_BatchSize; }
		self_t& batchSize(vec_len_t val) noexcept { m_BatchSize = val; return *this; }

		training_observer_t& observer() noexcept { return m_trainingObserver; }

		bool calcFullLossValue()const noexcept { return m_bCalcFullLossValue; }
		self_t& calcFullLossValue(bool cflv)noexcept { m_bCalcFullLossValue = cflv; return *this; }

		bool ImmediatelyDeinit()const noexcept { return m_bImmediatelyDeinit; }
		self_t& ImmediatelyDeinit(bool imd)noexcept { m_bImmediatelyDeinit = imd; return *this; }

		self_t& dropFProp4FullBatchErrorCalc(bool f)noexcept { m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch = f; return *this; }
		bool dropFProp4FullBatchErrorCalc()const noexcept { return m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch; }

		const bool evalNNFinalPerf()const noexcept { return !!m_pNNEvalFinalRes; }
		nnet_td_eval_results<real_t>& NNEvalFinalResults()const noexcept { NNTL_ASSERT(m_pNNEvalFinalRes);			return *m_pNNEvalFinalRes; }
		self_t& NNEvalFinalResults(nnet_td_eval_results<real_t>& er)noexcept { m_pNNEvalFinalRes = &er; 			return *this; }
		
	};

}
