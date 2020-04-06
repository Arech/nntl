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

#include "training_observer.h"
#include "./utils/vector_conditions.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	//structure to hold the results of NNet evaluation
	// obsolete
	/*template<typename RealT>
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
	};*/


	//////////////////////////////////////////////////////////////////////////
	// options of training algo
	template <typename RealT, typename TrainingObserver = training_observer_stdcout<RealT>
		, ::std::enable_if_t<::std::is_arithmetic<RealT>::value, int> = 0//just a guardian for correct instantiation
	>
	class nnet_train_opts : public math::smatrix_td {
		static_assert(::std::is_base_of<i_training_observer<typename TrainingObserver::real_t>, TrainingObserver>::value,
			"TrainingObserver template parameter must be derived from i_training_observer");

	public:
		typedef TrainingObserver training_observer_t;
		typedef nnet_train_opts<RealT, training_observer_t> self_t;
		
		typedef typename training_observer_t::real_t real_t;
		typedef typename training_observer_t::realmtx_t realmtx_t;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		training_observer_t m_trainingObserver;

		//nnet_td_eval_results<real_t>* m_pNNEvalFinalRes;

		vector_conditions m_vbEpochEval;

		//DivergenceCheck* vars describes how to check for nn algo divergence
		real_t m_DivergenceCheckThreshold;

		// m_BatchSize can be 0, which means full batch mode.
		// if m_maxFpropSize is not zero (zero means "use max of training/testing sizes"), then any required forward propagation 
		// during fprop will be done using at max batch sizes of m_maxFpropSize. That allows to perform computations that doesn't
		// fit into memory at once. Note that if set, it MUST be >= m_BatchSize
		vec_len_t m_BatchSize, m_maxFpropSize;

		int16_t m_DivergenceCheckLastEpoch;//set to zero to turn off divergence check

		bool m_bCalcFullLossValue;//if set to false, then only the main part of loss function will be calculated 
		// (i.e. additional summands such as L2 weight penalty value will be stripped.
		// This is done by skipping _layer_base::lossAddendum() calls for all layers)
		
		//setting to false will prevent nnet and all its components from deinitialization (except for some temporary memory, that 
		// don't store anything valuable) on .train() exit
		bool m_bImmediatelyDeinit;

		//deprecated
		//set this flag to true to skip forward pass during training set error calculation in full-batch mode
		//This will make error value report slightly wrong (errVal corresponds to the previous pass), but will make a significant speedup
		//bool m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch;

		void _ctor()noexcept {
			m_BatchSize = m_maxFpropSize = 0;
			m_DivergenceCheckLastEpoch = 5;
			m_DivergenceCheckThreshold = real_t(1e5);
			m_bCalcFullLossValue = true;
			m_bImmediatelyDeinit = false;
			//m_pNNEvalFinalRes = nullptr;
			//m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch = false;
		}

	public:
		~nnet_train_opts()noexcept {}
		nnet_train_opts(numel_cnt_t _maxEpoch, const bool& defVal = true)noexcept : m_vbEpochEval(_maxEpoch, defVal)
		{_ctor();}
		nnet_train_opts(numel_cnt_t _maxEpoch, numel_cnt_t stride)noexcept : m_vbEpochEval(_maxEpoch, stride)
		{		_ctor();	}
		nnet_train_opts(numel_cnt_t _maxEpoch, numel_cnt_t startsAt, numel_cnt_t stride)noexcept : m_vbEpochEval(_maxEpoch, startsAt, stride)
		{		_ctor();	}


		numel_cnt_t maxEpoch()const noexcept { return m_vbEpochEval.maxEpoch(); }
		vector_conditions& getCondEpochEval() noexcept { return m_vbEpochEval; }
		const vector_conditions& getCondEpochEval()const noexcept { return m_vbEpochEval; }

		real_t divergenceCheckThreshold() const noexcept { return m_DivergenceCheckThreshold; }
		self_t& divergenceCheckThreshold(real_t val) noexcept { m_DivergenceCheckThreshold = val; return *this; }

		numel_cnt_t divergenceCheckLastEpoch() const noexcept { return m_DivergenceCheckLastEpoch; }
		self_t& divergenceCheckLastEpoch(int16_t val) noexcept { m_DivergenceCheckLastEpoch = val; return *this; }

		//batchSize is a term for training. 0 means a full batch mode (whole training set at once)
		vec_len_t batchSize() const noexcept { 
			NNTL_ASSERT(m_maxFpropSize >= 0 && m_BatchSize >= 0);
			NNTL_ASSERT(m_maxFpropSize == 0 || m_maxFpropSize >= m_BatchSize);
			return m_BatchSize;
		}
		self_t& batchSize(vec_len_t val) noexcept {
			NNTL_ASSERT(m_maxFpropSize >= 0);
			if (val < 0 || (m_maxFpropSize != 0 && val > m_maxFpropSize)) {
				NNTL_ASSERT(!"WTF?! Trying to set invalid batch size!");
				//OK to die right now with noexcept
			#pragma warning(disable:4297)//function assumed not to throw
				throw ::std::logic_error("WTF?! Trying to set invalid batch size!");
			#pragma warning(default:4297)
			}
			m_BatchSize = val;
			return *this;
		}

		// maxFpropSize is a term similar to batchSize, but applies to fprop() phase only. I.e. it's more relevant (in terms
		// of its absolute value) to loss function value calculation, or evaluation of nnet performance on whole dataset.
		// if m_maxFpropSize is not zero (zero means "use max of training/testing sizes"), then any required forward propagation 
		// during fprop will be done using at max batch sizes of m_maxFpropSize. That allows to perform computations that doesn't
		// fit into memory at once. Note that if set, it MUST be >= m_BatchSize
		vec_len_t maxFpropSize()const noexcept {
			NNTL_ASSERT(m_maxFpropSize >= 0 && m_BatchSize >= 0);
			NNTL_ASSERT(m_maxFpropSize == 0 || m_maxFpropSize >= m_BatchSize);
			return m_maxFpropSize;
		}
		//note that setting it to nonzero requies batchSize() to be set first to nonzero value
		self_t& maxFpropSize(vec_len_t val)noexcept{
			NNTL_ASSERT(m_BatchSize >= 0);
			if (val < 0 || (val > 0 && 0 == m_BatchSize) || (val > 0 && val < m_BatchSize)) {
				NNTL_ASSERT(!"WTF?! Trying to set invalid maxFpropSize!");
				//OK to die right now with noexcept
			#pragma warning(disable:4297)//function assumed not to throw
				throw ::std::logic_error("WTF?! Trying to set invalid maxFpropSize!");
			#pragma warning(default:4297)
			}
			m_maxFpropSize = val;
			return *this;
		}

		training_observer_t& observer() noexcept { return m_trainingObserver; }

		bool calcFullLossValue()const noexcept { return m_bCalcFullLossValue; }
		self_t& calcFullLossValue(bool cflv)noexcept { m_bCalcFullLossValue = cflv; return *this; }

		bool ImmediatelyDeinit()const noexcept { return m_bImmediatelyDeinit; }
		self_t& ImmediatelyDeinit(bool imd)noexcept { m_bImmediatelyDeinit = imd; return *this; }

		/*
		 *deprecated
		self_t& dropFProp4FullBatchErrorCalc(bool f)noexcept { m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch = f; return *this; }
		bool dropFProp4FullBatchErrorCalc()const noexcept { return m_bDropFProp4TrainingSetErrorCalculationWhileFullBatch; }
		*/

		/* deprecated
		 *const bool evalNNFinalPerf()const noexcept { return !!m_pNNEvalFinalRes; }
		nnet_td_eval_results<real_t>& NNEvalFinalResults()const noexcept { NNTL_ASSERT(m_pNNEvalFinalRes);	return *m_pNNEvalFinalRes; }
		self_t& NNEvalFinalResults(nnet_td_eval_results<real_t>& er)noexcept { m_pNNEvalFinalRes = &er; 	return *this; }*/
		
	};

}
