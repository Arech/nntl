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

#include "training_observer.h"
#include "nnet_cond_epoch_eval.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	// options of training algo
	template <typename cond_epoch_eval = nnet_cond_epoch_eval, typename TrainingObserver = training_observer_stdcout>
	class nnet_train_opts {
		static_assert(std::is_base_of<i_training_observer, TrainingObserver>::value,
			"TrainingObserver template parameter must be derived from i_training_observer");

	public:
		typedef TrainingObserver training_observer_t;
		typedef cond_epoch_eval cond_epoch_eval_t;
		typedef nnet_train_opts<cond_epoch_eval_t, training_observer_t> self_t;
		typedef math_types::floatmtx_ty::vec_len_t batch_size_t;
		typedef math_types::float_ty float_t_;

		~nnet_train_opts()noexcept {}
		nnet_train_opts(cond_epoch_eval_t&& cee)noexcept : m_vbEpochEval(std::move(cee)), m_BatchSize(0),
			m_DivergenceCheckLastEpoch(5), m_DivergenceCheckThreshold(1e6f) {}

		self_t& setEpochEval(cond_epoch_eval_t&& cee)noexcept { m_vbEpochEval = std::forward(cee); return *this; }

		size_t maxEpoch()const noexcept { return m_vbEpochEval.maxEpoch(); }
		const cond_epoch_eval_t& getCondEpochEval()const noexcept { return m_vbEpochEval; }

		float_t_ divergenceCheckThreshold() const noexcept { return m_DivergenceCheckThreshold; }
		self_t& divergenceCheckThreshold(float_t_ val) noexcept { m_DivergenceCheckThreshold = val; return *this; }

		size_t divergenceCheckLastEpoch() const noexcept { return m_DivergenceCheckLastEpoch; }
		self_t& divergenceCheckLastEpoch(int16_t val) noexcept { m_DivergenceCheckLastEpoch = val; return *this; }

		batch_size_t batchSize() const noexcept { return m_BatchSize; }
		self_t& batchSize(batch_size_t val) noexcept { m_BatchSize = val; return *this; }

		training_observer_t& observer() noexcept { return m_trainingObserver; }

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		cond_epoch_eval_t m_vbEpochEval;

		//DivergenceCheck* vars describes how to check for nn algo divergence
		float_t_ m_DivergenceCheckThreshold;

		batch_size_t m_BatchSize;

		int16_t m_DivergenceCheckLastEpoch;//probably don't need a bigger type here

		training_observer_t m_trainingObserver;
	};

}
