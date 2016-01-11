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

#include <type_traits>
#include <algorithm>

#include "errors.h"
#include "_nnet_errs.h"

#include "utils.h"

#include "train_data.h"

#include "nnet_train_opts.h"
#include "nnet_def_interfaces.h"

namespace nntl {

	namespace _impl {

		struct DummyOnEpochEndCB {
			template<typename _nnet, typename _opts>
			bool operator()(_nnet& nn, _opts& opts, const size_t epochIdx) { 
				//return false to stop learning
				return true; 
			}
		};

	}

	// RngInterface must be a type or a pointer to a type
	template <typename LayersPack, typename MathInterface = nnet_def_interfaces::iMath_t, typename RngInterface= nnet_def_interfaces::iRng_t>
	class nnet : public _has_last_error<_nnet_errs> {
	public:
		typedef typename std::remove_pointer_t<MathInterface> imath_t;
		typedef typename imath_t::real_t real_t;
		typedef utils::own_or_use_ptr_t<MathInterface> imath_ptr_t;
		static_assert(std::is_base_of<math::_i_math<real_t>, imath_t>::value, "MathInterface type should be derived from _i_math");

		typedef typename std::remove_pointer_t<RngInterface> irng_t;
		typedef utils::own_or_use_ptr_t<RngInterface> irng_ptr_t;
		static_assert(std::is_base_of<rng::_i_rng, irng_t>::value, "RngInterface type should be derived from _i_rng");

		typedef LayersPack layers_pack_t;

		typedef train_data<real_t> train_data_t;

		typedef typename layers_pack_t::realmtx_t realmtx_t;
		//typedef typename realmtx_t::value_type real_t;
		typedef typename realmtx_t::vec_len_t vec_len_t;
		typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

		typedef typename layers_pack_t::realmtxdef_t realmtxdef_t;
		typedef typename layers_pack_t::floatmtxdef_array_t floatmtxdef_array_t;

		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		layers_pack_t& m_Layers;
		imath_ptr_t m_pMath;
		irng_ptr_t m_pRng;

		layer_index_t m_failedLayerIdx;

		bool m_bCalcFullLossValue;//set based on nnet_train_opts::calcFullLossValue() and the value, returned by layers init()

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & m_Layers;
		}

	public:
		~nnet()noexcept {}
		nnet(layers_pack_t& lp)noexcept : m_Layers(lp), m_failedLayerIdx(0){
			static_assert(m_pRng.bOwning && m_pMath.bOwning,"WTF?");
			_init_rng();
		}

		nnet(layers_pack_t& lp, MathInterface mathi) noexcept : m_Layers(lp), m_pMath(mathi), m_failedLayerIdx(0) {
			static_assert(std::is_pointer<MathInterface>::value, "mathi parameter must be pointer");
			static_assert(!m_pMath.bOwning && m_pRng.bOwning, "WTF?");
			_init_rng();
		}

		nnet(layers_pack_t& lp, RngInterface rngi) noexcept: m_Layers(lp), m_pRng(rngi), m_failedLayerIdx(0) {
			static_assert(std::is_pointer<RngInterface>::value, "rngi parameter must be pointer");
			static_assert(!m_pRng.bOwning && m_pMath.bOwning, "WTF?");
			_init_rng();
		}

		nnet(layers_pack_t& lp, MathInterface mathi, RngInterface rngi) noexcept : m_Layers(lp),
			m_pMath(mathi), m_pRng(rngi), m_failedLayerIdx(0)
		{
			static_assert(std::is_pointer<RngInterface>::value, "rngi parameter must be pointer");
			static_assert(std::is_pointer<MathInterface>::value, "mathi parameter must be pointer");
			static_assert(!m_pRng.bOwning && !m_pMath.bOwning, "WTF?");
			_init_rng();
		}

		std::string get_last_error_string()const noexcept {
			std::string les(get_last_error_str());

			//TODO: версии ошибок сюда

			/*if (ErrorCode::FailedToParseJson == get_last_error()) {
				les = les + " Rapidjson: " + rapidjson::GetParseError_En(get_parse_error())
					+ " Offset: " + std::to_string(get_parse_error_offset());
			}*/
			les += std::string(NNTL_STRING(" (layer#")) + std::to_string(static_cast<unsigned int>(m_failedLayerIdx)) + NNTL_STRING(")");
			return les;
		}

		template <typename _train_opts, typename _onEpochEndCB = _impl::DummyOnEpochEndCB>
		ErrorCode train(train_data_t& td, _train_opts& opts, _onEpochEndCB&& onEpochEndCB = _impl::DummyOnEpochEndCB())noexcept {
			if (td.empty()) return _set_last_error(ErrorCode::InvalidTD);

			const auto& train_x = td.train_x();
			const auto& train_y = td.train_y();
			const vec_len_t samplesCount = train_x.rows();
			const bool bTrainSetBigger = samplesCount >= td.test_x().rows();
			NNTL_ASSERT(samplesCount == train_y.rows());

			//X data must come with emulated biases included
			NNTL_ASSERT(train_x.emulatesBiases());
			NNTL_ASSERT(td.test_x().emulatesBiases());

			if (train_x.cols_no_bias() != m_Layers.input_layer().m_neurons_cnt) return _set_last_error(ErrorCode::InvalidInputLayerNeuronsCount);
			if (train_y.cols() != m_Layers.output_layer().m_neurons_cnt) return _set_last_error(ErrorCode::InvalidOutputLayerNeuronsCount);

			const bool bMiniBatch = opts.batchSize() > 0 && opts.batchSize() < samplesCount;
			const auto batchSize = bMiniBatch ? opts.batchSize() : samplesCount;
			if (!_batchSizeOk(td, batchSize)) return _set_last_error(ErrorCode::BatchSizeMustBeMultipleOfTrainDataLength);

			m_bCalcFullLossValue = opts.calcFullLossValue();
			//////////////////////////////////////////////////////////////////////////
			// perform layers initialization, gather temp memory requirements, then allocate and spread temp buffers
			_impl::layers_mem_requirements<real_t> LMR;
			{
				const auto le = m_Layers.init(bTrainSetBigger ? samplesCount : td.test_x().rows(), batchSize, LMR, m_pMath.get(), m_pRng.get());
				if (ErrorCode::Success != le.first) {
					m_failedLayerIdx = le.second;
					return _set_last_error(le.first);
				}
			}

			//scheduling deinitialization with scope_exit to drop worries about returns;
			utils::scope_exit layers_deinit([this, &opts]() {
				if(opts.ImmediatelyDeinit()) m_Layers.deinit(m_pMath.get());
			});
			NNTL_ASSERT(LMR.maxSingledLdANumel > 0);//there must be at least room to store dL/dA

			if (m_bCalcFullLossValue) m_bCalcFullLossValue = LMR.bHasLossAddendum;

			//dLdA is loss function derivative wrt activations. For the top level it's usually called an 'error' and defined like (data_y-a).
			// We use slightly more generalized approach and name it appropriately. It's computed by _i_activation_loss::dloss
			// and most time (for quadratic or crossentropy loss) it is (a-data_y) (we reverse common definition to get rid
			// of negation in dL/dA = -error for error=data_y-a)
			realmtx_t _batch_x, _batch_y;
			realmtxdef_t train_dLdA;
			floatmtxdef_array_t a_dLdA;

			// here is how we gonna spread temp buffers:
			// 1. LMR.maxMemLayerTrainingRequire goes into m_Layers.initMem() to be used during fprop() or bprop() computations
			// 2. 2*LMR.maxSingleActivationMtxNumel will be spread over 2 same sized dL/dA matrices (first will be the incoming dL/dA, the second will be
			//		"outgoing" i.e. for lower layer). This matrices will be used during bprop() by m_Layers.bprop()
			// 3. [samplesCount x train_y.cols()] will be used to compute loss function over whole training set (error matrix in particular)
			// 4. In minibatch version, there will be 2 additional matrices sized (batchSize, train_x.cols()) and (batchSize, train_y.cols())
			//		to handle _batch_x and _batch_y data
			const numel_cnt_t totalTempMemSize = LMR.maxMemLayerTrainingRequire + a_dLdA.size()*LMR.maxSingledLdANumel
				+ (bTrainSetBigger ? train_y.numel() : td.test_y().numel()) //realmtx_t::sNumel(samplesCount,train_y.cols())
				+ (bMiniBatch ? (realmtx_t::sNumel(batchSize, train_x.cols()) + realmtx_t::sNumel(batchSize, train_y.cols())) : 0);
			std::unique_ptr<real_t[]> tempMemStorage(new(std::nothrow)real_t[totalTempMemSize]);
			if (nullptr==tempMemStorage.get()) return _set_last_error(ErrorCode::CantAllocateMemoryForTempData);

			{
				numel_cnt_t spreadTempMemSize = 0;
				// 3.
				train_dLdA.useExternalStorage(&tempMemStorage[spreadTempMemSize], bTrainSetBigger ? train_y : td.test_y());
				spreadTempMemSize += train_dLdA.numel();

				//4. _batch_x and _batch_y if necessary
				if (bMiniBatch) {
					_batch_x.useExternalStorage(&tempMemStorage[spreadTempMemSize], batchSize, train_x.cols());
					spreadTempMemSize += _batch_x.numel();
					_batch_y.useExternalStorage(&tempMemStorage[spreadTempMemSize], batchSize, train_y.cols());
					spreadTempMemSize += _batch_y.numel();
				}

				//2. dLdA
				for (unsigned i = 0; i < a_dLdA.size(); ++i) {
					a_dLdA[i].useExternalStorage(&tempMemStorage[spreadTempMemSize], LMR.maxSingledLdANumel);
					spreadTempMemSize += LMR.maxSingledLdANumel;
				}

				// 1.
				if (LMR.maxMemLayerTrainingRequire > 0) {
					m_Layers.initMem(&tempMemStorage[spreadTempMemSize], LMR.maxMemLayerTrainingRequire);
					spreadTempMemSize += LMR.maxMemLayerTrainingRequire;
				}

				NNTL_ASSERT(spreadTempMemSize == totalTempMemSize);
			}
			realmtx_t& batch_x = bMiniBatch ? _batch_x : td.train_x_mutable();
			realmtx_t& batch_y = bMiniBatch ? _batch_y : td.train_y_mutable();

			std::vector<vec_len_t> vRowIdxs(bMiniBatch ? samplesCount : 0);
			if (bMiniBatch) {
				for (size_t i = 0; i < samplesCount; ++i) vRowIdxs[i] = static_cast<decltype(vRowIdxs)::value_type>(i);
			}

			//////////////////////////////////////////////////////////////////////////
			const auto& cee = opts.getCondEpochEval();
			const size_t maxEpoch = opts.maxEpoch();
			const vec_len_t numBatches = samplesCount / batchSize;
			const auto divergenceCheckLastEpoch = opts.divergenceCheckLastEpoch();

			if (! opts.observer().init(maxEpoch, train_y, td.test_y(), m_pMath.get())) return _set_last_error(ErrorCode::CantInitializeObserver);
			utils::scope_exit observer_deinit([&opts]() {
				opts.observer().deinit();
			});
			
			//making initial report
			opts.observer().on_training_start(samplesCount, td.test_x().rows(), train_x.cols_no_bias(), train_y.cols(), batchSize, LMR.totalParamsToLearn);
			if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();
			_report_training_fragment(-1, _calcLoss(train_x, train_y), td, std::chrono::nanoseconds(0), opts.observer());

			m_Layers.set_mode(0);//prepare for training (sets to batchSize, that's already stored in Layers)

			NNTL_ASSERT(std::chrono::steady_clock::is_steady);
			const auto trainingBeginsAt = std::chrono::steady_clock::now();//starting training timer
			auto epochPeriodBeginsAt = std::chrono::steady_clock::now();//starting epoch timer

			{
				//raising thread priorities for faster computation
				utils::prioritize_workers<utils::PriorityClass::Working, imath_t::ithreads_t> pw(m_pMath.get().ithreads());

				for (size_t epochIdx = 0; epochIdx < maxEpoch; ++epochIdx) {
					auto vRowIdxIt = vRowIdxs.begin();
					if (bMiniBatch) {
						//making random permutations to define which data rows will be used as batch data
						std::random_shuffle(vRowIdxIt, vRowIdxs.end(), m_pRng.get());
					}

					for (vec_len_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
						if (bMiniBatch) {
							m_pMath.get().mExtractRows(train_x, vRowIdxIt, batchSize, batch_x);
							m_pMath.get().mExtractRows(train_y, vRowIdxIt, batchSize, batch_y);
							vRowIdxIt += batchSize;
						}

						//TODO: denoising autoencoders support here

						m_Layers.fprop(batch_x);
						//fullbatch algorithms with quadratic or crossentropy loss function use dL/dA (==error) to compute loss function value.
						//But the dL/dA should be calculated during backward pass. We _can_ calculate it here, but because most of the time we will
						// learn NN with dropout or some other regularizer, output layer activation values in fact can (or "will"
						// in case of dropout) be distorted by that regularizer. Therefore there is no great point in computing error here and
						// pass it to bprop() to be able to reuse it later in loss function computation (where it's suitable only in case of
						// fullbatch learning and absence of dropout/regularizer). Therefore don't bother here with error and leave it to bprop()
						m_Layers.bprop(batch_y, a_dLdA);
					}

					const bool bInspectEpoch = cee(epochIdx);
					const bool bCheckForDivergence = epochIdx < divergenceCheckLastEpoch;
					if (bCheckForDivergence || bInspectEpoch) {
						if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();
						const auto trainLoss = _calcLoss(train_x, train_y);
						if (bCheckForDivergence && trainLoss >= opts.divergenceCheckThreshold()) {
							return _set_last_error(ErrorCode::NNDiverged);
						}

						if (bInspectEpoch) {
							const auto epochPeriodEnds = std::chrono::steady_clock::now();
							_report_training_fragment(epochIdx, trainLoss, td, epochPeriodEnds - epochPeriodBeginsAt, opts.observer());
							epochPeriodBeginsAt = epochPeriodEnds;//restarting period timer
						}
						m_Layers.set_mode(0);//restoring training mode after _calcLoss()
					}

					if (! onEpochEndCB(*this, opts, epochIdx)) break;
				}
			}
			opts.observer().on_training_end(std::chrono::steady_clock::now()- trainingBeginsAt);

			return _set_last_error(ErrorCode::Success);
		}

		layers_pack_t& get_layer_pack()noexcept { return m_Layers; }
		imath_t& get_iMath()const noexcept { return m_pMath.get(); }
		irng_t& get_iRng()const noexcept { return m_pRng.get(); }

	protected:

		void _init_rng()noexcept {
			if (irng_t::is_multithreaded) {
				m_pRng.get().set_ithreads(m_pMath.get().ithreads());
			}
		}

		//returns test loss
		template<typename Observer>
		const real_t _report_training_fragment(const size_t epoch, const real_t trainLoss, const train_data_t& td,
			const std::chrono::nanoseconds& tElapsed, Observer& obs) noexcept
		{
			//relaxing thread priorities (we don't know in advance what callback functions actually do, so better relax it)
			utils::prioritize_workers<utils::PriorityClass::Normal, imath_t::ithreads_t> pw(m_pMath.get().ithreads());

			const auto& activations = m_Layers.output_layer().get_activations();
			obs.inspect_results(td.train_y(), activations, false, m_pMath.get());

			const auto testLoss = _calcLoss(td.test_x(), td.test_y());
			obs.inspect_results(td.test_y(), activations, true, m_pMath.get());

			obs.on_training_fragment_end(epoch, trainLoss, testLoss, tElapsed);
			return testLoss;
		}

		bool _batchSizeOk(const train_data_t& td, vec_len_t batchSize)const noexcept {
			//TODO: соответствие оптимизатора и размера батча (RProp только фулбатчевый)
			double d = double(td.train_x().rows()) / double(batchSize);
			return  d == floor(d);
		}

		real_t _calcLoss(const realmtx_t& data_x, const realmtx_t& data_y) noexcept {
			NNTL_ASSERT(data_x.rows() == data_y.rows());
			//preparing for evaluation
			m_Layers.set_mode(data_x.rows());
			m_Layers.fprop(data_x);

			static_assert(std::is_base_of<activation::_i_activation_loss, layers_pack_t::output_layer_t::activation_f_t>::value,
				"Activation function class of output layer must implement activation::_i_activation_loss interface");

			auto lossValue = layers_pack_t::output_layer_t::activation_f_t::loss(m_Layers.output_layer().get_activations(), data_y, m_pMath.get());
			if (m_bCalcFullLossValue) lossValue += m_Layers.calcLossAddendum();
			return lossValue;
		}
	};

	template <typename LayersPack>
	inline nnet<LayersPack> make_nnet(LayersPack& lp)noexcept { return nnet<LayersPack>(lp); }

	template <typename LayersPack, typename RngInterface>
	inline nnet<LayersPack, RngInterface> make_nnet(LayersPack& lp, RngInterface rngi)noexcept { return nnet<LayersPack, RngInterface>(lp, rngi); }

	template <typename LayersPack, typename MathInterface, typename RngInterface>
	inline nnet<LayersPack, MathInterface, RngInterface> make_nnet(LayersPack& lp, MathInterface mathi, RngInterface rngi)noexcept 
	{ return nnet<LayersPack, MathInterface, RngInterface>(lp, mathi, rngi); }
}