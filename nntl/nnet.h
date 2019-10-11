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

#include <type_traits>
#include <algorithm>

#include "errors.h"
#include "_nnet_errs.h"
#include "utils.h"
#include "train_data.h"

#include "nnet_train_opts.h"
#include "common_nn_data.h"
//#include "utils\lambdas.h"

#include "interface/inspectors/gradcheck.h"

namespace nntl {

	//dummy callback for .train() function.
	struct NNetCB_OnEpochEnd_Dummy {
		template<typename _nnet, typename _opts>
		constexpr const bool operator()(_nnet& nn, _opts& opts, const size_t& epochIdx)const {
			NNTL_UNREF(nn); NNTL_UNREF(opts); NNTL_UNREF(epochIdx);
			//return false to stop learning
			return true;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	template <typename LayersPack>
	class nnet 
		: public _has_last_error<_nnet_errs>
		, public _impl::interfaces_keeper<typename LayersPack::interfaces_t>
	{
	private:
		typedef _impl::interfaces_keeper<typename LayersPack::interfaces_t> _base_class;

	public:
		typedef LayersPack layers_pack_t;

		typedef typename iMath_t::realmtx_t realmtx_t;
		typedef typename iMath_t::realmtxdef_t realmtxdef_t;
		
		typedef train_data<real_t> train_data_t;

		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		layers_pack_t& m_Layers;

		_impl::layers_mem_requirements m_LMR;

		::std::vector<real_t> m_pTmpStor;

		realmtx_t m_batch_x, m_batch_y;

		layer_index_t m_failedLayerIdx;

		bool m_bCalcFullLossValue;//set based on nnet_train_opts::calcFullLossValue() and the value, returned by layers init()
		bool m_bRequireReinit;//set this flag to require nnet object and its layers to reinitialize on next call

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			ar & m_Layers;
		}

	public:
		~nnet()noexcept {}

		//don't instantiate directly, better use make_nnet() helper function
		template<typename PInspT = ::std::nullptr_t, typename PMathT = ::std::nullptr_t, typename PRngT = ::std::nullptr_t>
		nnet(layers_pack_t& lp, PInspT pI=nullptr, PMathT pM=nullptr, PRngT pR=nullptr) noexcept
			: _base_class(pI, pM, pR), m_Layers(lp)
		{
			m_bRequireReinit = false;
			m_failedLayerIdx = 0;
			if (iRng_t::is_multithreaded) get_iRng().init_ithreads(get_iMath().ithreads());
		}

		::std::string get_last_error_string()const noexcept {
			::std::string les(get_last_error_str());

			//TODO: версии ошибок сюда

			/*if (ErrorCode::FailedToParseJson == get_last_error()) {
			les = les + " Rapidjson: " + rapidjson::GetParseError_En(get_parse_error())
			+ " Offset: " + ::std::to_string(get_parse_error_offset());
			}*/
			les += ::std::string(NNTL_STRING(" (layer#")) + ::std::to_string(static_cast<::std::uint64_t>(m_failedLayerIdx)) + NNTL_STRING(")");
			return les;
		}

		layers_pack_t& get_layer_pack()const noexcept { return m_Layers; }

		//call this to force nnet and its dependents to reinitialize 
		void require_reinit()noexcept { m_bRequireReinit = true; }

	protected:

		//#todo get rid of pTestEvalRes
		//returns test loss
		template<bool bPrioritizeThreads = true, typename Observer>
		const real_t _report_training_fragment(const size_t& epoch, const real_t& trainLoss, const train_data_t& td,
			const ::std::chrono::nanoseconds& tElapsed, Observer& obs, const bool& bTrainSetWasInspected = false,
			nnet_eval_results<real_t>*const pTestEvalRes=nullptr) noexcept
		{
			//relaxing thread priorities (we don't know in advance what callback functions actually do, so better relax it)
			//threads::prioritize_workers<threads::PriorityClass::Normal, iThreads_t> pw(get_iMath().ithreads());
			::std::conditional_t<bPrioritizeThreads
				, threads::prioritize_workers<threads::PriorityClass::Normal, iThreads_t>
				, threads::_impl::prioritize_workers_dummy<threads::PriorityClass::Normal, iThreads_t>
			> pw(get_iMath().ithreads());


			//const auto& activations = m_Layers.output_layer().get_activations();
			if (!bTrainSetWasInspected) obs.inspect_results(epoch, td.train_y(), false, *this);

			if (m_bCalcFullLossValue && m_LMR.bLossAddendumDependsOnActivations) m_Layers.prepToCalcLossAddendum();

			const auto testLoss = _calcLossNotifyInspector(&td.test_x(), td.test_y(), false);
			if (pTestEvalRes) {
				//saving training results
				pTestEvalRes->lossValue = testLoss;
				m_Layers.output_layer().get_activations().clone_to(pTestEvalRes->output_activations);
			}

			obs.inspect_results(epoch, td.test_y(), true, *this);

			obs.on_training_fragment_end(epoch, trainLoss, testLoss, tElapsed);
			return testLoss;
		}

		bool _batchSizeOk(const train_data_t& td, vec_len_t batchSize)const noexcept {
			//TODO: соответствие оптимизатора и размера батча (RProp только фулбатчевый)
			double d = double(td.train_x().rows()) / double(batchSize);
			return  d == floor(d);
		}

		void _fprop(const realmtx_t& data_x)noexcept {
			//preparing for evaluation
			set_mode_and_batch_size(data_x.rows());
			m_Layers.fprop(data_x);
		}

		real_t _calcLossNotifyInspector(const realmtx_t*const pData_x, const realmtx_t& data_y, const bool bTrainingData) noexcept {
			auto& iI = get_iInspect();
			iI.train_preCalcError(bTrainingData);
			const auto r = _calcLoss(pData_x, data_y);
			iI.train_postCalcError();
			return r;
		}

		template<bool bNormalizeToDataSize = true>
		real_t _calcLoss(const realmtx_t*const pData_x, const realmtx_t& data_y) noexcept {
			NNTL_ASSERT(!pData_x || pData_x->rows() == data_y.rows());
			NNTL_ASSERT(data_y.rows());
			if (pData_x) _fprop(*pData_x);

			auto lossValue = m_Layers.output_layer().calc_loss(data_y);
			if (m_bCalcFullLossValue) lossValue += m_Layers.calcLossAddendum();
			return bNormalizeToDataSize ? lossValue / data_y.rows() : lossValue;
		}

		const bool _is_initialized(const vec_len_t biggestFprop, const vec_len_t batchSize)const noexcept {
			return !m_bRequireReinit && get_common_data().is_initialized()
				&& biggestFprop <= get_common_data().max_fprop_batch_size()
				&& batchSize <= get_common_data().training_batch_size();
				//&& (0 == batchSize || batchSize == get_common_data().training_batch_size());
		}

		//batchSize==0 means that _init is called for use in fprop scenario only
		ErrorCode _init(const vec_len_t biggestFprop, vec_len_t batchSize = 0, const bool bMiniBatch = false
			, const size_t maxEpoch = 1, const vec_len_t numBatches = 1)noexcept
		{
			if (_is_initialized(biggestFprop, batchSize)) {
				//_processTmpStor(bMiniBatch, train_x_cols, train_y_cols, batchSize, pTtd);
				//looks like the call above is actually a bug. If the nnet is initalized, no work should be done with its memory
				get_iInspect().init_nnet(m_Layers.total_layers(), maxEpoch, numBatches);
				return ErrorCode::Success;
			}

			//#TODO we must be sure here that no internal objects settings will be hurt during deinit phase
			_deinit();
			
			get_iInspect().init_nnet(m_Layers.total_layers(), maxEpoch, numBatches);

			bool bInitFinished = false;
			utils::scope_exit _run_deinit([this, &bInitFinished]() {
				if (!bInitFinished) _deinit();
			});

			get_common_data().init(biggestFprop, batchSize);
			
			m_failedLayerIdx = 0;
			const auto le = m_Layers.init(get_common_data(), m_LMR);
			if (ErrorCode::Success != le.first) {
				m_failedLayerIdx = le.second;
				return le.first;
			}
			NNTL_ASSERT(batchSize == 0 || m_LMR.maxSingledLdANumel > 0);
			
			if (!get_iMath().init()) return ErrorCode::CantInitializeIMath;
			if (!get_iRng().init_rng()) return ErrorCode::CantInitializeIRng;

			const numel_cnt_t totalTempMemSize = _totalTrainingMemSize(bMiniBatch, batchSize);
			//m_pTmpStor.reset(new(::std::nothrow)real_t[totalTempMemSize]);
			//if (nullptr == m_pTmpStor.get()) return ErrorCode::CantAllocateMemoryForTempData;
			m_pTmpStor.resize(totalTempMemSize);
			
			const auto _memUsed = _processTmpStor(bMiniBatch, batchSize);
			NNTL_ASSERT(totalTempMemSize == _memUsed);

			bInitFinished = true;
			return ErrorCode::Success;
		}
		const numel_cnt_t _totalTrainingMemSize(const bool bMiniBatch, const vec_len_t batchSize)noexcept
			//, const vec_len_t train_x_cols, const vec_len_t train_y_cols)noexcept
		{
			// here is how we gonna spread temp buffers:
			// 1. LMR.maxMemLayerTrainingRequire goes into m_Layers.initMem() to be used during fprop() or bprop() computations
			// 2. 2*LMR.maxSingleActivationMtxNumel will be spread over 2 same sized dL/dA matrices (first will be the incoming dL/dA, the second will be
			//		"outgoing" i.e. for lower layer). This matrices will be used during bprop() by m_Layers.bprop()
			// 3. In minibatch version, there will be 2 additional matrices sized (batchSize, train_x.cols()) and (batchSize, train_y.cols())
			//		to handle _batch_x and _batch_y data
			
			const vec_len_t train_x_cols = vec_len_t(1) + m_Layers.input_layer().get_neurons_cnt()//1 for bias column
				, train_y_cols = m_Layers.output_layer().get_neurons_cnt();

			return m_LMR.maxMemLayerTrainingRequire
				+ (batchSize > 0 
					? m_Layers.m_a_dLdA.size()*m_LMR.maxSingledLdANumel
					+ (bMiniBatch ? (realmtx_t::sNumel(batchSize, train_x_cols) + realmtx_t::sNumel(batchSize, train_y_cols)) : 0)
					: 0);
		}

		numel_cnt_t _processTmpStor(const bool bMiniBatch, const vec_len_t batchSize)noexcept
		{
			NNTL_ASSERT(batchSize == 0 || m_pTmpStor.size() > 0);
			//auto tempMemStorage = m_pTmpStor.get();
			//NNTL_ASSERT(tempMemStorage);
			auto& tempMemStorage = m_pTmpStor;

			numel_cnt_t spreadTempMemSize = 0;

			if (batchSize > 0) {
// 				const vec_len_t train_x_cols = 1 + m_Layers.input_layer().get_neurons_cnt()//1 for bias column
// 					, train_y_cols = m_Layers.output_layer().get_neurons_cnt();

				//3. _batch_x and _batch_y if necessary
				if (bMiniBatch) {
					NNTL_ASSERT(batchSize);
					m_batch_x.useExternalStorage(&tempMemStorage[spreadTempMemSize], batchSize, m_Layers.input_layer().get_neurons_cnt() + 1, true);
					spreadTempMemSize += m_batch_x.numel();
					m_batch_x.set_biases();
					m_batch_y.useExternalStorage(&tempMemStorage[spreadTempMemSize], batchSize, m_Layers.output_layer().get_neurons_cnt());
					spreadTempMemSize += m_batch_y.numel();
				}

				//2. dLdA
				//#TODO: better move it to m_Layers
				for(auto& m : m_Layers.m_a_dLdA){
					m.useExternalStorage(&tempMemStorage[spreadTempMemSize], m_LMR.maxSingledLdANumel);
					spreadTempMemSize += m_LMR.maxSingledLdANumel;
				}
			}

			// 1.
			if (m_LMR.maxMemLayerTrainingRequire > 0) {//m_LMR.maxMemLayerTrainingRequire is a max(for fprop() and for bprop() reqs)
				m_Layers.initMem(&tempMemStorage[spreadTempMemSize], m_LMR.maxMemLayerTrainingRequire);
				spreadTempMemSize += m_LMR.maxMemLayerTrainingRequire;
			}

			return spreadTempMemSize;
		}

		void _deinit()noexcept {
			m_Layers.deinit();
			get_iRng().deinit_rng();
			get_iMath().deinit();
			get_common_data().deinit();
			m_bRequireReinit = false;
			m_LMR.zeros();
			m_batch_x.clear();
			m_batch_y.clear();
			m_pTmpStor.clear();
		}
		
		void set_mode_and_batch_size(const vec_len_t bs)noexcept {
			const bool bIsTraining = bs == 0;
			auto& cd = get_common_data();
			//cd.set_training_mode(bIsTraining);
			cd.set_mode_and_batch_size(bIsTraining, bIsTraining ? cd.training_batch_size() : bs);
			m_Layers.on_batch_size_change();
		}

	public:

		template <bool bPrioritizeThreads = true, typename TrainOptsT, typename OnEpochEndCbT = NNetCB_OnEpochEnd_Dummy>
		ErrorCode train(const train_data_t& td, TrainOptsT& opts, OnEpochEndCbT&& onEpochEndCB = NNetCB_OnEpochEnd_Dummy())noexcept
		{
			typedef ::std::conditional_t<bPrioritizeThreads
				, threads::prioritize_workers<threads::PriorityClass::Working, iThreads_t>
				, threads::_impl::prioritize_workers_dummy<threads::PriorityClass::Normal, iThreads_t>> PW_t;

			//just leave it here
			global_denormalized_floats_mode();

			if (td.empty()) return _set_last_error(ErrorCode::InvalidTD);

			auto& iI = get_iInspect();
			const auto& train_x = td.train_x();
			const auto& train_y = td.train_y();
			const vec_len_t samplesCount = train_x.rows();
			NNTL_ASSERT(samplesCount == train_y.rows());

			//X data must come with emulated biases included
			NNTL_ASSERT(train_x.emulatesBiases());
			NNTL_ASSERT(td.test_x().emulatesBiases());

			if (train_x.cols_no_bias() != m_Layers.input_layer().get_neurons_cnt()) return _set_last_error(ErrorCode::InvalidInputLayerNeuronsCount);
			if (train_y.cols() != m_Layers.output_layer().get_neurons_cnt()) return _set_last_error(ErrorCode::InvalidOutputLayerNeuronsCount);

			const bool bTrainSetBigger = samplesCount >= td.test_x().rows();
			const bool bMiniBatch = opts.batchSize() > 0 && opts.batchSize() < samplesCount;
			const bool bSaveNNEvalResults = opts.evalNNFinalPerf();

			const size_t maxEpoch = opts.maxEpoch();
			const auto lastEpoch = maxEpoch - 1;
			const vec_len_t batchSize = bMiniBatch ? opts.batchSize() : samplesCount;
			const vec_len_t numBatches = samplesCount / batchSize;

			if (!_batchSizeOk(td, batchSize)) return _set_last_error(ErrorCode::BatchSizeMustBeMultipleOfTrainDataLength);

			m_bCalcFullLossValue = opts.calcFullLossValue();
			//////////////////////////////////////////////////////////////////////////
			// perform layers initialization, gather temp memory requirements, then allocate and spread temp buffers
			auto ec = _init(bTrainSetBigger ? samplesCount : td.test_x().rows(), batchSize, bMiniBatch, maxEpoch, numBatches);
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			//scheduling deinitialization with scope_exit to forget about return statements
			utils::scope_exit layers_deinit([this, &opts]() {
				if (opts.ImmediatelyDeinit()) {
					_deinit();
				}
			});

			if (m_bCalcFullLossValue) m_bCalcFullLossValue = m_LMR.bHasLossAddendum;

			//dropping the const just for convenience. We mustn't modify any of the TD element
			realmtx_t& batch_x = bMiniBatch ? m_batch_x : const_cast<realmtxdef_t&>(td.train_x());
			realmtx_t& batch_y = bMiniBatch ? m_batch_y : const_cast<realmtxdef_t&>(td.train_y());
			NNTL_ASSERT(batch_x.emulatesBiases() && !batch_y.emulatesBiases());

			::std::vector<vec_len_t> vRowIdxs(bMiniBatch ? samplesCount : 0);
			if (bMiniBatch) {
				::std::iota(vRowIdxs.begin(), vRowIdxs.end(), 0);
				//for (size_t i = 0; i < samplesCount; ++i) vRowIdxs[i] = static_cast<decltype(vRowIdxs)::value_type>(i);
			}

			//////////////////////////////////////////////////////////////////////////
			const auto& cee = opts.getCondEpochEval();			
			const auto divergenceCheckLastEpoch = opts.divergenceCheckLastEpoch();

			//when we're in a fullbatch mode and nn output is the same in training and testing modes, then we may use
			//results of training's fprop() in error calculation to skip corresponding fprop() completely
			const bool bOptimFullBatchErrorCalc = !bMiniBatch && opts.dropFProp4FullBatchErrorCalc()
				&& !m_LMR.bOutputDifferentDuringTraining;

			if (bSaveNNEvalResults) opts.getCondEpochEval().verbose(lastEpoch);

			if (! opts.observer().init(maxEpoch, train_y, td.test_y(), get_iMath())) return _set_last_error(ErrorCode::CantInitializeObserver);
			utils::scope_exit observer_deinit([&opts]() {
				opts.observer().deinit();
			});
			
			//making initial report
			opts.observer().on_training_start(samplesCount, td.test_x().rows(), train_x.cols_no_bias(), train_y.cols(), batchSize, m_LMR.totalParamsToLearn);
			
			if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();

			{
				const auto lv = _calcLossNotifyInspector(&train_x, train_y, true);
				_report_training_fragment<bPrioritizeThreads>(static_cast<size_t>(-1), lv, td, ::std::chrono::nanoseconds(0), opts.observer());
			}
			set_mode_and_batch_size(0);//prepare for training (sets to batchSize, that's already stored in Layers)

#if NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH
			STDCOUTL("Denormals check... " << (get_iMath().ithreads().denormalsOnInAnyThread() ? "FAILED!!!" : "passed.") 
				<< ::std::endl << " Going to check it further each epoch and report failures only.");
#endif//NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH

			nnet_eval_results<real_t>* pTestEvalRes = nullptr;

			NNTL_ASSERT(::std::chrono::steady_clock::is_steady);
			const auto trainingBeginsAt = ::std::chrono::steady_clock::now();//starting training timer
			auto epochPeriodBeginsAt = ::std::chrono::steady_clock::now();//starting epoch timer

			{
				//raising thread priorities for faster computation
				//threads::prioritize_workers<threads::PriorityClass::Working, iThreads_t> pw(get_iMath().ithreads());
				PW_t pw(get_iMath().ithreads());

				for (size_t epochIdx = 0; epochIdx < maxEpoch; ++epochIdx) {
					iI.train_epochBegin(epochIdx);

					real_t trainLoss = ::std::numeric_limits<real_t>::max();
					const bool bInspectEpoch = cee(epochIdx);
					const bool bCheckForDivergence = epochIdx < divergenceCheckLastEpoch;
					const bool bCalcLoss = bInspectEpoch || bCheckForDivergence;
					const bool bLastEpoch = epochIdx == lastEpoch;
					const bool bOptFBErrCalcThisEpoch = bOptimFullBatchErrorCalc && bCalcLoss && !bLastEpoch;

					auto vRowIdxIt = vRowIdxs.begin();
					if (bMiniBatch) {
						//making random permutations to define which data rows will be used as batch data
						::std::random_shuffle(vRowIdxIt, vRowIdxs.end(), get_iRng());
					}

					for (vec_len_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
						iI.train_batchBegin(batchIdx);

						if (bMiniBatch) {
							get_iMath().mExtractRows(train_x, vRowIdxIt, batch_x);
							get_iMath().mExtractRows(train_y, vRowIdxIt, batch_y);
							vRowIdxIt += batchSize;
						}

						iI.train_preFprop(batch_x);
						m_Layers.fprop(batch_x);

						if (bOptFBErrCalcThisEpoch) {
							if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();
							trainLoss = _calcLossNotifyInspector(nullptr, batch_y, true);
							//we don't need to call set_mode_and_batch_size(0) here because we did not do fprop() in _calcLoss()
							if (bInspectEpoch) opts.observer().inspect_results(epochIdx, train_y, false, *this);
						}

						iI.train_preBprop(batch_y);
						m_Layers.bprop(batch_y);

						iI.train_batchEnd();
					}

					if (bCalcLoss) {
						if (!bOptFBErrCalcThisEpoch) {
							if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();
							trainLoss = _calcLossNotifyInspector(&train_x, train_y, true);
						}
						if (bCheckForDivergence && trainLoss >= opts.divergenceCheckThreshold())
							return _set_last_error(ErrorCode::NNDiverged);

						if (bInspectEpoch) {
							const auto epochPeriodEnds = ::std::chrono::steady_clock::now();

							if (bSaveNNEvalResults && bLastEpoch) {
								//saving training results
								auto& trr = opts.NNEvalFinalResults().trainSet;
								trr.lossValue = trainLoss;
								//we can call output_layer().get_activations() here because for the last epoch
								// bOptFBErrCalcThisEpoch is always ==false
								m_Layers.output_layer().get_activations().clone_to(trr.output_activations);
								pTestEvalRes = &opts.NNEvalFinalResults().testSet;
							}
							
							_report_training_fragment<bPrioritizeThreads>(epochIdx, trainLoss, td
								, epochPeriodEnds - epochPeriodBeginsAt, opts.observer(), bOptFBErrCalcThisEpoch, pTestEvalRes);

							epochPeriodBeginsAt = epochPeriodEnds;//restarting period timer
						}

// 						if (bInspectEpoch || !bOptFBErrCalcThisEpoch)
// 							set_mode_and_batch_size(0);//restoring training mode after _calcLoss()
					}

					iI.train_epochEnd();

#if NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH
					if (get_iMath().ithreads().denormalsOnInAnyThread()) {
						STDCOUTL("*** denormals check FAILED!!!");
					}
#endif//NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH

					if (!onEpochEndCB(*this, opts, epochIdx)) break;//mustn't forward here, onEpochEndCB is called multiple times

					if (bCalcLoss && (bInspectEpoch || !bOptFBErrCalcThisEpoch)) {
						set_mode_and_batch_size(0);//restoring training mode after _calcLoss()
						//moved the set_mode_and_batch_size() here after the call to onEpochEndCB() to allow onEpochEndCB to call this->fprop() on
						//any auxiliary (real test) dataset
					}
				}
			}
			opts.observer().on_training_end(::std::chrono::steady_clock::now()- trainingBeginsAt);

			return _set_last_error(ErrorCode::Success);
		}

		ErrorCode init4fixedBatchFprop(const vec_len_t dataSize)noexcept {
			NNTL_ASSERT(dataSize);
			const auto ec = _init(dataSize);
			if (ErrorCode::Success != ec) return _set_last_error(ec);
			set_mode_and_batch_size(dataSize);
			return _set_last_error(ec);
		}
		void doFixedBatchFprop(const realmtx_t& data_x)noexcept {
			NNTL_ASSERT(data_x.rows() == get_common_data().get_cur_batch_size());
			m_Layers.fprop(data_x);
		}

		ErrorCode fprop(const realmtx_t& data_x)noexcept {
			const auto ec = _init(data_x.rows());
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			_fprop(data_x);
			return _set_last_error(ec);
		}

		ErrorCode calcLoss(const realmtx_t& data_x, const realmtx_t& data_y, real_t& lossVal) noexcept {
			NNTL_ASSERT(data_x.rows() == data_y.rows());

			auto ec = _init(data_x.rows());
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();
			lossVal = _calcLoss(&data_x, data_y);
			return _set_last_error(ec);
		}

		ErrorCode eval(const realmtx_t& data_x, const realmtx_t& data_y, nnet_eval_results<real_t>& res)noexcept {
			NNTL_ASSERT(data_x.rows() == data_y.rows());

			auto ec = _init(data_x.rows());
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			if (m_bCalcFullLossValue) m_Layers.prepToCalcLossAddendum();

			res.lossValue = _calcLoss(&data_x, data_y);
			m_Layers.output_layer().get_activations().clone_to(res.output_activations);
			return _set_last_error(ec);
		}

		ErrorCode td_eval(train_data_t& td, nnet_td_eval_results<real_t>& res)noexcept {
			auto ec = eval(td.train_x(), td.train_y(), res.trainSet);
			if (ec != ErrorCode::Success) return ec;
			
			ec = eval(td.test_x(), td.test_y(), res.testSet);
			return ec;
		}
		
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// numeric gradient check
		// 
		
	protected:

		struct GradCheckFunctor {
			nnet& m_nn;
			const gradcheck_settings<real_t>& m_ngcSetts;

			layer_index_t m_failedLayerIdx;			
			_impl::gradcheck_mode m_mode;

			//there's no need to make these objects global besides getting rid of memory reallocations, so let it be
			::std::vector<neurons_count_t> m_grpIdx, m_subgrpIdxs;

			utils::dataHolder<real_t> m_data;

			const layer_index_t m_outputLayerIdx;

			//////////////////////////////////////////////////////////////////////////
			//affected state vars
			const vec_len_t m__origBatchSize;
			const bool m__bOrigInTraining;
			const bool m__bOrigCalcFullLossValue;
			
			//////////////////////////////////////////////////////////////////////////
		public:
			~GradCheckFunctor()noexcept{
				m_nn.get_iInspect().gc_deinit();

				m_nn.m_bCalcFullLossValue = m__bOrigCalcFullLossValue;
				m_nn.get_common_data().set_mode_and_batch_size(m__bOrigInTraining, m__origBatchSize);
				m_nn.m_Layers.on_batch_size_change();
				m_nn._unblockLearning();
			}
			GradCheckFunctor(nnet& n, const gradcheck_settings<real_t>& ngcSetts)noexcept
				: m_nn(n), m_ngcSetts(ngcSetts), m_outputLayerIdx(n.m_Layers.output_layer().get_layer_idx())
				//saving nnet mode & affected state
				, m__bOrigInTraining(n.get_common_data().is_training_mode())
				, m__bOrigCalcFullLossValue(n.m_bCalcFullLossValue)
				, m__origBatchSize(n.get_common_data().get_cur_batch_size())
			{
				NNTL_ASSERT(m_ngcSetts.evalSetts.dLdA_setts.relErrWarnThrsh <= m_ngcSetts.evalSetts.dLdA_setts.relErrFailThrsh);
				NNTL_ASSERT(m_ngcSetts.evalSetts.dLdW_setts.relErrWarnThrsh <= m_ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh);
				m_nn.get_iInspect().gc_init(m_ngcSetts.stepSize);

				m_nn._blockLearning();
				m_nn.m_bCalcFullLossValue = true;
			}
			//////////////////////////////////////////////////////////////////////////

			bool performCheck(const vec_len_t batchSize, const realmtx_t& data_x, const realmtx_t& data_y)noexcept {
				const vec_len_t biggestBatch = ::std::max(batchSize, m_ngcSetts.onlineBatchSize);
				m_data.init(biggestBatch == data_x.rows() ? 0 : biggestBatch, data_x, &data_y);
				
				bool bRet = false;
				NNTL_ASSERT(m_ngcSetts.onlineBatchSize > 0);
				if (m_ngcSetts.bVerbose) {
					STDCOUT(::std::endl << "Performing layerwise gradient check in online mode (check dL/dA and so on)");
					if (m_ngcSetts.onlineBatchSize > 1) {
						STDCOUTL(" with a custom batchSize = " << m_ngcSetts.onlineBatchSize);
					} else STDCOUT(::std::endl);
				}
				_launchCheck(_impl::gradcheck_mode::online, m_ngcSetts.onlineBatchSize);
				if (!m_failedLayerIdx) {
					if (m_ngcSetts.bVerbose) {
						STDCOUTL(::std::endl << "Performing layerwise gradient check in batch mode (check dL/dW and so on) with a batchSize = " << batchSize);
					}
					_launchCheck(_impl::gradcheck_mode::batch, batchSize);
					bRet = !m_failedLayerIdx;
				}

				m_data.deinit();
				return bRet;
			}

			layer_index_t getFailedLayerIdx()const noexcept { return m_failedLayerIdx; }
			
			template<typename LayerT> void operator()(LayerT& lyr) noexcept {
				if (m_failedLayerIdx) return;//do nothing, check has already been failed.

				//calling internal layers at first if applicable
				_checkInnerLayers(lyr);

				if (m_failedLayerIdx) return;//do nothing, check has already been failed.

				if (::std::any_of(m_ngcSetts.ignoreLayerIds.cbegin(), m_ngcSetts.ignoreLayerIds.cend()
					, [lidx = lyr.get_layer_idx()](const auto v)
				{
					return lidx == v;
				}))
				{
					if (m_ngcSetts.bVerbose) STDCOUTL("*** Skipping layer " << lyr.get_layer_name_str() << " by request.");
					return;
				}

				switch (m_mode) {
				case nntl::_impl::gradcheck_mode::batch:
					_doCheckdLdW(lyr);
					break;

				case nntl::_impl::gradcheck_mode::online:
					//we should skip output_layer dLdA checks, because actually we don't compute dL/dA for it. We proceed
					//straight to the dL/dZ in output_layer...
					// It's not very good idea (to skip output layer check) for the sake of purity, but if there's a bug in it,
					// we most likely encounter wrong dL/dA in underlying layers, therefore it seems acceptable solution.
					
					if (lyr.get_layer_idx() != m_outputLayerIdx) {
						if (m_ngcSetts.bVerbose) STDCOUT(lyr.get_layer_name_str() << ": ");
						//_doCheckdLdA(lyr);
						_checkdLdA(lyr.get_layer_idx(), lyr.get_neurons_cnt());
					}					
					break;

				default:
					NNTL_ASSERT(!"WTF?");
					break;
				}
			}

		protected:

			template<typename LayerT>
			::std::enable_if_t<is_layer_learnable<LayerT>::value> _doCheckdLdW(LayerT& lyr) noexcept {
				if (m_ngcSetts.bVerbose) STDCOUT(lyr.get_layer_name_str() << ": ");
				_checkdLdW(lyr.get_layer_idx(), lyr.get_neurons_cnt(), lyr.get_incoming_neurons_cnt());
			}
			template<typename LayerT>
			::std::enable_if_t<!is_layer_learnable<LayerT>::value> _doCheckdLdW(LayerT& ) const noexcept {}

			void _reset()noexcept {
				m_failedLayerIdx = 0;
				m_nn.get_iInspect().gc_reset();
			}
			void _setMode(_impl::gradcheck_mode mode)noexcept {
				_reset();
				m_mode = mode;
			}
			void _launchCheck(_impl::gradcheck_mode mode, const vec_len_t batchSize)noexcept {
				NNTL_ASSERT(batchSize);
				_setMode(mode);
				m_data.prepareToBatchSize(batchSize);
				_prepNetToBatchSize(true, batchSize);
				m_nn.m_Layers.for_each_packed_layer_exc_input_down(*this);
			}

			void _prepNetToBatchSize(const bool bTraining, const vec_len_t batchSize)noexcept {
				NNTL_ASSERT(batchSize);
				m_nn.get_common_data().set_mode_and_batch_size(bTraining, batchSize);
				m_nn.m_Layers.on_batch_size_change();
			}

			template<typename LayerT>
			::std::enable_if_t<is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT& lyr)noexcept {
				lyr.for_each_packed_layer_down(*this);
			}
			template<typename LayerT>
			::std::enable_if_t<!is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT&)const noexcept {}
			
			void _checkdLdA(const layer_index_t& lIdx, const neurons_count_t neuronsCnt
			//	, const realmtx_t* pDropoutMask, const real_t dropoutScaleInv = real_t(1.)
			)noexcept
			{
				//NNTL_ASSERT(!pDropoutMask || !pDropoutMask->emulatesBiases());

				const auto checkNeuronsCnt = m_ngcSetts.groupSetts.countToCheck(neuronsCnt);
				if (m_ngcSetts.bVerbose) STDCOUTL( checkNeuronsCnt << " dL/dA values out of total " << neuronsCnt << "... ");

				m_grpIdx.resize(neuronsCnt);
				::std::iota(m_grpIdx.begin(), m_grpIdx.end(), neurons_count_t(0));
				::std::random_shuffle(m_grpIdx.begin(), m_grpIdx.end(), m_nn.get_iRng());
				//m_nn.get_iRng().gen_vector_gtz(&m_grpIdx[0], checkNeuronsCnt, neuronsCnt - 1);
				
				const neurons_count_t maxZerodLdA = (checkNeuronsCnt*m_ngcSetts.evalSetts.dLdA_setts.percOfZeros) / 100;
				neurons_count_t zerodLdA = 0;

				//const auto doubleSs = m_ngcSetts.stepSize * 2;
				// numeric error is already batchsize-normalized. however, analytical does not.
				//const real_t numMult = static_cast<real_t>(m_ngcSetts.onlineBatchSize) / (m_ngcSetts.stepSize * 2);
				const real_t numMult = static_cast<real_t>(1) / (m_ngcSetts.stepSize * 2);

				auto& iI = m_nn.get_iInspect();
				for (neurons_count_t i = 0; i < checkNeuronsCnt; ++i) {
					if (m_failedLayerIdx) break;

					auto neurIdx = m_grpIdx[i];

					m_data.nextBatch(m_nn.get_iRng(), m_nn.get_iMath());

					const mtx_coords_t coords(0, neurIdx);
					const bool bLayerMayBeExcluded = ::std::any_of(m_ngcSetts.layerCanSkipExecIds.cbegin(), m_ngcSetts.layerCanSkipExecIds.cend(), [lIdx](const auto i) {
						return i == lIdx;
					});
					const size_t s = m_ngcSetts.bForceSeed ? ::std::time(0) : 0;
					iI.gc_prep_check_layer(lIdx, _impl::gradcheck_paramsGroup::dLdA, coords, bLayerMayBeExcluded);

					//_prepNetToBatchSize(false, m_data.batchX().rows());
					//we must not change the bTraining state, because this would screw the loss value with dropout for example

					iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_minus);
					const real_t LossMinus = _calcLossF(s);

					//const bool bActivationWasDroppedOut = pDropoutMask && (real_t(0.) == pDropoutMask->get(coords));

					iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_plus);
					//const real_t dLnum = bActivationWasDroppedOut ? real_t(0) : numMult*(_calcLossF(s) - LossMinus);
					const real_t dLnum = numMult*(_calcLossF(s) - LossMinus);

					iI.gc_set_phase(_impl::gradcheck_phase::df_analytical);
					//_prepNetToBatchSize(true, m_data.batchX().rows());
					if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
					m_nn.m_Layers.fprop(m_data.batchX());
					if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
					m_nn.m_Layers.bprop(m_data.batchY());
					const real_t dLan = iI.get_analytical_value();

					_checkErr(lIdx, dLan, dLnum, coords, maxZerodLdA, zerodLdA);
				}
				if (!m_failedLayerIdx){
					if (maxZerodLdA && zerodLdA) {
						STDCOUTL("Note: "<< zerodLdA << " values were zeroed (acceptable up to " << maxZerodLdA << ")");
					}
					if (m_ngcSetts.evalSetts.dLdA_setts.percOfZeros >= 100 && maxZerodLdA <= zerodLdA) {
						STDCOUTL("Warning: all dL/dA was zeroed. It's normal for a setups with LPHG, but error for others");
					}
					if(m_ngcSetts.bVerbose) STDCOUTL("Passed.");
				}
			}

			void _checkWeight(const layer_index_t& lIdx, const mtx_coords_t& coords
				, const neurons_count_t& maxZerodLdW, neurons_count_t& zerodLdW) noexcept
			{
				const auto doubleSs = m_ngcSetts.stepSize * 2;
				auto& iI = m_nn.get_iInspect();

				m_data.nextBatch(m_nn.get_iRng(), m_nn.get_iMath());

				const bool bLayerMayBeExcluded = ::std::any_of(m_ngcSetts.layerCanSkipExecIds.cbegin(), m_ngcSetts.layerCanSkipExecIds.cend(), [lIdx](const auto i) {
					return i == lIdx;
				});
				const size_t s = m_ngcSetts.bForceSeed ? ::std::time(0) : 0;

				iI.gc_prep_check_layer(lIdx, _impl::gradcheck_paramsGroup::dLdW, coords, bLayerMayBeExcluded);

				//_prepNetToBatchSize(false, m_data.batchX().rows());

				iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_minus);
				const real_t LossMinus = _calcLossF(s);

				const auto curBs = iI.get_real_batch_size();
				iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_plus);
				const real_t LossPlus = _calcLossF(s);

				const real_t dLnum = (LossPlus - LossMinus) / (doubleSs*curBs);

				iI.gc_set_phase(_impl::gradcheck_phase::df_analytical);
				//_prepNetToBatchSize(true, m_data.batchX().rows());
				if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
				m_nn.m_Layers.fprop(m_data.batchX());
				if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
				m_nn.m_Layers.bprop(m_data.batchY());
				const real_t dLan = iI.get_analytical_value();

				_checkErr(lIdx, dLan, dLnum, coords, maxZerodLdW, zerodLdW);
			}

			void _checkdLdW(const layer_index_t& lIdx, const neurons_count_t neuronsCnt, const neurons_count_t incNeuronsCnt)noexcept
			{
				const auto checkNeuronsCnt = m_ngcSetts.groupSetts.countToCheck(neuronsCnt);
				const auto checkIncWeightsCnt = m_ngcSetts.subgroupSetts.countToCheck(incNeuronsCnt);
				if (m_ngcSetts.bVerbose) STDCOUTL("dL/dW: " << checkNeuronsCnt << " neurons (out of total " << neuronsCnt
					<< ") with " << checkIncWeightsCnt << " incoming weights (total " << incNeuronsCnt << ")...");
				
				m_grpIdx.resize(neuronsCnt);
				::std::iota(m_grpIdx.begin(), m_grpIdx.end(), neurons_count_t(0));
				::std::random_shuffle(m_grpIdx.begin(), m_grpIdx.end(), m_nn.get_iRng());
				//m_nn.get_iRng().gen_vector_gtz(&m_grpIdx[0], checkNeuronsCnt, neuronsCnt - 1);

				m_subgrpIdxs.resize(incNeuronsCnt);
				::std::iota(m_subgrpIdxs.begin(), m_subgrpIdxs.end(), neurons_count_t(0));

				const neurons_count_t maxZerodLdW = (checkNeuronsCnt*(checkIncWeightsCnt + 1)*m_ngcSetts.evalSetts.dLdW_setts.percOfZeros) / 100;
				neurons_count_t zerodLdW = 0;

				//prev layer neurons weights
				for (neurons_count_t i = 0; i < checkNeuronsCnt; ++i) {
					if (m_failedLayerIdx) break;
					auto neurIdx = m_grpIdx[i];

					::std::random_shuffle(m_subgrpIdxs.begin(), m_subgrpIdxs.end(), m_nn.get_iRng());
					//m_nn.get_iRng().gen_vector_gtz(&m_subgrpIdxs[0], checkIncWeightsCnt, incNeuronsCnt - 1);

					for (neurons_count_t j = 0; j < checkIncWeightsCnt; ++j) {
						if (m_failedLayerIdx) break;
						_checkWeight(lIdx, mtx_coords_t(neurIdx, m_subgrpIdxs[j]), maxZerodLdW, zerodLdW);
					}
				}
				if (!m_failedLayerIdx){
					if (m_ngcSetts.bVerbose) STDCOUTL("Bias weights in dL/dW: " << checkNeuronsCnt 
						<< " biases (out of total " << neuronsCnt << ")...");

					//bias weights
					//m_nn.get_iRng().gen_vector_gtz(&m_grpIdx[0], checkNeuronsCnt, neuronsCnt - 1);
					::std::random_shuffle(m_grpIdx.begin(), m_grpIdx.end(), m_nn.get_iRng());
					for (neurons_count_t i = 0; i < checkNeuronsCnt; ++i) {
						if (m_failedLayerIdx) break;
						_checkWeight(lIdx, mtx_coords_t(m_grpIdx[i], incNeuronsCnt), maxZerodLdW, zerodLdW);
					}
				}				

				if (!m_failedLayerIdx){
					if (maxZerodLdW && zerodLdW) {
						STDCOUTL("Note: " << zerodLdW << " values were zeroed (acceptable up to " << maxZerodLdW << ")");
					}

					if (m_ngcSetts.evalSetts.dLdW_setts.percOfZeros >= 100 && maxZerodLdW == zerodLdW) {
						STDCOUTL("Warning: all dL/dW was zeroed. It can be OK for a setups with LPHG, but it's an error for others");
					}
					if( m_ngcSetts.bVerbose) STDCOUTL("Passed.");
				}
			}

			void _checkErr(const layer_index_t& lIdx, const real_t& dLan, const real_t& dLnum, const mtx_coords_t& coords
				, const neurons_count_t& maxZerodL, neurons_count_t& zerodL)noexcept
			{
				const char* checkName;
				const real_t* warnThrsh, *failThrsh;
				const auto grp = m_nn.get_iInspect().gc_getCurParamsGroup();
				switch (grp){
				case _impl::gradcheck_paramsGroup::dLdA:
					checkName = "dL/dA";
					warnThrsh = &m_ngcSetts.evalSetts.dLdA_setts.relErrWarnThrsh;
					failThrsh = &m_ngcSetts.evalSetts.dLdA_setts.relErrFailThrsh;
					break;

				case _impl::gradcheck_paramsGroup::dLdW:
					checkName = "dL/dW";
					warnThrsh = &m_ngcSetts.evalSetts.dLdW_setts.relErrWarnThrsh;
					failThrsh = &m_ngcSetts.evalSetts.dLdW_setts.relErrFailThrsh;
					break;

				default:
					NNTL_ASSERT(!"WTF??");
					_fail(lIdx, "Unexpected parameter group passed", coords);
					return;
				}

				if (0 == dLan && 0 == dLnum) {
					if (!(grp == _impl::gradcheck_paramsGroup::dLdW && lIdx == 1 && m_ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer)) {
						if (++zerodL > maxZerodL) {
							char _s[128+32];
							sprintf_s(_s, "Too many (%d of max %d) of %s equal to zero. If it is acceptable, adjust corresponding gradcheck_settings::evalSetts.percOfZerodLd*", zerodL, maxZerodL, checkName);
							_fail(lIdx, _s, coords);
						}
					}//yes, this is ugly hack to get rid of underlying layer's kinks
				} else {
					const real_t relErr = ::std::abs(dLan - dLnum) / ::std::max(::std::abs(dLan), ::std::abs(dLnum));
					NNTL_ASSERT(!::std::isnan(relErr) && !::std::isinf(relErr));

					if (relErr > *warnThrsh) {
						if (relErr < *failThrsh) {
							char _s[128];
							sprintf_s(_s, "Relative error = %g (thrsh %g). Probably not an error.", relErr, *warnThrsh);
							_warn(_s, coords);
						} else {
							char _s[128];
							sprintf_s(_s, "Relative error = %g (thrsh %g). Gradient might be wrong!", relErr, *failThrsh);
							_fail(lIdx, _s, coords);
						}
					}
				}
			}

			void _say_final(const mtx_coords_t& coords)const noexcept {
				STDCOUT(" coordinates: (" << coords.first << ", " << coords.second << "). Following data rows were used in the batch: ");
				auto it = m_data.curBatchIdxs();
				const auto itE = m_data.curBatchIdxsEnd();
				while (it < itE) {
					STDCOUT(*it++ << ",");
				}
				STDCOUT(::std::endl);
			}

			void _warn(const char* reason, const mtx_coords_t& coords)const noexcept {
				STDCOUT("Warning: " << reason << ::std::endl << "Entry");
				_say_final(coords);
			}

			void _fail(const layer_index_t lIdx, const char* reason, const mtx_coords_t& coords)noexcept {
				m_failedLayerIdx = lIdx;
				STDCOUT("FAILED!" << ::std::endl << reason << ::std::endl << "Failed entry ");
				_say_final(coords);
			}

			real_t _calcLossF(const size_t s)noexcept {
				m_nn.m_Layers.prepToCalcLossAddendum();//cleanup cached loss version to always recalculate it from scratch, because we aren't interested in any cheats here.
				if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
				m_nn.m_Layers.fprop(m_data.batchX());
				return m_nn._calcLoss<false>(nullptr, m_data.batchY());
			}
		};

	public:

		// Safe to call during training phase (from the onEpochEndCB-functor parameter passed to .train())
		// Does not affect the nnet state (besides changing RNG state)
		// If the gradient check fails assert message is shown and returns false
		// Requires the iInspect_t to be derived from inspectors::GradCheck<>
		// Read http://cs231n.github.io/neural-networks-3/#gradcheck before using this function.
		// NB: Most of our loss functions implementations are very simple and therefore numerically unstable.
		// The first and the only real use a value of loss function is actually here, in .gradcheck().
		bool gradcheck(const realmtx_t& data_x, const realmtx_t& data_y
			, const vec_len_t batchSize = 5
			, const gradcheck_settings<real_t>& ngcSetts = gradcheck_settings<real_t>())noexcept
		{
			static_assert(inspector::is_gradcheck_inspector<iInspect_t>::value
				, "In order to perform numeric gradient check derive nnet's inspector from inspectors::GradCheck!");

			NNTL_ASSERT(batchSize > 0);
			if (0 == batchSize) {
				STDCOUTL("batchSize must be at least 1");
				return false;
			}
			const auto biggestBatchSize = ::std::max(batchSize, ngcSetts.onlineBatchSize);

			NNTL_ASSERT(data_x.cols() == m_Layers.input_layer().get_neurons_cnt() + 1 && data_x.rows() >= biggestBatchSize);
			NNTL_ASSERT(data_y.cols() == m_Layers.output_layer().get_neurons_cnt() && data_y.rows() >= biggestBatchSize);
			NNTL_ASSERT(data_x.rows() == data_y.rows());
			if (
				!(data_x.cols() == m_Layers.input_layer().get_neurons_cnt() + 1 && data_x.rows() >= biggestBatchSize)
				|| !(data_y.cols() == m_Layers.output_layer().get_neurons_cnt() && data_y.rows() >= biggestBatchSize)
				|| !(data_x.rows() == data_y.rows())
				)
			{
				STDCOUTL("Wrong data sizes passed to " << NNTL_FUNCTION);
				return false;
			}

			if (!::std::is_same<real_t,double>::value) {
				STDCOUTL("*** warning: it's significantly better to perform gradient check with double floating point precision.");
			}
			//////////////////////////////////////////////////////////////////////////
			const auto ec = _init(biggestBatchSize, biggestBatchSize, false);
			if (ErrorCode::Success != ec) {
				STDCOUTL("Failed to init nnet object for gradcheck. Reason: " << get_error_str(ec));
				return false;
			}			
			
			//walking over each and every layer in the stack and checking the gradients
			GradCheckFunctor gcf(*this, ngcSetts);

			bool bOk = gcf.performCheck(batchSize, data_x, data_y);
			if (bOk) {
				STDCOUTL("Gradient checks passed!");
			}else{
				STDCOUTL("**** Gradient check failed within a layer with idx = " << gcf.getFailedLayerIdx());
			}
			return bOk;
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// use this function for unit-tesing only
		//batchSize==0 means that _init is called for use in fprop scenario only
		ErrorCode ___init(const vec_len_t biggestFprop, vec_len_t batchSize = 0, const bool bMiniBatch = false)noexcept
		{
			return _init(biggestFprop, batchSize, bMiniBatch);
		}

		//for unit-testing only!
		common_data_t& ___get_common_data()noexcept { return get_common_data(); }
	
	};

	/*template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp)noexcept { return nnet<LayersPack>(lp); }

	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iRng_t& iR)noexcept {
		return nnet<LayersPack>(lp, nullptr, nullptr, &iR);
	}
	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iMath_t& iM)noexcept {
		return nnet<LayersPack>(lp, nullptr, &iM);
	}
	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iMath_t& iM, typename LayersPack::iRng_t& iR)noexcept {
		return nnet<LayersPack>(lp, nullptr, &iM, &iR);
	}

	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI)noexcept { return nnet<LayersPack>(lp, &iI); }

	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI, typename LayersPack::iRng_t& iR)noexcept {
		return nnet<LayersPack>(lp, &iI, nullptr, &iR);
	}
	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI, typename LayersPack::iMath_t& iM)noexcept {
		return nnet<LayersPack>(lp, &iI, &iM);
	}
	template <typename LayersPack>
	inline constexpr nnet<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI, typename LayersPack::iMath_t& iM, typename LayersPack::iRng_t& iR)noexcept {
		return nnet<LayersPack>(lp, &iI, &iM, &iR);
	}*/

	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp)noexcept { return NnT<LayersPack>(lp); }

	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iRng_t& iR)noexcept {
		return NnT<LayersPack>(lp, nullptr, nullptr, &iR);
	}
	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iMath_t& iM)noexcept {
		return NnT<LayersPack>(lp, nullptr, &iM);
	}
	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iMath_t& iM, typename LayersPack::iRng_t& iR)noexcept {
		return NnT<LayersPack>(lp, nullptr, &iM, &iR);
	}

	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI)noexcept { return NnT<LayersPack>(lp, &iI); }

	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI, typename LayersPack::iRng_t& iR)noexcept {
		return NnT<LayersPack>(lp, &iI, nullptr, &iR);
	}
	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI, typename LayersPack::iMath_t& iM)noexcept {
		return NnT<LayersPack>(lp, &iI, &iM);
	}
	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& iI, typename LayersPack::iMath_t& iM, typename LayersPack::iRng_t& iR)noexcept {
		return NnT<LayersPack>(lp, &iI, &iM, &iR);
	}

}