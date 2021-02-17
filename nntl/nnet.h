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

#include <type_traits>
#include <algorithm>
#include <filesystem>//for weights persistence

#include "common.h"

#include "errors.h"
#include "_nnet_errs.h"
#include "utils.h"
#include "train_data.h"

#include "nnet_train_opts.h"
#include "common_nn_data.h"

#include "interface/inspectors/gradcheck.h"

//we're using .mat files as a storage for weight saving/loading
#include "../../nntl/nntl/_supp/io/matfile.h"

namespace nntl {

	//dummy callbacks for .train() function.
	struct NNetCB_OnEpochEnd_Dummy {
		constexpr bool operator()(size_t /*epochEnded*/)const noexcept {
			//return false to stop learning
			return true;
		}
	};
	struct NNetCB_OnInit_Dummy {
		constexpr auto operator()()const noexcept {
			//return non success such as PostInitStopFromCallback to break learning
			return _nnet_errs::ErrorCode::Success;
		}
	};
	//////////////////////////////////////////////////////////////////////////
	// If not mentioned explicitly in a function comment, any member function of the class #supportsBatchInRow (at least it should)
	// However, it was not extensively tested in bBatchInRow() mode, so double check
	template <typename LayersPack>
	class nnet 
		: public _has_last_error<_nnet_errs>
		, public _impl::interfaces_keeper<typename LayersPack::interfaces_t>
		, public DataSetsId
	{
	private:
		typedef _impl::interfaces_keeper<typename LayersPack::interfaces_t> _base_class;

	public:
		typedef LayersPack layers_pack_t;

 		typedef typename iMath_t::realmtx_t realmtx_t;
 		typedef typename iMath_t::realmtxdef_t realmtxdef_t;

		//typedef train_data<real_t> train_data_t;
		//typedef _i_train_data<real_t> i_train_data_t;
		
		template<bool bPrioritizeThreads>
		using threads_prioritizer_tpl = ::std::conditional_t<bPrioritizeThreads
			, threads::prioritize_workers<threads::PriorityClass::Working, iThreads_t>
			, threads::_impl::prioritize_workers_dummy<threads::PriorityClass::Normal, iThreads_t>>;

		template<bool bPrioritizeThreads>
		using threads_relaxer_tpl = ::std::conditional_t<bPrioritizeThreads
			, threads::prioritize_workers<threads::PriorityClass::Normal, iThreads_t>
			, threads::_impl::prioritize_workers_dummy<threads::PriorityClass::Normal, iThreads_t>>;
		
		//defining archive types for dumping/restoring weights
	#if NNTL_MATLAB_AVAILABLE
		typedef nntl_supp::imatfile<> weights_load_archive_t;
		typedef nntl_supp::omatfile<> weights_save_archive_t;
	#else //NNTL_MATLAB_AVAILABLE
		typedef void weights_load_archive_t;
		typedef void weights_save_archive_t;
	#endif //NNTL_MATLAB_AVAILABLE

		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		layers_pack_t& m_Layers;

		_impl::layers_mem_requirements m_LMR;

		::std::vector<real_t> m_pTmpStor;

		//realmtx_t m_batch_x, m_batch_y;

		layer_index_t m_failedLayerIdx{ 0 };

		bool m_bCalcFullLossValue;//set based on nnet_train_opts::calcFullLossValue() and the value, returned by layers init()
		bool m_bRequireReinit{false};//set this flag to require nnet object and its layers to reinitialize on next call

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
		{}

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
		void dont_require_reinit()noexcept { m_bRequireReinit = false; }

		//////////////////////////////////////////////////////////////////////////
		// weights persistence (depends on Matlab archive types this moment)
		
		template<bool C=::std::is_same<void, weights_load_archive_t>::value>
		::std::enable_if_t<C, bool> read_weights(const char*const)noexcept {
			STDCOUTL("## Failing nnet::read_weights() because Matlab support is not configured.");
			return false;
		}
		template<bool C = ::std::is_same<void, weights_save_archive_t>::value>
		::std::enable_if_t<C, bool> write_weights(const char*const)noexcept {
			STDCOUTL("## Failing nnet::write_weights() because Matlab support is not configured.");
			return false;
		}

		template<bool C = ::std::is_same<void, weights_load_archive_t>::value>
		::std::enable_if_t<!C, bool> read_weights(const char*const pFile)noexcept {
			NNTL_ASSERT(pFile);
			const char*const pfx = "nnet::read_weights() ";
			if (::std::experimental::filesystem::exists(pFile)) {
				STDCOUTL(pfx << "Found file '" << pFile << "'. Trying to read it...");

				weights_load_archive_t mf;
				mf.turn_off_all_options();
				mf.m_binary_options.set(serialization::CommonOptions::serialize_weights, true);

				auto ec = mf.open(pFile);
				if (decltype(mf)::ErrorCode::Success == ec) {
					mf >> *this;
					ec = mf.get_last_error();
					if (decltype(mf)::ErrorCode::Success == ec) {
						STDCOUTL(pfx << "Success!");
						return true;
					} else {
						STDCOUTL(pfx << "Failed to read weights, code = " << ec << "=" << mf.get_error_str(ec) << ".");
					}
				} else {
					STDCOUTL(pfx << "Failed to open file with code = " << ec << "=" << mf.get_error_str(ec) << ".");
				}
			} else {
				STDCOUTL(pfx << "Weights file '" << pFile << "' not found.");
			}
			return false;
		}
		template<bool C = ::std::is_same<void, weights_save_archive_t>::value>
		::std::enable_if_t<!C, bool> write_weights(const char*const pFile)noexcept {
			NNTL_ASSERT(pFile);
			const char*const pfx = "nnet::write_weights() ";
			STDCOUTL(pfx << "Going to write weights to file '" << pFile << "'");

			weights_save_archive_t mf;
			mf.turn_off_all_options();
			mf.m_binary_options.set(serialization::CommonOptions::serialize_weights, true);

			auto ec = mf.open(pFile);
			if (decltype(mf)::ErrorCode::Success == ec) {
				mf << *this;
				ec = mf.get_last_error();
				if (decltype(mf)::ErrorCode::Success == ec) {
					STDCOUTL(pfx << "Success!");
					return true;
				} else {
					STDCOUTL(pfx << "Failed to write weights, code = " << ec << "=" << mf.get_error_str(ec) << ".");
				}
			} else {
				STDCOUTL(pfx << "Failed to open file with code = " << ec << "=" << mf.get_error_str(ec) << ".");
			}
			return false;
		}
		
	protected:
		//////////////////////////////////////////////////////////////////////////
		// the following functions are used mostly in training process

		//returns train loss value
		template<typename TrainDataT, typename Observer>
		real_t _report_training_progress(const numel_cnt_t epoch, TrainDataT& td, const ::std::chrono::nanoseconds& tElapsed, Observer& obs) noexcept {
			//relaxing thread priorities (we don't know in advance what callback functions actually do, so better relax it)
			//threads_relaxer_tpl<bPrioritizeThreads> pw(get_iMath().ithreads());
			// note sure it's necessary so pending for removal. If should be restored - wrap each obs.report_results() call instead of
			// single object?

			m_Layers.resetCalcLossAddendum();//no need to additionally check m_bCalcFullLossValue, because
			// resetCalcLossAddendum() is extremely cheap operation

			const real_t trainLoss = calcLossAndReport(td, train_set_id, obs);
			const auto testLoss = calcLossAndReport(td, test_set_id, obs);

			obs.on_training_fragment_end(epoch, trainLoss, testLoss, tElapsed);			
			return trainLoss;
		}
		
		void _fprop(const realmtx_t& data_x)noexcept {
			//preparing for evaluation
			_set_mode_and_batch_size(data_x.batch_size());
			m_Layers.fprop(data_x);
		}

	public:
		//note that nnet object MUST already be initialized via train() or similar function to call
		//observer used here MUST be initialized in conjunction with this nnet object.
		// Intended to be used during/after training
		template<typename TrainDataT, typename Observer>
		real_t calcLossAndReport(TrainDataT& td, const data_set_id_t dataSetId, Observer& obs) noexcept {
			NNTL_UNREF(obs); //observer may have only dummy functions that gets removed by optimizer.
			NNTL_ASSERT(dataSetId >= 0 && dataSetId < td.datasets_count());
			NNTL_ASSERT(!td.empty());

			//resetting loss calculation if there's dependence on activation values
			if (m_LMR.bLossAddendumDependsOnActivations) m_Layers.resetCalcLossAddendum();
			//no need to additionally check m_bCalcFullLossValue, because resetCalcLossAddendum() is extremely cheap operation

			auto& iI = get_iInspect();
			iI.train_preCalcError(dataSetId);

			const auto& cd = get_const_common_data();
			const auto& activations = *m_Layers.output_layer().get_activations_storage();//we won't touch it until it's valid
			real_t lossVal(0);

			const numel_cnt_t batchesCnt = td.walk_over_set(dataSetId, cd);
			NNTL_ASSERT(batchesCnt > 0);

			obs.report_results_begin(dataSetId, batchesCnt);

			for (numel_cnt_t bi = 0; bi < batchesCnt; ++bi) {
				td.next_subset(bi, cd);
				lossVal += _calcLoss4batch(td.batchX(), td.batchY());

				NNTL_ASSERT(m_Layers.output_layer().is_activations_valid());
				NNTL_ASSERT(&activations == m_Layers.output_layer().get_activations_storage());//sanity check
				obs.report_results(bi, activations, td.batchY(), cd);
			}

			//after calculating loss based on batch data, we must add a loss based on layers properties/parameters
			if (m_bCalcFullLossValue) lossVal += m_Layers.calcLossAddendum();

			obs.report_results_end(lossVal);

			iI.train_postCalcError();
			return lossVal;
		}

		bool is_initialized(const vec_len_t biggestFprop, const vec_len_t batchSize = 0)const noexcept {
			return get_common_data().is_initialized()
				&& biggestFprop <= get_common_data().input_max_fprop_batch_size()
				&& batchSize <= get_common_data().input_training_batch_size();
		}

	protected:
		// note that the function only calculate loss that depends on model prediction and data_y.
		// It does NOT calculates and adds to loss any additional loss values that depends on layers weights and so on (m_Layers.calcLossAddendum())
		template<typename YT>
		real_t _calcLoss4batch_nonnormalized(const realmtx_t& data_x, const math::smatrix<YT>& data_y) noexcept {
			NNTL_ASSERT(data_x.batch_size() == data_y.batch_size());
			_fprop(data_x);

			return m_Layers.output_layer().calc_loss(data_y);
			//if (m_bCalcFullLossValue) lossValue += m_Layers.calcLossAddendum();
			//WRONG! Call it after each batch processed!
			//return lossValue;
		}		
		template<typename YT>
		real_t _calcLoss4batch(const realmtx_t& data_x, const math::smatrix<YT>& data_y) noexcept {
			return _calcLoss4batch_nonnormalized(data_x, data_y) / data_y.batch_size();
		}

		bool _is_initialized(const vec_len_t biggestFprop, const vec_len_t batchSize)const noexcept {
			return !m_bRequireReinit && is_initialized(biggestFprop, batchSize);
		}

		//
		// biggestFprop - the biggest expected batch size for inferencing/fprop
		// batchSize - the biggest expected batch size for training/bprop (as well as inferencing/fprop for training also)
		//		batchSize==0 means that _init is called for use in fprop scenario only
		ErrorCode _init(const vec_len_t biggestFprop, const vec_len_t batchSize = 0, const bool bMiniBatch = false
			, const numel_cnt_t maxEpoch = 1)noexcept
		{
			NNTL_ASSERT(biggestFprop > 0 && biggestFprop >= batchSize);
			if (_is_initialized(biggestFprop, batchSize)) {
				if (!get_iMath().init()) return ErrorCode::CantInitializeIMath;
				get_iInspect().init_nnet(m_Layers.total_layers(), maxEpoch);
				return ErrorCode::Success;
			}

			//#TODO we must be sure here that no internal objects settings will be hurt during deinit phase
			deinit();
			return _full_init(biggestFprop, batchSize, bMiniBatch, maxEpoch);
		}

		template<typename TdT>
		ErrorCode _init4train(TdT& td, vec_len_t& maxFPropSize, vec_len_t& maxBatchSize, bool& bMiniBatch
			, const numel_cnt_t maxEpoch, const bool bForceReinitTrainData)noexcept
		{
			if (bForceReinitTrainData || !td.is_initialized4train(maxFPropSize, maxBatchSize, bMiniBatch)) {
				const auto ec = td.init4train(get_iMath(), maxFPropSize, maxBatchSize, bMiniBatch);
				if (ErrorCode::Success != ec) return _set_last_error(ec);
			}
			NNTL_ASSERT(maxFPropSize > 0 && maxBatchSize > 0 && maxBatchSize <= maxFPropSize);

			if (_is_initialized(maxFPropSize, maxBatchSize)) {
				//must call get_iMath().init() b/c td.init* could do iM.preinit()
				if (!get_iMath().init()) return ErrorCode::CantInitializeIMath;
				get_iInspect().init_nnet(m_Layers.total_layers(), maxEpoch);
				return ErrorCode::Success;
			}

			//#TODO we must be sure here that no internal objects settings will be hurt during deinit phase
			deinit();
			//because of deinit, we have to allow td to call preinit() again.
			td.preinit_iMath(get_iMath());

			return _full_init(maxFPropSize, maxBatchSize, bMiniBatch, maxEpoch);
		}

		ErrorCode _full_init(const vec_len_t biggestFprop, const vec_len_t batchSize, const bool bMiniBatch, const numel_cnt_t maxEpoch)noexcept {
			NNTL_ASSERT(biggestFprop > 0 && biggestFprop >= batchSize && batchSize >= 0);
			//call deinit if appropriate in caller
			//deinit();

			get_iInspect().init_nnet(m_Layers.total_layers(), maxEpoch);

			bool bInitFinished = false;
			utils::scope_exit _run_deinit([this, &bInitFinished]()noexcept {
				if (!bInitFinished) deinit();
			});

			get_common_data().init(biggestFprop, batchSize);

			m_failedLayerIdx = 0;
			const auto le = m_Layers.init_layers(get_common_data(), m_LMR);
			if (ErrorCode::Success != le.first) {
				m_failedLayerIdx = le.second;
				return le.first;
			}

			if (batchSize > 0 && 0 == m_LMR.maxSingledLdANumel) m_LMR.maxSingledLdANumel = 1;//just to make assertions happy

			if (!get_iMath().init()) return ErrorCode::CantInitializeIMath;
			if (!get_iRng().init_rng()) return ErrorCode::CantInitializeIRng;
			//#TODO shouldn't we reseed RNG here?
			//#BUGBUG ??

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
			NNTL_UNREF(bMiniBatch);
			//const vec_len_t train_x_cols = vec_len_t(1) + m_Layers.input_layer().get_neurons_cnt()//1 for bias column
			//	, train_y_cols = m_Layers.output_layer().get_neurons_cnt();

			return m_LMR.maxMemLayerTrainingRequire
				+ (batchSize > 0 
					? m_Layers.m_a_dLdA.size()*m_LMR.maxSingledLdANumel
					//+ (bMiniBatch ? (realmtx_t::sNumel(batchSize, train_x_cols) + realmtx_t::sNumel(batchSize, train_y_cols)) : 0)
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
				NNTL_UNREF(bMiniBatch);
				/*if (bMiniBatch) {
					NNTL_ASSERT(batchSize);
					m_batch_x.useExternalStorage(&tempMemStorage[spreadTempMemSize], batchSize, m_Layers.input_layer().get_neurons_cnt() + 1, true);
					spreadTempMemSize += m_batch_x.numel();
					m_batch_x.set_biases();
					m_batch_y.useExternalStorage(&tempMemStorage[spreadTempMemSize], batchSize, m_Layers.output_layer().get_neurons_cnt());
					spreadTempMemSize += m_batch_y.numel();
				}*/

				//2. dLdA
				//#TODO: better move it to m_Layers
				for (auto& m : m_Layers.m_a_dLdA) {
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
		
		//if bs==0 then "set training mode with batchsize = cd.training_batch_size()", else set inference mode with batchsize==bs
		void _set_mode_and_batch_size(vec_len_t bs)noexcept {
			NNTL_ASSERT(bs >= 0);
			const bool bIsTraining = bs == 0;
			auto& cd = get_common_data();
			bs = bIsTraining ? cd.input_training_batch_size() : bs;
			if (cd.set_mode_and_batch_size(bIsTraining, bs)) {
				m_Layers.on_batch_size_change(cd.input_batch_size());
			}
		}

		//similar to _set_mode_and_batch_size(), but unconditionally sets training mode with given non zero batchsize
		void _set_training_mode_and_batch_size(const vec_len_t bs)noexcept {
			NNTL_ASSERT(bs > 0 && bs <= get_common_data().input_training_batch_size());
			if (get_common_data().set_mode_and_batch_size(true, bs)) {
				m_Layers.on_batch_size_change(get_common_data().input_batch_size());
			}
		}

		void _set_inference_mode_and_batch_size(const vec_len_t bs)noexcept {
			NNTL_ASSERT(bs > 0 && bs <= get_common_data().input_max_fprop_batch_size());
			if (get_common_data().set_mode_and_batch_size(false, bs)) {
				m_Layers.on_batch_size_change(get_common_data().input_batch_size());
			}
		}

	public:
		void deinit()noexcept {
			m_Layers.deinit_layers();
			get_iRng().deinit_rng();
			get_iMath().deinit();
			get_common_data().deinit();
			m_bRequireReinit = false;
			m_LMR.zeros();
			//m_batch_x.clear();
			//m_batch_y.clear();
			m_pTmpStor.clear();
			m_pTmpStor.shrink_to_fit();
		}

		// note that TrainDataT& td doesn't have a const modifier. That's because actually it has to be a statefull modifiable
		// resettable wrapper over some const data storage. So, it's ok to pass it here without const modifier, because the only
		// thing that is allowed to be changed in it is the wrapper state, but not the data itself.
		template <bool bPrioritizeThreads = true, typename TrainDataT, typename TrainOptsT
			, typename onEpochEndCbT = NNetCB_OnEpochEnd_Dummy
			, typename onNnetInitCbT = NNetCB_OnInit_Dummy
		>
		ErrorCode train(TrainDataT& td, TrainOptsT& opts
			, onEpochEndCbT&& onEpochEndCB = NNetCB_OnEpochEnd_Dummy()
			, onNnetInitCbT&& onInitCB = NNetCB_OnInit_Dummy()
		)noexcept
		{
			static_assert(is_train_data_intf<TrainDataT>::value, "td object MUST be derived from _i_train_data interface");
			static_assert(::std::is_same<typename TrainDataT::x_t, real_t>::value, "TrainDataT::x_t must be same as real_t");

			//just leave it here
			global_denormalized_floats_mode();

			if (td.empty()) return _set_last_error(ErrorCode::InvalidTD);
			if (td.xWidth() != m_Layers.input_layer().get_neurons_cnt()) return _set_last_error(ErrorCode::InvalidInputLayerNeuronsCount);
			if (!td.isSuitableForOutputOf(m_Layers.output_layer().get_neurons_cnt())) return _set_last_error(ErrorCode::InvalidOutputLayerNeuronsCount);

			//scheduling deinitialization with scope_exit to forget about return statements
			utils::scope_exit layers_deinit([this, &opts, &td]()noexcept {
				if (opts.ImmediatelyDeinit()) {
					td.deinit4all();
					deinit();
				}
			});

			const numel_cnt_t maxEpoch = opts.maxEpoch();
			vec_len_t maxFPropSize = opts.maxFpropSize(), maxBatchSize = opts.batchSize();
			const bool bRepOnlyTime = opts.bReportOnlyTime();
			bool bMiniBatch = true;//default value
			m_bCalcFullLossValue = opts.calcFullLossValue();

			//////////////////////////////////////////////////////////////////////////
			// perform layers initialization, gather temp memory requirements, then allocate and spread temp buffers
			auto ec = _init4train(td, maxFPropSize, maxBatchSize, bMiniBatch, maxEpoch, opts.bForceReinitTD());
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			if (m_bCalcFullLossValue) m_bCalcFullLossValue = m_LMR.bLossAddendumDependsOnWeights || m_LMR.bLossAddendumDependsOnActivations;

			ec = ::std::forward<onNnetInitCbT>(onInitCB)();
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			//////////////////////////////////////////////////////////////////////////
			const auto& reportEpochCond = opts.getCondEpochEval();			
			const auto divergenceCheckLastEpoch = opts.divergenceCheckLastEpoch();

			if (! opts.observer().init(maxEpoch, td, get_const_common_data())) return _set_last_error(ErrorCode::CantInitializeObserver);
			utils::scope_exit observer_deinit([&opts]()noexcept {
				opts.observer().deinit();
			});
			
			//making initial report
			opts.observer().on_training_start(td, maxBatchSize, maxFPropSize, m_LMR.totalParamsToLearn);
			
			// #note should depend on bPrioritizeThreads value to relax priorities for callbacks?
			_report_training_progress(-1, td, ::std::chrono::nanoseconds(0), opts.observer());
			_set_mode_and_batch_size(0);//prepare for training (sets to maxBatchSize, that's already stored in common_data structure)

#if NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH
			STDCOUTL("Denormals check... " << (get_iMath().ithreads().denormalsOnInAnyThread() ? "FAILED!!!" : "passed.") 
				<< ::std::endl << " Going to check it further each epoch and report failures only.");
#endif//NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH

			static_assert(::std::chrono::steady_clock::is_steady,"");
			const auto trainingBeginsAt = ::std::chrono::steady_clock::now();//starting training timer
			auto epochPeriodBeginsAt = ::std::chrono::steady_clock::now();//starting epoch timer

			{
				threads_prioritizer_tpl<bPrioritizeThreads> pw(get_iMath().ithreads());//raising thread priorities for faster computation
				auto& iI = get_iInspect();

				for (numel_cnt_t epochIdx = 0; epochIdx < maxEpoch; ++epochIdx) {					
					const numel_cnt_t numBatches = td.on_next_epoch(epochIdx, get_const_common_data());
					NNTL_ASSERT(numBatches > 0);
					iI.train_epochBegin(epochIdx, numBatches);

					for (numel_cnt_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
						iI.train_batchBegin(batchIdx);

						td.on_next_batch(batchIdx, get_const_common_data());

						const auto& batch_x = td.batchX();
						const auto& batch_y = td.batchY();
						NNTL_ASSERT(batch_x.emulatesBiases() && !batch_y.emulatesBiases());
						NNTL_ASSERT(batch_x.test_biases_strict());
						NNTL_ASSERT(batch_x.sample_size() == td.xWidth() && batch_y.sample_size() == td.yWidth());
						NNTL_ASSERT(batch_x.batch_size() == batch_y.batch_size() && batch_x.batch_size() == get_const_common_data().input_batch_size());
						NNTL_ASSERT(batch_x.batch_size() == maxBatchSize);

						iI.train_preFprop(batch_x);
						m_Layers.fprop(batch_x);

						iI.train_preBprop(batch_y);
						m_Layers.bprop(batch_y);

						iI.train_batchEnd();
					}

					const bool bCheckForDivergence = epochIdx < divergenceCheckLastEpoch;
					if (reportEpochCond(epochIdx) || bCheckForDivergence) {
						const auto epochPeriodEnds = ::std::chrono::steady_clock::now();
						const auto periodTime = epochPeriodEnds - epochPeriodBeginsAt;
						epochPeriodBeginsAt = epochPeriodEnds;//restarting period timer

						if (bRepOnlyTime && !bCheckForDivergence) {
							static constexpr char* szReportFmt = "% 3zd/%-3zd %3.1fs (time report only)";
							static constexpr unsigned uBufSize = 64;

							char szRep[uBufSize];
							const real_t secs = real_t(periodTime.count()) * (real_t(1.) / real_t(1e9));

							sprintf_s(szRep, uBufSize, szReportFmt, epochIdx + 1, maxEpoch, secs);
							STDCOUTL(szRep);
						} else {
							// #note should depend on bPrioritizeThreads value to relax priorities for callbacks?
							const auto trainLoss = _report_training_progress(epochIdx, td, periodTime, opts.observer());
							if (bCheckForDivergence && trainLoss >= opts.divergenceCheckThreshold())
								return _set_last_error(ErrorCode::NNDiverged);
						}
					}

					iI.train_epochEnd();

#if NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH
					if (get_iMath().ithreads().denormalsOnInAnyThread()) {
						STDCOUTL("*** denormals check FAILED!!!");
					}
#endif//NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH

					// #note should depend on bPrioritizeThreads value to relax priorities for callbacks?
					if (!onEpochEndCB(/**this, opts,*/ epochIdx)) break;

					//should always call _set_mode_and_batch_size(0) because onEpochEndCB could change this object state
					_set_mode_and_batch_size(0);
				}
			}

			const auto totalTrainTime = ::std::chrono::steady_clock::now() - trainingBeginsAt;
			if (bRepOnlyTime) {
				static constexpr char* szReportFmt = "%-3zd training epochs (%zd params) took %3.1fs (time report only)";
				static constexpr unsigned uBufSize = 128;

				char szRep[uBufSize];
				const real_t secs = real_t(totalTrainTime.count()) * (real_t(1.) / real_t(1e9));

				sprintf_s(szRep, uBufSize, szReportFmt, maxEpoch, m_LMR.totalParamsToLearn, secs);
				STDCOUTL(szRep);
			}else opts.observer().on_training_end(totalTrainTime);

			return _set_last_error(ErrorCode::Success);
		}

		//note that as there's no train_data given, it's caller responsibility to execute
		//td.preinit_iMath(get_iMath()); and get_iMath().init() afterwards to allow td object to safe use iMath's internal memory
		ErrorCode init4fixedBatchFprop(const vec_len_t fpropBatchSize)noexcept {
			NNTL_ASSERT(fpropBatchSize);
			const auto ec = _init(fpropBatchSize);
			if (ErrorCode::Success != ec) return _set_last_error(ec);
			_set_inference_mode_and_batch_size(fpropBatchSize);
			return _set_last_error(ec);
		}


		void prepare_to_doFixedBatchFprop(const vec_len_t bs)noexcept {
			_set_inference_mode_and_batch_size(bs);
		}
		//designed to be used in conjunction with _i_train_data::walk_over_set() + next_subset() + batchX()
		//batchX.batch_size() must never exceed the fpropBatchSize passed to init4fixedBatchFprop() if used in other context
		void doFixedBatchFprop(const realmtx_t& batchX)noexcept {
			// if the batch size passed to init*() was not multiple of the set size, then the last batch of the set will contain less data
			const auto bs = batchX.batch_size();
			if (bs != get_const_common_data().input_batch_size()) {
				_set_inference_mode_and_batch_size(bs);
			}
			m_Layers.fprop(batchX);
		}

		//note that as there's no train_data given, it's caller responsibility to execute
		//td.preinit_iMath(get_iMath()); and get_iMath().init() afterwards to allow td object to safe use iMath's internal memory
		ErrorCode fprop(const realmtx_t& data_x)noexcept {
			const auto ec = _init(data_x.batch_size());
			if (ErrorCode::Success != ec) return _set_last_error(ec);

			_fprop(data_x);
			return _set_last_error(ec);
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
				if (m_nn.get_common_data().set_mode_and_batch_size(m__bOrigInTraining, m__origBatchSize))
					m_nn.m_Layers.on_batch_size_change(m__origBatchSize);
				m_nn._unblockLearning();
			}
			GradCheckFunctor(nnet& n, const gradcheck_settings<real_t>& ngcSetts)noexcept
				: m_nn(n), m_ngcSetts(ngcSetts), m_outputLayerIdx(n.m_Layers.output_layer().get_layer_idx())
				//saving nnet mode & affected state
				, m__bOrigInTraining(n.get_common_data().is_training_mode())
				, m__bOrigCalcFullLossValue(n.m_bCalcFullLossValue)
				, m__origBatchSize(n.get_common_data().input_batch_size())
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
				m_data.init(biggestBatch == data_x.batch_size() ? 0 : biggestBatch, data_x, &data_y);
				
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

				if (_isLayerIdInList(m_ngcSetts.ignoreLayerIds, lyr.get_layer_idx())) {
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
					} else if (m_ngcSetts.bVerbose) STDCOUTL("*** NB: output layer dL/dA check is skipped by design. Assuming "
						"it's bugs (if any) will be caught by lower layers dL/dA check");
					break;

				default:
					NNTL_ASSERT(!"WTF?");
					break;
				}
			}

		protected:
			static bool _isLayerIdInList(const ::std::vector<layer_index_t>& vec, const layer_index_t lidx)noexcept {
				return ::std::any_of(vec.cbegin(), vec.cend(), [lidx](const auto v)noexcept { return lidx == v; });
			}

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
				auto& cd = m_nn.get_common_data();
				if (cd.set_mode_and_batch_size(bTraining, batchSize))
					m_nn.m_Layers.on_batch_size_change(cd.input_batch_size());
			}

			template<typename LayerT>
			::std::enable_if_t<is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT& lyr)noexcept {
				lyr.for_each_packed_layer_down(*this);
			}
			template<typename LayerT>
			::std::enable_if_t<!is_layer_pack<LayerT>::value> _checkInnerLayers(LayerT&)const noexcept {}
			
			void _checkdLdA(const layer_index_t lIdx, const neurons_count_t neuronsCnt
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
					const bool bLayerMayBeExcluded = _isLayerIdInList(m_ngcSetts.layerCanSkipExecIds, lIdx);

					const size_t s = m_ngcSetts.bForceSeed ? ::std::time(0) : 0;
					iI.gc_prep_check_layer(lIdx, _impl::gradcheck_paramsGroup::dLdA, coords, bLayerMayBeExcluded);

					//_prepNetToBatchSize(false, m_data.batchX().batch_size());
					//we must not change the bTraining state, because this would screw the loss value with dropout for example

					iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_minus);
					const real_t LossMinus = _calcLossF(s);

					//const bool bActivationWasDroppedOut = pDropoutMask && (real_t(0.) == pDropoutMask->get(coords));

					iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_plus);
					//const real_t dLnum = bActivationWasDroppedOut ? real_t(0) : numMult*(_calcLossF(s) - LossMinus);
					const real_t dLnum = numMult*(_calcLossF(s) - LossMinus);

					iI.gc_set_phase(_impl::gradcheck_phase::df_analytical);
					//_prepNetToBatchSize(true, m_data.batchX().batch_size());
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

			void _checkWeight(const layer_index_t lIdx, const mtx_coords_t& coords
				, const neurons_count_t& maxZerodLdW, neurons_count_t& zerodLdW) noexcept
			{
				const auto doubleSs = m_ngcSetts.stepSize * 2;
				auto& iI = m_nn.get_iInspect();

				m_data.nextBatch(m_nn.get_iRng(), m_nn.get_iMath());

				const bool bLayerMayBeExcluded = _isLayerIdInList(m_ngcSetts.layerCanSkipExecIds, lIdx);
				const size_t s = m_ngcSetts.bForceSeed ? ::std::time(0) : 0;

				iI.gc_prep_check_layer(lIdx, _impl::gradcheck_paramsGroup::dLdW, coords, bLayerMayBeExcluded);

				//_prepNetToBatchSize(false, m_data.batchX().batch_size());

				iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_minus);
				const real_t LossMinus = _calcLossF(s);

				const auto curBs = iI.get_real_batch_size();
				iI.gc_set_phase(_impl::gradcheck_phase::df_numeric_plus);
				const real_t LossPlus = _calcLossF(s);

				const real_t dLnum = (LossPlus - LossMinus) / (doubleSs*curBs);

				iI.gc_set_phase(_impl::gradcheck_phase::df_analytical);
				//_prepNetToBatchSize(true, m_data.batchX().batch_size());
				if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
				m_nn.m_Layers.fprop(m_data.batchX());
				if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);
				m_nn.m_Layers.bprop(m_data.batchY());
				const real_t dLan = iI.get_analytical_value();

				_checkErr(lIdx, dLan, dLnum, coords, maxZerodLdW, zerodLdW);
			}

			void _checkdLdW(const layer_index_t lIdx, const neurons_count_t neuronsCnt, const neurons_count_t incNeuronsCnt)noexcept
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
// 					if (m_ngcSetts.bVerbose && zerodLdW > 0) STDCOUTL("Note, that there was " << zerodLdW << "/" << maxZerodLdW
// 						<< " zeroed dL/dW's out of total " << checkIncWeightsCnt << " tested.");
				}
				if (!m_failedLayerIdx){
					if (m_ngcSetts.bVerbose) STDCOUTL("Bias weights in dL/dW: " << checkNeuronsCnt 
						<< " biases (out of total " << neuronsCnt << ")...");

					//const auto curZeroed = zerodLdW;
					//bias weights
					//m_nn.get_iRng().gen_vector_gtz(&m_grpIdx[0], checkNeuronsCnt, neuronsCnt - 1);
					::std::random_shuffle(m_grpIdx.begin(), m_grpIdx.end(), m_nn.get_iRng());
					for (neurons_count_t i = 0; i < checkNeuronsCnt; ++i) {
						if (m_failedLayerIdx) break;
						_checkWeight(lIdx, mtx_coords_t(m_grpIdx[i], incNeuronsCnt), maxZerodLdW, zerodLdW);
					}
// 					const auto zDiff = curZeroed - zerodLdW;
// 					if (m_ngcSetts.bVerbose && zDiff > 0) STDCOUTL("Note, that there was " << zDiff
// 						<< " zeroed bias's dL/dW's out of total " << checkNeuronsCnt << " tested.");
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

			void _checkErr(const layer_index_t lIdx, const real_t& dLan, const real_t& dLnum, const mtx_coords_t& coords
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
					++zerodL;
					if (!(grp == _impl::gradcheck_paramsGroup::dLdW && lIdx == 1 && m_ngcSetts.evalSetts.bIgnoreZerodLdWInUndelyingLayer)) {
						if (zerodL > maxZerodL) {
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
				STDCOUT(" coordinates: (" << coords.first << ", " << coords.second << "). Following data batches were used: ");
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
				auto& layrs = m_nn.m_Layers;

				layrs.resetCalcLossAddendum();//cleanup cached loss version to always recalculate it from scratch, because we aren't interested in any cheats here.
				if (m_ngcSetts.bForceSeed) m_nn.get_iRng().seed64(s);

				//return m_nn._calcLoss4batch_nonnormalized(m_data.batchX(), m_data.batchY());
				//we must NOT use nnet's function, because it changes nnet mode from training to inference. But should the mode
				// be changed, it'll screw the error value for some training-mode dependent algorithms, such as common Dropout
				// (it's actually an inverted dropout, it upscales survived activations proportional to 1/p, so
				// change the mode, and it'll remove the upscaling and probably change the error computed in a numerical way.
				// But analytical error will stay unchanged!)

				//auto& cd = m_nn.get_common_data();
				NNTL_ASSERT(m_nn.get_common_data().is_training_mode());//we MUST be in training mode
				//const auto batchSize = m_data.batchX().batch_size();
				NNTL_ASSERT(m_nn.get_common_data().input_batch_size() == m_data.batchX().batch_size());
				//if(cd.change_cur_batch_size(batchSize) != batchSize) layrs.on_batch_size_change();

				layrs.fprop(m_data.batchX());

				return layrs.output_layer().calc_loss(m_data.batchY()) + layrs.calcLossAddendum();
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

			NNTL_ASSERT(data_x.sample_size() == m_Layers.input_layer().get_neurons_cnt() && data_x.batch_size() >= biggestBatchSize);
			NNTL_ASSERT(data_y.sample_size() == m_Layers.output_layer().get_neurons_cnt() && data_y.batch_size() >= biggestBatchSize);
			NNTL_ASSERT(data_x.batch_size() == data_y.batch_size());
			if (
				!(data_x.sample_size() == m_Layers.input_layer().get_neurons_cnt() && data_x.batch_size() >= biggestBatchSize)
				|| !(data_y.sample_size() == m_Layers.output_layer().get_neurons_cnt() && data_y.batch_size() >= biggestBatchSize)
				|| !(data_x.batch_size() == data_y.batch_size())
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
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& _iI)noexcept { return NnT<LayersPack>(lp, &_iI); }

	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& _iI, typename LayersPack::iRng_t& iR)noexcept {
		return NnT<LayersPack>(lp, &_iI, nullptr, &iR);
	}
	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& _iI, typename LayersPack::iMath_t& iM)noexcept {
		return NnT<LayersPack>(lp, &_iI, &iM);
	}
	template <template<typename> class NnT = nnet, typename LayersPack = void>
	inline constexpr NnT<LayersPack> make_nnet(LayersPack& lp, typename LayersPack::iInspect_t& _iI, typename LayersPack::iMath_t& iM, typename LayersPack::iRng_t& iR)noexcept {
		return NnT<LayersPack>(lp, &_iI, &iM, &iR);
	}

}