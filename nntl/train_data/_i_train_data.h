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

#include <functional>

#include "../_nnet_errs.h"

namespace nntl {

	//should be inherited with virtual keyword to prevent names clash in multiple inheritance
	struct DataSetsId {
		typedef int data_set_id_t;
		static constexpr data_set_id_t invalid_set_id = -1;

		static constexpr data_set_id_t train_set_id = 0;
		static constexpr data_set_id_t test_set_id = 1;
		//derived classes may add another ids if necessary, however nnet::train will use only these two ids
		//and also note, that all train-related functions (init4train() & etc) works only with the id==train_set_id
		// Note that any data_set_id<0 is considered invalid
		
		//////////////////////////////////////////////////////////////////////////
		static constexpr unsigned flag_exclude_nothing = 0;
		static constexpr unsigned flag_exclude_dataX = 1;
		static constexpr unsigned flag_exclude_dataY = 2;

		static constexpr bool exclude_dataX(const unsigned f)noexcept { return f == flag_exclude_dataX; }
		static constexpr bool exclude_dataY(const unsigned f)noexcept { return f == flag_exclude_dataY; }

		//////////////////////////////////////////////////////////////////////////
		// supplementary datasets (besides train & test) naming stuff
		// LongestDatasetNameWNull defines a max number of bytes dataset name can occupy including terminating null char
		static constexpr unsigned LongestDatasetNameWNull = 32;
		//see how to use it below
		typedef ::std::function<void(data_set_id_t, char*, unsigned)> DatasetNamingFunc_t;
	};

	//template-less base class with core definitions
	struct _i_train_data_base : public virtual DataSetsId {
		typedef _nnet_errs::ErrorCode nnet_errors_t;
	};

	// interface to training data that is required at least by nnet::train()
	// Note that object with this interface is actually a statefull modifiable resettable wrapper over
	// some constant data storage. So, it's ok to pass it anywhere without const modifier, because the only
	// thing that can be changed via the interface is the state of that wrapper, but not the data itself.
	template<typename XT, typename YT /*= XT*/>
	class _i_train_data : public _i_train_data_base {
	public:
		typedef XT x_t;
		typedef YT y_t;
// 		typedef BaseT value_type;
// 		typedef BaseT real_t;
// 		typedef math::smatrix<value_type> realmtx_t;
// 		typedef math::smatrix_deform<real_t> realmtxdef_t;

		typedef math::smatrix<x_t> x_mtx_t;
		typedef math::smatrix<y_t> y_mtx_t;
		typedef math::smatrix_deform<x_t> x_mtxdef_t;
		typedef math::smatrix_deform<y_t> y_mtxdef_t;

		//////////////////////////////////////////////////////////////////////////
		// the following list contains the only functions and properties, that can be called or referenced
		// during nnet::train() execution by core nntl components (such as performance evaluators)
		// if anything else is called, it should be considered as a bug
		
		// if it is safe to cache training or testing set elsewhere, leave = true here.
		// If a derived class wants to prohibit caching (for example, because it has so big training set,
		// that it can't be returned/processed with a single pass), always set this to false in a derived class
		static constexpr bool allowExternalCachingOfSets = true;
		// When allowExternalCachingOfSets is set to true, trainset_samples_count() and total amount of samples
		// obtainable via calls to on_next_epoch()/on_next_batch()/batchX(), as well as walk_over_train_set()
		// /next_subset()/batchX() MUST be the same (not counting possible crops due to uneven batch size). The same
		// applies to testset_samples_count() and walk_over_test_set()/next_subset()/batchX(). Also note that
		// in that case biggest dataset_samples_count() must be less or equal to ::std::numeric_limits<vec_len_t>::max()
		// 

		//how many there are datasets? Must be at least 2 - train_set_id and test_set_id
		nntl_interface data_set_id_t datasets_count()const noexcept;

		//numel_cnt_t is intentionally used here to underscore, that actual training/testing set size may be huge
		//(in principle - bigger than vec_len_t may hold and bigger than it's possible to store in a single math::smatrix)
		// Note, for training set it must return non modified/non augmented training set size.
		// Caller MUST NOT assume that this exact count of training samples will be used during one training
		// epoch for allowExternalCachingOfSets==false
		nntl_interface numel_cnt_t dataset_samples_count(data_set_id_t dataSetId)const noexcept;

		nntl_interface numel_cnt_t trainset_samples_count()const noexcept;
		nntl_interface numel_cnt_t testset_samples_count()const noexcept;

		//note that caller MUST destroy obtained function before this object goes out of scope to permit
		//any kind of suitable lambda captures &etc.
		nntl_interface DatasetNamingFunc_t get_dataset_naming_function()const noexcept;

		nntl_interface bool empty()const noexcept;
		
		nntl_interface vec_len_t xWidth()const noexcept;
		nntl_interface vec_len_t yWidth()const noexcept;
		//Note: to make special loss functions that require yWidth() != output_layer::get_neurons_cnt() available, check suitabililty
		//of TD to train the nnet with the following function and then assume that batches will also have proper size
		nntl_interface bool isSuitableForOutputOf(neurons_count_t n)const noexcept;

		
		// Now the important part about training/testing sets and how they should be handled.
		// We need to make the following possible:
		// 1. training/inferencing on sets of any size (not limited to the amount of available RAM)
		//		so, that implies that nntl must be able to do inferencing/fprop using batches of data (aka minibatch inferencing)
		//			and then aggregate the resulting total loss value (training/bprop is already fit for the purpose)
		//			(also that means that network will output its predictions in sequential batches)
		// 2. _i_train_data<> is the only source of data to feed nnet::train() so it MUST support data
		//		augmentation. We must not restrict data augmentation techniques to some particular subset, so
		//		that implies that in theory even a small training set could be augmented to be infinitely huge.
		//		That in turn, implies that there should be a way to return a separate "training set to evaluate
		//		performance/loss value" (aka do minibatch inferencing) and a separate "training set for actual training"
		//		These two sets may have totally different size and content.
		//		Also that implies that it is the _i_train_data<> derived class who should decide how many batches of given
		//		size there should be for a given training epoch number

		//takes desired batch sizes, updates them if necessary to any suitable value and initializes internal state for training
		nntl_interface nnet_errors_t init4train(IN OUT vec_len_t& maxFPropSize, IN OUT vec_len_t& maxBatchSize, OUT bool& bMiniBatch)noexcept;
		//init4inference() does a subset of initialization required to perform inference on the given _i_train_data object
		nntl_interface nnet_errors_t init4inference(IN OUT vec_len_t& maxFPropSize)noexcept;

		nntl_interface void deinit4all()noexcept;

		nntl_interface bool is_initialized4train()const noexcept;
		nntl_interface bool is_initialized4inference()const noexcept;
		
		//////////////////////////////////////////////////////////////////////////
		// the following set of functions can be called only if init4train() was called earlier
		// 
		// notifies object about new training epoch start and must return total count of batches to execute over the training set.
		// batchSize can either be set directly via function parameter, or if the func.parameter is set to 0, will be fetched
		// from cd.get_cur_batch_size(), or if the func.param is negative - taken from default value (maxBatchSize returned by init4train())
		// Anyway, it is expected to be less then or equal to maxBatchSize returned by init4train().
		template<typename CommonDataT>
		nntl_interface numel_cnt_t on_next_epoch(const numel_cnt_t epochIdx, const CommonDataT& cd, vec_len_t batchSize = 0) noexcept;
		//on_next_epoch() may but doesn't have to obey epochIdx (depending on implementation)

		// for training (init4train()) only. Always pass batchIdx that correctly corresponds to this function call number
		// (so, it doesn't allow randomly select batch, it only takes away a burden of counting call numbers from the function)
		template<typename CommonDataT>
		nntl_interface void on_next_batch(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept;
		//on_next_batch() may but doesn't have to obey batchIdx (depending on implementation)

		//////////////////////////////////////////////////////////////////////////
		// the following set of functions can be called if init4train() or init4inference() were called before
		// 
		//for training & inference mode to return
		// - corresponding training set batches for given on_next_epoch(epochIdx) and on_next_batch(batchIdx) (if they were called just before)
		//		Each batch always contain the same amount of rows equal to the one was set to cd.get_cur_batch_size() during on_next_epoch()
		// - corresponding batches of data if walk_over_set()/next_subset() were called before
		//		Each batch contain at max maxFPropSize rows (set in init4inference())
		// Note 1: once the mode is changed, the caller MUST assume that underlying data, returned earlier by these functions, became a junk
		// Note 2: if the mode was set to exclude X or Y data (see excludeDataFlag argument of walk_over_set()), then
		//		corresponding batchX() or batchY() MUST NOT be called.
		nntl_interface const x_mtx_t& batchX()const noexcept;
		nntl_interface const y_mtx_t& batchY()const noexcept;
		// 		
		// walk_over_set() must be called before calling batchX/batchY functions to select which set (train or test, for example) to use
		// and to which batch size divide the dataset into.
		// batchSize can either be set directly via function parameter, or if the func.parameter is set to 0, will be fetched
		// from cd.get_cur_batch_size(), or if the func.param is negative - taken from default value (maxFPropSize returned by
		// init4train() or init4inference())
		// Anyway, batchSize must be less then or equal to maxFPropSize.
		// Returns the total count of calls to next_subset() required to walk over whole selected set with
		// current batch size. Note that the last batch may contain less samples than cd.get_cur_batch_size() returns during
		// running this function, so adjust-set batch size for the nnet accordingly.
		// Also note, that implementation MUST NOT permute the data samples of corresponding datasets and MUST always
		// return the data in the same order. Recall that if allowExternalCachingOfSets==true, then total amount of data returned
		// for the corresponding dataset MUST be equal to dataset_samples_count() (not true for allowExternalCachingOfSets==false)
		// The last argument excludeDataFlag may help to squeeze some performance when corresponding part of dataset is not needed.
		template<typename CommonDataT>
		nntl_interface numel_cnt_t walk_over_set(const data_set_id_t dataSetId, const CommonDataT& cd
			, vec_len_t batchSize = -1, const unsigned excludeDataFlag = flag_exclude_nothing)noexcept;

		//convenience wrappers around walk_over_set(). Always uses maxFPropSize returned from init*() as a batch size
		nntl_interface numel_cnt_t walk_over_train_set()noexcept;
		nntl_interface numel_cnt_t walk_over_test_set()noexcept;
		
		// similar to on_next_batch() prepares object to return corresponding data subset.
		// Always pass batchIdx that correctly corresponds to this function call number
		// (so, it doesn't allow randomly select batch, it only takes away a burden of counting call numbers from the function)
		template<typename CommonDataT>
		nntl_interface void next_subset(const numel_cnt_t batchIdx, const CommonDataT& cd)noexcept;
		//next_subset() may but doesn't have to obey batchIdx (depending on implementation)

	protected:
		_i_train_data()noexcept{}
	};

	//////////////////////////////////////////////////////////////////////////
	namespace _impl {
		template<class T>
		struct is_train_data_derived : public ::std::is_base_of<_i_train_data<typename T::x_t, typename T::y_t>, T> {};
	}
	template<typename T>
	using is_train_data_intf = ::std::conditional_t< utils::has_x_t_and_y_t<T>::value
		, _impl::is_train_data_derived<T>, ::std::false_type>;

	//////////////////////////////////////////////////////////////////////////

}

