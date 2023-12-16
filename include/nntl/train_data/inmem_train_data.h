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

//#include "../serialization/serialization.h"

#include <nntl/train_data/_train_data_simple.h>
#include <nntl/train_data/inmem_train_data_stor.h>

namespace nntl {
	//inmem_train_data is the simpliest implementation of _i_train_data interface that stores train/test sets directly in
	//memory and doesn't provide any augmentation facilities
	// If not mentioned explicitly in a function comment, any member function of the class #supportsBatchInRow (at least it should)
	template<typename FCT, typename XT, typename YT = XT>
	class _inmem_train_data
		: public _impl::_train_data_simple<FCT, XT, YT>
		, public inmem_train_data_stor<XT, YT>
	{
		typedef _impl::_train_data_simple<FCT, XT, YT> base_class_t;
		typedef inmem_train_data_stor<XT, YT> base_stor_t;

	public:
		//resolving definitions clash
		using base_class_t::x_t;
		using base_class_t::y_t;
		using base_class_t::x_mtx_t;
		using base_class_t::y_mtx_t;
		using base_class_t::x_mtxdef_t;
		using base_class_t::y_mtxdef_t;


	public:
		//////////////////////////////////////////////////////////////////////////
		// _i_train_data<> interface
		
		using base_stor_t::empty;

		numel_cnt_t dataset_samples_count(data_set_id_t dataSetId)const noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(dataSetId >= 0 && dataSetId <= 1);
			return X(dataSetId).batch_size();
		}

		vec_len_t xWidth()const noexcept { NNTL_ASSERT(!empty()); return train_x().sample_size(); }
		vec_len_t yWidth()const noexcept { NNTL_ASSERT(!empty()); return train_y().sample_size(); }

		//////////////////////////////////////////////////////////////////////////
		// other functions

		//////////////////////////////////////////////////////////////////////////
		//
		// Normalization support functions.
		// _fix_trainX_whole() must support train set by default, but may also support any dataset.
		// after _fix_trainX_whole()/*_cw() any subset of train X returned by the td must be scaled with given values.
		// If it was called more than 1 time, effect must be cumulative
		template<typename MtxUpdT, typename CommonDataT, typename StatsT>
		void _fix_trainX_whole(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st
			, const data_set_id_t dsId = train_set_id)noexcept
		{
			MtxUpdT::whole(cd.get_iMath(), get_self().X_mutable(dsId), st);
		}

		template<typename MtxUpdT, typename CommonDataT, typename StatsT>
		void _fix_trainX_cw(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralVector_tpl<StatsT>& allSt
			, const data_set_id_t dsId = train_set_id)noexcept
		{
			MtxUpdT::batchwise(cd.get_iMath(), get_self().X_mutable(dsId), allSt);
		}
	};

	template<typename XT, typename YT = XT>
	class inmem_train_data final : public _inmem_train_data<inmem_train_data<XT, YT>, XT, YT> {
		typedef _inmem_train_data<inmem_train_data<XT, YT>, XT, YT> _base_class_t;
	public:
		template<typename ... ArgsT>
		inmem_train_data(ArgsT&&... args)noexcept : _base_class_t(::std::forward<ArgsT>(args)...) {}
	};

}
