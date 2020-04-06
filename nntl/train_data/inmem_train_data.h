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

//#include "../serialization/serialization.h"

#include "_train_data_simple.h"
#include "simple_train_data_stor.h"

namespace nntl {
	//inmem_train_data is the simpliest implementation of _i_train_data interface that stores train/test sets directly in
	//memory and doesn't provide any augmentation facilities

	template<typename T>
	class inmem_train_data final
		: public _impl::_train_data_simple<inmem_train_data<T>, T>
		, public simple_train_data_stor<T>
	{
		typedef _impl::_train_data_simple<inmem_train_data<T>, T> base_class_t;
		typedef simple_train_data_stor<T> base_stor_t;

	public:
		//resolving definitions clash
		using base_class_t::value_type;
		using base_class_t::real_t;
		using base_class_t::realmtx_t;
		using base_class_t::realmtxdef_t;
		/*
		using base_class_t::data_set_id_t;
		using base_class_t::DatasetNamingFunc_t;

		using base_class_t::invalid_set_id;
		using base_class_t::train_set_id;
		using base_class_t::test_set_id;
		using base_class_t::LongestDatasetNameWNull;
		*/

	public:
		//////////////////////////////////////////////////////////////////////////
		// _i_train_data<> interface
		
		using base_stor_t::empty;

		numel_cnt_t dataset_samples_count(data_set_id_t dataSetId)const noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(dataSetId >= 0 && dataSetId <= 1);
			return X(dataSetId).rows();
		}

		vec_len_t xWidth()const noexcept { NNTL_ASSERT(!empty()); return train_x().cols_no_bias(); }
		vec_len_t yWidth()const noexcept { NNTL_ASSERT(!empty()); return train_y().cols_no_bias(); }

		//////////////////////////////////////////////////////////////////////////
		// other functions

	};

}
