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

#include "interfaces.h"

namespace nntl {
namespace _impl {

	//////////////////////////////////////////////////////////////////////////
	// this structure contains all common data shared between nn object and layers including
	// pointers to math&rng interfaces and some data related to current nn.train() call only.
	// This structure is expected to live within nn object (and share it lifetime) and can be reinitialized
	// to work with another train() session. Const reference to an instance of this structure is passed to each layer
	// during layer.init() call and should be stored in it to provide access to the data.
	// Also the same instance serves as a constainer/handler of objects of math&rng interfaces within a NNet object, therefore
	// make sure that non-const reference/pointer to the instance would have extremely restricted usage.
	template<typename InterfacesT>
	struct common_nn_data : public math::smatrix_td, public interfaces_td<InterfacesT> {
		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		//same for every train() session
		iMath_t* m_pMath;
		iRng_t* m_pRng;
		iInspect_t* m_pInspect;

		//could be different in different train() sessions.
		vec_len_t m_max_fprop_batch_size;//The biggest samples count for fprop(), usually this is data_x.rows()
		vec_len_t m_training_batch_size;//Fixed samples count for bprop(), usually it is a batchSize

		//////////////////////////////////////////////////////////////////////////
		// methods
	public:
		~common_nn_data()noexcept { 
			m_pMath = nullptr;
			m_pRng = nullptr;
			m_pInspect = nullptr;
			deinit();
		}
		common_nn_data()noexcept : m_pMath(nullptr), m_pRng(nullptr), m_pInspect(nullptr)
			, m_max_fprop_batch_size(0), m_training_batch_size(0)
		{}
		common_nn_data(iMath_t& im, iRng_t& ir, iInspect_t& iI)noexcept : m_pMath(&im), m_pRng(&ir), m_pInspect(&iI)
			, m_max_fprop_batch_size(0), m_training_batch_size(0)
		{}

		void setInterfacesFrom(const common_nn_data& other)noexcept {
			NNTL_ASSERT(!m_pMath && !m_pRng && !m_pInspect);
			m_pMath = &other.iMath();
			m_pRng = &other.iRng();
			m_pInspect = &other.iInspect();
		}

		void deinit()noexcept {
			m_max_fprop_batch_size = 0;
			m_training_batch_size = 0;
		}
		void init(vec_len_t fbs, vec_len_t bbs)noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size == 0 && m_training_batch_size == 0);
			NNTL_ASSERT(fbs >= bbs);//essential assumption
			m_max_fprop_batch_size = fbs;
			m_training_batch_size = bbs;
		}

		iMath_t& iMath()const noexcept { NNTL_ASSERT(m_pMath); return *m_pMath; }
		iRng_t& iRng()const noexcept { NNTL_ASSERT(m_pRng); return *m_pRng; }
		iInspect_t& iInspect()const noexcept { NNTL_ASSERT(m_pInspect); return *m_pInspect; }
		const vec_len_t max_fprop_batch_size()const noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size > 0);
			return m_max_fprop_batch_size;
		}
		const vec_len_t training_batch_size()const noexcept {
			//NNTL_ASSERT(m_training_batch_size >= 0);//batch size could be 0 to run fprop() only
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			return m_training_batch_size;
		}

		const bool is_initialized()const noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			return m_max_fprop_batch_size > 0;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// wrapper class to help deal with a pointer to common_nn_data<interfaces_t>
	template<typename InterfacesT>
	class _common_data_consumer : public interfaces_td<InterfacesT> {
	public:
		typedef common_nn_data<interfaces_t> common_data_t;

	protected:
		const common_data_t* m_pCommonData;

		void set_common_data(const common_data_t& cd)noexcept {
			NNTL_ASSERT(!m_pCommonData);
			m_pCommonData = &cd;
		}
		void clean_common_data()noexcept { m_pCommonData = nullptr; }

	public:
		~_common_data_consumer()noexcept {}
		_common_data_consumer()noexcept : m_pCommonData(nullptr) {}

		//////////////////////////////////////////////////////////////////////////
		// helpers to access common data 
		const common_data_t& get_common_data()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return *m_pCommonData;
		}
		iMath_t& get_iMath()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return m_pCommonData->iMath();
		}
		iRng_t& get_iRng()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return m_pCommonData->iRng();
		}
		iInspect_t& get_iInspect()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return m_pCommonData->iInspect();
		}
		const typename iMath_t::vec_len_t get_max_fprop_batch_size()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return m_pCommonData->max_fprop_batch_size();
		}
		const typename iMath_t::vec_len_t get_training_batch_size()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return m_pCommonData->training_batch_size();
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// the following class inherits common_nn_data and uses it as a storage for managed pointers to interfaces.
	// Could be used as a base class for nnet class

	template<typename InterfacesT>
	class interfaces_keeper 
		: private common_nn_data<InterfacesT>//must be inherited privately to hide non-relevant API
	{
	private:
		typedef common_nn_data<InterfacesT> _base_class;

	public:
		//restoring types visibility
		using _base_class::interfaces_t;
		using _base_class::iMath_t;
		using _base_class::iRng_t;
		using _base_class::iThreads_t;
		using _base_class::iInspect_t;
		using _base_class::real_t;

		using _base_class::vec_len_t;
		using _base_class::numel_cnt_t;
		using _base_class::mtx_size_t;

		typedef common_nn_data<interfaces_t> common_data_t;

	protected:
		const bool bOwnMath, bOwnRng, bOwnInspect;

	public:
		~interfaces_keeper()noexcept{
			if (bOwnMath) delete m_pMath;
			if (bOwnRng) delete m_pRng;
			if (bOwnInspect) delete m_pInspect;
		}
		interfaces_keeper(iInspect_t* pI = nullptr, iMath_t* pM = nullptr, iRng_t* pR = nullptr)noexcept
			: bOwnMath(!pM), bOwnRng(!pR), bOwnInspect(!pI)
		{
			m_pMath = (pM ? pM : new(std::nothrow) iMath_t);
			NNTL_ASSERT(m_pMath);
			m_pRng = (pR ? pR : new(std::nothrow) iRng_t);
			NNTL_ASSERT(m_pRng);
			m_pInspect = (pI ? pI : new(std::nothrow) iInspect_t);
			NNTL_ASSERT(m_pInspect);
		}

	protected:
		//only derived classes should be able to modify common_data_t
		common_data_t& get_common_data()noexcept { return *this; }

	public:
		const common_data_t& get_common_data()const noexcept { return *this; }
		iMath_t& get_iMath()const noexcept { NNTL_ASSERT(m_pMath); return *m_pMath; }
		iRng_t& get_iRng()const noexcept { NNTL_ASSERT(m_pRng); return *m_pRng; }
		iInspect_t& get_iInspect()const noexcept { NNTL_ASSERT(m_pInspect); return *m_pInspect; }
	};
}
}