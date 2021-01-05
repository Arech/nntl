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
	public:
		static constexpr bool bAllowToBlockLearning = inspector::is_gradcheck_inspector<iInspect_t>::value;

	protected:
		//////////////////////////////////////////////////////////////////////////
		//same for every train() session
		iMath_t* m_pMath;
		iRng_t* m_pRng;
		iInspect_t* m_pInspect;

		bool* m_pbNotLearningNow; //this is a special flag intended to block any internal state change during
		// normal learning process - useful to perform numeric gradient check while leaving the state intact.
		// It's not used if the nnet's inspector is not derived from inspector::GradCheck<>
		// Using a pointer to make sure every common_nn_data struct have the same value.

		//////////////////////////////////////////////////////////////////////////
		//could be different in different train() sessions.
		vec_len_t m_max_fprop_batch_size;//The biggest samples count for fprop(), usually this is data_x.rows()
		vec_len_t m_training_batch_size;//The biggest batch size for bprop(), usually it is a batchSize

		mutable vec_len_t m_cur_batch_size;

		bool m_bInTraining;

		//////////////////////////////////////////////////////////////////////////
		// methods
	public:
		~common_nn_data()noexcept { 
			m_pMath = nullptr;
			m_pRng = nullptr;
			m_pInspect = nullptr;
			m_pbNotLearningNow = nullptr;
			deinit();
		}
		common_nn_data()noexcept : m_pMath(nullptr), m_pRng(nullptr), m_pInspect(nullptr), m_pbNotLearningNow(nullptr)
			, m_max_fprop_batch_size(0), m_training_batch_size(0), m_cur_batch_size(0), m_bInTraining(false)
		{}
		common_nn_data(iMath_t& im, iRng_t& ir, iInspect_t& iI, bool& bNLNf)noexcept 
			: m_pMath(&im), m_pRng(&ir), m_pInspect(&iI), m_pbNotLearningNow(&bNLNf)
			, m_max_fprop_batch_size(0), m_training_batch_size(0), m_cur_batch_size(0), m_bInTraining(false)
		{}

		void setInterfacesFrom(const common_nn_data& other)noexcept {
			NNTL_ASSERT(!m_pMath && !m_pRng && !m_pInspect && !m_pbNotLearningNow);
			m_pMath = other.m_pMath;
			m_pRng = other.m_pRng;
			m_pInspect = other.m_pInspect;
			m_pbNotLearningNow = other.m_pbNotLearningNow;
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect && m_pbNotLearningNow);
		}

		void deinit()noexcept {
			m_max_fprop_batch_size = 0;
			m_training_batch_size = 0;
			m_cur_batch_size = 0;
		}
		void init(vec_len_t fbs, vec_len_t bbs)noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size == 0 && m_training_batch_size == 0);
			NNTL_ASSERT(fbs >= bbs);//essential assumption
			m_max_fprop_batch_size = fbs;
			m_training_batch_size = bbs;
			m_cur_batch_size = 0;
			m_bInTraining = false;
		}

		//////////////////////////////////////////////////////////////////////////
		//obsolete, use get_*()
		iMath_t& iMath()const noexcept { NNTL_ASSERT(m_pMath); return *m_pMath; }
		iRng_t& iRng()const noexcept { NNTL_ASSERT(m_pRng); return *m_pRng; }
		iThreads_t& iThreads()const noexcept { return iMath().ithreads(); }
		iInspect_t& iInspect()const noexcept { NNTL_ASSERT(m_pInspect); return *m_pInspect; }

		iMath_t& get_iMath()const noexcept { NNTL_ASSERT(m_pMath); return *m_pMath; }
		iRng_t& get_iRng()const noexcept { NNTL_ASSERT(m_pRng); return *m_pRng; }
		iThreads_t& get_iThreads()const noexcept { return get_iMath().ithreads(); }
		iInspect_t& get_iInspect()const noexcept { NNTL_ASSERT(m_pInspect); return *m_pInspect; }

		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<B, bool> isLearningBlocked()const noexcept { NNTL_ASSERT(m_pbNotLearningNow); return *m_pbNotLearningNow; }
		template<bool B = bAllowToBlockLearning>
		constexpr ::std::enable_if_t<!B, bool> isLearningBlocked()const noexcept { return false; }

		void set_training_mode(bool bTraining)noexcept { m_bInTraining = bTraining; }
		bool is_training_mode()const noexcept { return m_bInTraining; }

		//returns false if the same mode&batch has already been set
		bool set_mode_and_batch_size(const bool bTraining, const vec_len_t BatchSize)noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size > 0 && BatchSize > 0);
			NNTL_ASSERT((bTraining && m_training_batch_size > 0 && BatchSize <= m_training_batch_size)
				|| (!bTraining && BatchSize <= m_max_fprop_batch_size));
			
			if (bTraining == m_bInTraining && BatchSize == m_cur_batch_size) return false;

			m_bInTraining = bTraining;
			m_cur_batch_size = BatchSize;
			return true;
		}

		vec_len_t get_cur_batch_size()const noexcept { return m_cur_batch_size; }

		vec_len_t change_cur_batch_size(const vec_len_t bs)const noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size > 0 && bs > 0);
			NNTL_ASSERT((m_bInTraining && m_training_batch_size > 0 && bs <= m_training_batch_size)
				|| (!m_bInTraining && bs <= m_max_fprop_batch_size));

			const auto r = m_cur_batch_size;
			m_cur_batch_size = bs;
			return r;
		}

		vec_len_t max_fprop_batch_size()const noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size > 0);
			return m_max_fprop_batch_size;
		}
		vec_len_t training_batch_size()const noexcept {
			//NNTL_ASSERT(m_training_batch_size >= 0);//batch size could be 0 to run fprop() only
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			return m_training_batch_size;
		}
		vec_len_t biggest_batch_size()const noexcept {
			NNTL_ASSERT(m_pMath && m_pRng && m_pInspect);//must be preinitialized!
			NNTL_ASSERT(m_max_fprop_batch_size > 0);
			return ::std::max(m_training_batch_size, m_max_fprop_batch_size);
		}

		bool is_training_possible()const noexcept {
			return training_batch_size() > 0;
		}

		bool is_initialized()const noexcept {
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
		// note, that there's a dependency on this very definition of common_data_t in some _i_function/_i_activation - derived
		// classes. See _i_function::act_init() comment.

		static constexpr bool bAllowToBlockLearning = inspector::is_gradcheck_inspector<iInspect_t>::value;

	protected:
		const common_data_t* m_pCommonData;

		void set_common_data(const common_data_t& cd)noexcept {
			NNTL_ASSERT(!m_pCommonData);
			m_pCommonData = &cd;
		}
		void clean_common_data()noexcept { m_pCommonData = nullptr; }

	public:
		~_common_data_consumer()noexcept { clean_common_data(); }
		_common_data_consumer()noexcept : m_pCommonData(nullptr) {}

		//////////////////////////////////////////////////////////////////////////
		// helpers to access common data
		bool has_common_data()const noexcept { return !!m_pCommonData; }
		const common_data_t& get_common_data()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return *m_pCommonData;
		}
		iMath_t& get_iMath()const noexcept { NNTL_ASSERT(m_pCommonData); return m_pCommonData->iMath(); }
		iRng_t& get_iRng()const noexcept { NNTL_ASSERT(m_pCommonData); return m_pCommonData->iRng(); }
		iInspect_t& get_iInspect()const noexcept { NNTL_ASSERT(m_pCommonData); return m_pCommonData->iInspect(); }
		iThreads_t& get_iThreads()const noexcept{ NNTL_ASSERT(m_pCommonData); return m_pCommonData->iMath().ithreads(); }

		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<B, bool> isLearningBlocked()const noexcept {
			NNTL_ASSERT(m_pCommonData);
			return m_pCommonData->isLearningBlocked();
		}
		template<bool B = bAllowToBlockLearning>
		constexpr ::std::enable_if_t<!B, bool> isLearningBlocked() const noexcept { return false; }

		//////////////////////////////////////////////////////////////////////////
		//everything besides written above should be accessed via get_common_data()
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

		using _base_class::mtx_size_t;
		using _base_class::mtx_coords_t;

		typedef common_nn_data<interfaces_t> common_data_t;

	protected:
		const bool bOwnMath, bOwnRng, bOwnInspect;
		bool m_bLearningBlockedFlag;

	private:
		void _make_Inspector(iInspect_t* pI)noexcept {
			NNTL_ASSERT(pI);
			NNTL_ASSERT(!bOwnInspect);
			m_pInspect = pI;
		}
		void _make_Inspector(::std::nullptr_t)noexcept {
			NNTL_ASSERT(bOwnInspect);
			m_pInspect = new(::std::nothrow) iInspect_t;
		}
		void _make_Math(iMath_t* pM)noexcept {
			NNTL_ASSERT(!bOwnMath);
			NNTL_ASSERT(pM);
			m_pMath = pM;
		}
		void _make_Math(::std::nullptr_t)noexcept {
			NNTL_ASSERT(bOwnMath);
			m_pMath = new(::std::nothrow) iMath_t;
		}
		void _make_Rng(iRng_t* pR)noexcept {
			NNTL_ASSERT(!bOwnRng);
			NNTL_ASSERT(pR);
			m_pRng = pR;
		}
		void _make_Rng(::std::nullptr_t)noexcept {
			NNTL_ASSERT(bOwnRng);
			m_pRng = new(::std::nothrow) iRng_t;
		}

	public:
		~interfaces_keeper()noexcept{
			if (bOwnMath) delete m_pMath;
			if (bOwnRng) delete m_pRng;
			if (bOwnInspect) delete m_pInspect;
		}

		template<typename PInspT = ::std::nullptr_t, typename PMathT = ::std::nullptr_t, typename PRngT = ::std::nullptr_t>
		interfaces_keeper(PInspT pI = nullptr, PMathT pM = nullptr, PRngT pR = nullptr)noexcept
			: bOwnMath(!pM), bOwnRng(!pR), bOwnInspect(!pI), m_bLearningBlockedFlag(false)
		{
			_make_Math(pM);
			NNTL_ASSERT(m_pMath);
			_make_Rng(pR);
			NNTL_ASSERT(m_pRng);
			_make_Inspector(pI);
			NNTL_ASSERT(m_pInspect);
			m_pbNotLearningNow = &m_bLearningBlockedFlag;
		}

		

	protected:
		//only derived classes should be able to modify common_data_t
		common_data_t& get_common_data()noexcept { return *this; }

	public:
		const common_data_t& get_common_data()const noexcept { return *this; }
		const common_data_t& get_const_common_data()const noexcept { return *this; }

		iMath_t& get_iMath()const noexcept { NNTL_ASSERT(m_pMath); return *m_pMath; }
		iRng_t& get_iRng()const noexcept { NNTL_ASSERT(m_pRng); return *m_pRng; }
		iThreads_t& get_iThreads()const noexcept { return get_iMath().ithreads(); }
		iInspect_t& get_iInspect()const noexcept { NNTL_ASSERT(m_pInspect); return *m_pInspect; }

	protected:
		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<B, void> _blockLearning() noexcept { m_bLearningBlockedFlag = true; }
		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<B, void> _unblockLearning() noexcept { m_bLearningBlockedFlag = false; }

		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<!B, static void> _blockLearning() noexcept {
			static_assert(false, "this feature is designed to be used with a numeric gradient check!");
		}
		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<!B, static void> _unblockLearning() noexcept {
			static_assert(false, "this feature is designed to be used with a numeric gradient check!");
		}

	};
}
}