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

#include "../interface/math/smatrix.h"

namespace nntl {

	namespace _impl {
		//this class would be used internally to remove Dropout API completely, as well as a base class for all Dropout classes
		template<typename RealT>
		struct _No_Dropout_at_All : public math::smatrix_td {
			typedef RealT real_t;
			typedef math::smatrix<real_t> realmtx_t;
		};
	}

	//this class defines a dropout class interface as well as dummy class that removes dropout
	template<typename RealT>
	class NoDropout : public _impl::_No_Dropout_at_All<RealT> {
	protected:
		template<class Archive>
		static constexpr void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {}

		static constexpr bool _dropout_init(const vec_len_t training_batch_size, const neurons_count_t neurons_cnt)noexcept {
			return true;
		}
		static constexpr void _dropout_deinit() noexcept {}
		static constexpr void _dropout_on_batch_size_change(const vec_len_t batchSize) noexcept {}

		template<typename iMathT, typename iRngT, typename iInspectT>
		static constexpr void _dropout_apply(realmtx_t& activations, const bool bTrainingMode
			, iMathT& iM, iRngT& iR, iInspectT& _iI) noexcept {}

		template<typename iMathT, typename iInspectT>
		static constexpr void _dropout_restoreScaling(realmtx_t& dLdZ, realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {}

	public:
		static constexpr bool bDropout() noexcept { return false; }
		static constexpr real_t dropoutPercentActive() noexcept { return real_t(1.); }
		static constexpr void dropoutPercentActive(const real_t dpa) noexcept {
			if (dpa != real_t(1.) || dpa != real_t(0.)) {
				STDCOUTL("**BE AWARE: Trying to set dropout rate for a class without dropout implementation");
			}
		}

		// if it returns a real pointer, then an activation value is considered dropped out iff corresponding mask value is zero
		static constexpr const realmtx_t* _dropout_get_mask() noexcept { return nullptr; }
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// Some helpers
	template< class, class = ::std::void_t<> >
	struct has_Dropout_t : ::std::false_type { };
	// specialization recognizes types that do have a nested ::Dropout_t member:
	template< class T >
	struct has_Dropout_t<T, ::std::void_t<typename T::Dropout_t>> : ::std::true_type {};

	//////////////////////////////////////////////////////////////////////////
	template<typename DropoutT>
	using is_dummy_dropout = ::std::disjunction<
		::std::is_same < DropoutT, NoDropout<typename DropoutT::real_t>>
		, ::std::is_same < DropoutT, _impl::_No_Dropout_at_All<typename DropoutT::real_t>>
	>;

	//////////////////////////////////////////////////////////////////////////
	template< class, class = ::std::void_t<> >
	struct layer_has_dropout : ::std::false_type { };
	// specialization recognizes types that checks whether T has Dropout_t type defined and if so, whether it isn't dummy dropout
	template< class T >
	struct layer_has_dropout<T, ::std::void_t<typename T::Dropout_t>> : ::std::negation<is_dummy_dropout<typename T::Dropout_t>> {};

	//////////////////////////////////////////////////////////////////////////
	template<typename RealT>
	struct hlpr_layer_set_dropoutPercentActive {
		const RealT m_dpa;

		~hlpr_layer_set_dropoutPercentActive()noexcept {}
		hlpr_layer_set_dropoutPercentActive(const RealT dpa)noexcept : m_dpa(dpa) {}

		template<typename LayerT>
		::std::enable_if_t <!layer_has_dropout<LayerT>::value> operator()(LayerT& lyr)const noexcept {}

		template<typename LayerT>
		::std::enable_if_t<layer_has_dropout<LayerT>::value> operator()(LayerT& lyr)const noexcept {
			lyr.dropoutPercentActive(m_dpa);
		}
	};

}
