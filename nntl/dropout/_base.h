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

#pragma warning(disable : 4100)
	//this class defines a dropout class interface as well as dummy class that removes dropout
	template<typename RealT>
	class NoDropout : public _impl::_No_Dropout_at_All<RealT> {
	public:
		//This flag means, that all dropout implementation-related functions might be called not only during training phase,
		//but during evaluation too
		static constexpr bool bDropoutWorksAtEvaluationToo = false;

		//this flag means that the dropout algorithm doesn't change the activation value if it is zero.
		// for example, it is the case of classical dropout (it drops values to zeros), but not the case of AlphaDropout
		static constexpr bool bDropoutIsZeroStable = true;

	protected:
		//////////////////////////////////////////////////////////////////////////
		//The following functions will always be called no matter what
		template<class Archive>
		static constexpr void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {}
		static constexpr void _dropout_deinit() noexcept {}

		template<typename CommonDataT>
		static constexpr bool _dropout_init(const neurons_count_t /*neurons_cnt*/
			, const CommonDataT& /*CD*/, const _impl::_layer_init_data<CommonDataT>& /*lid*/)noexcept
		{
			return true;
		}

		template<typename CommonDataT>
		static constexpr void _dropout_on_batch_size_change(const CommonDataT& CD, const vec_len_t outgBS) noexcept {}

		//const realmtxdef_t& _dropout_get_original_activations()const noexcept{};

		//if _dropout_have_original_activations() returns true, then _dropout_get_original_activations() must be workable
		template<typename CommonDataT>
		static constexpr bool _dropout_has_original_activations(const CommonDataT& CD) noexcept { return false; }

		//////////////////////////////////////////////////////////////////////////
		//The following functions will be called only if bDropout() returns true
		template<typename CommonDataT>
		static constexpr void _dropout_apply(realmtx_t& activations, const CommonDataT& CD) noexcept {}

		template<typename CommonDataT>
		static constexpr void _dropout_restoreScaling(realmtx_t& dLdA, realmtx_t& activations, const CommonDataT& CD)noexcept {}

		//////////////////////////////////////////////////////////////////////////
	public:
		static constexpr bool bDropout() noexcept { return false; }
		static constexpr real_t dropoutPercentActive() noexcept { return real_t(1.); }
		static constexpr void dropoutPercentActive(const real_t dpa) noexcept {
			if (dpa != real_t(1.) || dpa != real_t(0.)) {
				STDCOUTL("**BE AWARE: Trying to set dropout rate for a class without dropout implementation");
			}
		}
	};
#pragma warning(default : 4100)

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
		::std::enable_if_t <!layer_has_dropout<LayerT>::value> operator()(LayerT&)const noexcept {}

		template<typename LayerT>
		::std::enable_if_t<layer_has_dropout<LayerT>::value> operator()(LayerT& lyr)const noexcept {
			lyr.dropoutPercentActive(m_dpa);
		}
	};

}
