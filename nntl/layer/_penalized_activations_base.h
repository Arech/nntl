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

#include "_layer_base.h"
//#include "_pack_.h"

#include "../loss_addendum/L1.h"
#include "../loss_addendum/L2.h"
#include "../loss_addendum/DeCov.h"

namespace nntl {

	class _PA_base_dummy {
	protected:
		template<typename CommonDataT>
		static constexpr bool _pab_init(const math::smatrix_td::mtx_size_t biggestMtx, const CommonDataT& CD)noexcept { return true; }
		static constexpr void _pab_deinit()noexcept {}

		static constexpr bool _pab_hasLossAddendum()noexcept {
			return false;
		}

		template<typename CommonDataT>
		static constexpr typename CommonDataT::real_t _pab_lossAddendum(
			const math::smatrix_deform<typename CommonDataT::real_t>& ThisActivations, const CommonDataT& CD) noexcept
		{
			return (typename CommonDataT::real_t)(0);
		}

		template<typename CommonDataT>
		static constexpr void _pab_fprop(const math::smatrix_deform<typename CommonDataT::real_t>& ThisActivations
			, const CommonDataT& CD)noexcept {}

		template<typename CommonDataT>
		static constexpr void _pab_update_dLdA(const math::smatrix<typename CommonDataT::real_t>& dLdA
			, const math::smatrix_deform<typename CommonDataT::real_t>& ThisActivations, CommonDataT& CD)noexcept{}
	};

	//must be default-constructible
	template<typename AddendumsTupleT>
	class _PA_base : private math::smatrix_td
	{
	public:
		//typedef ::std::tuple<LossAddsTs...> addendums_tuple_t;
		//static constexpr size_t addendums_count = sizeof...(LossAddsTs);
		typedef AddendumsTupleT addendums_tuple_t;
		static constexpr size_t addendums_count = ::std::tuple_size<AddendumsTupleT>::value;
		static_assert(addendums_count > 0, "Use the layer directly instead of LPA<>");

		static_assert(tuple_utils::is_tuple<addendums_tuple_t>::value, "Must be a tuple!");
		template<typename T>
		struct _addendums_props : ::std::true_type {
			static_assert(!::std::is_reference<T>::value, "Must not be a reference");
			static_assert(!::std::is_const<T>::value, "Must not be a const");
			static_assert(loss_addendum::is_loss_addendum<T>::value, "must be a real loss_addendum");
		};
		static_assert(tuple_utils::assert_each<addendums_tuple_t, _addendums_props >::value, "addendums_tuple_t must be assembled from proper objects!");

	protected:
		addendums_tuple_t m_addendumsTuple;

	private:
		typedef typename ::std::tuple_element<0, addendums_tuple_t>::type _first_addendum_t;
		typedef typename _first_addendum_t::real_t real_t;
		typedef typename math::smatrix<real_t> realmtx_t;
		typedef typename math::smatrix_deform<real_t> realmtxdef_t;

	public:
		addendums_tuple_t& addendums()noexcept { return m_addendumsTuple; }

		template<size_t idx>
		auto& addendum()noexcept { return ::std::get<idx>(m_addendumsTuple); }
		template<class LaT>
		auto& addendum()noexcept { return ::std::get<LaT>(m_addendumsTuple); }

	protected:
		template<typename CommonDataT>
		bool _pab_init(const mtx_size_t biggestMtx, const CommonDataT& CD)noexcept {
			bool b = true;
			tuple_utils::for_each_up(m_addendumsTuple, [&b, biggestMtx, &CD](auto& la) {
				b = b & la.init(biggestMtx, CD);
			});
			return b;
		}
		void _pab_deinit()noexcept {
			tuple_utils::for_each_up(m_addendumsTuple, [](auto& la) {
				la.deinit();
			});
		}

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		const bool _pab_hasLossAddendum()const noexcept {
			bool b = false;
			tuple_utils::for_each_up(m_addendumsTuple, [&b](const auto& la) noexcept {
				b |= la.bEnabled();
			});
			return b;
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		template<typename CommonDataT>
		real_t _pab_lossAddendum(const realmtxdef_t& ThisActivations, const CommonDataT& CD)const noexcept {
			real_t ret(0.0);

			//the only modification of activations we may use is stripping/restoring last (bias) column,
			//which is in fact not a modification from outside POV
			realmtxdef_t& act = const_cast<realmtxdef_t&>(ThisActivations);
			NNTL_ASSERT(act.emulatesBiases());
			const auto bRestoreBiases = act.hide_biases();

			tuple_utils::for_each_up(m_addendumsTuple, [&act, &ret, &CD](auto& la) {
				if (la.bEnabled()) {
					ret += la.lossAdd(act, CD);
				}
			});

			if (bRestoreBiases) act.restore_biases();

			return ret;
		}

	private:

		template<typename CommonDataT>
		struct call_on_fprop {
			const realmtxdef_t& act;
			const CommonDataT& CD;

			call_on_fprop(const realmtxdef_t& a, const CommonDataT& _cd)noexcept : act(a), CD(_cd) {}

			template<typename LAT>
			::std::enable_if_t<LAT::calcOnFprop> operator()(LAT& la)const noexcept {
				if (la.bEnabled()) {
					la.on_fprop(act, CD);
				}
			}
			template<typename LAT> ::std::enable_if_t<!LAT::calcOnFprop> operator()(LAT& la)const noexcept {}
		};

	protected:

		static constexpr bool bWorkOnFprop = tuple_utils::aggregate<::std::disjunction, loss_addendum::works_on_fprop, addendums_tuple_t>::value;
		
		template<typename CommonDataT, bool c = bWorkOnFprop>
		::std::enable_if_t<!c> _pab_fprop(const realmtxdef_t& ThisActivations, const CommonDataT& CD)noexcept {}

		template<typename CommonDataT, bool c = bWorkOnFprop>
		::std::enable_if_t<c> _pab_fprop(const realmtxdef_t& ThisActivations, const CommonDataT& CD)noexcept {
			//dropping const only to hide+restore biases
			realmtxdef_t& act = const_cast<realmtxdef_t&>(ThisActivations);
			NNTL_ASSERT(act.emulatesBiases());
			const auto bRestoreBiases = act.hide_biases();
			tuple_utils::for_each_up(m_addendumsTuple, call_on_fprop<CommonDataT>(act, CD));
			if (bRestoreBiases) act.restore_biases();
		}

		//#note: if we actually need to work with the output_layer, then there must be very special handling of this case because of how
		//bprop() is implemented for output_layer now
		template<typename CommonDataT>
		void _pab_update_dLdA(realmtx_t& dLdA, const realmtxdef_t& ThisActivations, const CommonDataT& CD)noexcept {
			//dropping const only to hide+restore biases
			realmtxdef_t& act = const_cast<realmtxdef_t&>(ThisActivations);
			NNTL_ASSERT(act.emulatesBiases());
			const auto bRestoreBiases = act.hide_biases();

			tuple_utils::for_each_up(m_addendumsTuple, [&act, &dLdA, &CD](auto& la) {
				if (la.bEnabled()) {
					la.dLossAdd(act, dLdA, CD);
				}
			});

			if (bRestoreBiases) act.restore_biases();
		}

	public:
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// consider the following as an example of how to enable, setup and use a custom _i_loss_addendum derived class object.
		/*
		DEPRECATED
		typedef loss_addendum::L1<real_t> LA_L1_t;//for the convenience
		typedef loss_addendum::L2<real_t> LA_L2_t;

		static constexpr auto idxL1 = tuple_utils::get_element_idx_impl<LA_L1_t, 0, LossAddsTs...>::value;
		//static constexpr auto idxL1 = tuple_utils::get_element_idx<LA_L1_t>(AddendumsTupleT());
		static constexpr bool bL1Available = (idxL1 < addendums_count);

		static constexpr auto idxL2 = tuple_utils::get_element_idx_impl<LA_L2_t, 0, LossAddsTs...>::value;
		//static constexpr auto idxL2 = tuple_utils::get_element_idx<LA_L2_t>(AddendumsTupleT());
		static constexpr bool bL2Available = (idxL2 < addendums_count);

		template<bool b = bL1Available>
		::std::enable_if_t<b, real_t> L1()const noexcept { return addendum<LA_L1_t>().scale(); }

		template<bool b = bL2Available>
		::std::enable_if_t<b, real_t> L2()const noexcept { return addendum<LA_L2_t>().scale(); }*/
	};

	//////////////////////////////////////////////////////////////////////////
	// Helper traits recognizer
	// primary template handles types that have no nested ::addendums_tuple_t member:
	template< class, class = ::std::void_t<> >
	struct can_penalize_activations : ::std::false_type { };
	// specialization recognizes types that do have a nested ::addendums_tuple_t member:
	template< class T >
	struct can_penalize_activations<T, ::std::void_t<typename T::addendums_tuple_t>> : ::std::true_type {};


	template<typename AddendumsTupleT>
	using _PA_base_selector = typename ::std::conditional<
		tuple_utils::is_tuple< typename tuple_utils::assert_tuple_or_void<AddendumsTupleT>::type >::value
		, _PA_base<AddendumsTupleT>
		, _PA_base_dummy
	>::type;
}
