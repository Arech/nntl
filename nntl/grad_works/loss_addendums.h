/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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

//this grad_works mixin introduces a wrapper over loss functions addendums such as L1 or L2 regularizers
// #include compatible addendums from the ./loss_addendum folder before this file 

#include "../utils/mixins.h"

#include "../loss_addendum/L1.h"
#include "../loss_addendum/L2.h"

namespace nntl {
namespace GW { //GW namespace is for grad_works mixins and other stuff, that helps to implement gradient processing voodooo things
	
#pragma warning(disable:4100)
	template<typename _FC, typename RealT, size_t MixinIdx>
	class Loss_Addendums_dummy : private math::smatrix_td {
	private:
// 		typedef _FC self_t;
// 		NNTL_METHODS_SELF();
// 		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	public:
		enum OptsList {
			opts_total = 0
		};

		static constexpr bool hasLossAddendum() noexcept { return false; }
		static constexpr real_t lossAddendum(const realmtx_t&) noexcept { return real_t(0.); }

	protected:
		static constexpr void _applyLossAddendums(realmtxdef_t& weights, realmtxdef_t& dLdW) noexcept {}
		static constexpr void _la_construct()noexcept {}
		static constexpr bool _la_init(const mtx_size_t biggestMtx)noexcept { return true; }
		static constexpr void _la_deinit()noexcept {}
	};
#pragma warning(default:4100)

	//////////////////////////////////////////////////////////////////////////

	//Don't use two loss addendums of the same type!!!
	template<typename _FC, typename RealT, size_t MixinIdx, class LossAddsTuple>
	class _Loss_Addendums : private math::smatrix_td {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	public:

		static_assert(tuple_utils::is_tuple<LossAddsTuple>::value, "Must be a tuple!");
		typedef LossAddsTuple addendums_tuple_t;
		static constexpr size_t addendums_count = ::std::tuple_size<addendums_tuple_t>::value;
		static_assert(addendums_count > 0, "Use Loss_Addendums_dummy if you don't need loss_addendums at all");

		static constexpr bool defRegularizersIgnoresBiasWeights = true;

		template<typename T>
		struct _addendums_props : ::std::true_type {
			static_assert(!::std::is_reference<T>::value, "Must not be a reference");
			static_assert(!::std::is_const<T>::value, "Must not be a const");
			static_assert(loss_addendum::is_loss_addendum<T>::value, "must be a real loss_addendum");
			static_assert(!T::calcOnFprop, "loss addendum for GradWorks must not be evaluated during fprop()");
		};
		static_assert(tuple_utils::assert_each<addendums_tuple_t, _addendums_props >::value, "addendums_tuple_t must be assembled from proper objects!");

	protected:
		addendums_tuple_t m_addendumsTuple;

	public:
		enum OptsList {
			//f_UseAddendum0 = 0,
			//f_UseAddendumLast = (addendums_count - 1),
			f_AddendumIgnoresBias0 = 0,
			f_AddendumIgnoresBiasLast = f_AddendumIgnoresBias0 + (addendums_count - 1),
			opts_total
		};

	public:
		addendums_tuple_t& addendums()noexcept { return m_addendumsTuple; }

		template<size_t idx>
		auto& addendum()noexcept { return ::std::get<idx>(m_addendumsTuple); }
		template<class LaT>
		auto& addendum()noexcept { return ::std::get<LaT>(m_addendumsTuple); }

// 		const bool useAddendum(const size_t& idx)const noexcept { 
// 			NNTL_ASSERT(idx < addendums_count);
// 			return get_opt(f_UseAddendum0 + idx);
// 		}
// 		template<class LaT>
// 		const bool useAddendum()const noexcept {
// 			constexpr auto idx = tuple_utils::get_element_idx<LaT, LossAddsTs...>();
// 			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
// 			return useAddendum(idx);
// 		}
// 
// 		void useAddendum(const size_t& idx, const bool& b) noexcept {
// 			NNTL_ASSERT(idx < addendums_count);
// 			set_opt(f_UseAddendum0 + idx, b);
// 		}
// 		template<class LaT>
// 		void useAddendum(const bool& b) noexcept {
// 			constexpr auto idx = tuple_utils::get_element_idx<LaT, LossAddsTs...>();
// 			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
// 			useAddendum(idx, b);
// 		}

		const bool addendumIgnoresBias(const size_t& idx)const noexcept {
			NNTL_ASSERT(idx < addendums_count);
			return get_opt(f_AddendumIgnoresBias0 + idx);
		}
		template<class LaT>
		const bool addendumIgnoresBias()const noexcept {
			constexpr auto idx = tuple_utils::tuple_element_idx_safe<LaT, addendums_tuple_t>::value;
			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
			return addendumIgnoresBias(idx);
		}

		void addendumIgnoresBias(const size_t& idx, const bool& b)noexcept {
			NNTL_ASSERT(idx < addendums_count);
			set_opt(f_AddendumIgnoresBias0 + idx, b);
		}
		template<class LaT>
		void addendumIgnoresBias(const bool& b)noexcept {
			constexpr auto idx = tuple_utils::tuple_element_idx_safe<LaT, addendums_tuple_t>::value;
			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
			addendumIgnoresBias(idx, b);
		}

		//////////////////////////////////////////////////////////////////////////

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		const bool hasLossAddendum()const noexcept {
			bool b = false;
// 			for (size_t optId = f_UseAddendum0; optId <= f_UseAddendumLast; ++optId) {
// 				b |= get_opt(optId);
// 			}
			tuple_utils::for_each_up(m_addendumsTuple, [&b](const auto& la) noexcept {
				b |= la.bEnabled();
			});
			return b;
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum(const realmtxdef_t& weights)const noexcept {
			real_t ret(0.0);

			//the only modification of weights we may use is stripping/restoring last (bias) column,
			//which is in fact not a modification from outside POV
			realmtxdef_t& _W = *(const_cast<realmtxdef_t*>(&weights));

			tuple_utils::for_each_up(m_addendumsTuple, [&_W, &ret, &CD = get_self().get_common_data(), this](auto& la) {
				typedef ::std::decay_t<decltype(la)> la_t;

				if (la.bEnabled()) {
					const auto bIgnoreBiases = addendumIgnoresBias<la_t>();
					if (bIgnoreBiases) _W.hide_last_col();
					ret += la.lossAdd(_W, CD);
					if (bIgnoreBiases) _W.restore_last_col();
				}
			});

			return ret;
		}

	protected:
		void _applyLossAddendums(realmtxdef_t& weights, realmtxdef_t& dLdW)const noexcept {
			tuple_utils::for_each_up(m_addendumsTuple, [&weights, &dLdW, &CD = get_self().get_common_data(), this](auto& la) {
				typedef ::std::decay_t<decltype(la)> la_t;
				static_assert(loss_addendum::is_loss_addendum<la_t>::value, "Every Loss_addendum class must implement loss_addendum::_i_loss_addendum<>");

				if (la.bEnabled()) {
					const auto bIgnoreBiases = addendumIgnoresBias<la_t>();
					if (bIgnoreBiases) { dLdW.hide_last_col(); weights.hide_last_col(); }
					la.dLossAdd(weights, dLdW, CD);
					if (bIgnoreBiases) { dLdW.restore_last_col(); weights.restore_last_col(); }
				}
			});
		}

		void _la_construct()noexcept {
			for (size_t idx = 0; idx < addendums_count; ++idx) {
				//useAddendum(idx, false);
				addendumIgnoresBias(idx, defRegularizersIgnoresBiasWeights);
			}
		}

		bool _la_init(const mtx_size_t biggestMtx)noexcept {
			bool b = true;
			tuple_utils::for_each_up(m_addendumsTuple, [&b, biggestMtx, &CD = get_self().get_common_data()](auto& la) {
				NNTL_UNREF(la);
				b = b & la.init(biggestMtx, CD);
			});
			return b;
		}
		void _la_deinit()noexcept {
			tuple_utils::for_each_up(m_addendumsTuple, [](auto& la) {
				NNTL_UNREF(la);
				la.deinit();
			});
		}

	public:
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// consider the following as an example of how to enable, setup and use a custom _i_loss_addendum derived class object.
		typedef loss_addendum::L1<real_t> LA_L1_t;//for the convenience
		typedef loss_addendum::L2<real_t> LA_L2_t;

		//static constexpr auto idxL1 = tuple_utils::get_element_idx_impl<LA_L1_t, 0, LossAddsTs...>::value;
		static constexpr auto idxL1 = tuple_utils::tuple_element_idx_safe<LA_L1_t, addendums_tuple_t>::value;
		static constexpr bool bL1Available = (idxL1 < addendums_count);

		//static constexpr auto idxL2 = tuple_utils::get_element_idx_impl<LA_L2_t, 0, LossAddsTs...>::value;
		static constexpr auto idxL2 = tuple_utils::tuple_element_idx_safe<LA_L2_t, addendums_tuple_t>::value;
		static constexpr bool bL2Available = (idxL2 < addendums_count);
		
		template<bool b = bL1Available>
		::std::enable_if_t<b, self_ref_t> L1(const real_t& l1, const bool& bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			auto& adn = addendum<LA_L1_t>();
			adn.scale(l1);
			//useAddendum<LA_L1_t>(adn.bEnabled());
			addendumIgnoresBias<LA_L1_t>(bIgnoreBiasWeights);
			return get_self();
		}
		template<bool b = bL1Available>
		//::std::enable_if_t<b, real_t> L1()const noexcept { return useAddendum<LA_L1_t>() ? addendum<LA_L1_t>().scale() : real_t(0.); }
		::std::enable_if_t<b, real_t> L1()const noexcept { return addendum<LA_L1_t>().scale(); }

		template<bool b = bL2Available>
		::std::enable_if_t<b, self_ref_t> L2(const real_t& l2, const bool& bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			auto& adn = addendum<LA_L2_t>();
			adn.scale(l2);
			//useAddendum<LA_L2_t>(adn.bEnabled());
			addendumIgnoresBias<LA_L2_t>(bIgnoreBiasWeights);
			return get_self();
		}
		template<bool b = bL2Available>
		//::std::enable_if_t<b, real_t> L2()const noexcept { return useAddendum<LA_L2_t>() ? addendum<LA_L2_t>().scale() : real_t(0.); }
		::std::enable_if_t<b, real_t> L2()const noexcept { return addendum<LA_L2_t>().scale(); }

	};

	// primary template handles types that have no nested ::addendums_tuple_t member:
	template< class, class = ::std::void_t<> >
	struct has_loss_addendums : ::std::false_type { };
	// specialization recognizes types that do have a nested ::addendums_tuple_t member:
	template< class T >
	struct has_loss_addendums<T, ::std::void_t<typename T::addendums_tuple_t>> : ::std::true_type {};


	//Don't use two loss addendums of the same type!!!
// 	template<typename ... LossAddsTs>
// 	struct Loss_Addendums_builder {
// 		template<typename _FC, typename RealT, size_t MixinIdx>
// 		using type = ::std::conditional_t<sizeof...(LossAddsTs)==0
// 			, Loss_Addendums_dummy<_FC, RealT, MixinIdx>
// 			, _Loss_Addendums<_FC, RealT, MixinIdx, ::std::make_tuple(LossAddsTs...)>
// 		>;
// 	};

	template<typename LossAddsTuple>
	struct Loss_Addendums_builder {
		template<typename _FC, typename RealT, size_t MixinIdx>
		using type = ::std::conditional_t<
			tuple_utils::is_tuple<typename tuple_utils::assert_tuple_or_void<LossAddsTuple>::type>::value
			, _Loss_Addendums<_FC, RealT, MixinIdx, LossAddsTuple>
			, Loss_Addendums_dummy<_FC, RealT, MixinIdx>
		>;
	};

	template<typename _FC, typename RealT, size_t MixinIdx>
	using Loss_Addendums_L1L2 = _Loss_Addendums<_FC, RealT, MixinIdx
		, ::std::tuple<loss_addendum::L1<RealT>, loss_addendum::L2<RealT>> >;

}
}