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

//this grad_works mixin introduces a wrapper over loss functions addendums such as L1 or L2 regularizers
// #include compatible addendums from the ./loss_addendum folder before this file 

#include "../utils/mixins.h"

#include "../loss_addendum/L1.h"
#include "../loss_addendum/L2.h"

namespace nntl {
namespace GW { //GW namespace is for grad_works mixins and other stuff, that helps to implement gradient processing voodooo things
	
	template<typename _FC, typename RealT, size_t MixinIdx>
	class Loss_Addendums_dummy : private math::smatrix_td {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	public:
		enum OptsList {
			opts_total = 0
		};

		constexpr bool hasLossAddendum()const noexcept { return false; }
		constexpr real_t lossAddendum(const realmtx_t&)const noexcept { return real_t(0.); }

	protected:
		void _applyLossAddendums(realmtxdef_t& weights, realmtxdef_t& dLdW)const noexcept {}
		void _la_construct()noexcept {}
	};

	//////////////////////////////////////////////////////////////////////////

	//Don't use two loss addendums of the same type!!!
	template<typename _FC, typename RealT, size_t MixinIdx, class... LossAddsTs>
	class _Loss_Addendums : private math::smatrix_td {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	public:

		typedef loss_addendum::L1<real_t> LA_L1_t;
		typedef loss_addendum::L2<real_t> LA_L2_t;

		typedef std::tuple<LossAddsTs...> addendums_tuple_t;
		static constexpr size_t addendums_count = sizeof...(LossAddsTs);
		static_assert(addendums_count > 0, "Use Loss_Addendums_dummy if you don't need loss_addendums at all");

		static constexpr auto idxL1 = tuple_utils::get_element_idx_impl<LA_L1_t, 0, LossAddsTs...>::value;
		static constexpr bool bL1Available = (idxL1 < addendums_count);

		static constexpr auto idxL2 = tuple_utils::get_element_idx_impl<LA_L2_t, 0, LossAddsTs...>::value;
		static constexpr bool bL2Available = (idxL2 < addendums_count);

		static constexpr bool defRegularizersIgnoresBiasWeights = true;

	protected:
		addendums_tuple_t m_addendumsTuple;

	public:
		enum OptsList {
			f_UseAddendum0 = 0,
			f_UseAddendumLast = (addendums_count - 1),
			f_AddendumIgnoresBias0,
			f_AddendumIgnoresBiasLast = f_AddendumIgnoresBias0 + (addendums_count - 1),
			opts_total
		};

	public:
		addendums_tuple_t& addendums()noexcept { return m_addendumsTuple; }

		template<size_t idx>
		auto& addendum()noexcept { return std::get<idx>(m_addendumsTuple); }
		template<class LaT>
		auto& addendum()noexcept { return std::get<LaT>(m_addendumsTuple); }

		const bool useAddendum(const size_t& idx)const noexcept { 
			NNTL_ASSERT(idx < addendums_count);
			return get_opt(f_UseAddendum0 + idx);
		}
		template<class LaT>
		const bool useAddendum()const noexcept {
			constexpr auto idx = tuple_utils::get_element_idx<LaT, LossAddsTs...>();
			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
			return useAddendum(idx);
		}

		void useAddendum(const size_t& idx, const bool& b) noexcept {
			NNTL_ASSERT(idx < addendums_count);
			set_opt(f_UseAddendum0 + idx, b);
		}
		template<class LaT>
		void useAddendum(const bool& b) noexcept {
			constexpr auto idx = tuple_utils::get_element_idx<LaT, LossAddsTs...>();
			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
			useAddendum(idx, b);
		}

		const bool addendumIgnoresBias(const size_t& idx)const noexcept {
			NNTL_ASSERT(idx < addendums_count);
			return get_opt(f_AddendumIgnoresBias0 + idx);
		}
		template<class LaT>
		const bool addendumIgnoresBias()const noexcept {
			constexpr auto idx = tuple_utils::get_element_idx<LaT, LossAddsTs...>();
			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
			return addendumIgnoresBias(idx);
		}

		void addendumIgnoresBias(const size_t& idx, const bool& b)noexcept {
			NNTL_ASSERT(idx < addendums_count);
			set_opt(f_AddendumIgnoresBias0 + idx, b);
		}
		template<class LaT>
		void addendumIgnoresBias(const bool& b)noexcept {
			constexpr auto idx = tuple_utils::get_element_idx<LaT, LossAddsTs...>();
			static_assert(idx < addendums_count, "Unknown loss_addendum type passed!");
			addendumIgnoresBias(idx, b);
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		template<bool b= bL1Available>
		std::enable_if_t<b, self_ref_t> L1(const real_t& l1, const bool& bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			addendum<LA_L1_t>().scale(l1);
			useAddendum<LA_L1_t>(l1 != real_t(0.0));
			addendumIgnoresBias<LA_L1_t>(bIgnoreBiasWeights);
			return get_self();
		}
		template<bool b = bL1Available>
		std::enable_if_t<b, const real_t&> L1()const noexcept { return addendum<LA_L1_t>().scale(); }

		template<bool b = bL2Available>
		std::enable_if_t<b, self_ref_t> L2(const real_t& l2, const bool& bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			addendum<LA_L2_t>().scale(l2);
			useAddendum<LA_L2_t>(l2 != real_t(0.0));
			addendumIgnoresBias<LA_L2_t>(bIgnoreBiasWeights);
			return get_self();
		}
		template<bool b = bL2Available>
		std::enable_if_t<b, const real_t&> L2()const noexcept { return addendum<LA_L2_t>().scale(); }

		//////////////////////////////////////////////////////////////////////////


		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		const bool hasLossAddendum()const noexcept {
			bool b = false;
			for (size_t optId = f_UseAddendum0; optId <= f_UseAddendumLast; ++optId) {
				b |= get_opt(optId);
			}
			return b;
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum(const realmtxdef_t& weights)const noexcept {
			real_t ret(0.0);

			//the only modification of weights we may use is stripping/restoring last (bias) column,
			//which is in fact not a modification from outside POV
			realmtxdef_t& _W = *(const_cast<realmtxdef_t*>(&weights));

			tuple_utils::for_each_up(m_addendumsTuple, [&_W, &ret, this](auto& la) {
				typedef std::decay_t<decltype(la)> la_t;

				if (useAddendum<la_t>()) {
					const auto bIgnoreBiases = addendumIgnoresBias<la_t>();
					if (bIgnoreBiases) _W.hide_last_col();
					ret += la.lossAdd(_W, get_self().get_iMath());
					if (bIgnoreBiases) _W.restore_last_col();
				}
			});

			return ret;
		}

// 		template<bool b = bL1Available>
// 		std::enable_if_t<b, const bool> use_L1_regularization()const noexcept { return useAddendum<LA_L1_t>(); }
// 		template<bool b = bL1Available>
// 		std::enable_if_t<!b, constexpr bool> use_L1_regularization()const noexcept { return false; }
// 
// 		template<bool b = bL2Available>
// 		std::enable_if_t<b, const bool> use_L2_regularization()const noexcept { return useAddendum<LA_L2_t>(); }
// 		template<bool b = bL2Available>
// 		std::enable_if_t<!b, constexpr bool> use_L2_regularization()const noexcept { return false; }

	protected:
		void _applyLossAddendums(realmtxdef_t& weights, realmtxdef_t& dLdW)const noexcept {
			tuple_utils::for_each_up(m_addendumsTuple, [&weights, &dLdW, this](auto& la) {
				typedef std::decay_t<decltype(la)> la_t;

				if (useAddendum<la_t>()) {
					const auto bIgnoreBiases = addendumIgnoresBias<la_t>();
					if (bIgnoreBiases) { dLdW.hide_last_col(); weights.hide_last_col(); }

					//ret += la.lossAdd(_W, get_self().get_iMath());
					la.dLossAdd(weights, dLdW, get_self().get_iMath());

					if (bIgnoreBiases) { dLdW.restore_last_col(); weights.restore_last_col(); }
				}
			});
		}

		void _la_construct()noexcept {
			for (size_t idx = 0; idx < addendums_count; ++idx) {
				useAddendum(idx, false);
				addendumIgnoresBias(idx, defRegularizersIgnoresBiasWeights);
			}
		}


	};

	// primary template handles types that have no nested ::addendums_tuple_t member:
	template< class, class = std::void_t<> >
	struct has_loss_addendums : std::false_type { };
	// specialization recognizes types that do have a nested ::addendums_tuple_t member:
	template< class T >
	struct has_loss_addendums<T, std::void_t<typename T::addendums_tuple_t>> : std::true_type {};


	//Don't use two loss addendums of the same type!!!
	template<typename ... LossAddsTs>
	struct Loss_Addendums_builder {
		template<typename _FC, typename RealT, size_t MixinIdx>
		using type = std::conditional_t<sizeof...(LossAddsTs)==0
			, Loss_Addendums_dummy<_FC, RealT, MixinIdx>
			, _Loss_Addendums<_FC, RealT, MixinIdx, LossAddsTs...>
		>;
	};

}
}