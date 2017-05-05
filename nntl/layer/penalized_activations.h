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

// layer_penalized_activations implements a layer wrapper that offers a way to impose some restrictions, such as L1 or L2, over
// a layer activations values.
// 
// #todo: _i_loss_addendum interface should be extended to allow optimizations (caching) for a fullbatch learning with full error calculation

#include "_layer_base.h"
//#include "_pack_.h"

#include "../loss_addendum/L1.h"
#include "../loss_addendum/L2.h"
#include "../loss_addendum/DeCov.h"

namespace nntl {

	//LayerTpl is a template of base layer whose activations should be penalized. See ../../tests/test_layer_penalized_activations.cpp for a use-case
	template<typename FinalPolymorphChild, template <class FpcT> class LayerTpl, typename ...LossAddsTs>
	class _layer_penalized_activations : public LayerTpl<FinalPolymorphChild>
	{
	private:
		typedef LayerTpl<FinalPolymorphChild> _base_class_t;
		static_assert(!is_layer_output<_base_class_t>::value, "What the reason to penalize output layer activations?");
		//#note: if we actually need to work with the output_layer, then there must be very special handling of this case because of how
		//bprop() is implemented for output_layer now

	public:
		typedef typename _base_class_t::real_t real_t;
		typedef typename _base_class_t::realmtx_t realmtx_t;
		typedef typename _base_class_t::realmtxdef_t realmtxdef_t;
		typedef typename _base_class_t::ErrorCode ErrorCode;
		typedef typename _base_class_t::_layer_init_data_t _layer_init_data_t;

		typedef typename _base_class_t::self_t self_t;
		typedef typename _base_class_t::self_ref_t self_ref_t;

		
		typedef std::tuple<LossAddsTs...> addendums_tuple_t;
		static constexpr size_t addendums_count = sizeof...(LossAddsTs);
		static_assert(addendums_count > 0, "Use LayerT directly instead of LPLA");
		
	protected:
		addendums_tuple_t m_addendumsTuple;

	public:
		~_layer_penalized_activations()noexcept{}

		template<typename... ArgsTs>
		_layer_penalized_activations(const char* pCustomName, ArgsTs... _args)noexcept
			: _base_class_t(pCustomName, _args...)
		{}

		static constexpr const char _defName[] = "lpa";
		
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		
		addendums_tuple_t& addendums()noexcept { return m_addendumsTuple; }

		template<size_t idx>
		auto& addendum()noexcept { return std::get<idx>(m_addendumsTuple); }
		template<class LaT>
		auto& addendum()noexcept { return std::get<LaT>(m_addendumsTuple); }

		//////////////////////////////////////////////////////////////////////////
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		const bool hasLossAddendum()const noexcept {
			bool b = false;
			tuple_utils::for_each_up(m_addendumsTuple, [&b](const auto& la) noexcept {
				b |= la.bEnabled();
			});
			return b ? b : _base_class_t::hasLossAddendum();
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept {
			real_t ret(0.0);

			//the only modification of activations we may use is stripping/restoring last (bias) column,
			//which is in fact not a modification from outside POV
			realmtxdef_t& act = const_cast<realmtxdef_t&>(get_self().get_activations());
			NNTL_ASSERT(act.emulatesBiases());
			const auto bRestoreBiases = act.hide_biases();

			tuple_utils::for_each_up(m_addendumsTuple, [&act, &ret, &iM = get_self().get_iMath()](auto& la) {
				if (la.bEnabled()) {
					ret += la.lossAdd(act, iM);
				}
			});

			if (bRestoreBiases) act.restore_biases();

			return ret + _base_class_t::lossAddendum();
		}

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			const auto r = _base_class_t::init(lid, pNewActivationStorage);
			lid.bLossAddendumDependsOnActivations = true;
			return r;
		}
		
	protected:

		//#note: if we actually need to work with the output_layer, then there must be very special handling of this case because of how
		//bprop() is implemented for output_layer now
		void _update_dLdA(realmtxdef_t& dLdA)noexcept {
			realmtxdef_t& act = const_cast<realmtxdef_t&>(get_self().get_activations());
			NNTL_ASSERT(act.emulatesBiases());
			const auto bRestoreBiases = act.hide_biases();

			tuple_utils::for_each_up(m_addendumsTuple, [&act, &dLdA, &iM = get_self().get_iMath(), this](auto& la) {
				typedef std::decay_t<decltype(la)> la_t;
				static_assert(loss_addendum::is_loss_addendum<la_t>::value, "Every Loss_addendum class must implement loss_addendum::_i_loss_addendum<>");

				if (la.bEnabled()) {
					la.dLossAdd(act, dLdA, iM);
				}
			});

			if (bRestoreBiases) act.restore_biases();
		}

	public:
		template <typename LowerLayerT>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayerT& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayerT>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			//#TODO this routine shouldn't be invisible for an iInspector, however if we to pass dLdA to an inspector here, it
			//should break layer's gradient check

// 			auto& iI = get_self().get_iInspect();
// 			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			_update_dLdA(dLdA);

			const auto ret = _base_class_t::bprop(dLdA, lowerLayer, dLdAPrev);

			//iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}
		
// 	private:
// 		//support for boost::serialization
// 		#TODO
// 		friend class boost::serialization::access;
// 		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
// 			ar & serialization::make_named_struct(m_laLayer.get_layer_name_str().c_str(), m_laLayer);
// 		}

	public:
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// consider the following as an example of how to enable, setup and use a custom _i_loss_addendum derived class object.

		typedef loss_addendum::L1<real_t> LA_L1_t;//for the convenience
		typedef loss_addendum::L2<real_t> LA_L2_t;

		static constexpr auto idxL1 = tuple_utils::get_element_idx_impl<LA_L1_t, 0, LossAddsTs...>::value;
		static constexpr bool bL1Available = (idxL1 < addendums_count);

		static constexpr auto idxL2 = tuple_utils::get_element_idx_impl<LA_L2_t, 0, LossAddsTs...>::value;
		static constexpr bool bL2Available = (idxL2 < addendums_count);

		template<bool b = bL1Available>
		std::enable_if_t<b, self_ref_t> L1(const real_t& l1)noexcept {
			auto& adn = addendum<LA_L1_t>();
			adn.scale(l1);
			return get_self();
		}
		template<bool b = bL1Available>
		std::enable_if_t<b, real_t> L1()const noexcept { return addendum<LA_L1_t>().scale(); }

		template<bool b = bL2Available>
		std::enable_if_t<b, self_ref_t> L2(const real_t& l2)noexcept {
			auto& adn = addendum<LA_L2_t>();
			adn.scale(l2);
			return get_self();
		}
		template<bool b = bL2Available>
		std::enable_if_t<b, real_t> L2()const noexcept { return addendum<LA_L2_t>().scale(); }
	};

	template<template <class FpcT> class LayerTpl, typename ...LossAddsTs>
	class LPA final : public _layer_penalized_activations<LPA<LayerTpl, LossAddsTs...>, LayerTpl, LossAddsTs...> {
	public:
		~LPA() noexcept {};
		template<typename... ArgsTs>
		LPA(const char* pCustomName, ArgsTs... _args) noexcept
			: _layer_penalized_activations<LPA<LayerTpl, LossAddsTs...>, LayerTpl, LossAddsTs...>(pCustomName, _args...) {};
	};

	template <template <class FpcT> class LayerTpl, typename ..._T>
	using layer_penalized_activations = typename LPA<LayerTpl, _T...>;
}