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
// Be sure to install https://support.microsoft.com/en-us/help/3207317/visual-c-optimizer-fixes-for-visual-studio-2015-update-3
// if your are going to use this class.
// 
// #todo: _i_loss_addendum interface should be extended to allow optimizations (caching) for a fullbatch learning with full error calculation

#include "_penalized_activations_base.h"

namespace nntl {

	//LayerTpl is a template of base layer whose activations should be penalized. See ../../tests/test_layer_penalized_activations.cpp for a use-case
	//If possible prefer the _LPA<> over the _LPPA<>.
	
	// However, in many even not very complicated cases of LayerTpl definition, compiler's bugs may leave you no option
	// but to use _LPPA<>. ---- looks like this is no longer true with a new compiler fix	
	// Be sure to install https://support.microsoft.com/en-us/help/3207317/visual-c-optimizer-fixes-for-visual-studio-2015-update-3
	// if your are going to use this class.

	template<typename FinalPolymorphChild, template <class FpcT> class LayerTpl, typename LossAddsTuple>
	class _LPA : public LayerTpl<FinalPolymorphChild>, public _PA_base<LossAddsTuple>
	{
	private:
		typedef LayerTpl<FinalPolymorphChild> _base_class_t;
		static_assert(!is_layer_output<_base_class_t>::value, "What the reason to penalize output layer activations?");
		//#note: if we actually need to work with the output_layer, then there must be very special handling of this case because of how
		//bprop() is implemented for output_layer now

		typedef _PA_base<LossAddsTuple> _PAB_t;

	public:
		typedef typename _base_class_t::real_t real_t;
		typedef typename _base_class_t::realmtx_t realmtx_t;
		typedef typename _base_class_t::realmtxdef_t realmtxdef_t;
		typedef typename _base_class_t::ErrorCode ErrorCode;
		typedef typename _base_class_t::_layer_init_data_t _layer_init_data_t;

		typedef typename _base_class_t::self_t self_t;
		typedef typename _base_class_t::self_ref_t self_ref_t;


	public:
		~_LPA()noexcept{}

		template<typename... ArgsTs>
		_LPA(const char* pCustomName, ArgsTs... _args)noexcept
			: _base_class_t(pCustomName, _args...)
		{}

		static constexpr const char _defName[] = "lpa";
		
		//////////////////////////////////////////////////////////////////////////
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		const bool hasLossAddendum()const noexcept {
			const bool b = _PAB_t::_pab_hasLossAddendum();
			return b ? b : _base_class_t::hasLossAddendum();
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept {
			return _PAB_t::_pab_lossAddendum(get_self().get_activations(), get_self().get_iMath())
				+ (_base_class_t::hasLossAddendum() ? _base_class_t::lossAddendum() : real_t(0));
		}

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			const auto r = _base_class_t::init(lid, pNewActivationStorage);
			lid.bLossAddendumDependsOnActivations = true;
			return r;
		}
		
		template <typename LowerLayerT>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayerT& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayerT>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			//#TODO this routine shouldn't be invisible for an iInspector, however if we to pass dLdA to an inspector here, it
			//should break layer's gradient check

			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			_PAB_t::_pab_update_dLdA(dLdA, get_self().get_activations(), get_self().get_iMath(), iI);

			iI.bprop_finaldLdA(dLdA);

			const auto ret = _base_class_t::bprop(dLdA, lowerLayer, dLdAPrev);

			iI.bprop_end(ret ? dLdAPrev : dLdA);

			return ret;
		}
		
// 	private:
// 		//support for ::boost::serialization
// 		#TODO
// 		friend class ::boost::serialization::access;
// 		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
// 			ar & serialization::make_named_struct(m_laLayer.get_layer_name_str().c_str(), m_laLayer);
// 		}
	};

	template<template <class FpcT> class LayerTpl, typename ...LossAddsTs>
	class LPA final : public _LPA<LPA<LayerTpl, LossAddsTs...>, LayerTpl, ::std::tuple<LossAddsTs...>> {
	public:
		~LPA() noexcept {};
		template<typename... ArgsTs>
		LPA(const char* pCustomName, ArgsTs... _args) noexcept
			: _LPA<LPA<LayerTpl, LossAddsTs...>, LayerTpl, ::std::tuple<LossAddsTs...>>(pCustomName, _args...) {};
	};

	template <template <class FpcT> class LayerTpl, typename ..._T>
	using layer_penalized_activations = typename LPA<LayerTpl, _T...>;
}