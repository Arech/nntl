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

// layer_extender implements a layer wrapper that offers a way to impose some restrictions, such as L1 or L2, over
// a layer activations values, as well as adding dropout algorithm
// 
// Be sure to install https://support.microsoft.com/en-us/help/3207317/visual-c-optimizer-fixes-for-visual-studio-2015-update-3
// if your are going to use this class.
// 
// #todo: _i_loss_addendum interface should be extended to allow optimizations (caching) for a fullbatch learning with full error calculation

#include "_penalized_activations_base.h"
#include "../dropout.h"

#include "_pack_traits.h"

namespace nntl {

	//LayerTpl is a template of a base layer. See ../../tests/test_layer_penalized_activations.cpp for a use-case
	// Be sure to install https://support.microsoft.com/en-us/help/3207317/visual-c-optimizer-fixes-for-visual-studio-2015-update-3
	// if your are going to use this class.

	namespace _impl {
		template<typename _ToV>
		struct surrogateRealT {
			typedef typename ::std::tuple_element_t<0, _ToV>::real_t type;
		};
		template<>
		struct surrogateRealT<void> {
			typedef void type;
		};

		template <typename DropoutTOrVoid, typename _ToV>
		using dropoutTVoidHandler = ::std::conditional_t < ::std::is_same<void, DropoutTOrVoid>::value
			, NoDropout < typename surrogateRealT<_ToV>::type >
			, DropoutTOrVoid>;
	}

	template<typename FinalPolymorphChild
		, template <class FpcT> class LayerTpl
		, typename LossAddsTupleTOrVoid
		, typename DropoutTOrVoid
	>
		class _LEx
		: public LayerTpl<FinalPolymorphChild>
		, public _PA_base_selector<LossAddsTupleTOrVoid>
		, public _impl::dropoutTVoidHandler<DropoutTOrVoid, LossAddsTupleTOrVoid>
	{
	private:
		typedef LayerTpl<FinalPolymorphChild> _base_class_t;
		static_assert(!is_layer_output<_base_class_t>::value, "What the reason to penalize output layer activations?");
		//#note: if we actually need to work with the output_layer, then there must be very special handling of this case because of how
		//bprop() is implemented for output_layer now

		typedef _PA_base_selector<LossAddsTupleTOrVoid> _PAB_t;

	public:
		//have to forward declarations (probably a MSVC bug)
		typedef typename _base_class_t::real_t real_t;
		typedef typename _base_class_t::numel_cnt_t numel_cnt_t;
		typedef typename _base_class_t::realmtx_t realmtx_t;
		typedef typename _base_class_t::realmtxdef_t realmtxdef_t;
		typedef typename _base_class_t::ErrorCode ErrorCode;
		typedef typename _base_class_t::_layer_init_data_t _layer_init_data_t;
		typedef typename _base_class_t::iInspect_t iInspect_t;
		typedef typename _base_class_t::interfaces_t interfaces_t;

		typedef typename _base_class_t::self_t self_t;
		typedef typename _base_class_t::self_ref_t self_ref_t;

		typedef _impl::dropoutTVoidHandler<DropoutTOrVoid, LossAddsTupleTOrVoid> Dropout_t;
		static_assert(::std::is_same<typename Dropout_t::real_t, real_t>::value, "Invalid real_t");

		static constexpr bool bActivationPenalizationAvailable = can_penalize_activations<_PAB_t>::value;
		static constexpr bool bDropoutAvailable = !is_dummy_dropout<Dropout_t>::value;
		static_assert(bActivationPenalizationAvailable || bDropoutAvailable, "What for?");

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & boost::serialization::base_object<_base_class_t>(*this);

			_dropout_serialize(ar, version);
			//#todo something similar should be done with _PAB_t
		}

	public:
		~_LEx()noexcept {}

		template<typename... ArgsTs>
		_LEx(ArgsTs&&... _args)noexcept : _base_class_t(::std::forward<ArgsTs>(_args)...)
		{}

		static constexpr const char _defName[] = "lex";

		//////////////////////////////////////////////////////////////////////////
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		template<bool c = bActivationPenalizationAvailable>
		::std::enable_if_t<c, bool> hasLossAddendum()const noexcept {
			const bool b = _PAB_t::_pab_hasLossAddendum();
			return b || _base_class_t::hasLossAddendum();
		}
		template<bool c = bActivationPenalizationAvailable>
		::std::enable_if_t<!c, bool> hasLossAddendum()const noexcept { return _base_class_t::hasLossAddendum(); }

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		template<bool c = bActivationPenalizationAvailable>
		::std::enable_if_t<c, real_t> lossAddendum()const noexcept {
			// 			return _PAB_t::_pab_lossAddendum(get_activations(), get_iMath())
			// 				+ (_base_class_t::hasLossAddendum() ? _base_class_t::lossAddendum() : real_t(0));
			// 			//the commented code above costs about 2% of run time comparing to variant without _base_class_t::hasLossAddendum() call
			return _PAB_t::_pab_lossAddendum(get_activations(), get_common_data())
				+ _base_class_t::lossAddendum();
		}
		template<bool c = bActivationPenalizationAvailable>
		::std::enable_if_t<!c, real_t> lossAddendum()const noexcept { return _base_class_t::lossAddendum(); }

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

	private:
		template<bool c = bDropoutAvailable>
		constexpr ::std::enable_if_t<!c, bool> _init_do(_layer_init_data_t& )const noexcept { return true; }

		template<bool c = bDropoutAvailable>
		::std::enable_if_t<c, bool> _init_do(_layer_init_data_t& lid) noexcept {
			if (!_dropout_init(get_neurons_cnt(), get_common_data()))
				return false;

			if (get_common_data().is_training_possible()) {
				lid.bOutputDifferentDuringTraining = true;
			}

			if (is_activations_shared()) {
				STDCOUTL("** Attention! Layer " << get_layer_name_str() 
					<< " is configured to have a Dropout, however it's activations are shared. Sure there's no double dropout application?");
			}

			return true;
		}

	public:
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			const auto ec = _base_class_t::init(lid, pNewActivationStorage);
			if (ec != ErrorCode::Success) return ec;

			const auto as = get_activations_size();
			if (!_PAB_t::_pab_init(mtx_size_t(as.first, as.second - 1), get_common_data()))
				return ErrorCode::CantInitializePAB;

			if (bActivationPenalizationAvailable) {
				lid.bHasLossAddendum = true;
				lid.bLossAddendumDependsOnActivations = true;
			}

			return _init_do(lid) ? ErrorCode::Success : ErrorCode::DropoutInitFailed;
		}

		void deinit()noexcept {
			_dropout_deinit();
			_PAB_t::_pab_deinit();
			_base_class_t::deinit();
		}
		
		//////////////////////////////////////////////////////////////////////////

		template<bool c = bDropoutAvailable>
		::std::enable_if_t<c> on_batch_size_change(const real_t learningRateScale, real_t*const pNewActivationStorage = nullptr)noexcept {
			_base_class_t::on_batch_size_change(learningRateScale, pNewActivationStorage);
			_dropout_on_batch_size_change(get_common_data());
		}

		template<bool c = bDropoutAvailable>
		::std::enable_if_t<!c> on_batch_size_change(const real_t learningRateScale, real_t*const pNewActivationStorage = nullptr)noexcept {
			_base_class_t::on_batch_size_change(learningRateScale, pNewActivationStorage);
		}

	protected:
		//////////////////////////////////////////////////////////////////////////
		// fprop() handling for various underlying layer types
		//////////////////////////////////////////////////////////////////////////		
		//we shouldn't apply dLdA penalty to the gating layer, it's useless but moreover it may even hurt DeCov

		template<typename T = _base_class_t, bool ba = bActivationPenalizationAvailable>
		::std::enable_if_t< ba && is_pack_gated<T>::value> _fprop4PA() noexcept {
			static_assert(_base_class_t::gate_neurons_count > 0, "invalid gate_neurons_count");
			const auto& CD = get_common_data();
#pragma warning(disable : 4127)
			if (_PAB_t::bAnyRequiresOnFprop && CD.is_training_mode()) {
				//we won't modify activations, just a trick to create a wrapper matrix 
				auto& act = const_cast<realmtxdef_t&>(get_activations());
				NNTL_ASSERT(act.emulatesBiases());

				//first _base_class_t::gate_neurons_count neurons in activations are for the gate. Just dropping them
				auto nogateAct = act.submatrix_cols_no_bias(_base_class_t::gate_neurons_count, act.cols_no_bias() - _base_class_t::gate_neurons_count);
				_PAB_t::_pab_fprop(nogateAct, CD);
			}
#pragma warning(default : 4127)
		}

		template<typename T = _base_class_t, bool ba = bActivationPenalizationAvailable>
		::std::enable_if_t<ba && !is_pack_gated<T>::value> _fprop4PA() noexcept {
			const auto& CD = get_common_data();
			if (CD.is_training_mode()) {
				_PAB_t::_pab_fprop(get_activations(), CD);
			}
		}

		template<typename T = _base_class_t, bool ba = bActivationPenalizationAvailable>
		constexpr ::std::enable_if_t<!ba> _fprop4PA() const noexcept {}

		//////////////////////////////////////////////////////////////////////////
		// for Dropout
		template<typename T = _base_class_t, bool ba = bDropoutAvailable>
		::std::enable_if_t<ba && is_pack_gated<T>::value && !Dropout_t::bDropoutIsZeroStable> _fprop4Dropout() noexcept {
			static_assert(false, "Don't put LEx with a non zero stable dropout over the gated/LPO layer. Do the inverse!");
			// changing absent activations to a non-zero value _may_ screw upper layers. It's better to move dropout inside
			// of individual inner layers
		}
		template<typename T = _base_class_t, bool ba = bDropoutAvailable>
		::std::enable_if_t<ba && !(is_pack_gated<T>::value && !Dropout_t::bDropoutIsZeroStable)> _fprop4Dropout() noexcept {
			//no need to make special handling of gating layers etc.
			if (bDropout()) {
				_dropout_apply(_get_activations_mutable(), get_common_data());
			}
		}
		template<typename T = _base_class_t, bool ba = bDropoutAvailable>
		constexpr ::std::enable_if_t<!ba> _fprop4Dropout() const noexcept {}

	public:
		//////////////////////////////////////////////////////////////////////////
		template<typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			_base_class_t::fprop(lowerLayer);
			_fprop4PA();
			_fprop4Dropout();
		}


	protected:

		template<typename T = _base_class_t, bool ba = bActivationPenalizationAvailable>
		::std::enable_if_t< ba && is_pack_gated<T>::value> _update_dLdA4PA(realmtxdef_t& dLdA) noexcept {
			//we won't modify activations, just a trick to create a wrapper matrix 
			auto& act = const_cast<realmtxdef_t&>(get_activations());
			NNTL_ASSERT(act.emulatesBiases() && act.size_no_bias() == dLdA.size());
			NNTL_ASSERT(!dLdA.emulatesBiases());

			//first _base_class_t::gate_neurons_count neurons in activations are for the gate. Just dropping them
			auto nogateAct = act.submatrix_cols_no_bias(_base_class_t::gate_neurons_count, act.cols_no_bias() - _base_class_t::gate_neurons_count);
			auto nogatedLdA = dLdA.submatrix_cols_no_bias(_base_class_t::gate_neurons_count, dLdA.cols_no_bias() - _base_class_t::gate_neurons_count);

			_PAB_t::_pab_update_dLdA(nogatedLdA, nogateAct, get_common_data());
		}

		template<typename T = _base_class_t, bool ba = bActivationPenalizationAvailable>
		::std::enable_if_t<ba && !is_pack_gated<T>::value> _update_dLdA4PA(realmtx_t& dLdA) noexcept {
			_PAB_t::_pab_update_dLdA(dLdA, get_activations(), get_common_data());
		}

		template<typename T = _base_class_t, bool ba = bActivationPenalizationAvailable>
		::std::enable_if_t<!ba> _update_dLdA4PA(realmtx_t&) const noexcept {}

		//////////////////////////////////////////////////////////////////////////

		template<bool c = bDropoutAvailable>
		::std::enable_if_t<c> _update_dLdA4Dropout(realmtx_t& dLdA) noexcept {
			if (bDropout()) {
				//we must cancel activations that was dropped out by the mask (should they've been restored by activation_f_t::df())
				//and restore the scale of dL/dA according to 1/p
				//because the true scaled_dL/dA = 1/p * computed_dL/dA
				//we must undo the scaling of activations done by inverted dropout in order to obtain correct activation values
				//It must be done as a basis to obtain correct dA/dZ
				_dropout_restoreScaling(dLdA, _get_activations_mutable(), get_common_data());
			}
		}

		template<bool c = bDropoutAvailable>
		constexpr ::std::enable_if_t<!c> _update_dLdA4Dropout(realmtx_t&)const noexcept{}

	public:
		template <typename LowerLayerT>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayerT& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayerT>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			auto& iI = get_iInspect();
			iI.bprop_begin(get_layer_idx(), dLdA);

			//first we erase dL/dA for dropped out activations and scale dL/dA for others and then
			// restore activations magnitude after dropout
			_update_dLdA4Dropout(dLdA);

			//then apply penalizations
			_update_dLdA4PA(dLdA);

			iI.bprop_finaldLdA(dLdA);

			const auto ret = _base_class_t::bprop(dLdA, lowerLayer, dLdAPrev);
			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template<template <class FpcT> class LayerTpl, typename ...LossAddsTs>
	class LPA final : public _LEx<LPA<LayerTpl, LossAddsTs...>, LayerTpl, ::std::tuple<LossAddsTs...>, void> {
	public:
		~LPA() noexcept {};
		template<typename... ArgsTs>
		LPA(ArgsTs&&... _args) noexcept : _LEx<LPA<LayerTpl, LossAddsTs...>, LayerTpl, ::std::tuple<LossAddsTs...>, void>
			(::std::forward<ArgsTs>(_args)...) {};
	};

	template <template <class FpcT> class LayerTpl, typename ..._T>
	using layer_penalized_activations = typename LPA<LayerTpl, _T...>;

	//////////////////////////////////////////////////////////////////////////

	template<template <class FpcT> class LayerTpl, typename LossAddsTupleT>
	class LPAt final : public _LEx<LPAt<LayerTpl, LossAddsTupleT>, LayerTpl, LossAddsTupleT, void> {
	public:
		~LPAt() noexcept {};
		template<typename... ArgsTs>
		LPAt(ArgsTs&&... _args) noexcept : _LEx<LPAt<LayerTpl, LossAddsTupleT>, LayerTpl, LossAddsTupleT, void>
			(::std::forward<ArgsTs>(_args)...) {};
	};

	//////////////////////////////////////////////////////////////////////////
	template<template <class FpcT> class LayerTpl, typename DropoutT>
	class LDO final : public _LEx<LDO<LayerTpl, DropoutT>, LayerTpl, void, DropoutT> {
	public:
		~LDO() noexcept {};
		template<typename... ArgsTs>
		LDO(ArgsTs&&... _args) noexcept
			: _LEx<LDO<LayerTpl, DropoutT>, LayerTpl, void, DropoutT>(::std::forward<ArgsTs>(_args)...) {};
	};

	template <template <class FpcT> class LayerTpl, typename DropoutT>
	using layer_dropout = typename LDO<LayerTpl, DropoutT>;

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template<template <class FpcT> class LayerTpl, typename DropoutT, typename ...LossAddsTs>
	class LPA_DO final : public _LEx<LPA_DO<LayerTpl, DropoutT, LossAddsTs...>, LayerTpl, ::std::tuple<LossAddsTs...> , DropoutT >	{
	public:
		~LPA_DO() noexcept {};
		template<typename... ArgsTs>
		LPA_DO(ArgsTs&&... _args) noexcept
			: _LEx<LPA_DO<LayerTpl, DropoutT, LossAddsTs...>, LayerTpl, ::std::tuple<LossAddsTs...>, DropoutT
			>(::std::forward<ArgsTs>(_args)...) {};
	};
	
	//////////////////////////////////////////////////////////////////////////

	template<template <class FpcT> class LayerTpl, typename DropoutT, typename LossAddsTupleT>
	class LPAt_DO final : public _LEx<LPAt_DO<LayerTpl, DropoutT, LossAddsTupleT>, LayerTpl, LossAddsTupleT, DropoutT>	{
	public:
		~LPAt_DO() noexcept {};
		template<typename... ArgsTs>
		LPAt_DO( ArgsTs&&... _args) noexcept
			: _LEx<LPAt_DO<LayerTpl, DropoutT, LossAddsTupleT>, LayerTpl, LossAddsTupleT, DropoutT
			>(::std::forward<ArgsTs>(_args)...) {};
	};
}