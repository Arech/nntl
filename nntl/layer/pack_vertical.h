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

// layer_pack_vertical is a collection of vertically linked layers. I.e. it defines a layer (an object with _i_layer interface), that
// consists of a set of other vertically linked layers:
// 
//       \  |  |  |  |  |  |  /
// |-----layer_pack_vertical------|
// |     \  |  |  |  |  |  |  /   |
// |    ---some_layer_last-----   |
// |    /    |    |   |   |   \   |
// |                              |
// |  . . . . . . . . . . . . . . |
// |      \  |  |  |  |  |  /     |
// |  ------some_layer_first----  |
// /    /  |  |  |  |  |  |  \    |
// |------------------------------|
//      /  |  |  |  |  |  |  \
//
// layer_pack_vertical uses all neurons of the last layer as its activation units and passes all of its input
// to the input of the first layer.
// 
#include "_pack_.h"
#include "_tuple_utils.h"
#include "../utils.h"

namespace nntl {

	template<typename FinalPolymorphChild, typename LayrsRefTuple>
	class _LPV : public _layer_base_forwarder<FinalPolymorphChild
		, typename ::std::remove_reference<typename ::std::tuple_element<0, LayrsRefTuple>::type>::type::interfaces_t>
		, public _impl::m_prop_stops_bprop_marker<typename ::std::remove_reference
		<typename ::std::tuple_element<::std::tuple_size<LayrsRefTuple>::value - 1, LayrsRefTuple>::type>::type>
	{
	private:
		typedef _layer_base_forwarder<FinalPolymorphChild,
			typename ::std::remove_reference<typename ::std::tuple_element<0, LayrsRefTuple>::type>::type::interfaces_t
		> _base_class_t;

	public:
		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

		typedef const LayrsRefTuple _layers;

		static_assert(tuple_utils::is_tuple<LayrsRefTuple>::value, "Must be a tuple!");
		static constexpr size_t layers_count = ::std::tuple_size<_layers>::value;

		static_assert(layers_count > 1, "For vertical pack with a single inner layer use that layer instead");

		typedef ::std::remove_reference_t<::std::tuple_element_t<0, _layers>> lowmost_layer_t;
		typedef ::std::remove_reference_t<::std::tuple_element_t<layers_count - 1, _layers>> topmost_layer_t;
		//fprop() goes from the lowmost_layer_t to the topmost_layer_t layer.

		static constexpr bool bAssumeFPropOnly = is_layer_stops_bprop<topmost_layer_t>::value;

		template<typename T>
		struct _layers_props : ::std::true_type {
			static_assert(::std::is_lvalue_reference<T>::value, "Must be a reference to a layer");

			typedef ::std::remove_reference_t<T> LT;
			static_assert(! ::std::is_const< LT >::value, "Must not be a const");
			static_assert(! is_layer_input<LT>::value && !is_layer_output<LT>::value, "Inner layers of _LPV mustn't be input or output layers!");
			static_assert(::std::is_base_of<_i_layer<real_t>, LT>::value, "must derive from _i_layer");
		};
		static_assert(tuple_utils::assert_each<_layers, _layers_props>::value, "LayrsRefTuple must be assembled from proper objects!");

		typedef typename lowmost_layer_t::_layer_init_data_t _layer_init_data_t;
		typedef typename lowmost_layer_t::common_data_t common_data_t;

	protected:
		//we need 2 matrices for bprop()
		typedef ::std::array<realmtxdef_t*, 2> realmtxdefptr_array_t;

	protected:
		_layers m_layers;
		
	protected:
		
		//////////////////////////////////////////////////////////////////////////
		//
	protected:
		//this is how we going to initialize layer indexes.
		//template <typename LCur, typename LPrev> friend void _init_layers::operator()(LCur&& lc, LPrev&& lp, bool bFirst)noexcept;
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(inc_neurons_cnt > 0);
			_base_class_t::_preinit_layer(ili);

			_impl::_preinit_layers initializer(ili, inc_neurons_cnt);
			tuple_utils::for_eachwp_up(m_layers, initializer);
		}

		//first layer is lowmost layer
		lowmost_layer_t& lowmost_layer()const noexcept { return ::std::get<0>(m_layers); }
		//last layer is topmost layer
		topmost_layer_t& topmost_layer()const noexcept { return ::std::get<layers_count - 1>(m_layers); }

	public:
		//used by _layer_base_forwarder<> functions to forward various data from the topmost_layer()
		auto& _forwarder_layer()const noexcept { return topmost_layer(); }

		~_LPV()noexcept {}
		_LPV(const char* pCustomName, const LayrsRefTuple& layrs)noexcept
			: _base_class_t(pCustomName), m_layers(layrs)
		{}
		_LPV(const char* pCustomName, LayrsRefTuple&& layrs)noexcept
			: _base_class_t(pCustomName), m_layers(::std::move(layrs))
		{}

		static constexpr const char _defName[] = "lpv";
				
		//////////////////////////////////////////////////////////////////////////
		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_up(m_layers, [&func{ f }](auto& l)noexcept {
				call_F_for_each_layer(func, l);//mustn't forward, because lambda is called multiple times
			});
		}
		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			tuple_utils::for_each_down(m_layers, [&func{ f }](auto& l)noexcept {
				call_F_for_each_layer_down(func, l);
			});
		}

		//This will apply f to every layer, packed in tuple no matter whether it is a _pack_* kind of layer or no
		template<typename _Func>
		void for_each_packed_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_up(m_layers, ::std::forward<_Func>(f));//we're using f only once, so let for_each_up care how to work with it
		}
		template<typename _Func>
		void for_each_packed_layer_down(_Func&& f)const noexcept {
			tuple_utils::for_each_down(m_layers, ::std::forward<_Func>(f));
		}

		//overriding _layer_base_forwarder<> implementation.
		neurons_count_t get_incoming_neurons_cnt()const noexcept { return  lowmost_layer().get_incoming_neurons_cnt(); }

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept {
			bool b = false;
			for_each_packed_layer([&b](auto& l) {				b |= l.hasLossAddendum();			});
			return b;
		}
		//returns a loss function summand, that's caused by this layer
		real_t lossAddendum()const noexcept {
			real_t la(.0);
			for_each_packed_layer([&la](auto& l) {				la += l.lossAddendum();			});
			return la;
		}

		//////////////////////////////////////////////////////////////////////////
		//
		ErrorCode layer_init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			_base_class_t::layer_init();
			ErrorCode ec = ErrorCode::Success;
			layer_index_t failedLayerIdx = 0;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().layer_deinit();
			});

			//we must initialize encapsulated layers and find out their initMem() requirements. Things to consider:
			// - we'll be passing dLdA and dLdAPrev arguments of bprop() down to layer stack, therefore we must
			//		propagate/return max() of layer's max_dLdA_numel as ours lid.max_dLdA_numel.
			// - layers will be called sequentially, therefore they are safe to use a shared memory.
			auto initD = lid.dupe();
			tuple_utils::for_each_exc_last_up(m_layers, [&ec, &initD, &lid, &failedLayerIdx](auto& l)noexcept {
				if (ErrorCode::Success == ec) {
					initD.pass_to_upper_layer();
					ec = l.layer_init(initD);
					if (ErrorCode::Success == ec) {
						lid.aggregate_from(initD);
					} else failedLayerIdx = l.get_layer_idx();
				}
			});
			//separate initialization for the top layer.
			//doubling the code by intention, because some layers can be incompatible with pNewActivationStorage specification
			if (ErrorCode::Success == ec) {
				initD.pass_to_upper_layer();
				ec = topmost_layer().layer_init(initD, pNewActivationStorage);
				if (ErrorCode::Success == ec) {
					lid.aggregate_from(initD);
					lid.outgBS = initD.outgBS;
				} else failedLayerIdx = topmost_layer().get_layer_idx();
			}

			//must be called after first inner layer initialization complete - see our get_iInspect() implementation
			get_iInspect().init_layer(get_layer_idx(), get_layer_name_str(), get_layer_type_id());

			//#TODO need some way to return failedLayerIdx
			if (ErrorCode::Success == ec) bSuccessfullyInitialized = true;
			return ec;
		}

		void layer_deinit() noexcept {
			for_each_packed_layer([](auto& l) {l.layer_deinit(); });
			_base_class_t::layer_deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			//we'd just pass the pointer data down to the layer stack
			for_each_packed_layer([=](auto& l) {l.initMem(ptr, cnt); });
		}

		vec_len_t on_batch_size_change(vec_len_t incBatchSize, real_t*const pNewActivationStorage = nullptr)noexcept {
			tuple_utils::for_each_exc_last_up(m_layers, [&incBatchSize](auto& lyr)noexcept {
				incBatchSize = lyr.on_batch_size_change(incBatchSize);
			});
			return topmost_layer().on_batch_size_change(incBatchSize, pNewActivationStorage);
		}

	protected:
		template<typename LLWrapT>
		void _lpv_fprop(const realmtx_t& prevAct)noexcept {
			NNTL_ASSERT(prevAct.test_biases_strict());
			auto& iI = get_iInspect();
			iI.fprop_begin(get_layer_idx(), prevAct, get_common_data().is_training_mode());

			lowmost_layer().fprop(LLWrapT(prevAct));

			tuple_utils::for_eachwp_up(m_layers, [](auto& lcur, auto& lprev, const bool)noexcept {
				lcur.fprop(lprev);
			});
			
			iI.fprop_activations(get_activations());
			iI.fprop_end(get_activations());
		}

		template<typename LLWrapT>
		unsigned _lpv_bprop(realmtxdef_t& dLdA, realmtxdef_t& dLdAPrev, const realmtx_t& prevAct)noexcept {
			static constexpr bool bPrevLayerWBprop = is_layer_with_bprop<LLWrapT>::value;
			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT(dLdA.size() == topmost_layer().get_activations().size_no_bias());
			NNTL_ASSERT(!bPrevLayerWBprop || dLdAPrev.size() == prevAct.size_no_bias());

			auto& iI = get_iInspect();
			iI.bprop_begin(get_layer_idx(), dLdA);
			iI.bprop_finaldLdA(dLdA);

			realmtxdefptr_array_t a_dLdA = { &dLdA, &dLdAPrev };
			unsigned mtxIdx = 0;
			//bool bContBprop = true;

			//tuple_utils::for_eachwn_downfullbp(m_layers, [&mtxIdx, &a_dLdA/*, &bContBprop*/](auto& lcur, auto& lprev, const bool)noexcept {
			tuple_utils::for_each_down4bprop(m_layers, [&mtxIdx, &a_dLdA/*, &bContBprop*/](auto& lcur, auto& lprev)noexcept {
				//if (bContBprop && lcur.bDoBProp()) {
					const unsigned nextMtxIdx = mtxIdx ^ 1;
					a_dLdA[nextMtxIdx]->deform_like_no_bias(lprev.get_activations());
					NNTL_ASSERT(lprev.get_activations().test_biases_strict());
					NNTL_ASSERT(a_dLdA[mtxIdx]->size() == lcur.get_activations().size_no_bias());

					const unsigned bAlternate = lcur.bprop(*a_dLdA[mtxIdx], lprev, *a_dLdA[nextMtxIdx]);

					NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
					NNTL_ASSERT(lprev.get_activations().test_biases_strict());
					mtxIdx ^= bAlternate;
				//} else bContBprop = false;
			});

			//if (bContBprop && lowmost_layer().bDoBProp()) {
				const unsigned nextMtxIdx = mtxIdx ^ 1;
				if (bPrevLayerWBprop) {
					a_dLdA[nextMtxIdx]->deform_like_no_bias(prevAct);
				} else a_dLdA[nextMtxIdx]->deform(0, 0); 
				const unsigned bAlternate = lowmost_layer().bprop(*a_dLdA[mtxIdx], LLWrapT(prevAct), *a_dLdA[nextMtxIdx]);
				NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
				mtxIdx ^= bAlternate;
			//}

			NNTL_ASSERT(prevAct.test_biases_strict());

			iI.bprop_end(mtxIdx ? dLdAPrev : dLdA);
			return mtxIdx;
		}

	public:
		//////////////////////////////////////////////////////////////////////////
		//variation of fprop for normal layer
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			get_self()._lpv_fprop<_impl::wrap_trainable_layer<LowerLayer>>(lowerLayer.get_activations());
		}

		template <typename LowerLayer>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			static_assert(!bAssumeFPropOnly, "");
			return get_self()._lpv_bprop<_impl::wrap_trainable_layer<LowerLayer>>(dLdA, dLdAPrev, lowerLayer.get_activations());
		}

	private:
		//support for ::boost::serialization
		friend class ::boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			for_each_packed_layer([&ar](auto& l) {
				ar & serialization::make_named_struct(l.get_layer_name_str().c_str(), l);
			});
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _LPV
	// If you need to derive a new class, derive it from _LPV (to make static polymorphism work)
	template <typename ...Layrs>
	class LPV final : public _LPV<LPV<Layrs...>, ::std::tuple<Layrs&...>>
	{
	public:
		LPV(Layrs&... layrs) noexcept
			: _LPV<LPV<Layrs...>, ::std::tuple<Layrs&...>>(nullptr, ::std::tie(layrs...)) {};
		LPV(const char* pCustomName, Layrs&... layrs) noexcept
			: _LPV<LPV<Layrs...>, ::std::tuple<Layrs&...>>(pCustomName, ::std::tie(layrs...)) {};
	};

	template <typename ..._T>
	using layer_pack_vertical = typename LPV<_T...>;

	template <typename ...Layrs> inline constexpr
	LPV <Layrs...> make_layer_pack_vertical(Layrs&... layrs) noexcept {
		return LPV<Layrs...>(layrs...);
	}
	template <typename ...Layrs> inline constexpr
	LPV <Layrs...> make_layer_pack_vertical(const char* pCustomName, Layrs&... layrs) noexcept {
		return LPV<Layrs...>(pCustomName, layrs...);
	}

	//////////////////////////////////////////////////////////////////////////
	template <typename LayrsTuple>
	class LPVt final : public _LPV<LPVt<LayrsTuple>, LayrsTuple>
	{
	public:
		~LPVt() noexcept {};
		LPVt(const LayrsTuple& layrs) noexcept : _LPV<LPVt<LayrsTuple>, LayrsTuple>(nullptr, layrs) {};
		LPVt(LayrsTuple&& layrs) noexcept : _LPV<LPVt<LayrsTuple>, LayrsTuple>(nullptr, ::std::move(layrs)) {};

		LPVt(const char* pCustomName, const LayrsTuple& layrs) noexcept
			: _LPV<LPVt<LayrsTuple>, LayrsTuple>(pCustomName, layrs) {};

		LPVt(const char* pCustomName, LayrsTuple&& layrs) noexcept
			: _LPV<LPVt<LayrsTuple>, LayrsTuple>(pCustomName, ::std::move(layrs)) {};
	};
}
