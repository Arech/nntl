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
#include "../utils.h"

namespace nntl {

	template<typename FinalPolymorphChild, typename ...Layrs>
	class _layer_pack_vertical : public _layer_base_forwarder<FinalPolymorphChild,
		typename std::remove_reference<typename std::tuple_element<0, const std::tuple<Layrs&...>>::type>::type::interfaces_t>
	{
	private:
		typedef _layer_base_forwarder<FinalPolymorphChild,
			typename std::remove_reference<typename std::tuple_element<0, const std::tuple<Layrs&...>>::type>::type::interfaces_t
		> _base_class_t;

	public:
		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

		typedef const std::tuple<Layrs&...> _layers;

		static constexpr size_t layers_count = sizeof...(Layrs);
		static_assert(layers_count > 1, "For vertical pack with a single inner layer use that layer instead");
		typedef typename std::remove_reference<typename std::tuple_element<0, _layers>::type>::type lowmost_layer_t;
		typedef typename std::remove_reference<typename std::tuple_element<layers_count - 1, _layers>::type>::type topmost_layer_t;
		//fprop() goes from the lowmost_layer_t to the topmost_layer_t layer.

		//the first layer mustn't be input layer, the last - can't be output layer
		static_assert(!std::is_base_of<m_layer_input, lowmost_layer_t>::value, "First layer can't be the input layer!");
		static_assert(!std::is_base_of<m_layer_output, topmost_layer_t>::value, "Last layer can't be the output layer!");

		typedef typename lowmost_layer_t::_layer_init_data_t _layer_init_data_t;
		typedef typename lowmost_layer_t::common_data_t common_data_t;

	protected:
		//we need 2 matrices for bprop()
		typedef std::array<realmtxdef_t*, 2> realmtxdefptr_array_t;

	protected:
		_layers m_layers;

	private:
		layer_index_t m_layerIdx;

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
			NNTL_ASSERT(!m_layerIdx && inc_neurons_cnt > 0);

			if (m_layerIdx) abort();
			m_layerIdx = ili.newIndex();

			_impl::_preinit_layers initializer(ili, inc_neurons_cnt);
			tuple_utils::for_eachwp_up(m_layers, initializer);
		}

		//first layer is lowmost layer
		lowmost_layer_t& lowmost_layer()const noexcept { return std::get<0>(m_layers); }
		//last layer is topmost layer
		topmost_layer_t& topmost_layer()const noexcept { return std::get<layers_count - 1>(m_layers); }

	public:
		//used by _layer_base_forwarder<> functions to forward various data from the topmost_layer()
		auto& _forwarder_layer()const noexcept { return topmost_layer(); }

		~_layer_pack_vertical()noexcept {}
		_layer_pack_vertical(const char* pCustomName, Layrs&... layrs)noexcept 
			: _base_class_t(pCustomName), m_layers(layrs...), m_layerIdx(0)
		{
			//#todo this better be done with a single static_assert in the class scope
			tuple_utils::for_each_up(m_layers, [](auto& l)noexcept {
				static_assert(!std::is_base_of<m_layer_input, decltype(l)>::value && !std::is_base_of<m_layer_output, decltype(l)>::value,
					"Inner layers of _layer_pack_vertical mustn't be input or output layers!");
			});
		}
		static constexpr const char _defName[] = "lpv";
				
		//////////////////////////////////////////////////////////////////////////
		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_up(m_layers, [&func{ std::forward<_Func>(f) }](auto& l)noexcept {
				call_F_for_each_layer(std::forward<_Func>(func), l);
			});
		}
		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			tuple_utils::for_each_down(m_layers, [&func{ std::forward<_Func>(f) }](auto& l)noexcept {
				call_F_for_each_layer_down(std::forward<_Func>(func), l);
			});
		}

		//This will apply f to every layer, packed in tuple no matter whether it is a _pack_* kind of layer or no
		template<typename _Func>
		void for_each_packed_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_up(m_layers, std::forward<_Func>(f));
		}
		template<typename _Func>
		void for_each_packed_layer_down(_Func&& f)const noexcept {
			tuple_utils::for_each_down(m_layers, std::forward<_Func>(f));
		}

		const layer_index_t& get_layer_idx() const noexcept { return m_layerIdx; }

		//overriding _layer_base_forwarder<> implementation.
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { return  get_self().lowmost_layer().get_incoming_neurons_cnt(); }

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept {
			bool b = false;
			get_self().for_each_packed_layer([&b](auto& l) {				b |= l.hasLossAddendum();			});
			return b;
		}
		//returns a loss function summand, that's caused by this layer
		real_t lossAddendum()const noexcept {
			real_t la(.0);
			get_self().for_each_packed_layer([&la](auto& l) {				la += l.lossAddendum();			});
			return la;
		}

		//////////////////////////////////////////////////////////////////////////
		//in order to pass correct initialization values to underlying topmost layer, we must define a .OuterLayerCustomFlag1Eval()
/*
deprecated:
		template<typename PhlsTupleT>
		std::enable_if_t< _impl::layer_has_OuterLayerCustomFlag1Eval<topmost_layer_t, PhlsTupleT, _layer_init_data_t>::value, bool>
			OuterLayerCustomFlag1Eval(const PhlsTupleT& lphTuple, const _layer_init_data_t& lphLid)const noexcept
		{
			return topmost_layer().OuterLayerCustomFlag1Eval(lphTuple, lphLid);
		}*/

		//////////////////////////////////////////////////////////////////////////
		//
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			_base_class_t::init();
			ErrorCode ec = ErrorCode::Success;
			layer_index_t failedLayerIdx = 0;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			//we must initialize encapsulated layers and find out their initMem() requirements. Things to consider:
			// - we'll be passing dLdA and dLdAPrev arguments of bprop() down to layer stack, therefore we must
			//		propagate/return max() of layer's max_dLdA_numel as ours lid.max_dLdA_numel.
			// - layers will be called sequentially, therefore they are safe to use a shared memory.
			auto initD = lid.dupe();
			tuple_utils::for_each_exc_last_up(m_layers, [&ec, &initD, &lid, &failedLayerIdx](auto& l)noexcept {
				if (ErrorCode::Success == ec) {
					initD.clean_passing(lid); // there are currently no IN flags/variables in _layer_init_data_t structure,
					// that must be propagated to every layer in a stack, therefore we're using the default clean_using() form.
					ec = l.init(initD);
					if (ErrorCode::Success == ec) {
						lid.update(initD);
					} else failedLayerIdx = l.get_layer_idx();
				}
			});
			//doubling the code by intention, because some layers can be incompatible with pNewActivationStorage specification
			if (ErrorCode::Success == ec) {
				initD.clean_using(lid);//we must propagate any IN flags set in the .lid variable to the topmost layer being initialized.
				ec = get_self().topmost_layer().init(initD, pNewActivationStorage);
				if (ErrorCode::Success == ec) {
					lid.update(initD);
				} else failedLayerIdx = get_self().topmost_layer().get_layer_idx();
			}

			//must be called after first inner layer initialization complete - see our get_iInspect() implementation
			get_iInspect().init_layer(get_self().get_layer_idx(), get_self().get_layer_name_str(), get_self().get_layer_type_id());

			//#TODO need some way to return failedLayerIdx
			if (ErrorCode::Success == ec) bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			get_self().for_each_packed_layer([](auto& l) {l.deinit(); });
			_base_class_t::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			//we'd just pass the pointer data down to the layer stack
			get_self().for_each_packed_layer([=](auto& l) {l.initMem(ptr, cnt); });
		}

		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			tuple_utils::for_each_exc_last_up(m_layers, [](auto& lyr)noexcept {
				lyr.on_batch_size_change();
			});
			get_self().topmost_layer().on_batch_size_change(pNewActivationStorage);
		}

		//////////////////////////////////////////////////////////////////////////
		//variation of fprop for normal layer
		template <typename LowerLayer>
		std::enable_if_t<!_impl::is_layer_wrapper<LowerLayer>::value> fprop(const LowerLayer& lowerLayer)noexcept
		{
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			get_self().fprop(_impl::trainable_layer_wrapper<LowerLayer>(lowerLayer.get_activations()));
		}
		//variation of fprop for layerwrappers
		template <typename LowerLayerWrapper>
		std::enable_if_t<_impl::is_layer_wrapper<LowerLayerWrapper>::value> fprop(const LowerLayerWrapper& lowerLayer)noexcept
		{
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().get_common_data().is_training_mode());

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self().lowmost_layer().fprop(lowerLayer);
			tuple_utils::for_eachwp_up(m_layers, [](auto& lcur, auto& lprev, const bool)noexcept {
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
				lcur.fprop(lprev);
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
			});
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.fprop_end(get_self().get_activations());
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			NNTL_ASSERT(dLdA.size() == topmost_layer().get_activations().size_no_bias());
			NNTL_ASSERT( (std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			realmtxdefptr_array_t a_dLdA = { &dLdA, &dLdAPrev };
			unsigned mtxIdx = 0;

			tuple_utils::for_eachwn_downfullbp(m_layers, [&mtxIdx, &a_dLdA](auto& lcur, auto& lprev, const bool)noexcept {
				const unsigned nextMtxIdx = mtxIdx ^ 1;
				a_dLdA[nextMtxIdx]->deform_like_no_bias(lprev.get_activations());
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
				NNTL_ASSERT(a_dLdA[mtxIdx]->size() == lcur.get_activations().size_no_bias());

				const unsigned bAlternate = lcur.bprop(*a_dLdA[mtxIdx], lprev, *a_dLdA[nextMtxIdx]);

				NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
				mtxIdx ^= bAlternate;
			});

			const unsigned nextMtxIdx = mtxIdx ^ 1;
			if (std::is_base_of<m_layer_input, LowerLayer>::value) {
				a_dLdA[nextMtxIdx]->deform(0, 0);
			}else a_dLdA[nextMtxIdx]->deform_like_no_bias(lowerLayer.get_activations());
			const unsigned bAlternate = get_self().lowmost_layer().bprop(*a_dLdA[mtxIdx], lowerLayer, *a_dLdA[nextMtxIdx]);
			NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
			mtxIdx ^= bAlternate;

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.bprop_end(mtxIdx ? dLdAPrev : dLdA);
			return mtxIdx;
		}
		
		//#TODO we should adopt topmost_layer().is_trivial_drop_samples() function signature here, but it look like non-trivial
		//to detect if the constexpr attribute was used.
		const bool is_trivial_drop_samples() const noexcept { return get_self().topmost_layer().is_trivial_drop_samples(); }

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			get_self().topmost_layer().drop_samples(mask, bBiasesToo);
		}

	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			get_self().for_each_packed_layer([&ar](auto& l) {
				ar & serialization::make_named_struct(l.get_layer_name_str().c_str(), l);
			});
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_vertical
	// If you need to derive a new class, derive it from _layer_pack_vertical (to make static polymorphism work)
	template <typename ...Layrs>
	class LPV final
		: public _layer_pack_vertical<LPV<Layrs...>, Layrs...>
	{
	public:
		~LPV() noexcept {};
		LPV(Layrs&... layrs) noexcept
			: _layer_pack_vertical<LPV<Layrs...>, Layrs...>(nullptr, layrs...) {};
		LPV(const char* pCustomName, Layrs&... layrs) noexcept
			: _layer_pack_vertical<LPV<Layrs...>, Layrs...>(pCustomName, layrs...) {};
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
}
