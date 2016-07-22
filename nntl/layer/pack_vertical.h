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
	class _layer_pack_vertical : public _i_layer<typename std::remove_reference<typename std::tuple_element<0, const std::tuple<Layrs&...>>
		::type>::type::iMath_t::real_t>
	{
	public:
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;

		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

		typedef const std::tuple<Layrs&...> _layers;

		static constexpr size_t layers_count = sizeof...(Layrs);
		static_assert(layers_count > 1, "For vertical pack with a single inner layer use that layer instead");
		typedef typename std::remove_reference<typename std::tuple_element<0, _layers>::type>::type first_layer_t;
		typedef typename std::remove_reference<typename std::tuple_element<layers_count - 1, _layers>::type>::type last_layer_t;
		//fprop() moves from first to last layer.

		//the first layer mustn't be input layer, the last - can't be output layer
		static_assert(!std::is_base_of<m_layer_input, first_layer_t>::value, "First layer can't be the input layer!");
		static_assert(!std::is_base_of<m_layer_output, last_layer_t>::value, "Last layer can't be the output layer!");

		typedef typename first_layer_t::iMath_t iMath_t;
		typedef typename first_layer_t::iRng_t iRng_t;
		typedef typename first_layer_t::_layer_init_data_t _layer_init_data_t;

	protected:
		//we need 2 matrices for bprop()
		typedef std::array<realmtxdef_t*, 2> realmtxdefptr_array_t;

	protected:
		_layers m_layers;

	private:
		layer_index_t m_layerIdx;

		//////////////////////////////////////////////////////////////////////////
		//
	protected:
		//this is how we going to initialize layer indexes.
		//template <typename LCur, typename LPrev> friend void _init_layers::operator()(LCur&& lc, LPrev&& lp, bool bFirst)noexcept;
		friend class _impl::_preinit_layers;
		void _preinit_layer(layer_index_t& idx, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx && idx > 0 && inc_neurons_cnt > 0);

			if (m_layerIdx) abort();
			m_layerIdx = idx;

			_impl::_preinit_layers initializer(idx + 1, inc_neurons_cnt);
			utils::for_eachwp_up(m_layers, initializer);
			idx = initializer._idx;
		}

		first_layer_t& first_layer()const noexcept { return std::get<0>(m_layers); }
		last_layer_t& last_layer()const noexcept { return std::get<layers_count - 1>(m_layers); }

	public:
		~_layer_pack_vertical()noexcept {}
		_layer_pack_vertical(Layrs&... layrs)noexcept : m_layers(layrs...), m_layerIdx(0) {
			//#todo this better be done with a single static_assert in the class scope
			utils::for_each_up(m_layers, [](auto& l)noexcept {
				static_assert(!std::is_base_of<m_layer_input, decltype(l)>::value && !std::is_base_of<m_layer_output, decltype(l)>::value,
					"Inner layers of _layer_pack_vertical mustn't be input or output layers!");
			});
		}

		self_ref_t get_self() noexcept {
			static_assert(std::is_base_of<_layer_pack_vertical, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_pack_vertical");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<_layer_pack_vertical, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_pack_vertical");
			return static_cast<self_cref_t>(*this);
		}

		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func& f)const noexcept {
			utils::for_each_up(m_layers, [&f](auto& l)noexcept {
				call_F_for_each_layer(f, l);
			});
		}
		//This will apply f to every layer, packed in tuple no matter whether it is a _pack_* kind of layer or no
		template<typename _Func>
		void for_each_packed_layer(_Func& f)const noexcept {
			utils::for_each_up(m_layers, f);
		}

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { return get_self().last_layer().get_neurons_cnt(); }
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { return  get_self().first_layer().get_incoming_neurons_cnt(); }

		const realmtx_t& get_activations()const noexcept { return get_self().last_layer().get_activations(); }

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, "lpv%d", static_cast<unsigned>(get_self().get_layer_idx()));
		}
		std::string get_layer_name_str()const noexcept {
			constexpr size_t ml = 16;
			char n[ml];
			get_self().get_layer_name(n, ml);
			return std::string(n);
		}

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

		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept {
			utils::for_each_exc_last_up(m_layers, [batchSize](auto& lyr)noexcept {
				lyr.set_mode(batchSize);
			});
			get_self().last_layer().set_mode(batchSize, pNewActivationStorage);
		}

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			ErrorCode ec = ErrorCode::Success;
			layer_index_t failedLayerIdx = 0;

			auto initD = lid.dupe();
			utils::for_each_exc_last_up(m_layers, [&](auto& l)noexcept {
				if (ErrorCode::Success == ec) {
					initD.clean();
					ec = l.init(initD);
					if (ErrorCode::Success == ec) {
						lid.update(initD);
					} else failedLayerIdx = l.get_layer_idx();
				}
			});
			//doubling the code by intention, because some layers can be incompatible with pNewActivationStorage specification
			if (ErrorCode::Success == ec) {
				initD.clean();
				ec = get_self().last_layer().init(initD, pNewActivationStorage);
				if (ErrorCode::Success == ec) {
					lid.update(initD);
				} else failedLayerIdx = get_self().last_layer().get_layer_idx();
			}
			//#TODO need some way to return failedLayerIdx
			return ec;
		}

		void deinit() noexcept {
			get_self().for_each_packed_layer([](auto& l) {l.deinit(); });
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			get_self().for_each_packed_layer([=](auto& l) {l.initMem(ptr, cnt); });
		}

		//variation of fprop for normal layer
		template <typename LowerLayer>
		std::enable_if_t<!_impl::is_layer_wrapper<LowerLayer>::value> fprop(const LowerLayer& lowerLayer)noexcept
		{
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			//STDCOUTL("In nonspecial "<<get_layer_name_str());
			get_self().fprop(_impl::trainable_layer_wrapper<LowerLayer>(lowerLayer.get_activations()));
		}
		//variation of fprop for layerwrappers
		template <typename LowerLayerWrapper>
		std::enable_if_t<_impl::is_layer_wrapper<LowerLayerWrapper>::value> fprop(const LowerLayerWrapper& lowerLayer)noexcept
		{
			//STDCOUTL("In special " << get_layer_name_str());
			get_self().first_layer().fprop(lowerLayer);
			utils::for_eachwp_up(m_layers, [](auto& lcur, auto& lprev, const bool)noexcept {
				lcur.fprop(lprev);
			});
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			//STDCOUTL("bprop begin " << get_layer_name_str());

			NNTL_ASSERT(dLdA.size() == last_layer().get_activations().size_no_bias());
			NNTL_ASSERT( (std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			realmtxdefptr_array_t a_dLdA = { &dLdA, &dLdAPrev };
			unsigned mtxIdx = 0;

			utils::for_eachwn_downfullbp(m_layers, [&mtxIdx, &a_dLdA](auto& lcur, auto& lprev, const bool)noexcept {
				const unsigned nextMtxIdx = mtxIdx ^ 1;
				a_dLdA[nextMtxIdx]->deform_like_no_bias(lprev.get_activations());
				const unsigned bAlternate = lcur.bprop(*a_dLdA[mtxIdx], lprev, *a_dLdA[nextMtxIdx]);
				NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
				mtxIdx ^= bAlternate;
			});

			const unsigned nextMtxIdx = mtxIdx ^ 1;
			if (std::is_base_of<m_layer_input, LowerLayer>::value) {
				a_dLdA[nextMtxIdx]->deform(0, 0);
			}else a_dLdA[nextMtxIdx]->deform_like_no_bias(lowerLayer.get_activations());
			const unsigned bAlternate = get_self().first_layer().bprop(*a_dLdA[mtxIdx], lowerLayer, *a_dLdA[nextMtxIdx]);
			NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
			mtxIdx ^= bAlternate;

			//STDCOUTL("bprop end " << get_layer_name_str() << ", returns = " << mtxIdx);
			return mtxIdx;
		}

	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			get_self().for_each_packed_layer([&ar](auto& l) {
				constexpr size_t maxStrlen = 16;
				char lName[maxStrlen];
				l.get_layer_name(lName, maxStrlen);
				ar & serialization::make_named_struct(lName, l);
			});
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_vertical
	// If you need to derive a new class, derive it from _layer_pack_vertical (to make static polymorphism work)
	/*template <typename ...Layrs>
	class layer_pack_vertical final
		: public _layer_pack_vertical<layer_pack_vertical<Layrs...>, Layrs...>
	{
	public:
		~layer_pack_vertical() noexcept {};
		layer_pack_vertical(Layrs&... layrs) noexcept
			: _layer_pack_vertical<layer_pack_vertical<Layrs...>, Layrs...>(layrs...){};
	};

	template <typename ...Layrs> inline
		layer_pack_vertical <Layrs...> make_layer_pack_vertical(Layrs&... layrs) noexcept {
		return layer_pack_vertical<Layrs...>(layrs...);
	}*/

	template <typename ...Layrs>
	class LPV final
		: public _layer_pack_vertical<LPV<Layrs...>, Layrs...>
	{
	public:
		~LPV() noexcept {};
		LPV(Layrs&... layrs) noexcept
			: _layer_pack_vertical<LPV<Layrs...>, Layrs...>(layrs...) {};
	};

	template <typename ..._T>
	using layer_pack_vertical = typename LPV<_T...>;

	template <typename ...Layrs> inline
		LPV <Layrs...> make_layer_pack_vertical(Layrs&... layrs) noexcept {
		return LPV<Layrs...>(layrs...);
	}
}
