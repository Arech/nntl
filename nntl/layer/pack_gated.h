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

// NOT FINISHED YET!

// layer_pack_gated is a layer wrapper, that uses underlying layer activation values as it's own activation values
// when a value of special gating neuron is 1, or uses zeros as activation values if the value of gating neuron is 0.
//											 | | | | | | | |
//			|--------------layer_pack_gated-----------------------|
//			|				(0 or 1)		 | | | | | | | |	  |
//			|		   gate----------->x     \ | | | | | | /      |
//			|			|		          |--underlying_layer--|  |
//			|			/					  /  |  |  |  \		  |
//		--->|>----------					  |  |  |  |  |       |
//	   /	|								  |  |  |  |  |       |
//	   |	-------------------------------------------------------
//	   |									  |  |  |  |  | 
//
// layer_pack_gated has the same number of neurons and incoming neurons, as underlying_layer.
// Gating neuron is implemented as a (any) layer with a single neuron and is passed by reference
// during layer_pack_gated construction. The only requirement is that gating neuron (layer) must be processed by NN earlier,
// than corresponding layer_pack_gated (i.e. is must receive a smaller layer_index during preinit() phase)
// Usually layer_pack_gated is used in conjunction with layer_identity (to pass gating neuron value from layer_input)
// and layer_pack_horizontal to assemble NN architecture.
// 

#include "_pack_.h"
#include "../utils.h"

namespace nntl {

	template<typename FinalPolymorphChild, typename UnderlyingLayer>
	class _layer_pack_gated : public _i_layer<typename UnderlyingLayer::real_t> {
	public:
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;

		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

		typedef UnderlyingLayer underlying_layer_t;

		typedef typename underlying_layer_t::iMath_t iMath_t;
		typedef typename underlying_layer_t::iRng_t iRng_t;
		typedef typename underlying_layer_t::_layer_init_data_t _layer_init_data_t;

	private:
		layer_index_t m_layerIdx;

	protected:
		underlying_layer_t& m_undLayer;

		//realmtxdef_t m_gatingMask;

		//////////////////////////////////////////////////////////////////////////
		//
	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(layer_index_t& idx, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx && idx > 0 && inc_neurons_cnt > 0);

			if (m_layerIdx) abort();
			m_layerIdx = idx;

			_impl::_preinit_layers initializer(idx + 1, inc_neurons_cnt);
			initializer(m_undLayer);
			idx = initializer._idx;
		}

	public:
		~_layer_pack_gated()noexcept {}
		_layer_pack_gated(UnderlyingLayer& ulayer)noexcept : m_undLayer(ulayer), m_layerIdx(0) {}

		self_ref_t get_self() noexcept {
			static_assert(std::is_base_of<_layer_pack_gated, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_pack_gated");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<_layer_pack_gated, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_pack_gated");
			return static_cast<self_cref_t>(*this);
		}

		underlying_layer_t& underlying_layer()const noexcept { return m_undLayer; }

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { return get_self().underlying_layer().get_neurons_cnt(); }
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { return  get_self().underlying_layer().get_incoming_neurons_cnt(); }

		const realmtx_t& get_activations()const noexcept { return get_self().underlying_layer().get_activations(); }

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, "lpg%d", static_cast<unsigned>(get_self().get_layer_idx()));
		}
		std::string get_layer_name_str()const noexcept {
			constexpr size_t ml = 16;
			char n[ml];
			get_self().get_layer_name(n, ml);
			return std::string(n);
		}

		//////////////////////////////////////////////////////////////////////////
		// btw, in most cases we just pass function request to underlying layer. There are only two exceptions:
		// fprop (where we call original fprop() first and then turn off activations forbidden by the gate) and
		// bprop (where we clean dLdA by the gating mask, and then pass that dLdA to original bprop())

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept {
			return get_self().underlying_layer().hasLossAddendum();
		}
		//returns a loss function summand, that's caused by this layer
		real_t lossAddendum()const noexcept {
			return get_self().underlying_layer().lossAddendum();
		}

		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept {
			get_self().underlying_layer().set_mode(batchSize, pNewActivationStorage);
		}

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			return get_self().underlying_layer().init(lid, pNewActivationStorage);
		}

		void deinit() noexcept {
			get_self().underlying_layer().deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			get_self().underlying_layer().initMem(ptr, cnt);
		}

		//////////////////////////////////////////////////////////////////////////
		// gating functions
	protected:
		//construct a mask of ones and zeros based on gating neuron. The mask has to have a size of activations units of
		//underlying layer. It has ones for a rows of activations that are allowed by gating neuron and zeros for forbidden
		// rows. The mask is applied to activations and dLdA with a simple inplace elementwise multiplication.
		void make_gating_mask()noexcept {
		}

		//applies gating mask to a matrix. The matrix has to have a size of gating mask.
		void apply_gating_mask(realmtx_t& A)const noexcept {

		}


	public:
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
			first_layer().fprop(lowerLayer);
			utils::for_eachwp_up(m_layers, [](auto& lcur, auto& lprev, const bool)noexcept {
				lcur.fprop(lprev);
			});
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			//STDCOUTL("bprop begin " << get_layer_name_str());

			NNTL_ASSERT(dLdA.size() == last_layer().get_activations().size_no_bias());
			NNTL_ASSERT((std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

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
			} else a_dLdA[nextMtxIdx]->deform_like_no_bias(lowerLayer.get_activations());
			const unsigned bAlternate = first_layer().bprop(*a_dLdA[mtxIdx], lowerLayer, *a_dLdA[nextMtxIdx]);
			NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
			mtxIdx ^= bAlternate;

			//STDCOUTL("bprop end " << get_layer_name_str() << ", returns = " << mtxIdx);
			return mtxIdx;
		}

	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			for_each_packed_layer([&ar](auto& l) {
				constexpr size_t maxStrlen = 16;
				char lName[maxStrlen];
				l.get_layer_name(lName, maxStrlen);
				ar & serialization::make_named_struct(lName, l);
			});
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_fully_connected
	// If you need to derive a new class, derive it from _layer_fully_connected (to make static polymorphism work)

	/*
	template <typename ...Layrs>
	class LPV final
		: public _layer_pack_gated<LPV<Layrs...>, Layrs...>
	{
	public:
		~LPV() noexcept {};
		LPV(Layrs&... layrs) noexcept
			: _layer_pack_gated<LPV<Layrs...>, Layrs...>(layrs...) {};
	};

	template <typename ..._T>
	using layer_pack_gated = typename LPV<_T...>;

	template <typename ...Layrs> inline
		LPV <Layrs...> make_layer_pack_gated(Layrs&... layrs) noexcept {
		return LPV<Layrs...>(layrs...);
	}
	*/
}
