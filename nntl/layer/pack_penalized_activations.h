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

// Same as layer_penalized_activations, but gets underlying layer by reference
// 
// #todo: _i_loss_addendum interface should be extended to allow optimizations (caching) for a fullbatch learning with full error calculation

#include "_penalized_activations_base.h"

namespace nntl {

	//If possible prefer the _LPA<> over the _LPPA<>. However, in complicated
	// cases compiler's bugs may leave you no option but to use _LPPA<>.
	template<typename FinalPolymorphChild, class LayerT, typename LossAddsTuple>
	class _LPPA 
		: public _layer_base_forwarder<FinalPolymorphChild, typename LayerT::interfaces_t>
		, public _PA_base<LossAddsTuple>
	{
	private:
		typedef _layer_base_forwarder<FinalPolymorphChild, typename LayerT::interfaces_t> _base_class_t;

		static_assert(!is_layer_output<LayerT>::value, "What the reason to penalize output layer activations?");
		//#note: if we actually need to work with the output_layer, then there must be very special handling of this case because of how
		//bprop() is implemented for output_layer now

	public:
		using _base_class_t::real_t;
		using _base_class_t::realmtx_t;
		using _base_class_t::realmtxdef_t;
		typedef typename LayerT::_layer_init_data_t _layer_init_data_t;
		typedef typename LayerT::common_data_t common_data_t;

		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

	protected:
		LayerT& m_undLayer;

	private:
		layer_index_t m_layerIdx;

	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx && inc_neurons_cnt > 0);

			if (m_layerIdx) abort();
			m_layerIdx = ili.newIndex();
			_impl::_preinit_layers initializer(ili, inc_neurons_cnt);
			initializer(m_undLayer);
		}

	public:
		~_LPPA()noexcept {}
		_LPPA(const char* pCustomName, LayerT& ul)noexcept
			: _base_class_t(pCustomName), m_undLayer(ul), m_layerIdx(0)
		{}

		static constexpr const char _defName[] = "lppa";

		//////////////////////////////////////////////////////////////////////////
		const layer_index_t& get_layer_idx() const noexcept { return m_layerIdx; }
		//used by _layer_base_forwarder<> functions to forward various data from the topmost_layer()
		auto& _forwarder_layer()const noexcept { return m_undLayer; }
		//////////////////////////////////////////////////////////////////////////
		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			call_F_for_each_layer(::std::forward<_Func>(f), m_undLayer);
		}
		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			call_F_for_each_layer_down(::std::forward<_Func>(f), m_undLayer);
		}
		template<typename _Func> void for_each_packed_layer(_Func&& f)const noexcept {
			::std::forward<_Func>(f)(m_undLayer);
		}
		template<typename _Func> void for_each_packed_layer_down(_Func&& f)const noexcept { 
			::std::forward<_Func>(f)(m_undLayer);
		}


		//////////////////////////////////////////////////////////////////////////
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		const bool hasLossAddendum()const noexcept {
			const bool b = _pab_hasLossAddendum();
			return b ? b : m_undLayer.hasLossAddendum();
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept {
			return _pab_lossAddendum(get_self().get_activations(), get_self().get_iMath())
				+ (m_undLayer.hasLossAddendum() ? m_undLayer.lossAddendum() : real_t(0));
		}

		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			_base_class_t::init();
			const auto r = m_undLayer.init(lid, pNewActivationStorage);
			lid.bLossAddendumDependsOnActivations = true;
			return r;
		}
		void deinit() noexcept {
			m_undLayer.deinit();
			_base_class_t::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept { m_undLayer.initMem(ptr, cnt); }

		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			m_undLayer.on_batch_size_change(pNewActivationStorage);
		}

		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().get_common_data().is_training_mode());
			
			m_undLayer.fprop(lowerLayer);
			iI.fprop_activations(get_self().get_activations());
			iI.fprop_end(get_self().get_activations());
		}


		template <typename LowerLayerT>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayerT& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayerT>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			_pab_update_dLdA(dLdA, get_self().get_activations(), get_self().get_iMath(), iI);

			iI.bprop_finaldLdA(dLdA);

			const auto ret = m_undLayer.bprop(dLdA, lowerLayer, dLdAPrev);

			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}

	private:
		//support for ::boost::serialization
		// 		#TODO
		friend class ::boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			//ar & serialization::make_named_struct(m_undLayer.get_layer_name_str().c_str(), m_undLayer);
		}


		//#TODO we should adopt last_layer().drop_activations_is_trivial() function signature here, but it look like non-trivial
		//to detect if the constexpr attribute was used.
		//If the m_tiledLayer.drop_activations_is_trivial() then ours drop_activations() is trivial too
		const bool is_trivial_drop_samples() const noexcept { return m_undLayer.is_trivial_drop_samples(); }

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			m_undLayer.drop_samples(mask, bBiasesToo);
		}
	};

	template<class LayerT, typename ...LossAddsTs>
	class LPPA final : public _LPPA<LPPA<LayerT, LossAddsTs...>, LayerT, ::std::tuple<LossAddsTs...>> {
	public:
		~LPPA() noexcept {};
		LPPA(const char* pCustomName, LayerT& ul) noexcept
			: _LPPA<LPPA<LayerT, LossAddsTs...>, LayerT, ::std::tuple<LossAddsTs...>>(pCustomName, ul) {};
	};

	template <class LayerT, typename ..._T>
	using layer_pack_penalized_activations = typename LPPA<LayerT, _T...>;
}