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

// layer_pack_horizontal (LPH) provides a way to concatenate activation matrices of a set of layers into a single
// activation matrix, i.e. gather a set of layers into a single layer.
// Moreover, LPH allows to feed different ranges of
// underlying activation units into different set of layers.
// 
//    \  |  |  |  |     |  |  |  | /
// |------layer_pack_horizontal-------|
// |  \  |  |  |  |  .  |  |  |  | /  |
// |   |--layer1--|  .  |--layerN--|  | 
// |    / | | | | |  .  | | | | | \   |
// |----------------------------------|
//      / | | | | |  .  | | | | | \
//
// 
#include "_lph_base.h"

namespace nntl {
	
	template<typename FinalPolymorphChild, typename PHLsTuple>
	class _LPH : public _impl::_LPH_base<FinalPolymorphChild, PHLsTuple> {
		typedef _impl::_LPH_base<FinalPolymorphChild, PHLsTuple> _base_class_t;

	public:
		~_LPH()noexcept {}
		_LPH(const char* pCustomName, const PHLsTuple& phls)noexcept : _base_class_t(pCustomName, phls)	{}
		_LPH(const char* pCustomName, PHLsTuple&& phls)noexcept : _base_class_t(pCustomName, ::std::move(phls))	{}

		//the next c-tor is deprecated. Don't use it if possible
		template<typename... PHLsT>
		_LPH(const char* pCustomName, PHLsT&&... phls) noexcept
			: _base_class_t(pCustomName, ::std::make_tuple<::std::remove_reference_t<PHLsT>...>(::std::forward<PHLsT>(phls)...))
		{}

		static constexpr const char _defName[] = "lph";

	protected:
		template<typename LLWrapT>
		void _lph_fprop(const realmtx_t& prevAct)noexcept {
			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(m_activations.bBatchInColumn() && prevAct.bBatchInColumn());

			auto& iI = get_iInspect();
			iI.fprop_begin(get_layer_idx(), prevAct, get_common_data().is_training_mode());

			tuple_utils::for_each_up(m_phl_tuple, [&prevAct, pTBS = m_pTmpBiasStorage](const auto& phl) {
				phl.l.fprop(LLWrapT(prevAct, pTBS, phl.coord));
			});

			NNTL_ASSERT(prevAct.test_biases_strict());			
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());

			iI.fprop_activations(m_activations);
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		// in order to implement backprop for the inner layers, we must provide them with a correct dLdA and dLdAPrev, each of which must
		// address at least _layer_init_data_t::max_dLdA_numel elements, that layers returned during init() phase.
		// Some things to consider:
		// - there might be a compound layer in m_phl_tuple (such as layer_pack_vertical). That means, that it may require a far bigger
		// max_dLdA_numel, than might be provided by slicing dLdA passed to this function as argument to corresponding parts. So we'll
		// need a way to protect out-of-slice data from being overwritten by layer.bprop() (because we allow layer.brop() to use dLdA&dLdAPrev
		// almost without restrictions)
		// - inner layers may use the same (intersecting) lowerLayer activation units (because we don't require inner layers to use different
		// lower layer activations). That means, that after we'll get their individual dLdAPrev, we must aggregate them into resulting dLdAPrev.
		// 
		// Therefore it looks much more safe to allocate and use for inner layers special dLdA and dLdAPrev matrices, that are independent
		// from dLdA&dLdAPrev, passed to this function as argument. It's possible however to reuse passed dLdA&dLdAPrev for that task, but
		// it would require significantly more complicated and error-prone code to keep all data safe
		template<typename LLWrapT>
		unsigned _lph_bprop(realmtxdef_t& dLdA, realmtxdef_t& dLdAPrev, const realmtx_t& prevAct)noexcept {
			static constexpr bool bPrevLayerWBprop = is_layer_with_bprop<LLWrapT>::value;
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(dLdA.bBatchInColumn() && dLdAPrev.bBatchInColumn() && prevAct.bBatchInColumn() && m_activations.bBatchInColumn());
			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT(get_common_data().is_training_mode());
			NNTL_ASSERT(dLdA.size() == m_activations.size_no_bias());
			NNTL_ASSERT(get_incoming_neurons_cnt() == prevAct.sample_size());
			NNTL_ASSERT(!bPrevLayerWBprop || dLdAPrev.size() == prevAct.size_no_bias());

			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

			auto& iI = get_iInspect();
			iI.bprop_begin(get_layer_idx(), dLdA);
			iI.bprop_finaldLdA(dLdA);

			// We'll copy corresponding parts of dLdA into m_innerdLdA and on inner layer.bprop() return we'll ADD corresponding dLdA to dLdAPrev passed
			if (bPrevLayerWBprop) dLdAPrev.zeros();

			neurons_count_t firstNeuronOfs = get_neurons_cnt();

			//The order of traversing is EXTREMELY IMPORTANT for gating layers, for example (they might expect a gating layer to be
			// processed first during fprop() and last during bprop()). Therefore we must go backwards here!
			tuple_utils::for_each_down(m_phl_tuple, [&firstNeuronOfs, &prevAct, &dLdA, &dLdAPrev
				, &_innerdLdA = m_innerdLdA, &_innerdLdAPrev = m_innerdLdAPrev, _pTmpBiasStorage = m_pTmpBiasStorage
				, &_Math = get_iMath()](const auto& phl)
			{
				static constexpr bool bPrevLayerWBprop = is_layer_with_bprop<LLWrapT>::value;
				auto& lyr = phl.l;

				NNTL_ASSERT(firstNeuronOfs >= lyr.get_neurons_cnt());
				firstNeuronOfs -= lyr.get_neurons_cnt();

				//setting up the _innerdLdA
				_innerdLdA.deform_like_no_bias(lyr.get_activations());
				NNTL_ASSERT(firstNeuronOfs + _innerdLdA.cols() <= dLdA.cols());
				NNTL_ASSERT(_innerdLdA.rows() == dLdA.rows());
				::std::memcpy(_innerdLdA.data(), dLdA.colDataAsVec(firstNeuronOfs), _innerdLdA.byte_size());

				//#consider если для каждого внутреннего слоя помнить max_dLdA_numel, и она окажется меньше _innerdLdA.numel() и
				//_innerdLdAPrev.numel, то можно избежать копирования dLdA в _innerdLdA передавая данные напрямую, адресуя
				// их внутри dLdA - в этом случае соседние данные dLdA других слоёв останутся в безопасности.
				// Однако, в большинстве случаев условие будет не выполняться (т.к. внутренние слои представляют собой обычно
				// жирные фиче-детекторы с меньшем числом выходов, чем внутренних нейронов - у них dLdA для внутренних слоёв
				// больше располагаемой тут dLdA внешнего слоя)

				//setting up the _innerdLdAPrev
				if (bPrevLayerWBprop) {
					_innerdLdAPrev.deform(dLdAPrev.rows(), phl.coord.m_count);
				} else _innerdLdAPrev.deform(0, 0);

				const auto switchMtxs = lyr.bprop(_innerdLdA, LLWrapT(prevAct, _pTmpBiasStorage, phl.coord), _innerdLdAPrev);

				if (bPrevLayerWBprop) {
					const auto& curdLdAPrev = switchMtxs ? _innerdLdAPrev : _innerdLdA;
					NNTL_ASSERT(curdLdAPrev.size() == realmtx_t::mtx_size_t(dLdAPrev.rows(), phl.coord.m_count));

					//saving curdLdAPrev to dLdAPrev
					_Math.vAdd_ip(dLdAPrev.colDataAsVec(phl.coord.m_offset)
						, curdLdAPrev.data(), curdLdAPrev.numel());
				}
			});
			NNTL_ASSERT(firstNeuronOfs == 0);
			NNTL_ASSERT(prevAct.test_biases_strict());

			iI.bprop_end(dLdAPrev);
			return 1;
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop<real_t>, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			get_self()._lph_fprop<_impl::wrap_part_trainable_layer<LowerLayer>>(lowerLayer.get_activations());
		}

		template <typename LowerLayer>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			static_assert(!bAssumeFPropOnly, "");
			//NNTL_ASSERT(get_self().bDoBProp());
			return get_self()._lph_bprop<_impl::wrap_part_trainable_layer<LowerLayer>>(dLdA, dLdAPrev, lowerLayer.get_activations());
		}

	};

	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _LPH
	// If you need to derive a new class, derive it from _LPH (to make static polymorphism work)

	//to shorten class name to get rid of C4503
	template <typename ...PHLsT>
	class LPH final : public _LPH < LPH<PHLsT...>, ::std::tuple<PHLsT...>>
	{
	public:
		~LPH() noexcept {};
		LPH(const PHLsT&... phls) noexcept
			: _LPH<LPH<PHLsT...>, ::std::tuple<PHLsT...>>(nullptr, ::std::make_tuple(phls...)) {};

		LPH(PHLsT&&... phls) noexcept
			: _LPH<LPH<PHLsT...>, ::std::tuple<PHLsT...>>(nullptr, ::std::make_tuple(::std::move(phls)...)) {};

		LPH(const char* pCustomName, const PHLsT&... phls) noexcept
			: _LPH<LPH<PHLsT...>, ::std::tuple<PHLsT...>>(pCustomName, ::std::make_tuple(phls...)) {};

		LPH(const char* pCustomName, PHLsT&&... phls) noexcept
			: _LPH<LPH<PHLsT...>, ::std::tuple<PHLsT...>>(pCustomName, ::std::make_tuple( ::std::move(phls)...)) {};
	};

	//////////////////////////////////////////////////////////////////////////
	template <typename PHLsTuple>
	class LPHt final : public _LPH < LPHt<PHLsTuple>, PHLsTuple> {
	public:
		~LPHt() noexcept {};
		LPHt(const PHLsTuple& phls) noexcept : _LPH<LPHt<PHLsTuple>, PHLsTuple>(nullptr, phls) {};

		LPHt(PHLsTuple&& phls) noexcept : _LPH<LPHt<PHLsTuple>, PHLsTuple>(nullptr, ::std::move(phls)) {};

		LPHt(const char* pCustomName, const PHLsTuple& phls) noexcept
			: _LPH<LPHt<PHLsTuple>, PHLsTuple>(pCustomName, phls) {};

		LPHt(const char* pCustomName, PHLsTuple&& phls) noexcept
			: _LPH<LPHt<PHLsTuple>, PHLsTuple>(pCustomName, ::std::move(phls)) {};
	};

	template <typename ..._T>
	using layer_pack_horizontal = typename LPHt<::std::tuple<_T...>>;

	template <typename ...PHLsT> inline constexpr
	auto make_layer_pack_horizontal(PHLsT&&... phls) noexcept {
		return LPHt<::std::tuple<::std::remove_reference_t<PHLsT>...>>(::std::make_tuple(::std::forward<PHLsT>(phls)...));
	}
	template <typename ...PHLsT> inline constexpr
	auto make_layer_pack_horizontal(const char* pCustomName, PHLsT&&... phls) noexcept {
		return  LPHt<::std::tuple<::std::remove_reference_t<PHLsT>...>>(pCustomName, ::std::make_tuple(::std::forward<PHLsT>(phls)...));
	}

}
