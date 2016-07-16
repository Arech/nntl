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

// layer_identity defines a layer that just passes it's incoming data as activation units.
// It's intended to be used within layer_pack_horizontal (and can't be used elsewhere)

#include <type_traits>

#include "_layer_base.h"

namespace nntl {

	template<typename Interfaces, typename FinalPolymorphChild>
	class _layer_identity : public _i_layer<typename Interfaces::iMath_t::real_t> {
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;

		typedef typename Interfaces::iMath_t iMath_t;
		static_assert(std::is_base_of<math::_i_math<real_t>, iMath_t>::value, "Interfaces::iMath type should be derived from _i_math");

		typedef typename Interfaces::iRng_t iRng_t;
		static_assert(std::is_base_of<rng::_i_rng, iRng_t>::value, "Interfaces::iRng type should be derived from _i_rng");

		typedef _impl::_layer_init_data<iMath_t, iRng_t> _layer_init_data_t;

		//////////////////////////////////////////////////////////////////////////
		//members section (in "biggest first" order)
	private:
		realmtxdef_t m_activations;

		vec_len_t m_max_fprop_batch_size, m_training_batch_size;

		neurons_count_t m_neurons_cnt;
		neurons_count_t m_incoming_neurons_cnt;
		layer_index_t m_layerIdx;

	public:
		~_layer_identity()noexcept {}
		_layer_identity()noexcept : m_neurons_cnt(0), m_incoming_neurons_cnt(0), m_layerIdx(0) {}

		self_ref_t get_self() noexcept {
			static_assert(std::is_base_of<_layer_base, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_base<FinalPolymorphChild>");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<_layer_base, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_base<FinalPolymorphChild>");
			return static_cast<self_cref_t>(*this);
		}

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { 
			NNTL_ASSERT(m_neurons_cnt);
			return m_neurons_cnt;
		}
		const neurons_count_t get_incoming_neurons_cnt()const noexcept {
			NNTL_ASSERT(m_incoming_neurons_cnt);
			return m_incoming_neurons_cnt;
		}

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, "id%d", static_cast<unsigned>(get_layer_idx()));
		}
		std::string get_layer_name_str()const noexcept {
			constexpr size_t ml = 16;
			char n[ml];
			get_self().get_layer_name(n, ml);
			return std::string(n);
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		constexpr const real_t lossAddendum()const noexcept { return real_t(0.0); }

		//////////////////////////////////////////////////////////////////////////
		const realmtx_t& get_activations()const noexcept { return m_activations; }

		// pNewActivationStorage MUST be specified (we're expecting to be incapsulated into layer_pack_horizontal)
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(pNewActivationStorage);

			NNTL_ASSERT(m_neurons_cnt);

			m_max_fprop_batch_size = lid.max_fprop_batch_size;
			m_training_batch_size = lid.training_batch_size;

			m_activations.useExternalStorage(pNewActivationStorage, m_max_fprop_batch_size, m_neurons_cnt + 1, true);

			return ErrorCode::Success;
		}

		void deinit() noexcept {
			m_activations.clear();
		}
		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {}

		// pNewActivationStorage MUST be specified (we're expecting to be incapsulated into layer_pack_horizontal)
		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(pNewActivationStorage);

			const bool bTraining = batchSize == 0;

			NNTL_ASSERT(m_activations.emulatesBiases() && m_activations.bDontManageStorage() && m_neurons_cnt);
			//m_neurons_cnt + 1 for biases
			m_activations.useExternalStorage(pNewActivationStorage, bTraining ? m_training_batch_size : batchSize, m_neurons_cnt + 1, true);
			//should not restore biases here, because for compound layers its a job for their fprop() implementation
		}

		void _fprop(const realmtx_t& prevActivations)noexcept {
			NNTL_ASSERT(prevActivations.size() == m_activations.size());
			NNTL_ASSERT(m_activations.bDontManageStorage());
			//just copying the data from prevActivations to m_activations
			const bool r = prevActivations.cloneTo(m_activations);
			NNTL_ASSERT(r);
		}

		//we're restricting the use of layer_identity to layer_pack_horizontal only
		template <typename LowerLayerWrapper>
		std::enable_if_t<_impl::is_layer_wrapper<LowerLayerWrapper>::value> fprop(const LowerLayerWrapper& lowerLayer)noexcept{
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayerWrapper>::value, "Template parameter LowerLayerWrapper must implement _i_layer_fprop");
			get_self()._fprop(lowerLayer.get_activations());
		}

		template <typename LowerLayerWrapper>
		std::enable_if_t<_impl::is_layer_wrapper<LowerLayerWrapper>::value, const unsigned>
			bprop(realmtx_t& dLdA, const LowerLayerWrapper& lowerLayer, realmtx_t& dLdAPrev)noexcept
		{
			//nothing to do here
			return 0;//indicating that dL/dA for previous layer is actually in dLdA parameter (not in dLdAPrev)
		}


		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(layer_index_t& idx, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(!m_layerIdx && !m_incoming_neurons_cnt && !m_neurons_cnt);
			NNTL_ASSERT(idx > 0 && inc_neurons_cnt > 0);

			if (m_layerIdx || m_incoming_neurons_cnt || m_neurons_cnt || !idx || !inc_neurons_cnt) abort();

			m_neurons_cnt = inc_neurons_cnt;
			m_incoming_neurons_cnt = inc_neurons_cnt;
			m_layerIdx = idx;
		}

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			//nothing to do here at this moment
		}

	};

	template <typename Interfaces = nnet_def_interfaces>
	class layer_identity final : public _layer_identity<Interfaces, layer_identity<Interfaces>>
	{
	public:
		~layer_identity() noexcept {};
		layer_identity() noexcept : _layer_identity<Interfaces, layer_identity<Interfaces>> () {};
	};

}
