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

// layer_identity defines a layer that just passes it's incoming neurons as activation neurons without any processing.
// Can be used to pass a source data unmodified to upper feature detectors.
// layer_identity_gate can also serve as a gating source for layer_pack_gated.
// It's intended to be used within a layer_pack_horizontal (and can't/shouldn't be used elsewhere)

#include <type_traits>

#include "_layer_base.h"

namespace nntl {

	template<typename Interfaces, typename FinalPolymorphChild>
	class _layer_identity : public _layer_base<Interfaces, FinalPolymorphChild>, public m_layer_autoneurons_cnt
	{
	private:
		typedef _layer_base<Interfaces, FinalPolymorphChild> _base_class;

		//////////////////////////////////////////////////////////////////////////
		//members section (in "biggest first" order)
	protected:
		realmtxdef_t m_activations;

	public:
		~_layer_identity()noexcept {}
		_layer_identity(const char* pCustomName = nullptr)noexcept : _base_class(0, pCustomName) {
			m_activations.will_emulate_biases();
		}

		static constexpr const char _defName[] = "id";

		//////////////////////////////////////////////////////////////////////////
		const realmtxdef_t& get_activations()const noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			return m_activations;
		}
		const mtx_size_t get_activations_size()const noexcept { return m_activations.size(); }

		// pNewActivationStorage MUST be specified (we're expecting to be encapsulated into a layer_pack_horizontal)
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage)noexcept {
			NNTL_ASSERT(pNewActivationStorage);

			auto ec = _base_class::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			NNTL_ASSERT(get_self().get_neurons_cnt());
			m_activations.useExternalStorage(pNewActivationStorage
				, get_self().get_max_fprop_batch_size(), get_self().get_neurons_cnt() + 1, true);
			return ec;
		}

		void deinit() noexcept {
			m_activations.clear();
			_base_class::deinit();
		}
		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {}

		// pNewActivationStorage MUST be specified (we're expecting to be encapsulated into layer_pack_horizontal)
		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage)noexcept {
			NNTL_ASSERT(pNewActivationStorage);

			m_bActivationsValid = false;
			m_bTraining = batchSize == 0;

			NNTL_ASSERT(m_activations.emulatesBiases() && m_activations.bDontManageStorage() && get_self().get_neurons_cnt());
			//m_neurons_cnt + 1 for biases
			m_activations.useExternalStorage(pNewActivationStorage
				, m_bTraining ? get_self().get_training_batch_size() : batchSize, get_self().get_neurons_cnt() + 1, true);
			//should not restore biases here, because for compound layers its a job for their fprop() implementation
		}
	protected:

		void _fprop(const realmtx_t& prevActivations)noexcept {
			NNTL_ASSERT(prevActivations.size() == m_activations.size());
			NNTL_ASSERT(m_activations.bDontManageStorage());

			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), prevActivations, m_bTraining);

			// just copying the data from prevActivations to m_activations
			// We must copy the data, because layer_pack_horizontal uses its own storage for activations, therefore
			// we can't just use the m_activations as an alias to prevActivations - we have to physically copy the data
			// to a new storage within layer_pack_horizontal activations
			const bool r = prevActivations.cloneTo(m_activations);
			NNTL_ASSERT(r);

			iI.fprop_end(m_activations);
		}

	public:
		//we're restricting the use of layer_identity to layer_pack_horizontal only
		template <typename LowerLayerWrapper>
		std::enable_if_t<_impl::is_layer_wrapper<LowerLayerWrapper>::value> fprop(const LowerLayerWrapper& lowerLayer)noexcept
		{
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayerWrapper>::value, "Template parameter LowerLayerWrapper must implement _i_layer_fprop");
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			get_self()._fprop(lowerLayer.get_activations());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			m_bActivationsValid = true;
		}

		template <typename LowerLayerWrapper>
		std::enable_if_t<_impl::is_layer_wrapper<LowerLayerWrapper>::value, const unsigned>
			bprop(realmtx_t& dLdA, const LowerLayerWrapper& lowerLayer, realmtx_t& dLdAPrev)noexcept
		{
			m_bActivationsValid = false;
			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			//nothing to do here
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.bprop_end(dLdA);
			return 0;//indicating that dL/dA for previous layer is actually in dLdA parameter (not in dLdAPrev)
		}


		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(inc_neurons_cnt > 0);
			NNTL_ASSERT(get_self().get_neurons_cnt() == inc_neurons_cnt);

			_base_class::_preinit_layer(ili, inc_neurons_cnt);
			//get_self()._set_neurons_cnt(inc_neurons_cnt);
		}

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template<typename Interfaces, typename FinalPolymorphChild>
	class _layer_identity_gate : public _layer_identity<Interfaces, FinalPolymorphChild>
		, public _i_layer_gate<typename Interfaces::iMath_t::real_t>
	{
	private:
		typedef _layer_identity<Interfaces, FinalPolymorphChild> _base_class;

	public:
		using _base_class::real_t;

		//////////////////////////////////////////////////////////////////////////
		//members section (in "biggest first" order)
	protected:
		realmtxdef_t m_gate;

	public:
		~_layer_identity_gate()noexcept {}
		_layer_identity_gate(const char* pCustomName = nullptr)noexcept : _base_class(pCustomName) {
			m_gate.dont_emulate_biases();
		}
		static constexpr const char _defName[] = "idg";

		//////////////////////////////////////////////////////////////////////////
		const realmtx_t& get_gate()const noexcept { return m_gate; }
		const vec_len_t get_gate_width()const noexcept { return get_self().get_neurons_cnt(); }

		// pNewActivationStorage MUST be specified (we're expecting to be encapsulated into a layer_pack_horizontal)
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage)noexcept {
			NNTL_ASSERT(pNewActivationStorage);

			auto ec = _base_class::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			NNTL_ASSERT(get_self().get_neurons_cnt());
			m_gate.useExternalStorage(pNewActivationStorage
				, get_self().get_max_fprop_batch_size(), get_self().get_neurons_cnt(), false);
			return ec;
		}

		void deinit() noexcept {
			m_gate.clear();
			_base_class::deinit();
		}
		// pNewActivationStorage MUST be specified (we're expecting to be encapsulated into layer_pack_horizontal)
		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage)noexcept {
			NNTL_ASSERT(pNewActivationStorage);
			_base_class::set_mode(batchSize, pNewActivationStorage);

			NNTL_ASSERT(m_bTraining == (batchSize == 0));
			m_gate.useExternalStorage(pNewActivationStorage
				, m_bTraining ? get_self().get_training_batch_size() : batchSize, get_self().get_neurons_cnt(), false);
		}

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			//#todo must correctly call base class serialize()
			//NNTL_ASSERT(!"must correctly call base class serialize()");
			ar & boost::serialization::base_object<_base_class>(*this);
			//ar & serialization::serialize_base_class<_base_class>(*this);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//
	// 
	template <typename Interfaces = d_interfaces>
	class layer_identity final : public _layer_identity<Interfaces, layer_identity<Interfaces>>
	{
	public:
		~layer_identity() noexcept {};
		layer_identity(const char* pCustomName = nullptr) noexcept 
			: _layer_identity<Interfaces, layer_identity<Interfaces>> (pCustomName) {};
	};

	template <typename Interfaces = d_interfaces>
	class layer_identity_gate final : public _layer_identity_gate<Interfaces, layer_identity_gate<Interfaces>>
	{
	public:
		~layer_identity_gate() noexcept {};
		layer_identity_gate(const char* pCustomName = nullptr) noexcept 
			: _layer_identity_gate<Interfaces, layer_identity_gate<Interfaces>>(pCustomName) {};
	};

}
