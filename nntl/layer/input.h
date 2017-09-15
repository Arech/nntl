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

#include "_layer_base.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	// class to derive from when making final input layer. Need it to propagate correct FinalPolymorphChild to
	// static polymorphism implementation here and in layer__base
	template<typename FinalPolymorphChild, typename Interfaces>
	class _layer_input 
		: public m_layer_input
		, public _layer_base<FinalPolymorphChild, Interfaces>
	{
	private:
		typedef _layer_base<FinalPolymorphChild, Interfaces> _base_class;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		const realmtx_t* m_pActivations;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		::std::enable_if_t<Archive::is_saving::value> serialize(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			if (m_pActivations && utils::binary_option<true>(ar, serialization::serialize_data_x)) 
				ar & serialization::make_nvp("data_x", * const_cast<realmtx_t*>(m_pActivations));
		}
		template<class Archive>
		::std::enable_if_t<Archive::is_loading::value> serialize(Archive & , const unsigned int ) {
		}

	public:

		_layer_input(const char* pCustomName, const neurons_count_t _neurons_cnt)noexcept 
			: _base_class(_neurons_cnt, pCustomName), m_pActivations(nullptr)
		{};
		~_layer_input() noexcept {};
		static constexpr const char _defName[] = "inp";

		const realmtx_t& get_activations()const noexcept {
			NNTL_ASSERT(m_pActivations);
			NNTL_ASSERT(m_bActivationsValid);
			return *m_pActivations;
		}
		const realmtx_t* get_activations_storage()const noexcept { return m_pActivations; }
		const mtx_size_t get_activations_size()const noexcept { 
			NNTL_ASSERT(m_pActivations);
			return m_pActivations->size();
		}
		const bool is_activations_shared()const noexcept {
			const auto r = _base_class::is_activations_shared();
			NNTL_ASSERT(!r || m_activations.bDontManageStorage());//shared activations can't manage their own storage
			return r;
		}

		ErrorCode init(_layer_init_data_t& lid)noexcept {
			auto ec = _base_class::init(lid);
			if (ErrorCode::Success != ec) return ec;

			m_pActivations = nullptr;
			return ec;
		}
		void deinit()noexcept {
			m_pActivations = nullptr;
			_base_class::deinit();
		}


		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept { NNTL_UNREF(ptr); NNTL_UNREF(cnt); }
		void on_batch_size_change()noexcept {
			m_bActivationsValid = false;
		}

		void fprop(const realmtx_t& data_x)noexcept {
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(),data_x, get_self().get_common_data().is_training_mode());

			NNTL_ASSERT(data_x.test_biases_ok());
			m_pActivations = &data_x;

			iI.fprop_activations(*m_pActivations);
			iI.fprop_end(*m_pActivations);
			m_bActivationsValid = true;
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtx_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;
			//auto& iI = get_self().get_iInspect();
			//iI.bprop_begin(get_self().get_layer_idx(), dLdA);
			// iI.bprop_finaldLdA(dLdA);

			//iI.bprop_end(dLdAPrev);
			return 1;
		}

		static constexpr bool is_trivial_drop_samples()noexcept {
			static_assert(false, "layer_input doesn't support the drop_samples()");
			return false;
		}

		static constexpr void left_after_drop_samples(const numel_cnt_t nNZElems)noexcept {
			NNTL_UNREF(nNZElems);
		}

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo, const numel_cnt_t nNZElems)noexcept {
			NNTL_UNREF(mask); NNTL_UNREF(bBiasesToo); NNTL_UNREF(nNZElems);
			static_assert(false, "layer_input doesn't support the drop_samples()");
		}
		
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		constexpr const bool hasLossAddendum()const noexcept { return false; }

	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 == inc_neurons_cnt);
			_base_class::_preinit_layer(ili, inc_neurons_cnt);
			NNTL_ASSERT(0 == get_self().get_layer_idx());
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_input
	// If you need to derive a new class, derive it from _layer_input (to make static polymorphism work)
	template < typename Interfaces = d_interfaces>
	class layer_input final : public _layer_input<layer_input<Interfaces>, Interfaces> {
	public:
		~layer_input() noexcept {};
		layer_input(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr) noexcept 
			: _layer_input<layer_input<Interfaces>, Interfaces>(pCustomName, _neurons_cnt) {};
		layer_input(const char* pCustomName, const neurons_count_t _neurons_cnt) noexcept
			: _layer_input<layer_input<Interfaces>, Interfaces>(pCustomName, _neurons_cnt) {};
	};

}