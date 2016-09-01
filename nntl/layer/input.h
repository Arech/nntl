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
	template<typename Interfaces, typename FinalPolymorphChild>
	class _layer_input : public m_layer_input, public _layer_base<Interfaces, FinalPolymorphChild> {
	private:
		typedef _layer_base<Interfaces, FinalPolymorphChild> _base_class;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		const realmtx_t* m_pActivations;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		std::enable_if_t<Archive::is_saving::value> serialize(Archive & ar, const unsigned int version) {
			if (m_pActivations && utils::binary_option<true>(ar, serialization::serialize_data_x)) 
				ar & serialization::make_nvp("data_x", * const_cast<realmtx_t*>(m_pActivations));
		}
		template<class Archive>
		std::enable_if_t<Archive::is_loading::value> serialize(Archive & ar, const unsigned int version) {
		}

	public:

		_layer_input(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr)noexcept :
			_base_class(_neurons_cnt, pCustomName), m_pActivations(nullptr) {};
		~_layer_input() noexcept {};
		static constexpr const char* _defName = "inp";

		const realmtx_t& get_activations()const noexcept {
			NNTL_ASSERT(m_pActivations);
			return *m_pActivations;
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


		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {}
		void set_mode(vec_len_t batchSize)noexcept {
			m_bTraining = 0 == batchSize;
		}

		void fprop(const realmtx_t& data_x)noexcept {
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(),data_x, m_bTraining);

			NNTL_ASSERT(data_x.test_biases_ok());
			m_pActivations = &data_x;

			iI.fprop_end(*m_pActivations);
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtx_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			//iI.bprop_end(dLdAPrev);
			return 1;
		}
		
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		constexpr const bool hasLossAddendum()const noexcept { return false; }

	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(layer_index_t& idx, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 == idx);
			NNTL_ASSERT(0 == inc_neurons_cnt);
			_base_class::_preinit_layer(idx, inc_neurons_cnt);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_input
	// If you need to derive a new class, derive it from _layer_input (to make static polymorphism work)
	template < typename Interfaces = d_interfaces>
	class layer_input final : public _layer_input<Interfaces, layer_input<Interfaces>> {
	public:
		~layer_input() noexcept {};
		layer_input(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr) noexcept 
			: _layer_input<Interfaces, layer_input<Interfaces>>(_neurons_cnt, pCustomName) {};
	};

}