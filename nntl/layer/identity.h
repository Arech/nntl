/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

// LI defines a layer that just passes it's incoming neurons as activation neurons without any processing.
// Can be used to pass a source data unmodified to some upper feature detectors.
// 
// LIG can also serve as a gating source for the layer_pack*gated.

#include <type_traits>

#include "_activation_storage.h"

namespace nntl {

	template<typename FinalPolymorphChild, typename Interfaces>
	class _LI : public _impl::_act_stor<FinalPolymorphChild, Interfaces>, public m_layer_autoneurons_cnt
	{
		typedef _impl::_act_stor<FinalPolymorphChild, Interfaces> _base_class_t;

	public:
		static constexpr bool bLayerToleratesNoBiases = true;
		static constexpr bool bLayerHasTrivialBProp = true;
		
	public:
		~_LI()noexcept {}
		_LI(const char* pCustomName)noexcept : _base_class_t(0, pCustomName) {}
		
		static constexpr const char _defName[] = "li";

		static constexpr bool hasLossAddendum()noexcept { return false; }
		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		static constexpr real_t lossAddendum()noexcept { return real_t(0.); }

		//////////////////////////////////////////////////////////////////////////
	protected:

		void _li_fprop(const realmtx_t& prevActivations)noexcept {
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(prevActivations.size_no_bias() == m_activations.size_no_bias());
			NNTL_ASSERT(m_activations.rows() == get_common_data().get_cur_batch_size());

			auto& iI = get_iInspect();
			iI.fprop_begin(get_layer_idx(), prevActivations, get_common_data().is_training_mode());

			// just copying the data from prevActivations to m_activations
			// We must copy the data, because layer_pack_horizontal uses its own storage for activations, therefore
			// we can't just use the m_activations as an alias to prevActivations - we have to physically copy the data
			// to a new storage within layer_pack_horizontal activations
			//const bool r = prevActivations.clone_to(m_activations);
			// we mustn't touch the bias column!
			const auto r = prevActivations.copy_data_skip_bias(m_activations);
			NNTL_ASSERT(r);

			m_bActivationsValid = true;
			
			iI.fprop_activations(m_activations);
			iI.fprop_end(m_activations);
		}

		void _li_bprop(const realmtx_t& dLdA)noexcept {
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(dLdA.size() == m_activations.size_no_bias());
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT(m_activations.rows() == get_common_data().get_cur_batch_size());

			m_bActivationsValid = false;
			auto& iI = get_iInspect();
			iI.bprop_begin(get_layer_idx(), dLdA);
			iI.bprop_finaldLdA(dLdA);

			//nothing to do here

			iI.bprop_end(dLdA);
		}

	public:
		template <typename LowerLayerT>
		void fprop(const LowerLayerT& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayerT>::value, "Template parameter LowerLayerT must implement _i_layer_fprop");
			get_self()._li_fprop(lowerLayer.get_activations());
		}

		template <typename LowerLayerT>
		unsigned bprop(realmtx_t& dLdA, const LowerLayerT& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			NNTL_UNREF(dLdAPrev); NNTL_UNREF(lowerLayer);
			get_self()._li_bprop(dLdA);
			return 0;//indicating that dL/dA for a previous layer is actually in the dLdA parameter (not in the dLdAPrev)
		}

		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(inc_neurons_cnt > 0);
			NNTL_ASSERT(get_neurons_cnt() == inc_neurons_cnt);

			_base_class_t::_preinit_layer(ili, inc_neurons_cnt);
			//_set_neurons_cnt(inc_neurons_cnt);
		}

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class ::boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int ) {
			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	template<typename FinalPolymorphChild, int iBinarize1e6, bool bDoBinarizeGate, typename Interfaces>
	class _LIG : public _LI<FinalPolymorphChild, Interfaces>, public m_layer_gate {
	private:
		typedef _LI<FinalPolymorphChild, Interfaces> _base_class_t;

	public:
		//using _base_class_t::real_t;

		static constexpr real_t sBinarizeFrac = real_t(iBinarize1e6) / real_t(1e6);
		static constexpr bool sbBinarizeGate = bDoBinarizeGate;

		//////////////////////////////////////////////////////////////////////////
		//members section (in "biggest first" order)
	protected:
		//gate matrix does not have a bias column
		// if the gate width>=2 AND it's requested by *pack_gated (it request the biases if its activations are not shared),
		// bias column additionally allocated and processed separately
		//realmtxdef_t m_gate;

	public:
		~_LIG()noexcept {}
		_LIG(const char* pCustomName)noexcept : _base_class_t(pCustomName) {}

		static constexpr const char _defName[] = "lig";

		//////////////////////////////////////////////////////////////////////////
		/*const realmtx_t& get_gate()const noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT(!m_gate.emulatesBiases() && m_gate.cols() == get_neurons_cnt());
			return m_gate;
		}
		const realmtx_t* get_gate_storage()const noexcept {
			NNTL_ASSERT(!m_gate.emulatesBiases() && m_gate.cols() == get_neurons_cnt());
			return &m_gate;
		}
		vec_len_t get_gate_width()const noexcept { return get_neurons_cnt(); }*/

	protected:
		//we can't use an alias to activations if we must binarize the gate first.
		// If we're under another gate, we still may safely use an alias to m_activations
		/*static constexpr bool _bAllocateGate() noexcept {
			return sbBinarizeGate;
		}*/

	public:
		/*ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			auto ec = _base_class_t::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			const auto nc = get_neurons_cnt();
			NNTL_ASSERT(nc);
			const auto bbs = get_common_data().biggest_batch_size();

			NNTL_ASSERT(!m_gate.emulatesBiases());
			if (_bAllocateGate()){
				if (!m_gate.resize(bbs, nc)) {
					return ErrorCode::CantAllocateMemoryForGatingMask;
				}
			} else {
				//we'll be using an alias to activations
				m_gate.useExternalStorage_no_bias(m_activations);
			}

			bSuccessfullyInitialized = true;
			return ErrorCode::Success;
		}
		
		void deinit() noexcept {
			m_gate.clear();
			_base_class_t::deinit();
		}
		
		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			_base_class_t::on_batch_size_change(pNewActivationStorage);

			NNTL_ASSERT(!m_gate.emulatesBiases());

			const vec_len_t batchSize = get_common_data().get_cur_batch_size();

			if (_bAllocateGate()) {
				NNTL_ASSERT(!m_gate.bDontManageStorage());
				m_gate.deform_rows(batchSize);
			} else {
				NNTL_ASSERT(m_gate.bDontManageStorage());
				m_gate.useExternalStorage_no_bias(m_activations);
			}
		}
		*/

	protected:

		// version without binarization
		template<bool bg = sbBinarizeGate>
		::std::enable_if_t<!bg> _make_gate()noexcept {
			NNTL_ASSERT(m_activations.isBinaryStrictNoBias());
		}
		// version with binarization
		template<bool bg = sbBinarizeGate>
		::std::enable_if_t<bg> _make_gate()noexcept {
			//NNTL_ASSERT(_bAllocateGate());			
			get_iMath().ewBinarize_ip(m_activations, sBinarizeFrac, real_t(0.), real_t(1.));
			NNTL_ASSERT(m_activations.isBinaryStrictNoBias());
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			_base_class_t::fprop(lowerLayer);
			//NNTL_ASSERT(!m_gate.empty() && !m_gate.emulatesBiases() && m_gate.size() == m_activations.size_no_bias() && m_gate.cols() == get_gate_width());
			get_self()._make_gate();
		}
		

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class ::boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int) {
			ar & ::boost::serialization::base_object<_base_class_t>(*this);
			//ar & serialization::serialize_base_class<_base_class_t>(*this);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template <typename Interfaces = d_interfaces>
	class LI final : public _LI<LI<Interfaces>, Interfaces>
	{
	public:
		~LI() noexcept {};
		LI(const char* pCustomName = nullptr) noexcept 
			: _LI<LI<Interfaces>, Interfaces> (pCustomName) {};
	};

	template <typename Interfaces = d_interfaces>
	using layer_identity = LI<Interfaces>;

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	template <int iBinarize1e6, bool bDoBinarizeGate, typename Interfaces = d_interfaces>
	class LIGf final : public _LIG<LIGf<iBinarize1e6, bDoBinarizeGate, Interfaces>, iBinarize1e6, bDoBinarizeGate, Interfaces> {
		typedef _LIG<LIGf<iBinarize1e6, bDoBinarizeGate, Interfaces>, iBinarize1e6, bDoBinarizeGate, Interfaces> _base_class_t;
	public:
		~LIGf() noexcept {};
		LIGf(const char* pCustomName = nullptr) noexcept : _base_class_t(pCustomName) {};
	};
	
	template <typename Interfaces = d_interfaces>
	using LIGFI = LIGf<0, false, Interfaces>;

	template <int iBinarize1e6, typename Interfaces = d_interfaces>
	using LIG = LIGf<iBinarize1e6, true, Interfaces>;
}
