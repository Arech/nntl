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

// layer_pack_gated is a layer wrapper, that uses underlying layer activation values as it's own activation values
// when a value of special gating neuron (incapsulated in a layer obeying _i_layer_gate interface) is 1,
// or uses zeros as activation values if the value of gating neuron is 0.
// Learning process (backpropagation) is also controlled by the gating neuron. When it is closed, no information flows
// down to underlying level and therefore it doesn't update its weights (I guess, it is effectively trains the layer
// on only a subset of original data that is "opened" by the gate)
// 
//											 | | | | | | | |
//			|--------------layer_pack_gated-----------------------|
//			|								 | | | | | | | |	  |
//			|		  gate--->(0 or 1)MUL    \ | | | | | | /      |
//			|			|		          |--underlying_layer--|  |
//			|			/					  /  |  |  |  \		  |
//		--->|>----------					  |  |  |  |  |       |
//	   /	|								  |  |  |  |  |       |
//	   |	-------------------------------------------------------
//	   |									  |  |  |  |  | 
//
// layer_pack_gated has the same number of neurons and incoming neurons, as underlying_layer.
// Gating neuron is implemented as a layer, obeying _i_layer_gate inteface. This layer must have a single neuron
// and is passed by reference during layer_pack_gated construction. The only requirement is that gating
// neuron (layer) must be processed by NN earlier, than corresponding layer_pack_gated (i.e. is must receive
// a smaller layer_index during preinit() phase)
// Usually layer_pack_gated is used in conjunction with layer_identity_gate (to pass gating neuron value from layer_input)
// and layer_pack_horizontal to assemble NN architecture.
// Look for examples in unit tests.
// 

#include "_pack_.h"
#include "../utils.h"

namespace nntl {

	//nBinarize1e6 - if this parameter has non-zero value, then gating neuron values are binarized according
	//to relation to value of real_t(nBinarize1e6/1e6)

	template<typename FinalPolymorphChild, typename UnderlyingLayer, typename GatingLayer, int32_t nBinarize1e6=500000>
	class _layer_pack_gated : public _cpolym_layer_base<FinalPolymorphChild, typename UnderlyingLayer::real_t> {
	public:
		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

		typedef UnderlyingLayer underlying_layer_t;
		static_assert(std::is_base_of<_i_layer<real_t>, UnderlyingLayer>::value,
			"UnderlyingLayer parameter must implement _i_layer interface!");
		//We require a UnderlyingLayer to have a normal activation matrix (with a bias column), therefore
		//UnderlyingLayer mustn't be a layer, that is directly embedded into layer_pack_horizontal (layers
		// directly embedded into LPH share activation storage and their bias columns intersects with corresponding first
		// activation column of neighbors layers)
		static_assert(!_impl::is_layer_wrapper<UnderlyingLayer>::value, "UnderlyingLayer mustn't be a layer wrapper!");

		typedef GatingLayer gating_layer_t;
		static_assert(std::is_base_of<_i_layer_gate<real_t>, GatingLayer>::value,
			"GatingLayer parameter must implement _i_layer_gate interface!");
		static_assert(std::is_base_of<_i_layer_trainable, GatingLayer>::value,
			"Template parameter GatingLayer must implement _i_layer_trainable");

		typedef typename underlying_layer_t::iMath_t iMath_t;
		typedef typename underlying_layer_t::iRng_t iRng_t;
		typedef typename underlying_layer_t::_layer_init_data_t _layer_init_data_t;
		typedef typename underlying_layer_t::common_data_t common_data_t;

		static constexpr real_t sBinarizeFrac = real_t(nBinarize1e6) / 1000000;
		static constexpr bool sbBinarizeGate = (0 != nBinarize1e6);

	protected:
		underlying_layer_t& m_undLayer;
		const gating_layer_t& m_gatingLayer;
		
		// gating matrix has the same size as activations of m_undLayer (EXcluding biases).
		// Each row of the mask has either a value of one, provided that a corresponding gating
		// neuron "is opened", or the value of zero in the other case
		realmtxdef_t m_gatingMask;

	private:
		layer_index_t m_layerIdx;

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
			NNTL_ASSERT(m_layerIdx > m_gatingLayer.get_layer_idx());

			_impl::_preinit_layers initializer(idx + 1, inc_neurons_cnt);
			initializer(m_undLayer);
			idx = initializer._idx;
		}

	public:
		~_layer_pack_gated()noexcept {}
		_layer_pack_gated(UnderlyingLayer& ulayer, const GatingLayer& glayer)noexcept :
			m_undLayer(ulayer), m_gatingLayer(glayer), m_layerIdx(0)
		{
			//gating mask works on biases also, but we shouldn't emulate them (causes unnecessary calls to fill_biases())
			m_gatingMask.dont_emulate_biases();
		}

		//and apply function _Func(auto& layer) to underlying layer
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			call_F_for_each_layer(std::forward<_Func>(f), m_undLayer);
		}

		underlying_layer_t& underlying_layer()const noexcept { return m_undLayer; }
		const gating_layer_t& gating_layer()const noexcept { return m_gatingLayer; }

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { return m_undLayer.get_neurons_cnt(); }
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { return  m_undLayer.get_incoming_neurons_cnt(); }

		const realmtxdef_t& get_activations()const noexcept { return m_undLayer.get_activations(); }

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, sbBinarizeGate ? "lpg%d" : "lpgfi%d", static_cast<unsigned>(get_self().get_layer_idx()));
		}

		//////////////////////////////////////////////////////////////////////////
		// helpers to access common data 
		// #todo this implies, that the following functions are described in _i_layer interface. It's not the case at this moment.
		const common_data_t& get_common_data()const noexcept { return m_undLayer.get_common_data(); }
		iMath_t& get_iMath()const noexcept { return m_undLayer.get_iMath(); }
		iRng_t& get_iRng()const noexcept { return m_undLayer.get_iRng();	}
		const vec_len_t get_max_fprop_batch_size()const noexcept { return m_undLayer.get_max_fprop_batch_size(); }
		const vec_len_t get_training_batch_size()const noexcept { return m_undLayer.get_training_batch_size(); }

		//////////////////////////////////////////////////////////////////////////
		// btw, in most cases we just pass function request to underlying layer. There are only two exceptions:
		// fprop (where we call original fprop() first and then turn off activations forbidden by the gate) and
		// bprop (where we clean dLdA by the gating mask, and then pass that dLdA to original bprop())

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept {
			return m_undLayer.hasLossAddendum();
		}
		//returns a loss function summand, that's caused by this layer
		real_t lossAddendum()const noexcept {
			//#BUGBUG loss values could depend on activation values, therefore depend on gating mask.
			//Though we can skip this little(?) bug now
			return m_undLayer.lossAddendum();
		}

		//////////////////////////////////////////////////////////////////////////
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(1 == m_gatingLayer.get_gate_width());
			auto ec = m_undLayer.init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			//we must resize gatingMask here to the size of underlying_layer activations, however gatingMask mustn't
			//have an emulated bias column
			NNTL_ASSERT(m_undLayer.get_activations().emulatesBiases());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			NNTL_ASSERT(m_undLayer.get_activations().size() != realmtx_t::mtx_size_t(0, 0));
			if (m_gatingMask.resize(m_undLayer.get_activations().size_no_bias())) {
				if (sbBinarizeGate) {
					//we'll use iMath internal memory storage for binarized source matrix, therefore we must notify iMath about it
					get_self().get_iMath().preinit(
						realmtx_t::sNumel(get_self().get_max_fprop_batch_size(), m_gatingLayer.get_gate_width())
					);
				}
			}else return ErrorCode::CantAllocateMemoryForGatingMask;

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			m_undLayer.deinit();
			m_gatingMask.clear();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			m_undLayer.initMem(ptr, cnt);
		}

		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept {
			m_undLayer.set_mode(batchSize, pNewActivationStorage);

			NNTL_ASSERT(m_undLayer.get_activations().emulatesBiases());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			//we must deform the mask to fit new underlying activations size
			m_gatingMask.deform_rows(m_undLayer.get_activations().rows());
			NNTL_ASSERT(m_gatingMask.size() == m_undLayer.get_activations().size_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		// gating functions
	protected:
		//construct a mask of ones and zeros based on gating neuron. The mask has to have a size of activations units of
		//underlying layer. It has ones for a rows of activations that are allowed by gating neuron and zeros for forbidden
		// rows. The mask is applied to activations and dLdA with a simple inplace elementwise multiplication.
		template<bool bg= sbBinarizeGate>
		std::enable_if_t<!bg,self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(1 == m_gatingLayer.get_gate_width());
			NNTL_ASSERT(1 == m_gatingLayer.get_gate().cols());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			get_self().get_iMath().mCloneCol(m_gatingLayer.get_gate(), m_gatingMask);
			return get_self();
		}

		template<bool bg= sbBinarizeGate>
		std::enable_if_t<bg, self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(1 == m_gatingLayer.get_gate_width());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			auto& origGate = m_gatingLayer.get_gate();
			NNTL_ASSERT(1 == origGate.cols());

			auto& iM = get_self().get_iMath();
			auto pTmpStor = iM._get_thread_temp_raw_storage(origGate.numel());
			NNTL_ASSERT(pTmpStor);
			
			realmtx_t gate(pTmpStor, origGate);
			iM.ewBinarize(gate, origGate, sBinarizeFrac);

			//iM.mCloneCols(gate, m_gatingMask, &m_gatingMask.cols());
			iM.mCloneCol(gate, m_gatingMask);
			return get_self();
		}

		//applies gating mask to a matrix.
		//A must either have a size of underlying activations matrix, or a size of dL/dA (which has one column less than
		// underlying activations)
		void apply_gating_mask(realmtxdef_t& A) noexcept {
			NNTL_ASSERT(!A.empty() && !m_gatingMask.empty());
			NNTL_ASSERT(A.rows() == m_gatingMask.rows());
			NNTL_ASSERT(m_gatingMask.cols() > 0);
			NNTL_ASSERT( (A.cols() == m_gatingMask.cols() && !A.emulatesBiases()) 
				|| (A.cols() == (m_gatingMask.cols() + 1) && A.emulatesBiases()));
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			const bool bHideBiases = A.cols() == (m_gatingMask.cols() + 1);
			if (bHideBiases) A.hide_last_col();

			NNTL_ASSERT(A.size() == m_gatingMask.size());
			get_self().get_iMath().evMul_ip(A, m_gatingMask);

			if (bHideBiases) A.restore_last_col();
		}


	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			NNTL_ASSERT(m_gatingMask.size() == m_undLayer.get_activations().size_no_bias());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			m_undLayer.fprop(lowerLayer);
			get_self().make_gating_mask<>().apply_gating_mask( *const_cast<realmtxdef_t*>(&m_undLayer.get_activations()) );
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			NNTL_ASSERT(m_gatingMask.size() == m_undLayer.get_activations().size_no_bias());
			NNTL_ASSERT(m_undLayer.get_activations().size_no_bias() == dLdA.size());
			NNTL_ASSERT((std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			get_self().apply_gating_mask(dLdA);
			const unsigned ret = m_undLayer.bprop(dLdA, lowerLayer, dLdAPrev);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			return ret;
		}

	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			if (utils::binary_option<true>(ar, serialization::serialize_gating_mask)) ar & NNTL_SERIALIZATION_NVP(m_gatingMask);
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & serialization::make_nvp("gating_layer_id", m_gatingLayer.get_layer_name_str());
			}
			ar & serialization::make_named_struct(m_undLayer.get_layer_name_str(), m_undLayer);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_gated
	// If you need to derive a new class, derive it from _layer_pack_gated (to make static polymorphism work)

	template<typename UnderlyingLayer, typename GatingLayer>
	class LPG final
		: public _layer_pack_gated<LPG<UnderlyingLayer, GatingLayer>, UnderlyingLayer, GatingLayer>
	{
	public:
		~LPG() noexcept {};
		LPG(UnderlyingLayer& u, const GatingLayer& g) noexcept
			: _layer_pack_gated<LPG<UnderlyingLayer, GatingLayer>, UnderlyingLayer, GatingLayer>(u,g){};
	};

	template<typename UnderlyingLayer, typename GatingLayer>
	using layer_pack_gated = typename LPG<UnderlyingLayer, GatingLayer>;

	template <typename UnderlyingLayer, typename GatingLayer> inline
		LPG <UnderlyingLayer, GatingLayer> make_layer_pack_gated(UnderlyingLayer& u, const GatingLayer& g) noexcept
	{
		return LPG <UnderlyingLayer, GatingLayer>(u,g);
	}

	//////////////////////////////////////////////////////////////////////////
	// this specialization of _layer_pack_gated skips binarization of gate and is expected to work
	// on a gate data, obtained directly from layer_input
	template<typename UnderlyingLayer, typename GatingLayer>
	class LPGFI final
		: public _layer_pack_gated<LPGFI<UnderlyingLayer, GatingLayer>, UnderlyingLayer, GatingLayer, 0>
	{
	public:
		~LPGFI() noexcept {};
		LPGFI(UnderlyingLayer& u, const GatingLayer& g) noexcept
			: _layer_pack_gated<LPGFI<UnderlyingLayer, GatingLayer>, UnderlyingLayer, GatingLayer, 0>(u,g){};
	};

	template<typename UnderlyingLayer, typename GatingLayer>
	using layer_pack_gated_from_input = typename LPGFI<UnderlyingLayer, GatingLayer>;

	template <typename UnderlyingLayer, typename GatingLayer> inline
		LPGFI <UnderlyingLayer, GatingLayer> make_layer_pack_gated_from_input(UnderlyingLayer& u, const GatingLayer& g) noexcept
	{
		return LPGFI <UnderlyingLayer, GatingLayer>(u, g);
	}
}
