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
// when a value of external (to the layer_pack_gated object) gating neuron (incapsulated in an other layer obeying _i_layer_gate interface) is 1,
// and uses zeros as activation values otherwise (if the value of the gating neuron is 0).
// Learning process (backpropagation) is controlled by the gating neuron. When it is closed, no information flows
// down to the underlying layer and therefore it doesn't update its weights. It effectively trains the underlying layer only
// on a subset of the original dataset that is "opened" by the gate.
// 
//											 | | | | | | | |
//			|--------------layer_pack_gated(LPG)------------------|
//			|								 | | | | | | | |	  |
//			|		  gate--->(0 or 1)MUL    \ | | | | | | /      |
//			|			|		          |--underlying_layer--|  |
//			|			/					  /  |  |  |  \		  |
//		--->|>----------					  |  |  |  |  |       |
//	   /	|								  |  |  |  |  |       |
//	   |	-------------------------------------------------------
//	   |									  |  |  |  |  | 
//
// layer_pack_gated has the same number of neurons and incoming neurons, as the underlying_layer.
// The underlying_layer object should be instantiated and passed to the layer_pack_gated constructor only (i.e. it mustn't be a part
// of a layers stack).
// Gating neuron is implemented as a layer, obeying _i_layer_gate interface, such as layer_identity_gate.
// This layer must have a single neuron and it is passed by reference during layer_pack_gated construction.
// The only requirement is that the gating neuron (layer) must be processed by NN earlier, than the corresponding
// layer_pack_gated (i.e. gating neuron/layer must receive a smaller layer_index during preinit() phase)
// 
// Look for examples in unit tests.
//
// Implementation discussion:
// In order to emulate an absence of a data sample, the data sample bias unit must also be affected by the gating value.
// Biases in NNTL are implemented as
// a last column of an activation matrix and that brings some issues into a straightforward implementation of gating. 
// Being a part of an activation matrix, biases are not actually a semantic part of a layer and aren't expected to be managed
// by the layer itself. Biases are just added to the activation matrix to make an upper layer's life easier.
// Therefore an extra care must be taken when the gating is about to be applied to the bias column.
// A LPG must apply gating values to a bias column iff:
// - the LPG is a single/sole layer <==> i.e. it doesn't share its activations space with other layers
//		(It's not the same as the layer is given pNewActivationStorage - it can be given it in some other cases while it's still
//			a single layer, for example in layer_pack_tile setup)
// - the LPG is the last (rightmost) layer in 2-layers layer_pack_horizontal (LPH), provided that the first layer is its gating layer
// In all other cases LPG must not touch a bit in a bias column.
// 

#include "_pack_.h"
#include "../utils.h"

namespace nntl {

	//nBinarize1e6 - if this parameter has non-zero value, then gating neuron values are binarized according
	//to relation to value of real_t(nBinarize1e6/1e6)

	template<typename FinalPolymorphChild, typename UnderlyingLayer, typename GatingLayer, int32_t nBinarize1e6=500000>
	class _layer_pack_gated 
		: public _cpolym_layer_base<FinalPolymorphChild, typename UnderlyingLayer::real_t>
		, public interfaces_td<typename UnderlyingLayer::interfaces_t>
	{
	private:
		typedef _cpolym_layer_base<FinalPolymorphChild, typename UnderlyingLayer::real_t> _base_class;

	public:
		using _base_class::real_t;

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

		typedef typename underlying_layer_t::_layer_init_data_t _layer_init_data_t;
		typedef typename underlying_layer_t::common_data_t common_data_t;

		static constexpr real_t sBinarizeFrac = real_t(nBinarize1e6) / 1000000;
		static constexpr bool sbBinarizeGate = (0 != nBinarize1e6);

	protected:
		underlying_layer_t& m_undLayer;
		const gating_layer_t& m_gatingLayer;
		
		// gating matrix is 1 column binary matrix that has the same number of rows as there are activations of m_undLayer
		// It is allocated when we had to binarize the gate OR we'd have the .drop_samples() called. Else it's just an alias to the gate
		realmtxdef_t m_gatingMask;

	private:
		layer_index_t m_layerIdx;

	protected:

		bool m_bApplyGateToBiases;

	private:
		bool m_bIsDropSamplesMightBeCalled;//taken from init()

		//////////////////////////////////////////////////////////////////////////
		//
	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx && inc_neurons_cnt > 0);

			if (m_layerIdx) abort();
			m_layerIdx = ili.newIndex();
			//if it asserts, then m_gatingLayer resides later in layers stack which is wrong.
			NNTL_ASSERT(m_layerIdx > m_gatingLayer.get_layer_idx());

			_impl::_preinit_layers initializer(ili, inc_neurons_cnt);
			initializer(m_undLayer);
		}

	private:
		template<bool b> struct _defNameS {};
		template<> struct _defNameS<true> { static constexpr const char n[] = "lpg"; };
		template<> struct _defNameS<false> { static constexpr const char n[] = "lpgfi"; };
	public:
		static constexpr const char _defName[sizeof(_defNameS<sbBinarizeGate>::n)] = _defNameS<sbBinarizeGate>::n;
		//static constexpr const char _defName[] = sbBinarizeGate ? "lpg" : "lpgfi";
		
		~_layer_pack_gated()noexcept {}
		_layer_pack_gated(UnderlyingLayer& ulayer, const GatingLayer& glayer, const char* pCustomName = nullptr)noexcept 
			: _base_class(pCustomName), m_undLayer(ulayer), m_gatingLayer(glayer), m_layerIdx(0)
			, m_bApplyGateToBiases(false), m_bIsDropSamplesMightBeCalled(false)
		{
			m_gatingMask.dont_emulate_biases();
		}

		//and apply function _Func(auto& layer) to underlying layer
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			call_F_for_each_layer(std::forward<_Func>(f), m_undLayer);
		}
		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			call_F_for_each_layer_down(std::forward<_Func>(f), m_undLayer);
		}

		underlying_layer_t& underlying_layer()const noexcept { return m_undLayer; }
		const gating_layer_t& gating_layer()const noexcept { return m_gatingLayer; }

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { return m_undLayer.get_neurons_cnt(); }
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { return  m_undLayer.get_incoming_neurons_cnt(); }

		const realmtxdef_t& get_activations()const noexcept { return m_undLayer.get_activations(); }
		const mtx_size_t get_activations_size()const noexcept { return m_undLayer.get_activations_size(); }
		const bool is_activations_shared()const noexcept { return m_undLayer.is_activations_shared(); }

	protected:
		const bool is_drop_samples_mbc()const noexcept { return m_bIsDropSamplesMightBeCalled; }

	public:
		//////////////////////////////////////////////////////////////////////////
		// helpers to access common data 
		// #todo this implies, that the following functions are described in _i_layer interface. It's not the case at this moment.
		const common_data_t& get_common_data()const noexcept { return m_undLayer.get_common_data(); }
		iMath_t& get_iMath()const noexcept { return m_undLayer.get_iMath(); }
		iRng_t& get_iRng()const noexcept { return m_undLayer.get_iRng();	}
		iInspect_t& get_iInspect()const noexcept { return m_undLayer.get_iInspect(); }
		const vec_len_t get_max_fprop_batch_size()const noexcept { return m_undLayer.get_max_fprop_batch_size(); }
		const vec_len_t get_training_batch_size()const noexcept { return m_undLayer.get_training_batch_size(); }
		const vec_len_t get_biggest_batch_size()const noexcept { return m_undLayer.get_biggest_batch_size(); }
		const bool isTrainingMode()const noexcept { return m_undLayer.isTrainingMode(); }

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
		template<typename PhlsTupleT>
		bool OuterLayerCustomFlag1Eval(const PhlsTupleT& lphTuple, const _layer_init_data_t& lphLid)const noexcept {
			return !lphLid.bActivationsShareSpace  //this ensures, that the parent LPH doesn't share it's own activation space
				&& 2 == std::tuple_size<PhlsTupleT>::value  //the following tests for the only case when we're allowed to update a bias column
				//&& &m_gatingLayer == &(std::get<0>(lphTuple).l)
				//&& this == &(std::get<1>(lphTuple).l);
 				&& reinterpret_cast<const void*>(&m_gatingLayer) == reinterpret_cast<const void*>(&(std::get<0>(lphTuple).l))
 				&& reinterpret_cast<const void*>(this) == reinterpret_cast<const void*>(&(std::get<1>(lphTuple).l));
			//we're doing type-less pointer base comparision, because actual types of layers stored in tuple may differ from
			//what we're having in variables.
		}

	protected:
		//we must allocate gating mask IFF we must binarize gate or we expect drop_samples() to be called
		const bool _bAllocateGatingMask()const noexcept {
			return sbBinarizeGate || is_drop_samples_mbc();
		}

	public:
		//////////////////////////////////////////////////////////////////////////
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			_base_class::init();
			NNTL_ASSERT(1 == m_gatingLayer.get_gate_width());

			//we're allowed to apply gate to the bias column IFF, we don't share our activation space (in that case bias column might
			// contain activation values of other layers), or if we do, we're actually inside a LPH, that doesn't share it's activations,
			// the size of the LPH is 2 and we're on the second place, while the first place is occupied by our gating layer
			// (The .OuterLayerCustomFlag1Eval() function defines the value of lid.bLPH_CustomFlag1)
			m_bApplyGateToBiases = !lid.bActivationsShareSpace || lid.bLPH_CustomFlag1;

			m_bIsDropSamplesMightBeCalled = lid.bDropSamplesMightBeCalled;
			const bool bOrig_LphFlag1 = lid.bLPH_CustomFlag1;
			lid.bDropSamplesMightBeCalled = true;
			lid.bLPH_CustomFlag1 = m_bApplyGateToBiases;//if we're working over another LPG, it'll help it to make a right decision.
			auto ec = m_undLayer.init(lid, pNewActivationStorage);
			lid.bDropSamplesMightBeCalled = m_bIsDropSamplesMightBeCalled;
			lid.bLPH_CustomFlag1 = bOrig_LphFlag1;

			if (ErrorCode::Success != ec) return ec;

			//must be called after m_undLayer.init() because see get_iInspect() implementation
			get_iInspect().init_layer(get_self().get_layer_idx(), get_self().get_layer_name_str(), get_self().get_layer_type_id());

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});
			
			NNTL_ASSERT(!m_gatingMask.emulatesBiases() && m_gatingMask.empty());
			// we're to allocate m_gatingMask IFF we had to binarize gate values OR we expect .drop_samples() to be called
			if (_bAllocateGatingMask()) {
				if (!m_gatingMask.resize(m_undLayer.get_activations_size().first, m_gatingLayer.get_gate_width()))
					return ErrorCode::CantAllocateMemoryForGatingMask;
			} //else //we'll use it as an alias to the gate
			

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			m_bApplyGateToBiases = false;
			m_bIsDropSamplesMightBeCalled = false;
			m_undLayer.deinit();
			m_gatingMask.clear();
			_base_class::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			m_undLayer.initMem(ptr, cnt);
		}

		void set_batch_size(const vec_len_t batchSize, real_t*const pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(batchSize > 0);

			m_undLayer.set_batch_size(batchSize, pNewActivationStorage);

			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			if (_bAllocateGatingMask()) {
				NNTL_ASSERT(!m_gatingMask.empty() && !m_gatingMask.bDontManageStorage());
				//we must deform the mask to fit new underlying activations size
				NNTL_ASSERT(batchSize == m_undLayer.get_activations_size().first);
				m_gatingMask.deform_rows(batchSize);
				NNTL_ASSERT(m_gatingMask.cols() == 1);
			} else {
				NNTL_ASSERT(m_gatingMask.empty() || m_gatingMask.bDontManageStorage());
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// gating functions
	protected:
		//construct a mask of ones and zeros based on gating neuron.The mask is applied to activations and dLdA
		template<bool bg= sbBinarizeGate>
		std::enable_if_t<!bg,const realmtx_t&> make_gating_mask()noexcept {
			NNTL_ASSERT(1 == m_gatingLayer.get_gate_width() && m_gatingLayer.get_gate().isBinary());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			if (_bAllocateGatingMask()) {
				NNTL_ASSERT(!m_gatingMask.empty() && !m_gatingMask.bDontManageStorage());
				NNTL_ASSERT(m_gatingLayer.get_gate().size() == m_gatingMask.size() && m_gatingMask.rows() == m_undLayer.get_activations().rows());
				m_gatingLayer.get_gate().copy_to(m_gatingMask);
			} else {
				NNTL_ASSERT(m_gatingMask.empty() || m_gatingMask.bDontManageStorage());
				//we won't update the gate here, it's just an alias, therefore const_cast is safe
				m_gatingMask.useExternalStorage(const_cast<realmtx_t&>(m_gatingLayer.get_gate()));
			}
			return m_gatingMask;
		}

		template<bool bg= sbBinarizeGate>
		std::enable_if_t<bg, const realmtx_t&> make_gating_mask()noexcept {
			NNTL_ASSERT(_bAllocateGatingMask());
			NNTL_ASSERT(1 == m_gatingLayer.get_gate_width());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases() && !m_gatingMask.empty() && !m_gatingMask.bDontManageStorage());
			NNTL_ASSERT(m_gatingMask.size() == m_gatingLayer.get_gate().size() && m_gatingMask.rows() == m_undLayer.get_activations().rows());

			get_self().get_iMath().ewBinarize(m_gatingMask, m_gatingLayer.get_gate(), sBinarizeFrac);

			NNTL_ASSERT(m_gatingMask.isBinary());
			return m_gatingMask;
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop<real_t>, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().isTrainingMode());

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			m_undLayer.fprop(lowerLayer);
			NNTL_ASSERT(!_bAllocateGatingMask() || (!m_gatingMask.empty() && m_gatingMask.rows() == m_undLayer.get_activations().rows()));
			
			m_undLayer.drop_samples(get_self().make_gating_mask<>(), m_bApplyGateToBiases);

			//ensure the bias column has correct flags
			NNTL_ASSERT((m_bApplyGateToBiases && m_undLayer.get_activations().isHoleyBiases() 
				&& !m_undLayer.is_activations_shared() && m_undLayer.get_activations().test_biases_holey()
				) || (!m_bApplyGateToBiases && !m_undLayer.get_activations().isHoleyBiases()
					&& (m_undLayer.is_activations_shared() || m_undLayer.get_activations().test_biases_strict())));

			iI.fprop_end(m_undLayer.get_activations());
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable<real_t>, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			
			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			NNTL_ASSERT(m_gatingMask.rows() == m_undLayer.get_activations().rows());
			NNTL_ASSERT(m_undLayer.is_activations_shared() || m_undLayer.get_activations().test_biases_ok());
			NNTL_ASSERT(m_undLayer.get_activations().size_no_bias() == dLdA.size());
			NNTL_ASSERT((std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			// we must zero the dLdA entries for activations that was removed by the gating mask during fprop()
			NNTL_ASSERT(!dLdA.empty() && !dLdA.emulatesBiases() && !m_gatingMask.empty() && !m_gatingMask.emulatesBiases());
			NNTL_ASSERT(dLdA.rows() == m_gatingMask.rows() && 1 == m_gatingMask.cols());
			NNTL_ASSERT(m_gatingMask.size() == m_gatingLayer.get_gate().size() && m_gatingMask.rows() == m_undLayer.get_activations().rows()); //this assert also checks whether the gate is still usable
			NNTL_ASSERT(m_gatingMask.isBinary());
			//get_self().get_iMath().evMul_ip(dLdA, m_gatingMask);
			get_self().get_iMath().mrwMulByVec(dLdA, m_gatingMask.data());

			const unsigned ret = m_undLayer.bprop(dLdA, lowerLayer, dLdAPrev);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}

		static constexpr bool is_trivial_drop_samples()noexcept { return false; }

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			NNTL_ASSERT(get_self().is_drop_samples_mbc() && _bAllocateGatingMask());
			//it should not be possible when we're aren't allowed to update biases, while getting a request to update them...
			NNTL_ASSERT(m_bApplyGateToBiases || !bBiasesToo);
			NNTL_ASSERT(m_gatingMask.isBinary());
			
			//update our mask
			NNTL_ASSERT(mask.size()==m_gatingMask.size() && mask.isBinary());
			get_self().get_iMath().evMul_ip(m_gatingMask, mask);

			//apply to the underlying layer
			m_undLayer.drop_samples(m_gatingMask, bBiasesToo);

			//ensure the bias column has correct flags
			NNTL_ASSERT((m_bApplyGateToBiases && m_undLayer.get_activations().isHoleyBiases()
				&& !m_undLayer.is_activations_shared() && m_undLayer.get_activations().test_biases_holey()
				) || (!m_bApplyGateToBiases && !m_undLayer.get_activations().isHoleyBiases()
					&& (m_undLayer.is_activations_shared() || m_undLayer.get_activations().test_biases_strict())));
		}


	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			if (utils::binary_option<true>(ar, serialization::serialize_gating_mask)) ar & NNTL_SERIALIZATION_NVP(m_gatingMask);
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				size_t li = m_gatingLayer.get_layer_idx();
				ar & serialization::make_nvp("gating_layer_idx", li);
			}
			ar & serialization::make_named_struct(m_undLayer.get_layer_name_str().c_str(), m_undLayer);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_gated
	// If you need to derive a new class, derive it from _layer_pack_gated (to make static polymorphism work)

	template<typename UnderlyingLayer, typename GatingLayer, int32_t nBinarize1e6 = 500000>
	class LPG final
		: public _layer_pack_gated<LPG<UnderlyingLayer, GatingLayer, nBinarize1e6>, UnderlyingLayer, GatingLayer>
	{
	public:
		~LPG() noexcept {};
		LPG(UnderlyingLayer& u, const GatingLayer& g, const char* pCustomName = nullptr) noexcept
			: _layer_pack_gated<LPG<UnderlyingLayer, GatingLayer, nBinarize1e6>, UnderlyingLayer, GatingLayer, nBinarize1e6>(u,g,pCustomName){};
	};

	template<typename UnderlyingLayer, typename GatingLayer, int32_t nBinarize1e6 = 500000>
	using layer_pack_gated = typename LPG<UnderlyingLayer, GatingLayer, nBinarize1e6>;

	template <typename UnderlyingLayer, typename GatingLayer> inline
		LPG <UnderlyingLayer, GatingLayer> make_layer_pack_gated(UnderlyingLayer& u, const GatingLayer& g, const char* pCustomName = nullptr) noexcept
	{
		return LPG <UnderlyingLayer, GatingLayer>(u, g, pCustomName);
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
		LPGFI(UnderlyingLayer& u, const GatingLayer& g, const char* pCustomName = nullptr) noexcept
			: _layer_pack_gated<LPGFI<UnderlyingLayer, GatingLayer>, UnderlyingLayer, GatingLayer, 0>(u,g,pCustomName){};
	};

	template<typename UnderlyingLayer, typename GatingLayer>
	using layer_pack_gated_from_input = typename LPGFI<UnderlyingLayer, GatingLayer>;

	template <typename UnderlyingLayer, typename GatingLayer> inline
		LPGFI <UnderlyingLayer, GatingLayer> make_layer_pack_gated_from_input(UnderlyingLayer& u, const GatingLayer& g, const char* pCustomName = nullptr) noexcept
	{
		return LPGFI <UnderlyingLayer, GatingLayer>(u, g,pCustomName);
	}
}
