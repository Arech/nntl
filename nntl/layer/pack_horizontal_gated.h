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

// this is a gated version of layer_pack_horizontal, that allows to selectively turn on/off output
// and learning for a horizontal pack of feature detectors based on their corresponding gates values.
// One may think of this pack as an optimized version of a set of individual layer_pack_gated's, that
// concatenates gates values and their corresponding feature detectors outputs into a single activation matrix.
// 
// For example: let's imagine one have a 3 optional feature sets for a problem. How to process them?
// One could make the following source data vector: {g1,g2,g3,x1,x2,x3} where g1,g2 and g3 are binary gate values
// that shows whether the feature data present, and x1,x2 and x3 are feature data sets. Then one may feed gate values
// into a single layer_identity_gated id1 and x1,x2 and x3 feed into corresponding feature detector layers set 
// fd1,fd2 and fd3 (for example, each feature detector could be a set of 2 fully_connected_layers stacked one
// on top of the other with a layer_pack_vertical). Then these 4 layers (id1, fd1,fd2 and fd3) could be bounded together
// into a single layer_pack_horizontal_gated that uses corresponding neuron of id1 to control the output and
// the learning process of layers fd1,fd2 and fd3.
// 

#include "pack_horizontal.h"
#include <array>

namespace nntl {

	// The first layer in PHLsT must be a gating layer. Its width must be the same as count of the other layers in PHLsT
	// We can't make the gating layer to be non first, because it must be on top of fprop() processing queue to
	// obtain correct gating values.
	// 
	//nBinarize1e6 - if this parameter has non-zero value, then gating neuron values are binarized according
	//to relation to value of real_t(nBinarize1e6/1e6).
	template<typename FinalPolymorphChild, int32_t nBinarize1e6, typename ...PHLsT>
	class _layer_pack_horizontal_gated : public _layer_pack_horizontal<FinalPolymorphChild, PHLsT...>
	{
	private:
		typedef _layer_pack_horizontal<FinalPolymorphChild, PHLsT...> _base_class;

	public:
		static_assert(std::is_base_of<_i_layer_gate<real_t>, first_layer_t>::value,
			"First layer within _layer_pack_horizontal_gated MUST be a gate with a corresponding width!");
		//BTW, gate width must be equal to phl_count-1. We can't check it with a static_assert, but
		// will check it later in runtime		
		typedef first_layer_t gating_layer_t;

		static constexpr size_t gated_layers_count = phl_count - 1;

		static constexpr real_t sBinarizeFrac = real_t(nBinarize1e6) / real_t(1e6);
		static constexpr bool sbBinarizeGate = (0 != nBinarize1e6);

	protected:
		// gating matrix has the width equal to the sum of all activation widths of embedded layers (except gating layer),
		// excluding their biases.
		// Each row of the mask has either a value of one, provided that a corresponding gating
		// neuron "is opened", or the value of zero in the other case
		realmtxdef_t m_gatingMask;

		//the following vector variable will hold neurons count of every embedded layer (excluding the gate layer)
		//we need it to make a correct gating mask
		typedef std::array<vec_len_t, gated_layers_count> column_spec_t;
		column_spec_t m_colSpec;

		friend class _impl::_preinit_layers;

	public:
		~_layer_pack_horizontal_gated()noexcept {}
		_layer_pack_horizontal_gated(const char* pCustomName, PHLsT&... phls)noexcept : _base_class(pCustomName, phls...) {
			//at this point all neuron counts in gating layers must be initialized and therefore we can check whether
			//the whole m_phl_tuple pack has correct number of layers
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == gated_layers_count);
			m_gatingMask.dont_emulate_biases();

			//filling m_colSpec
			m_colSpec.fill(vec_len_t(-1));
			vec_len_t* pA = &m_colSpec[0];
			NNTL_DEBUG_DECLARE(size_t s = 0);
			utils::for_each_exc_first_up(m_phl_tuple, [&pA NNTL_DEBUG_ARG(&s)](auto& phl)noexcept {
				NNTL_ASSERT(++s <= gated_layers_count);
				*pA++ = phl.l.get_neurons_cnt();
			});
			NNTL_ASSERT(s == gated_layers_count);
			NNTL_ASSERT(std::accumulate(m_colSpec.begin(), m_colSpec.end(), vec_len_t(0)) == get_self().get_neurons_cnt() - get_self().gating_layer().get_neurons_cnt());
		}
		static constexpr const char* _defName = sbBinarizeGate ? "lphg" : "lphgfi";

		const gating_layer_t& gating_layer()const noexcept { return get_self().first_layer(); }

		//#TODO returns a loss function summand, that's caused by this layer. Should take gating mask into account (and
		//that's a real trouble)
		//real_t lossAddendum()const noexcept {
			//#BUGBUG loss values could depend on activation values, therefore depend on gating mask.
			//Though we can skip this little(?) bug now
			//return m_undLayer.lossAddendum();
		//}

		//////////////////////////////////////////////////////////////////////////
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == gated_layers_count);

			auto ec = _base_class::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			//we must resize gatingMask here to the size of underlying_layer activations, however gatingMask mustn't
			//have an emulated bias column
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			NNTL_ASSERT(get_self().get_neurons_cnt() > get_self().gating_layer().get_neurons_cnt());
			if (m_gatingMask.resize(get_self().get_max_fprop_batch_size(),get_self().get_neurons_cnt()- get_self().gating_layer().get_neurons_cnt())) {
				if (sbBinarizeGate) {
					//we'll use iMath internal memory storage for binarized source matrix, therefore we must notify iMath about it
					get_self().get_iMath().preinit(
						realmtx_t::sNumel(get_self().get_max_fprop_batch_size(), get_self().gating_layer().get_gate_width())
					);
				}
			}else return ErrorCode::CantAllocateMemoryForGatingMask;

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			m_gatingMask.clear();
			_base_class::deinit();
		}

		void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept {
			_base_class::set_mode(batchSize, pNewActivationStorage);
			
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			//we must deform the mask to fit new underlying activations size
			m_gatingMask.deform_rows(get_self().gating_layer().get_activations().rows());
			NNTL_ASSERT(m_gatingMask.rows() == last_layer().get_activations().rows());
			NNTL_ASSERT(m_gatingMask.cols() == get_self().get_neurons_cnt() - get_self().gating_layer().get_neurons_cnt());
		}

		//////////////////////////////////////////////////////////////////////////
		// gating functions
	protected:
		//construct a mask of ones and zeros based on gating neuron. The mask has to have a size of activations units of
		//underlying layer. It has ones for a rows of activations that are allowed by gating neuron and zeros for forbidden
		// rows. The mask is applied to activations and dLdA with a simple inplace elementwise multiplication.
		// 
		// version without binarization optimized for 1 layer
		template<bool bg = sbBinarizeGate, size_t lc = gated_layers_count>
		std::enable_if_t<!bg && lc==1, self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(1 == get_self().gating_layer().get_gate_width());
			NNTL_ASSERT(1 == get_self().gating_layer().get_gate().cols());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			get_self().get_iMath().mCloneCol(get_self().gating_layer().get_gate(), m_gatingMask);
			return get_self();
		}
		// version without binarization for many layers
		template<bool bg = sbBinarizeGate, size_t lc = gated_layers_count>
		std::enable_if_t< (!bg && lc>1), self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == get_self().gating_layer().get_gate().cols());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			get_self().get_iMath().mCloneCols(get_self().gating_layer().get_gate(), m_gatingMask, &m_colSpec[0]);
			return get_self();
		}
		// version with binarization optimized for 1 layer
		template<bool bg = sbBinarizeGate, size_t lc = gated_layers_count>
		std::enable_if_t<bg && lc==1, self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(1 == get_self().gating_layer().get_gate_width());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			auto& origGate = get_self().gating_layer().get_gate();
			NNTL_ASSERT(1 == origGate.cols());

			auto& iM = get_self().get_iMath();
			auto pTmpStor = iM._get_thread_temp_raw_storage(origGate.numel());
			NNTL_ASSERT(pTmpStor);

			realmtx_t gate(pTmpStor, origGate);
			iM.ewBinarize(gate, origGate, sBinarizeFrac);

			iM.mCloneCol(gate, m_gatingMask);
			return get_self();
		}
		// version with binarization for many layer
		template<bool bg = sbBinarizeGate, size_t lc = gated_layers_count>
		std::enable_if_t< (bg && lc > 1), self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			auto& origGate = get_self().gating_layer().get_gate();
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == origGate.cols());

			auto& iM = get_self().get_iMath();
			auto pTmpStor = iM._get_thread_temp_raw_storage(origGate.numel());
			NNTL_ASSERT(pTmpStor);

			realmtx_t gate(pTmpStor, origGate);
			iM.ewBinarize(gate, origGate, sBinarizeFrac);

			iM.mCloneCols(gate, m_gatingMask, &m_colSpec[0]);
			return get_self();
		}

		//applies gating mask to a matrix (skipping matrix cols, that are used by gating layer itself)
		//A must either have a size of underlying activations matrix, or a size of dL/dA (which has one column less than
		// underlying activations)
		void apply_gating_mask(realmtxdef_t& A) noexcept {
			NNTL_ASSERT(!A.empty() && !m_gatingMask.empty());
			NNTL_ASSERT(A.rows() == m_gatingMask.rows());
			NNTL_ASSERT(m_gatingMask.cols() > 0);

			const auto skipCols = get_self().gating_layer().get_neurons_cnt();

			NNTL_ASSERT(A.cols_no_bias() == get_self().get_neurons_cnt());
			NNTL_ASSERT((A.cols()- skipCols == m_gatingMask.cols() && !A.emulatesBiases())
				|| (A.cols()- skipCols == (m_gatingMask.cols() + 1) && A.emulatesBiases()));
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			//const bool bHideBiases = A.cols() == (m_gatingMask.cols() + 1 + skipCols);
			//if (bHideBiases) A.hide_last_col();

			realmtx_t realA;
			realA.useExternalStorage(A.colDataAsVec(skipCols), A.rows(), get_self().get_neurons_cnt() - skipCols);

			NNTL_ASSERT(realA.size() == m_gatingMask.size());
			get_self().get_iMath().evMul_ip(realA, m_gatingMask);

			//if (bHideBiases) A.restore_last_col();
		}
		void finish_fprop()noexcept {
			get_self()
				.make_gating_mask<>()
				.apply_gating_mask(*const_cast<realmtxdef_t*>(&get_self().get_activations()));
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), m_bTraining);

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			NNTL_ASSERT(m_gatingMask.rows() == get_self().get_activations().rows());
			NNTL_ASSERT(m_gatingMask.rows() == lowerLayer.get_activations().rows());
			NNTL_ASSERT(m_gatingMask.cols() == get_self().get_neurons_cnt() - get_self().gating_layer().get_neurons_cnt());

			_base_class::fprop(lowerLayer);
			get_self().finish_fprop();
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.fprop_end(m_activations);
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			NNTL_ASSERT(m_gatingMask.rows() == get_self().get_activations().rows());
			NNTL_ASSERT(m_gatingMask.rows() == lowerLayer.get_activations().rows());
			NNTL_ASSERT(m_gatingMask.cols() == get_self().get_neurons_cnt() - get_self().gating_layer().get_neurons_cnt());

			NNTL_ASSERT(get_self().get_activations().size_no_bias() == dLdA.size());
			NNTL_ASSERT((std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			get_self().apply_gating_mask(dLdA);
			const unsigned ret = _base_class::bprop(dLdA, lowerLayer, dLdAPrev);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}

	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			if (utils::binary_option<true>(ar, serialization::serialize_gating_mask)) ar & NNTL_SERIALIZATION_NVP(m_gatingMask);
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				size_t li = get_self().gating_layer().get_layer_idx();
				ar & serialization::make_nvp("gating_layer_id", li);
			}
			//ar & serialization::make_named_struct(m_undLayer.get_layer_name_str(), m_undLayer);
			//ar & serialization::serialize_base_class<_base_class>(*this);
			ar & boost::serialization::base_object<_base_class>(*this);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_horizontal_gated
	// If you need to derive a new class, derive it from _layer_pack_horizontal_gated (to make static polymorphism work)

	//to shorten class name to get rid of C4503
	template <typename ...PHLsT>
	class LPHG final
		: public _layer_pack_horizontal_gated<LPHG<PHLsT...>, 500000, PHLsT...>
	{
	public:
		~LPHG() noexcept {};
		LPHG(PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHG<PHLsT...>, 500000, PHLsT...>(nullptr, phls...) {};
		LPHG(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHG<PHLsT...>, 500000, PHLsT...>(pCustomName, phls...) {};
	};

	template <typename ..._T>
	using layer_pack_horizontal_gated = typename LPHG<_T...>;

	template <typename ...PHLsT> inline constexpr
	LPHG <PHLsT...> make_layer_pack_horizontal_gated(PHLsT&... phls) noexcept {
		return LPHG<PHLsT...>(phls...);
	}
	template <typename ...PHLsT> inline constexpr
	LPHG <PHLsT...> make_layer_pack_horizontal_gated(const char* pCustomName, PHLsT&... phls) noexcept {
		return LPHG<PHLsT...>(pCustomName, phls...);
	}
	//////////////////////////////////////////////////////////////////////////

	template <typename ...PHLsT>
	class LPHGFI final
		: public _layer_pack_horizontal_gated<LPHGFI<PHLsT...>, 0, PHLsT...>
	{
	public:
		~LPHGFI() noexcept {};
		LPHGFI(PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHGFI<PHLsT...>, 0, PHLsT...>(nullptr, phls...) {};
		LPHGFI(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHGFI<PHLsT...>, 0, PHLsT...>(pCustomName, phls...) {};
	};

	template <typename ..._T>
	using layer_pack_horizontal_gated_from_input = typename LPHGFI<_T...>;

	template <typename ...PHLsT> inline constexpr
	LPHGFI <PHLsT...> make_layer_pack_horizontal_gated_from_input(PHLsT&... phls) noexcept {
		return LPHGFI<PHLsT...>(phls...);
	}
	template <typename ...PHLsT> inline constexpr
	LPHGFI <PHLsT...> make_layer_pack_horizontal_gated_from_input(const char* pCustomName, PHLsT&... phls) noexcept {
		return LPHGFI<PHLsT...>(pCustomName, phls...);
	}

}