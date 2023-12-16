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

// This is a modification of LPH that allows NN to deal with optional datasets.
// 
// For example: let's imagine we have k optional feature sets for a problem. How to process them and how to
// train corresponding feature detectors? One could make the following source data vector:
// {g1, ... , gk, x1, ... , xk} where g_i is the binary (gating) scalar value that shows whether the feature vector x_i is present.
// Then one may feed gating values {g1, ... , gk} into a single LIG object lig and {x1, ... , xk} into corresponding
// feature detector layers sets fd1, ... , fdk (for example, each feature detector could be a set of 2 fully_connected_layers stacked one
// on top of the other with a layer_pack_vertical). Then these k+1 layers {lig, fd1, ... , fdk} could be bounded together
// into a single layer_pack_horizontal_optional that uses corresponding neuron of lig to control the output and
// the learning process/gradient flow of each of {fd1, ... , fdk}.
//
// Important thing to consider: if you're going to use LPHO as a standalone layer (i.e. when it won't be a part of other
// LPH with an always-present data), then make sure that the case when all gates are closed (i,e, when all g_i == 0) is impossible.
// Else it may screw the gradient propagation.

#include <nntl/layer/_lph_base.h>

namespace nntl {
	
	//bAddDataNotPresentFeature is a switch that appends to LPHO activations a feature that equals to !gate value
	// One may treat them as an offsetting bias for absent features.
	// I don't know yet if it's helpful thing or not (looks like it was a bad idea, but leave it here for some more tests)

	template<typename FinalPolymorphChild, bool bAddDataNotPresentFeature, typename PHLsTuple>
	class _LPHO : public _impl::_LPH_base<FinalPolymorphChild, PHLsTuple> {
		typedef _impl::_LPH_base<FinalPolymorphChild, PHLsTuple> _base_class_t;

	public:
		static constexpr size_t gated_layers_count = phl_count - 1;
		//neurons of the gate always take the first columns of activations matrix
		static constexpr neurons_count_t gate_neurons_count = static_cast<neurons_count_t>(gated_layers_count*(1 + bAddDataNotPresentFeature));

		static_assert(is_layer_gate<first_layer_t>::value, "First layer within _LPHO MUST be a gate with a width of gated_layers_count!");
		typedef first_layer_t gating_layer_t;

	protected:
		//array of previous activations with gate applied
		::std::array<realmtxdef_t, gated_layers_count> m_aPrevActs;

		//storage for matrices of m_aPrevActs
		::std::unique_ptr<real_t[]> m_prevActsStor;

		//////////////////////////////////////////////////////////////////////////
	public:
		~_LPHO()noexcept {}
		_LPHO(const char* pCustomName, const PHLsTuple& phls)noexcept : _base_class_t(pCustomName, phls
			, static_cast<neurons_count_t>(bAddDataNotPresentFeature*gated_layers_count))
		{}
		_LPHO(const char* pCustomName, PHLsTuple&& phls)noexcept : _base_class_t(pCustomName, ::std::move(phls)
			, static_cast<neurons_count_t>(bAddDataNotPresentFeature*gated_layers_count))
		{}
		
		static constexpr const char _defName[] = "lpho";
		
	public:

		gating_layer_t& gating_layer()noexcept { return first_layer(); }

		template<typename _Func>
		void for_each_gated_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_exc_first_up(m_phl_tuple, [&f](auto& phl)noexcept {
				f(phl.l);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		ErrorCode _init_phls(_layer_init_data_t& lid)noexcept {
			NNTL_ASSERT(m_activations.bBatchInColumn() && !m_activations.empty());
			const auto origLid = lid.exact_dupe();

			//first we must initialize the gating layer
			if (gated_layers_count != gating_layer().get_neurons_cnt()) {
				STDCOUTL("*** " << get_layer_name_str() << ": Gating layer must have the same neurons count as there are layers under the gate!");
				NNTL_ASSERT(!"Gating layer must have the same neurons count as there are layers under the gate!");
				abort();
			}
			
			//#todo flag for inner layers to strip biases in the topmost layer?

			//we need to forward the gate values directly to the activations matrix
			auto initD = origLid.exact_dupe();
			ErrorCode ec = gating_layer().layer_init(initD, m_activations.data());
			if (ErrorCode::Success != ec) return ec;

			NNTL_ASSERT(initD.outgBS.isValid());
			const auto commonOutgBS = initD.outgBS;
			lid.aggregate_from(initD);

			//then initialize layers under the gate
			layer_index_t failedLayerIdx = 0;
			for_each_gated_layer([&ec, &failedLayerIdx, &lid, &origLid, &commonOutgBS](auto& l)noexcept {
				if (ErrorCode::Success == ec) {
					auto initD = origLid.exact_dupe();

					//#todo flag for inner layers to strip biases in the topmost layer?
					ec = l.layer_init(initD, nullptr);
					if (ErrorCode::Success == ec) {
						if (commonOutgBS != initD.outgBS) {
							STDCOUTL("Error: every PHL'ed layer must produce the same outgoing batch sizes! Not true for the first layer and "
								<< l.get_layer_name_str());
							NNTL_ASSERT(!"Error: every PHL'ed layer must produce the same outgoing batch sizes!");
							ec = ErrorCode::InvalidBatchSizeCombination;
							return;
						}
						lid.aggregate_from(initD);
					} else failedLayerIdx = l.get_layer_idx();
				}
			});
			if (ErrorCode::Success == ec)
				lid.outgBS = commonOutgBS;
			return ec;
		}

		ErrorCode _init_self(const _layer_init_data_t& lid)noexcept {
			//we must check that there's no overlapping in receptive fields of inner layers
			neurons_count_t totalIncomingNC = 0;
			tuple_utils::for_each_up(m_phl_tuple, [&totalIncomingNC](const auto& phl)noexcept {
				totalIncomingNC += phl.l.get_incoming_neurons_cnt();
			});
			if (totalIncomingNC != get_incoming_neurons_cnt()) {
				STDCOUTL("*** " << get_layer_name_str() << ": inner layers can't have overlapping receptive fields!");
				NNTL_ASSERT(!"*** _LPHO: inner layers can't have overlapping receptive fields!");
				abort();
			}
			//however, it's not harder to use the totalIncomingNC variable later instead of get_incoming_neurons_cnt(),
			//so lets stick to totalIncomingNC
			totalIncomingNC -= static_cast<neurons_count_t>(gated_layers_count);//remove gating neurons 

			//allocate memory for gated previous layers activations (still have to use biggest_batch_size(), because
			//a gate might be completely open (all ones), and have to add +1 to neurons count to account bias column
			// for the last/rightmost layer
			const auto biggestIncBS = lid.incBS.biggest();
			real_t* ptr = new(::std::nothrow) real_t[realmtx_t::sNumel(biggestIncBS, totalIncomingNC + 1)];

			if (ptr) {
				//storing ptr
				m_prevActsStor.reset(ptr);
				//now we must redistribute the storage under ptr to activation matrices
				size_t glIdx = 0;
				for_each_gated_layer([biggestIncBS, &ptr, &glIdx, &actArr = m_aPrevActs](const auto& l)noexcept {
					const auto pnc = l.get_incoming_neurons_cnt();
					NNTL_ASSERT(pnc);
					auto& prevAct = actArr[glIdx++];
					prevAct.useExternalStorage(ptr, biggestIncBS, pnc+1, true, false);//adding one column for biases
					ptr += realmtx_t::sNumel(biggestIncBS, pnc);//not including bias column here, as it'll be substituted as it is done in ordinary LPH
				});
				NNTL_ASSERT(glIdx == gated_layers_count);
			}
			return ptr ? ErrorCode::Success : ErrorCode::CantAllocateMemoryForInnerLLActivations;
		}

		void layer_deinit() noexcept {
			for (auto& e : m_aPrevActs) e.clear();
			m_prevActsStor.reset(::std::nullptr_t());
			_base_class_t::layer_deinit();
		}

		vec_len_t on_batch_size_change(const vec_len_t incBatchSize, real_t*const pNewActivationStorage = nullptr)noexcept {
			//passing the on_batch_size_change to the pre-base class
			const auto outgBS = _pre_LPH_base_class_t::on_batch_size_change(incBatchSize, pNewActivationStorage);
			//m_upperLayerLRScale = learningRateScale;
			//we don't need to call on_batch_size_change() on every inner layer except for the gate now, because the batch size for them
			//depends on a gating neuron column content
			const auto gtbs = gating_layer().on_batch_size_change(incBatchSize, m_activations.data());
			NNTL_ASSERT(gtbs == outgBS);
			return outgBS;
		}

	protected:

		template<bool c = bAddDataNotPresentFeature>
		::std::enable_if_t<c> _fprop_add_no_present_feature()noexcept {
			const auto& gate = gating_layer().get_activations();
			NNTL_ASSERT(gate._isBinaryStrictNoBias());
			NNTL_ASSERT(gate.cols_no_bias() == static_cast<vec_len_t>(gated_layers_count) && m_activations.rows() == gate.rows());
			
			realmtx_t gateCompl(m_activations.colDataAsVec(static_cast<vec_len_t>(gated_layers_count)), m_activations.rows()
				, static_cast<vec_len_t>(gated_layers_count), false);

			get_iMath().evOneCompl(gate, gateCompl);
		}

		template<bool c = bAddDataNotPresentFeature>
		::std::enable_if_t<!c> _fprop_add_no_present_feature()noexcept {
			//just common asserts only
			NNTL_ASSERT(gating_layer().get_activations()._isBinaryStrictNoBias());
			NNTL_ASSERT(gating_layer().get_activations().cols_no_bias() == static_cast<vec_len_t>(gated_layers_count)
				&& m_activations.rows() == gating_layer().get_activations().rows());
		}

		template<typename LLWrapT>
		void _lpho_fprop(const realmtx_t& prevAct)noexcept {
			NNTL_ASSERT(prevAct.test_biases_strict() && prevAct.bBatchInColumn());
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(prevAct.rows() == m_activations.rows());

			auto& iI = get_iInspect();
			iI.fprop_begin(get_layer_idx(), prevAct, get_common_data().is_training_mode());

			//0. calling fprop() of the gate
			gating_layer().fprop(LLWrapT(prevAct, m_pTmpBiasStorage
				, ::std::get<0>(m_phl_tuple).coord, !layer_tolerates_no_biases<gating_layer_t>::value));
			//adding <values_not_present> feature if necessary
			_fprop_add_no_present_feature();
			//now the first gate_neurons_count columns of m_activations is correct.

			//1. we must calculate batch sizes for inner layers (they depends on a corresponding gating neuron value),
			// prepare individual activations and call on_batch_size_change() for layers
			neurons_count_t ofs = gate_neurons_count, lIdx=0;
			tuple_utils::for_each_exc_first_up(m_phl_tuple, [&prevAct, &act = m_activations, &iM = get_iMath()
				, &ofs, &lIdx, &aPA = m_aPrevActs, &gate = gating_layer().get_activations()](const auto& phl)noexcept
			{
				const real_t*const pG = gate.colDataAsVec(lIdx);
				const vec_len_t nzc = static_cast<vec_len_t>(iM.vCountNonZeros(pG, gate.rows()));

				//updating the storage of rows extracted from curPrevAct
				realmtxdef_t& gatedPrevAct = aPA[lIdx];
				NNTL_ASSERT(!gatedPrevAct.empty() && gatedPrevAct.emulatesBiases() && gatedPrevAct.cols_no_bias() == phl.l.get_incoming_neurons_cnt());
				gatedPrevAct.deform_rows(nzc);

				const auto nc = phl.l.get_neurons_cnt();
				NNTL_ASSERT(ofs + nc <= act.cols_no_bias());
				realmtx_t curAct(act.colDataAsVec(ofs), act.rows(), nc, false);

				if (nzc) {
					//changing the batch size and notifying the layer about it
					phl.l.on_batch_size_change(nzc, nullptr);

					//constructing alias to relevant columns of prevAct
					NNTL_ASSERT(phl.coord.m_offset + phl.coord.m_count <= prevAct.cols_no_bias());
					NNTL_ASSERT(phl.coord.m_count == phl.l.get_incoming_neurons_cnt());
					//const_cast here is just a trick to get necessary pointer. We won't modify the data under it
					const realmtx_t curPrevAct(const_cast<real_t*>(prevAct.colDataAsVec(phl.coord.m_offset))
						, prevAct.rows(), phl.coord.m_count, false);

					// fetching relevant rows into gatedPrevAct
					iM.mExtractRowsByMask(curPrevAct, pG, gatedPrevAct);
					gatedPrevAct.set_biases();

					//doing fprop with gatedPrevAct
					phl.l.fprop(_impl::wrap_trainable_layer<LLWrapT>(gatedPrevAct));

					//pushing the layer's activations to our's activations				
					iM.mFillRowsByMask(phl.l.get_activations(), pG, curAct);
				} else {
					//just zeroing current activations
					curAct.zeros();
				}

				//getting ready to a next layer
				ofs += nc;
				++lIdx;
			});
			NNTL_ASSERT(lIdx == gated_layers_count);
			NNTL_ASSERT(prevAct.test_biases_strict());			
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());

			iI.fprop_activations(m_activations);
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		template<typename LLWrapT>
		unsigned _lpho_bprop(realmtxdef_t& dLdA, realmtxdef_t& dLdAPrev, const realmtx_t& prevAct)noexcept {
			static constexpr bool bPrevLayerWBprop = is_layer_with_bprop<LLWrapT>::value;
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
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

			// BTW, inner layers can't overlap their receptive fields, therefore we may write their dLdAPrev directly into our's dLdAPrev variable

			NNTL_ASSERT(!m_innerdLdA.emulatesBiases() && !m_innerdLdAPrev.emulatesBiases());

			neurons_count_t firstNeuronOfs = get_neurons_cnt(), lIdx = gated_layers_count;
			tuple_utils::for_each_exc_first_down(m_phl_tuple, [&firstNeuronOfs, &dLdA, &dLdAPrev
				, &lIdx, &aPA = m_aPrevActs, &gate = gating_layer().get_activations()
				, &_innerdLdA = m_innerdLdA, &_innerdLdAPrev = m_innerdLdAPrev, _pTmpBiasStorage = m_pTmpBiasStorage
				, &_Math = get_iMath(), bbs = m_biggestIncBS](const auto& phl)
			{
				static constexpr bool bPrevLayerWBprop = is_layer_with_bprop<LLWrapT>::value;
				auto& lyr = phl.l;

				NNTL_ASSERT(lIdx > 0);
				const real_t*const pG = gate.colDataAsVec(--lIdx);

				NNTL_ASSERT(firstNeuronOfs >= lyr.get_neurons_cnt());
				firstNeuronOfs -= lyr.get_neurons_cnt();
				NNTL_ASSERT(phl.coord.m_count == lyr.get_incoming_neurons_cnt());

				const auto& prA = aPA[lIdx];
				NNTL_ASSERT(prA.emulatesBiases());
				NNTL_ASSERT(prA.rows() == _Math.vCountNonZeros(pG, gate.rows()));

				if (prA.rows()) {
					//setting up the _innerdLdA
					_innerdLdA.deform_like_no_bias(lyr.get_activations());
					NNTL_ASSERT(firstNeuronOfs + _innerdLdA.cols() <= dLdA.cols());
					NNTL_ASSERT(_innerdLdA.rows() == prA.rows());
					NNTL_ASSERT(prA.size_no_bias() == mtx_size_t(_innerdLdA.rows(), lyr.get_incoming_neurons_cnt()));
					auto curdLdA = dLdA.submatrix_cols_no_bias(firstNeuronOfs, _innerdLdA.cols());
					_Math.mExtractRowsByMask(curdLdA, pG, _innerdLdA);

					//we also must upscale dLdA to reflect the proper batch size --- should we?
					//_Math.evMulC_ip(_innerdLdA, real_t(dLdA.rows()) / real_t(_innerdLdA.rows()));

					//setting up the _innerdLdAPrev
					if (bPrevLayerWBprop) {
						_innerdLdAPrev.deform(_innerdLdA.rows(), phl.coord.m_count);
					} else _innerdLdAPrev.deform(0, 0);

					const auto switchMtxs = lyr.bprop(_innerdLdA
						, LLWrapT(prA, _pTmpBiasStorage, realmtx_t::sNumel(bbs, lyr.get_incoming_neurons_cnt())), _innerdLdAPrev);

					if (bPrevLayerWBprop) {
						const auto& gatedCurdLdAPrev = switchMtxs ? _innerdLdAPrev : _innerdLdA;
						NNTL_ASSERT(gatedCurdLdAPrev.size() == prA.size_no_bias());

						//saving curdLdAPrev to dLdAPrev
						auto curdLdAPrev = dLdAPrev.submatrix_cols_no_bias(phl.coord.m_offset, phl.coord.m_count);

						_Math.mFillRowsByMask(gatedCurdLdAPrev, pG, curdLdAPrev);
					}
				} else {
					//gate is completely closed and nothing to do here except for zeroing corresponding region of dLdAPrev
					if (bPrevLayerWBprop) {
						auto curdLdAPrev = dLdAPrev.submatrix_cols_no_bias(phl.coord.m_offset, phl.coord.m_count);
						curdLdAPrev.zeros();
					}
				}				
			});
			NNTL_ASSERT(firstNeuronOfs == gate_neurons_count);
			NNTL_ASSERT(prevAct.test_biases_strict());
			
			//doing bprop() for the gating layer
			const auto& gate_phl = ::std::get<0>(m_phl_tuple);
			NNTL_ASSERT(bPrevLayerWBprop || (dLdAPrev.rows() == 0 && dLdAPrev.cols() == 0));
			if (layer_has_trivial_bprop<gating_layer_t>::value) {
				//just resetting corresponding dLdAPrev
				if (bPrevLayerWBprop) {
					auto curdLdAPrev = dLdAPrev.submatrix_cols_no_bias(gate_phl.coord.m_offset, gate_phl.coord.m_count);
					curdLdAPrev.zeros();
				}
				//and don't call bprop at all
			} else {
				//calling bprop()
				m_innerdLdA.deform_like_no_bias(gating_layer().get_activations());
				if (bPrevLayerWBprop) {
					m_innerdLdAPrev.deform(m_innerdLdA.rows(), gate_phl.coord.m_count);
				} else m_innerdLdAPrev.deform(0, 0); 

				const auto switchMtxs = gating_layer().bprop(m_innerdLdA, LLWrapT(prevAct, m_pTmpBiasStorage, gate_phl.coord), m_innerdLdAPrev);
				if (bPrevLayerWBprop) {
					const auto& curdLdAPrev = switchMtxs ? m_innerdLdAPrev : m_innerdLdA;
					::std::memcpy(dLdAPrev.colDataAsVec(gate_phl.coord.m_offset), curdLdAPrev.data(), curdLdAPrev.byte_size());
				}
			}

			iI.bprop_end(dLdAPrev);
			return 1;
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop<real_t>, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			get_self()._lpho_fprop<_impl::wrap_part_trainable_layer<LowerLayer>>(lowerLayer.get_activations());
		}

		template <typename LowerLayer>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			static_assert(!bAssumeFPropOnly, "");
			return get_self()._lpho_bprop<_impl::wrap_part_trainable_layer<LowerLayer>>(dLdA, dLdAPrev, lowerLayer.get_activations());
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _LPHO
	// If you need to derive a new class, derive it from _LPHO (to make static polymorphism work)

	template <bool bAddDataNotPresentFeature, typename ...PHLsT>
	class LPHO final : public _LPHO < LPHO<bAddDataNotPresentFeature, PHLsT...>, bAddDataNotPresentFeature, ::std::tuple<PHLsT...>>
	{
		typedef _LPHO < LPHO<bAddDataNotPresentFeature, PHLsT...>, bAddDataNotPresentFeature, ::std::tuple<PHLsT...>> _base_class_t;
	public:
		~LPHO() noexcept {};
		LPHO(const PHLsT&... phls) noexcept : _base_class_t(nullptr, ::std::make_tuple(phls...)) {};

		LPHO(PHLsT&&... phls) noexcept : _base_class_t(nullptr, ::std::make_tuple(::std::move(phls)...)) {};

		LPHO(const char* pCustomName, const PHLsT&... phls) noexcept : _base_class_t(pCustomName, ::std::make_tuple(phls...)) {};

		LPHO(const char* pCustomName, PHLsT&&... phls) noexcept : _base_class_t(pCustomName, ::std::make_tuple( ::std::move(phls)...)) {};
	};

	//////////////////////////////////////////////////////////////////////////
	template <bool bAddDataNotPresentFeature, typename PHLsTuple>
	class LPHOt final : public _LPHO<LPHOt<bAddDataNotPresentFeature, PHLsTuple>, bAddDataNotPresentFeature, PHLsTuple> {
		typedef _LPHO<LPHOt<bAddDataNotPresentFeature, PHLsTuple>, bAddDataNotPresentFeature, PHLsTuple> _base_class_t;
	public:
		~LPHOt() noexcept {};
		LPHOt(const PHLsTuple& phls) noexcept : _base_class_t(nullptr, phls) {};

		LPHOt(PHLsTuple&& phls) noexcept : _base_class_t(nullptr, ::std::move(phls)) {};

		LPHOt(const char* pCustomName, const PHLsTuple& phls) noexcept : _base_class_t(pCustomName, phls) {};

		LPHOt(const char* pCustomName, PHLsTuple&& phls) noexcept : _base_class_t(pCustomName, ::std::move(phls)) {};
	};

	template<typename PHLsTuple>
	using LPHOt_true = LPHOt<true, PHLsTuple>;

	template<typename PHLsTuple>
	using LPHOt_false = LPHOt<false, PHLsTuple>;
}
