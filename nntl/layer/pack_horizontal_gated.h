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
// 
// For example: let's imagine one have 3 optional feature sets for a problem. How to process them?
// One could make the following source data vector: {g1,g2,g3,x1,x2,x3} where g1,g2 and g3 are binary gate values
// that shows whether the feature data present, and x1,x2 and x3 are feature data sets. Then one may feed gate values
// into a single layer_identity_gate id1 and x1,x2 and x3 feed into corresponding feature detector layers sets 
// fd1,fd2 and fd3 (for example, each feature detector could be a set of 2 fully_connected_layers stacked one
// on top of the other with a layer_pack_vertical). Then these 4 layers (id1, fd1,fd2 and fd3) could be bounded together
// into a single layer_pack_horizontal_gated that uses corresponding neuron of id1 to control the output and
// the learning process of each of {fd1, fd2, fd3}.
// BTW: bias get zeroed only when the layer doesn't share it's activations AND all the gating values are zeros
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
	// 
	template<typename FinalPolymorphChild, int32_t nBinarize1e6, typename PHLsTuple, typename AddendumsTupleT = void>
	class _layer_pack_horizontal_gated : public _layer_pack_horizontal<FinalPolymorphChild, PHLsTuple, AddendumsTupleT>
	{
	private:
		typedef _layer_pack_horizontal<FinalPolymorphChild, PHLsTuple, AddendumsTupleT> _base_class;

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
		// gating matrix has the width equal to the gate width.
		// Each row of the mask has either a value of one, provided that a corresponding gating
		// neuron "is opened", or the value of zero in the other case.
		// We'll allocate it if we expect drop_samples() to be called OR if we had to binarize the gate
		// Else it'll be the gate alias
		realmtxdef_t m_gatingMask;

		//this is a special 1-column mask for biases (we'll update bases only if we don't share the activation storage)
		//It'll be an alias to m_gatingMask iff there's only one gated layer
		realmtxdef_t m_biasGatingMask;

		static constexpr bool is_biasGatingMask_anAlias = 1 == gated_layers_count;

		//if we expect drop_samples() to be called on us, we have to store a mask passed to drop_samples() to apply
		// it later to dLdA columns, that corresponds to the gating layer. It's a single column matrix
		realmtxdef_t m_dropSamplesGatingMask;

		bool m_bDropSamplesWasCalled;

		//this flag differs from the one instantiated in the parent _layer_base. This flag informs whether the
		//_layer_pack_horizontal_gated::drop_samples() could be called (this depends on init(lid)). The flag
		//inside the _layer_base class informs whether the _layer_pack_horizontal::drop_samples() could be called (it'll be set to true
		// by our's init() procedure). We won't redefine the .is_drop_samples_mbc() here and leave it as is.
		// Derived classes must choose the proper variable depending on the task
		bool m_bIsGatedDropSamplesMightBeCalled;

		friend class _impl::_preinit_layers;

	private:
		template<bool b> struct _defNameS {};
		template<> struct _defNameS<true> { static constexpr const char n[] = "lphg"; };
		template<> struct _defNameS<false> { static constexpr const char n[] = "lphgfi"; };
	public:
		//#BUGBUG this trick doesn't work, _defName remains NULLed!
		static constexpr const char _defName[sizeof(_defNameS<sbBinarizeGate>::n)] = _defNameS<sbBinarizeGate>::n;

		~_layer_pack_horizontal_gated()noexcept {}
		_layer_pack_horizontal_gated(const char* pCustomName, const PHLsTuple& phls)noexcept : _base_class(pCustomName, phls)
			, m_bIsGatedDropSamplesMightBeCalled(false), m_bDropSamplesWasCalled(false)
		{
			//at this point all neuron counts in gating layers must be initialized and therefore we can check whether
			//the whole m_phl_tuple pack has correct number of layers
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == gated_layers_count);
			m_gatingMask.dont_emulate_biases();
			m_biasGatingMask.dont_emulate_biases();
			m_dropSamplesGatingMask.dont_emulate_biases();
		}

		const gating_layer_t& gating_layer()const noexcept { return get_self().first_layer(); }

		void get_gating_info(_impl::GatingContext<real_t>& ctx)const noexcept {
			ctx.pGatingMask = &m_gatingMask;
			ctx.colsDescr.clear();
			ctx.nongatedIds.clear();
			ctx.nongatedIds.insert(gating_layer().get_layer_idx());
			vec_len_t cIdx = 0;
			tuple_utils::for_each_exc_first_up(m_phl_tuple, [&cIdx, &cDescr = ctx.colsDescr](const auto& phl)noexcept {
				cDescr[phl.l.get_layer_idx()] = cIdx++;
			});
		}

		//#TODO returns a loss function summand, that's caused by this layer. Should take gating mask into account (and
		//that's a bit of issue)
		//real_t lossAddendum()const noexcept {
			//#BUGBUG loss values could depend on activation values, therefore depend on gating mask value.
			//Though we can skip this little(?) bug now
			//return m_undLayer.lossAddendum();
		//}

	protected:
		const bool _bAllocateGatingMask()const noexcept {
			return sbBinarizeGate || m_bIsGatedDropSamplesMightBeCalled;
		}
		const bool _bApplyGateToBiases()const noexcept {
			return !get_self().is_activations_shared();
		}

	public:
		//////////////////////////////////////////////////////////////////////////
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == gated_layers_count);

			m_bIsGatedDropSamplesMightBeCalled = lid.bDropSamplesMightBeCalled;
			lid.bDropSamplesMightBeCalled = true;
			auto ec = _base_class::init(lid, pNewActivationStorage);
			lid.bDropSamplesMightBeCalled = m_bIsGatedDropSamplesMightBeCalled;
			if (ErrorCode::Success != ec) return ec;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			//we must resize gatingMask here to the size of underlying_layer activations, however gatingMask mustn't
			//have an emulated bias column
			NNTL_ASSERT(get_self().get_neurons_cnt() > gated_layers_count);
			const auto biggestBatchSize = get_self().get_common_data().biggest_batch_size();

			NNTL_ASSERT(!m_gatingMask.emulatesBiases() && m_gatingMask.empty());
			if (_bAllocateGatingMask()) {
				if (!m_gatingMask.resize(biggestBatchSize, gated_layers_count))
					return ErrorCode::CantAllocateMemoryForGatingMask;
			} //else //we'll use it as an alias to the gate
			
			NNTL_ASSERT(!m_biasGatingMask.emulatesBiases() && m_biasGatingMask.empty());
			if (_bApplyGateToBiases()) {
				if (is_biasGatingMask_anAlias) {
					NNTL_ASSERT(1 == m_gatingMask.cols());
					//we'll initialize it later, when the m_gatingMask will surely be available
				} else {
					if (!m_biasGatingMask.resize(biggestBatchSize, 1))
						return ErrorCode::CantAllocateMemoryForGatingMask;
				}
			}

			m_bDropSamplesWasCalled = false;
			if (m_bIsGatedDropSamplesMightBeCalled) {
				//m_dropSamplesGatingMask will be applied only during bprop(), therefore it's size should rely on training_batch_size()
				if (!m_dropSamplesGatingMask.resize(get_self().get_common_data().training_batch_size(),1))
					return ErrorCode::CantAllocateMemoryForGatingMask;
			}

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit() noexcept {
			m_bIsGatedDropSamplesMightBeCalled = false;
			m_bDropSamplesWasCalled = false;
			m_dropSamplesGatingMask.clear();
			m_gatingMask.clear();
			m_biasGatingMask.clear();
			_base_class::deinit();
		}

		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			_base_class::on_batch_size_change(pNewActivationStorage);

			const vec_len_t batchSize = get_self().get_common_data().get_cur_batch_size();
			if (_bAllocateGatingMask()) {
				NNTL_ASSERT(!m_gatingMask.empty() && !m_gatingMask.bDontManageStorage());
				//we must deform the mask to fit new underlying activations size
				NNTL_ASSERT(batchSize == get_self().get_activations_size().first);
				m_gatingMask.deform_rows(batchSize);
				NNTL_ASSERT(m_gatingMask.cols() == gated_layers_count);
			}else{
				NNTL_ASSERT(m_gatingMask.empty() || m_gatingMask.bDontManageStorage());
			}
			
			if (_bApplyGateToBiases()) {
				NNTL_ASSERT(!m_biasGatingMask.emulatesBiases());
				if (is_biasGatingMask_anAlias) {
					NNTL_ASSERT(m_biasGatingMask.empty() || (m_biasGatingMask.bDontManageStorage() && 1 == m_biasGatingMask.cols()));
				} else {
					NNTL_ASSERT(!m_biasGatingMask.empty() && !m_biasGatingMask.bDontManageStorage() && 1 == m_biasGatingMask.cols());
					m_biasGatingMask.deform_rows(batchSize);
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// gating functions
	protected:
		void reinit_biasGatingMask()noexcept {
			if (_bApplyGateToBiases()) {
				NNTL_ASSERT(!m_biasGatingMask.emulatesBiases() && m_biasGatingMask.cols() == 1);
				if (is_biasGatingMask_anAlias) {
					NNTL_ASSERT(m_biasGatingMask.empty() || m_biasGatingMask.bDontManageStorage());
					NNTL_ASSERT(m_gatingMask.cols() == 1);
					m_biasGatingMask.useExternalStorage(m_gatingMask);
				} else {
					NNTL_ASSERT(!m_biasGatingMask.bDontManageStorage());
					NNTL_ASSERT(m_biasGatingMask.rows() == m_gatingMask.rows() && 1 == m_biasGatingMask.cols());
					//now we must create m_biasGatingMask using m_gatingMask
					//safe to use OR here, because there will be only two types of bit patterns, that corresponds to
					// real_t(1.) and real_t(0.). The later one actually consists of zero bits.
					get_self().get_iMath().mrwOr(m_gatingMask, m_biasGatingMask.data());
				}
				NNTL_ASSERT(m_biasGatingMask.isBinary());
			}
		}

		// version without binarization
		template<bool bg = sbBinarizeGate>
		std::enable_if_t< !bg, self_ref_t> make_gating_mask()noexcept {
			const auto& gate = get_self().gating_layer().get_gate();

			NNTL_ASSERT(gate.isBinary());
			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == gate.cols() && gate.cols() == gated_layers_count);
			NNTL_ASSERT(gate.rows() == get_activations().rows());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());

			if (_bAllocateGatingMask()) {
				NNTL_ASSERT(!m_gatingMask.empty() && !m_gatingMask.bDontManageStorage());
				NNTL_ASSERT(gate.size() == m_gatingMask.size());
				gate.copy_to(m_gatingMask);
			} else {
				NNTL_ASSERT(m_gatingMask.empty() || m_gatingMask.bDontManageStorage());
				//we won't update the gate here, it's just an alias, therefore const_cast is safe
				m_gatingMask.useExternalStorage(const_cast<realmtx_t&>(gate));
			}
			
			//finally we must update m_biasGatingMask -- it'll be used if we don't share ours activation matrix
			reinit_biasGatingMask();

			return get_self();
		}
		// version with binarization
		template<bool bg = sbBinarizeGate>
		std::enable_if_t<bg, self_ref_t> make_gating_mask()noexcept {
			NNTL_ASSERT(_bAllocateGatingMask());
			const auto& gate = get_self().gating_layer().get_gate();

			NNTL_ASSERT(get_self().gating_layer().get_gate_width() == gate.cols() && gate.cols() == gated_layers_count);
			NNTL_ASSERT(gate.rows() == get_activations().rows());
			NNTL_ASSERT(!m_gatingMask.emulatesBiases());
			NNTL_ASSERT(!m_gatingMask.empty() && !m_gatingMask.bDontManageStorage());
			NNTL_ASSERT(gate.size() == m_gatingMask.size());

			get_self().get_iMath().ewBinarize(m_gatingMask, gate, sBinarizeFrac);

			//finally we must update m_biasGatingMask -- it'll be used if we don't share ours activation matrix
			reinit_biasGatingMask();

			return get_self();
		}

		//applies gating mask to a matrix (skipping matrix cols, that are used by gating layer itself)
		//A must have a size of dL/dA (which has one column less than underlying activations)
		void drop_dLdA_by_mask(realmtx_t& dLdA) noexcept {
			NNTL_ASSERT(!dLdA.empty() && !dLdA.emulatesBiases() && !m_gatingMask.empty() & !m_gatingMask.emulatesBiases());			
			NNTL_ASSERT(dLdA.rows() == m_gatingMask.rows());
			NNTL_ASSERT(dLdA.cols() == get_self().get_neurons_cnt());
			NNTL_ASSERT(m_gatingMask.cols() == gated_layers_count);
			NNTL_ASSERT(m_gatingMask.isBinary());

			realmtx_t ind_dLdA;
			vec_len_t ofs = gating_layer().get_neurons_cnt(), lNum=0;
			auto &iM = get_self().get_iMath();

			NNTL_ASSERT(get_self().get_common_data().is_training_mode());
			if (m_bIsGatedDropSamplesMightBeCalled && m_bDropSamplesWasCalled) {
				NNTL_ASSERT(!m_dropSamplesGatingMask.empty() && m_dropSamplesGatingMask.rows() == dLdA.rows() && 1 == m_dropSamplesGatingMask.cols());
				//we must also apply mask to corresponding dLdA columns
				ind_dLdA.useExternalStorage(dLdA.data(), dLdA.rows(), ofs, false);
				iM.mrwMulByVec(ind_dLdA, m_dropSamplesGatingMask.data());
			} else {
				NNTL_ASSERT(!m_bDropSamplesWasCalled || m_bIsGatedDropSamplesMightBeCalled);
			}

			tuple_utils::for_each_exc_first_up(m_phl_tuple
				, [&ind_dLdA, &dLdA, &ofs, &lNum, &mask = this->m_gatingMask, &iM](const auto& phl)
			{
				const auto nc = phl.l.get_neurons_cnt();
				ind_dLdA.useExternalStorage(dLdA.colDataAsVec(ofs), dLdA.rows(), nc, false);
				ofs += nc;
				iM.mrwMulByVec(ind_dLdA, mask.colDataAsVec(lNum++));
			});
		}

		// applies gating mask to activations
		void drop_samples_by_mask(const bool bApplyToBiases) noexcept {
			NNTL_ASSERT(_bApplyGateToBiases() || !bApplyToBiases);
			NNTL_ASSERT(m_gatingMask.isBinary() && m_gatingMask.cols() == gated_layers_count);

			realmtx_t rMask;
			vec_len_t lNum = 0;
			tuple_utils::for_each_exc_first_up(m_phl_tuple
				, [&lNum, &rMask, &mask = this->m_gatingMask, &iM = get_self().get_iMath()](const auto& phl)
			{
				rMask.useExternalStorage(mask.colDataAsVec(lNum++), mask.rows(), 1, false);
				phl.l.drop_samples(rMask, false);
			});

			if (bApplyToBiases) {
				//applying bias mask to biases
				NNTL_ASSERT(!m_biasGatingMask.empty() && m_biasGatingMask.isBinary() && 1 == m_biasGatingMask.cols());
				m_activations.copy_biases_from(m_biasGatingMask.data());
			}

			//ensure the bias column has correct flags
			NNTL_ASSERT((_bApplyGateToBiases() && m_activations.isHoleyBiases() && !is_activations_shared() && m_activations.test_biases_holey()
				) || (!_bApplyGateToBiases() && !m_activations.isHoleyBiases() && (is_activations_shared() || m_activations.test_biases_strict())));
		}

		void finish_fprop(const bool bApplyToBiases)noexcept {
			get_self()
				.make_gating_mask<>()
				.drop_samples_by_mask(bApplyToBiases);
		}

	public:
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().get_common_data().is_training_mode());

			if (m_bIsGatedDropSamplesMightBeCalled && get_self().get_common_data().is_training_mode()) {
				m_bDropSamplesWasCalled = false;
			}

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

 			NNTL_ASSERT(!_bAllocateGatingMask() || (!m_gatingMask.empty() && m_gatingMask.rows() == m_activations.rows() 
					&& m_gatingMask.rows() == lowerLayer.get_activations().rows()
					&& m_gatingMask.cols() == get_self().gating_layer().get_gate_width()));

			
			_base_class::fprop(lowerLayer);
			iI.fprop_activations(m_activations);

			get_self().finish_fprop(_bApplyGateToBiases());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.fprop_end(m_activations);
		}

		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			NNTL_ASSERT(get_self().get_common_data().is_training_mode());
			NNTL_ASSERT(m_gatingMask.rows() == get_self().get_activations().rows());
			NNTL_ASSERT(m_gatingMask.rows() == lowerLayer.get_activations().rows());
			NNTL_ASSERT(m_gatingMask.cols() == get_self().gating_layer().get_gate_width());

			NNTL_ASSERT(get_self().get_activations().size_no_bias() == dLdA.size());
			NNTL_ASSERT((std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			get_self().drop_dLdA_by_mask(dLdA);

			iI.bprop_finaldLdA(dLdA);

			const unsigned ret = _base_class::bprop(dLdA, lowerLayer, dLdAPrev);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}

		static constexpr bool is_trivial_drop_samples()noexcept { return false; }

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT(m_bIsGatedDropSamplesMightBeCalled && _bAllocateGatingMask());
			//it should not be possible when we're aren't allowed to update biases, while getting a request to update them...
			NNTL_ASSERT(_bApplyGateToBiases() || !bBiasesToo);
			NNTL_ASSERT(m_gatingMask.isBinary());
			NNTL_ASSERT(mask.isBinary() && 1 == mask.cols());
			
			if (get_self().get_common_data().is_training_mode()) {
				m_bDropSamplesWasCalled = true;
				NNTL_ASSERT(!m_dropSamplesGatingMask.empty());
				m_dropSamplesGatingMask.deform_like(mask);
				mask.copy_to(m_dropSamplesGatingMask);
			}

			//first we must update the gating layer activations
			get_self().gating_layer().drop_samples(mask, false);

			//then - rebuild the gating mask based on the update and then reapply it to layer activations
			get_self().finish_fprop(bBiasesToo);
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
		: public _layer_pack_horizontal_gated<LPHG<PHLsT...>, 500000, std::tuple<PHLsT...>>
	{
	public:
		~LPHG() noexcept {};
		LPHG(PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHG<PHLsT...>, 500000, std::tuple<PHLsT...>>(nullptr, std::make_tuple(phls...)) {};
		LPHG(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHG<PHLsT...>, 500000, std::tuple<PHLsT...>>(pCustomName, std::make_tuple(phls...)) {};
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
	//////////////////////////////////////////////////////////////////////////

	template <typename ...PHLsT>
	class LPHGFI final
		: public _layer_pack_horizontal_gated<LPHGFI<PHLsT...>, 0, std::tuple<PHLsT...>>
	{
	public:
		~LPHGFI() noexcept {};
		LPHGFI(PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHGFI<PHLsT...>, 0, std::tuple<PHLsT...>>(nullptr, std::make_tuple(phls...)) {};
		LPHGFI(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHGFI<PHLsT...>, 0, std::tuple<PHLsT...>>(pCustomName, std::make_tuple(phls...)) {};
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

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	
	template <typename AddendumsTupleT, typename ...PHLsT>
	class LPHG_PA final
		: public _layer_pack_horizontal_gated<LPHG_PA<AddendumsTupleT, PHLsT...>, 500000, std::tuple<PHLsT...>, AddendumsTupleT>
	{
	public:
		~LPHG_PA() noexcept {};
		LPHG_PA(PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHG_PA<AddendumsTupleT, PHLsT...>, 500000, std::tuple<PHLsT...>, AddendumsTupleT>(nullptr, std::make_tuple(phls...)) {};
		LPHG_PA(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHG_PA<AddendumsTupleT, PHLsT...>, 500000, std::tuple<PHLsT...>, AddendumsTupleT>(pCustomName, std::make_tuple(phls...)) {};
	};

	template <typename AddendumsTupleT, typename ...PHLsT>
	class LPHGFI_PA final
		: public _layer_pack_horizontal_gated<LPHGFI_PA<AddendumsTupleT, PHLsT...>, 0, std::tuple<PHLsT...>, AddendumsTupleT>
	{
	public:
		~LPHGFI_PA() noexcept {};
		LPHGFI_PA(PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHGFI_PA<AddendumsTupleT, PHLsT...>, 0, std::tuple<PHLsT...>, AddendumsTupleT>(nullptr, std::make_tuple(phls...)) {};
		LPHGFI_PA(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal_gated<LPHGFI_PA<AddendumsTupleT, PHLsT...>, 0, std::tuple<PHLsT...>, AddendumsTupleT>(pCustomName, std::make_tuple(phls...)) {};
	};
}