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

// layer_pack_horizontal (LPH) provides a way to concatenate activation matrices of a set of layers into a single
// activation matrix, i.e. gather a set of layers into a single layer.
// Moreover, LPH allows to feed different ranges of
// underlying activation units into different set of layers.
// 
//    \  |  |  |  |     |  |  |  | /
// |------layer_pack_horizontal-------|
// |  \  |  |  |  |  .  |  |  |  | /  |
// |   |--layer1--|  .  |--layerN--|  | 
// |    / | | | | |  .  | | | | | \   |
// |----------------------------------|
//      / | | | | |  .  | | | | | \
//
// 
#include "_pack_.h"
#include "../utils.h"

#include "_penalized_activations_base.h"

namespace nntl {
	
	//AddendumsTupleT was introduced only to overcome compiler bug when using _layer_penalized_activations<> wrapper

	template<typename FinalPolymorphChild, typename PHLsTuple, typename AddendumsTupleT = void>
	class _layer_pack_horizontal 
		: public _layer_base<FinalPolymorphChild, typename ::std::remove_reference<typename ::std::tuple_element<0, PHLsTuple>::type>::type::phl_original_t::interfaces_t>
		, public _penalized_activations_base_selector<AddendumsTupleT>
	{
	private:
		typedef _layer_base<FinalPolymorphChild, typename ::std::remove_reference<typename ::std::tuple_element<0, PHLsTuple>
			::type>::type::phl_original_t::interfaces_t> _base_class_t;

	public:
		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;

		using _base_class_t::real_t;
		using _base_class_t::realmtx_t;
		using _base_class_t::realmtxdef_t;

		//typedef const ::std::tuple<PHLsT...> _phl_tuple;
		typedef const PHLsTuple _phl_tuple;

		//static constexpr size_t phl_count = sizeof...(PHLsT);
		static constexpr size_t phl_count = ::std::tuple_size<PHLsTuple>::value;
		static_assert(phl_count > 1, "For a pack with a single inner layer use that layer instead");

		static_assert(is_PHL<typename ::std::remove_reference<typename ::std::tuple_element<0, _phl_tuple>::type>::type>::value, "_layer_pack_horizontal must be assembled from PHL objects!");
		static_assert(is_PHL<typename ::std::remove_reference<typename ::std::tuple_element<phl_count - 1, _phl_tuple>::type>::type>::value, "_layer_pack_horizontal must be assembled from PHL objects!");

		typedef typename ::std::remove_reference<typename ::std::tuple_element<0, _phl_tuple>::type>::type::phl_original_t first_layer_t;
		typedef typename ::std::remove_reference<typename ::std::tuple_element<phl_count - 1, _phl_tuple>::type>::type::phl_original_t last_layer_t;

	protected:
		_phl_tuple m_phl_tuple;
		realmtxdef_t m_activations;//its content assembled from individual activations of inner layers in-place

		// addresses at max m_max_fprop_batch_size elements to store a column of data of previous layer activation,
		// that will be replaced by biases during fprop()/bprop() phase.
		real_t* m_pTmpBiasStorage;


		numel_cnt_t m_layers_max_dLdA_numel;//max lid.max_dLdA_numel gathered from inner layers during init() phase. This value is used 
		// to allocate dLdA matricies to pass to layers bprop()

		realmtxdef_t m_innerdLdA, m_innerdLdAPrev;
				
		//////////////////////////////////////////////////////////////////////////
		//
	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(inc_neurons_cnt > 0);

			_base_class_t::_preinit_layer(ili, inc_neurons_cnt);

			_impl::_preinit_layers initializer(ili, inc_neurons_cnt);
			if (initializer.preparePHLCheck()) {
				tuple_utils::for_each_up(m_phl_tuple, initializer);
				if (!initializer.PHLCheck()) {
					NNTL_ASSERT(!"All lower layer activations must be covered by a set of inner layers of _layer_pack_horizontal!");
					//#todo: probably need a better way to return error
					abort();
				}
			} else {
				NNTL_ASSERT(!"Failed to prepare for PHLCheck / No memory, probably");
				//#todo: probably need a better way to return error
				abort();
			}
		}

		//NB: first/last means only the first or the last ELEMENT of the m_phl_tuple. It has nothing in common with neurons range.
		//Neurons range determined by PHLs of m_phl_tuple only.
		first_layer_t& first_layer()const noexcept { return ::std::get<0>(m_phl_tuple).l; }
		last_layer_t& last_layer()const noexcept { return ::std::get<phl_count - 1>(m_phl_tuple).l; }
		
	private:
		static const neurons_count_t _calcNeuronsCnt(const _phl_tuple& tupl) noexcept {
			neurons_count_t nc = 0;
			tuple_utils::for_each_up(tupl, [&nc](const auto& phl)noexcept {
				typedef ::std::remove_reference_t<decltype(phl)> PHL_t;
				typedef typename PHL_t::phl_original_t Layer_t;

				static_assert(::std::is_same<real_t, typename Layer_t::real_t>::value, "Invalid real_t");
				static_assert(is_PHL<PHL_t>::value, "_layer_pack_horizontal must be assembled from PHL objects!");
				static_assert(!is_layer_input<Layer_t>::value && !is_layer_output<Layer_t>::value
					, "Inner layers of _layer_pack_horizontal mustn't be input or output layers!");
				static_assert(::std::is_base_of<_i_layer<real_t>, Layer_t>::value, "Each layer must derive from i_layer");

				nc += phl.l.get_neurons_cnt();
			});
			return nc;
		}

	public:
		~_layer_pack_horizontal()noexcept {}
		_layer_pack_horizontal(const char* pCustomName, const PHLsTuple& phls)noexcept
			: _base_class_t(_calcNeuronsCnt(phls),pCustomName)
			, m_phl_tuple(phls), m_pTmpBiasStorage(nullptr), m_layers_max_dLdA_numel(0)
		{
			m_activations.will_emulate_biases();
		}
		static constexpr const char _defName[] = "lph";

		const realmtxdef_t& get_activations()const noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			return m_activations;
		}
		const realmtxdef_t* get_activations_storage()const noexcept { return &m_activations; }
		const mtx_size_t get_activations_size()const noexcept { return m_activations.size(); }

		const bool is_activations_shared()const noexcept {
			const auto r = _base_class_t::is_activations_shared();
			NNTL_ASSERT(!r || m_activations.bDontManageStorage());//shared activations can't manage their own storage
			return r;
		}
		
		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_up(m_phl_tuple, [&func{ ::std::forward<_Func>(f) }](auto& phl)noexcept {
				call_F_for_each_layer(::std::forward<_Func>(func), phl.l);
			});
		}

		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			tuple_utils::for_each_down(m_phl_tuple, [&func{ ::std::forward<_Func>(f) }](auto& phl)noexcept {
				call_F_for_each_layer_down(::std::forward<_Func>(func), phl.l);
			});
		}

		//This will apply f to every layer, packed in tuple no matter whether it is a _pack_* kind of layer or no
		template<typename _Func>
		void for_each_packed_layer(_Func&& f)const noexcept {
			tuple_utils::for_each_up(m_phl_tuple, [&func{ ::std::forward<_Func>(f) }](auto& phl)noexcept {
				::std::forward<_Func>(func)(phl.l);
			});
		}		
		template<typename _Func>
		void for_each_packed_layer_down(_Func&& f)const noexcept {
			tuple_utils::for_each_down(m_phl_tuple, [&func{ ::std::forward<_Func>(f) }](auto& phl)noexcept {
				::std::forward<_Func>(func)(phl.l);
			});
		}

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept {
			bool b = _pab_hasLossAddendum();
			if (!b) {
				get_self().for_each_packed_layer([&b](auto& l) {				b |= l.hasLossAddendum();			});
			}
			return b;
		}
		//returns a loss function summand, that's caused by this layer
		real_t lossAddendum()const noexcept {
			real_t la(.0);
			get_self().for_each_packed_layer([&la](auto& l) {				la += l.lossAddendum();			});
			return la + _pab_lossAddendum(get_self().get_activations(), get_self().get_iMath());
		}

	/*
	 *deprecated:
	 *protected:
		template<typename InnerLayerT>
		::std::enable_if_t<_impl::layer_has_OuterLayerCustomFlag1Eval<InnerLayerT, decltype(m_phl_tuple), _layer_init_data_t>::value,bool>
			call_layers_OuterLayerCustomFlag1Eval(const InnerLayerT& lyr, const _layer_init_data_t& origLid)noexcept
		{
			const auto v = lyr.OuterLayerCustomFlag1Eval(m_phl_tuple, origLid);
			//STDCOUTL("OuterLayerCustomFlag1Eval for " << lyr.get_layer_name_str() << " returned " << v);
			return v;
		}

		template<typename InnerLayerT>
		::std::enable_if_t<!_impl::layer_has_OuterLayerCustomFlag1Eval<InnerLayerT, decltype(m_phl_tuple), _layer_init_data_t>::value, constexpr bool>
			call_layers_OuterLayerCustomFlag1Eval(const InnerLayerT& lyr, const _layer_init_data_t&)noexcept
		{
			//return ((STDCOUTL("returning default for " << lyr.get_layer_name_str())), false);
			return false;
		}*/

	public:
		//////////////////////////////////////////////////////////////////////////
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			auto ec = _base_class_t::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			const auto biggestBatchSize = get_self().get_common_data().biggest_batch_size();

			//allocating m_activations
			NNTL_ASSERT(m_activations.emulatesBiases());
			if (pNewActivationStorage) {
				m_activations.useExternalStorage(pNewActivationStorage, biggestBatchSize, get_self().get_neurons_cnt() + 1, true);
			} else {
				if (!m_activations.resize(biggestBatchSize, get_self().get_neurons_cnt()))
					return ErrorCode::CantAllocateMemoryForActivations;
			}

			//now we must initialize encapsulated layers and that's a tricky part.
			// - we can't pass dLdA & dLdAPrev to these layers, because they could also be compound and may require
			//		a far bigger matrix sizes than dLdA allows. Therefore we must calculate the biggest max_dLdA_numel 
			//		for these layers (including dLdAPrev!!!) and allocate 2 corresponding matrices to be used as dLdA & dLdAPrev.
			//		However, ours .max_dLdA_numel should be set to a normal value for a simple layer [batchSize, neuronsCount]
			// - We should aggregate (by max()) layer's initMem() requirements and add them to ours
			NNTL_ASSERT(0 == lid.max_dLdA_numel && 0 == lid.maxMemFPropRequire && 0 == lid.maxMemTrainingRequire);

			_layer_init_data_t origLid = lid;

			const auto bLidActShSp = lid.bActivationsShareSpace;
			lid.bActivationsShareSpace = true;

			layer_index_t failedLayerIdx = 0;
			auto initD = lid.dupe();
			neurons_count_t firstNeuronOfs = 0, maxIncNeuronsCnt = 0;//we need maxIncNeuronsCnt to calculate the biggest possible internal dLdAPrev
			get_self().for_each_packed_layer([&, &act = m_activations](auto& l)noexcept {
				if (ErrorCode::Success == ec) {
					maxIncNeuronsCnt = ::std::max(maxIncNeuronsCnt, l.get_incoming_neurons_cnt());

					initD.clean_using(lid);//we must propagate any IN flags set in the .lid variable to the layer being initialized.
					//initD.bLPH_CustomFlag1 = call_layers_OuterLayerCustomFlag1Eval(l, origLid);
					ec = l.init(initD, act.colDataAsVec(firstNeuronOfs));
					if (ErrorCode::Success == ec) {
						lid.update(initD);
						firstNeuronOfs += l.get_neurons_cnt();
					} else failedLayerIdx = l.get_layer_idx();
				}
			});
			NNTL_ASSERT(firstNeuronOfs + 1 == m_activations.cols());
			lid.bActivationsShareSpace = bLidActShSp;

			if (ErrorCode::Success == ec) {
				//Now we should append our own training requirements (reqs of this (top-level) LPH layer)

				//we'll need a column-vector of length biggestBatchSize to store a data column of previous layer activation,
				// that will be substituted by biases for fprop() of one inner layer
				lid.maxMemFPropRequire += biggestBatchSize;

				if (get_self().get_common_data().is_training_possible()) {
					const auto& _training_batch_size = get_self().get_common_data().training_batch_size();
					//adding backprop requirements
					// saving lid.max_dLdA_numel, gathered from inner layers and substituting it for this layer's max_dLdA_numel
					m_layers_max_dLdA_numel = ::std::max(lid.max_dLdA_numel
						, realmtx_t::sNumel(_training_batch_size, maxIncNeuronsCnt));
					//The first argument of max() - lid.max_dLdA_numel - describes the biggest dLdA size, the second
					//argument (realmtx_t::sNumel(get_self().training_batch_size(), maxIncNeuronsCnt)) describes the biggest
					// dLdAPrev.

					//The biggest "outside" dLdA is limited to the ours activation matrix size (we can't send dLdA passed as bprop() argument
					//to encapsulated layer's bprop())
					lid.max_dLdA_numel = realmtx_t::sNumel(_training_batch_size, get_self().get_neurons_cnt());
					
					//reserving memory (additional to what encapsulated layers require) for two inner dLdA matrices
					// AND a column-vector of length biggestBatchSize to store a data column of previous layer activation
					lid.maxMemTrainingRequire += biggestBatchSize + 2 * m_layers_max_dLdA_numel;
				}
			}

			if (ErrorCode::Success == ec) bSuccessfullyInitialized = true;

			lid.bLossAddendumDependsOnActivations = can_penalize_activations<FinalPolymorphChild>::value;

			//#TODO need some way to return failedLayerIdx
			return ec;
		}

		void deinit() noexcept {
			get_self().for_each_packed_layer([](auto& l) {l.deinit(); });
			m_activations.clear();
			m_pTmpBiasStorage = nullptr;
			m_layers_max_dLdA_numel = 0;
			m_innerdLdA.clear();
			m_innerdLdAPrev.clear();
			_base_class_t::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			//for fprop()
			const auto _biggest_batch_size = static_cast<numel_cnt_t>(get_self().get_common_data().biggest_batch_size());
			NNTL_ASSERT(ptr && cnt >= _biggest_batch_size);
			m_pTmpBiasStorage = ptr;
			ptr += _biggest_batch_size;
			cnt -= _biggest_batch_size;

			if (get_self().get_common_data().is_training_possible()) {
				NNTL_ASSERT(cnt >= 2 * m_layers_max_dLdA_numel);
				m_innerdLdA.useExternalStorage(ptr, m_layers_max_dLdA_numel, false);
				ptr += m_layers_max_dLdA_numel;
				m_innerdLdAPrev.useExternalStorage(ptr, m_layers_max_dLdA_numel, false);
				ptr += m_layers_max_dLdA_numel;
				cnt -= 2 * m_layers_max_dLdA_numel;
			}

			get_self().for_each_packed_layer([=](auto& l) {l.initMem(ptr, cnt); });
		}

		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			const vec_len_t batchSize = get_self().get_common_data().get_cur_batch_size();
			NNTL_ASSERT(batchSize > 0 && batchSize <= get_self().get_common_data().biggest_batch_size());
			NNTL_ASSERT(m_activations.emulatesBiases());
			// now we must resize m_activations and update activations of inner layers with on_batch_size_change variation
			m_bActivationsValid = false;

			bool bRestoreBiases;
			if (pNewActivationStorage) {
				NNTL_ASSERT(m_activations.bDontManageStorage());
				//m_neurons_cnt + 1 for biases
				m_activations.useExternalStorage(pNewActivationStorage, batchSize, get_self().get_neurons_cnt() + 1, true);
				//should not restore biases here, because for compound layers its a job for their fprop() implementation
				bRestoreBiases = false;
			} else {
				NNTL_ASSERT(!m_activations.bDontManageStorage());
				m_activations.deform_rows(batchSize);
				bRestoreBiases = batchSize != get_self().get_common_data().biggest_batch_size();
			}

			neurons_count_t firstNeuronOfs = 0;
			get_self().for_each_packed_layer([&act = m_activations, &firstNeuronOfs](auto& lyr)noexcept {
				//we're just setting memory to store activation values of inner layers here.
				//there's no need to play with biases here.
				lyr.on_batch_size_change(act.colDataAsVec(firstNeuronOfs));
				firstNeuronOfs += lyr.get_neurons_cnt();
			});
			NNTL_ASSERT(firstNeuronOfs + 1 == m_activations.cols());

			if (bRestoreBiases) m_activations.set_biases();			
			NNTL_ASSERT(pNewActivationStorage || m_activations.test_biases_ok());
		}

		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop<real_t>, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().get_common_data().is_training_mode());

			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());

			//restoring biases, should they were altered in drop_samples()
			if (m_activations.isHoleyBiases() && !get_self().is_activations_shared()) {
				m_activations.set_biases();
			}

			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			tuple_utils::for_each_up(m_phl_tuple, [&act = lowerLayer.get_activations(), pTmpBiasStorage = m_pTmpBiasStorage](const auto& phl) {
				phl.l.fprop(_impl::trainable_partial_layer_wrapper<LowerLayer>(act, pTmpBiasStorage, phl.coord));
			});
			
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.fprop_activations(m_activations);
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		//redefine in derived class if necessary
		void _applydLdAPenalty(realmtx_t& dLdA)noexcept {
			_pab_update_dLdA(dLdA, get_self().get_activations(), get_self().get_iMath(), get_self().get_iInspect());
		}

		// in order to implement backprop for the inner layers, we must provide them with a correct dLdA and dLdAPrev, each of which must
		// address at least _layer_init_data_t::max_dLdA_numel elements, that layers returned during init() phase.
		// Some things to consider:
		// - there might be a compound layer in m_phl_tuple (such as layer_pack_vertical). That means, that it may require a far bigger
		// max_dLdA_numel, than might be provided by slicing dLdA passed to this function as argument to corresponding parts. So we'll
		// need a way to protect out-of-slice data from being overwritten by layer.bprop() (because we allow layer.brop() to use dLdA&dLdAPrev
		// almost without restrictions)
		// - inner layers may use the same (intersecting) lowerLayer activation units (because we don't require inner layers to use different
		// lower layer activations). That means, that after we'll get their individual dLdAPrev, we must aggregate them into resulting dLdAPrev.
		// 
		// Therefore it looks much more safe to allocate and use for inner layers special dLdA and dLdAPrev matrices, that are independent
		// from dLdA&dLdAPrev, passed to this function as argument. It's possible however to reuse passed dLdA&dLdAPrev for that task, but
		// it would require significantly more complicated and error-prone code to keep all data safe
		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			NNTL_ASSERT(m_bActivationsValid);

			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			get_self()._applydLdAPenalty(dLdA);

			iI.bprop_finaldLdA(dLdA);

			m_bActivationsValid = false;

			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());
			NNTL_ASSERT(get_self().get_common_data().is_training_mode());
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());
			//NNTL_ASSERT(m_activations.test_biases_ok());
			NNTL_ASSERT(dLdA.size() == m_activations.size_no_bias());
			NNTL_ASSERT((::std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());
			
			// We'll copy corresponding parts of dLdA into m_innerdLdA and on inner layer.bprop() return we'll ADD corresponding dLdA to dLdAPrev passed
			if (!::std::is_base_of<m_layer_input, LowerLayer>::value) dLdAPrev.zeros();
			
			neurons_count_t firstNeuronOfs = get_self().get_neurons_cnt();
			
			//The order of traversing is EXTREMELY IMPORTANT for gating layers, for example (they might expect a gating layer to be
			// processed first during fprop() and last during bprop()). Therefore we must go backwards here!
			tuple_utils::for_each_down(m_phl_tuple, [&firstNeuronOfs, &lowerLayer, &dLdA, &dLdAPrev
				, _training_batch_size = get_self().get_common_data().get_cur_batch_size(), &_Math = get_self().get_iMath(), this](const auto& phl)
			{
				auto& lyr = phl.l;

				NNTL_ASSERT(firstNeuronOfs >= lyr.get_neurons_cnt());
				firstNeuronOfs -= lyr.get_neurons_cnt();
				
				constexpr bool bLowerLayerIsInput = ::std::is_base_of<m_layer_input, LowerLayer>::value;

				//setting up the m_innerdLdA
				m_innerdLdA.deform_like_no_bias(lyr.get_activations());
				NNTL_ASSERT(firstNeuronOfs + m_innerdLdA.cols() <= dLdA.cols());
				NNTL_ASSERT(m_innerdLdA.rows() == dLdA.rows() && _training_batch_size == m_innerdLdA.rows());
				memcpy(m_innerdLdA.data(), dLdA.colDataAsVec(firstNeuronOfs), m_innerdLdA.byte_size());

				//#consider если для каждого внутреннего слоя помнить max_dLdA_numel, и она окажется меньше m_innerdLdA.numel() и
				//m_innerdLdAPrev.numel, то можно избежать копирования dLdA в m_innerdLdA передавая данные напрямую, адресуя
				// их внутри dLdA - в этом случае соседние данные dLdA других слоёв останутся в безопасности.
				// Однако, в большинстве случаев условие будет не выполняться (т.к. внутренние слои представляют собой обычно
				// жирные фиче-детекторы с меньшем числом выходов, чем внутренних нейронов - у них dLdA для внутренних слоёв
				// больше располагаемой тут dLdA внешнего слоя)

				//setting up the m_innerdLdAPrev
				if (bLowerLayerIsInput) {
					m_innerdLdAPrev.deform(0, 0);
				}else m_innerdLdAPrev.deform(_training_batch_size, phl.coord.m_count);
				NNTL_ASSERT(bLowerLayerIsInput || m_innerdLdAPrev.rows() == dLdAPrev.rows());

				const auto switchMtxs = lyr.bprop( m_innerdLdA,
					_impl::trainable_partial_layer_wrapper<LowerLayer>(lowerLayer.get_activations(), m_pTmpBiasStorage, phl.coord)
					, m_innerdLdAPrev);

				if (!bLowerLayerIsInput) {
					NNTL_ASSERT(switchMtxs ? m_innerdLdAPrev.size() == realmtx_t::mtx_size_t(_training_batch_size, phl.coord.m_count)
						: m_innerdLdA.size() == realmtx_t::mtx_size_t(_training_batch_size, phl.coord.m_count));
					//saving m_innerdLdAPrev to dLdAPrev
					_Math.vAdd_ip(dLdAPrev.colDataAsVec(phl.coord.m_offset), switchMtxs ? m_innerdLdAPrev.data() : m_innerdLdA.data()
						, realmtx_t::sNumel(_training_batch_size, phl.coord.m_count));
				}
			});
			NNTL_ASSERT(firstNeuronOfs == 0);
			NNTL_ASSERT(lowerLayer.get_activations().test_biases_ok());

			iI.bprop_end(dLdAPrev);
			return 1;
		}

		//////////////////////////////////////////////////////////////////////////

		const bool is_trivial_drop_samples()const noexcept {
			bool b = true;
			get_self().for_each_packed_layer([&b](const auto& l) {
				b = b & l.is_trivial_drop_samples();
			});
			return b;
		}

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT(get_self().is_drop_samples_mbc());
			NNTL_ASSERT(!get_self().is_activations_shared() || !bBiasesToo);
			NNTL_ASSERT(!mask.emulatesBiases() && 1 == mask.cols() && m_activations.rows() == mask.rows() && mask.isBinary());
			NNTL_ASSERT(m_activations.emulatesBiases());

			if (is_trivial_drop_samples()) {//there should be NO get_self() in front of is_trivial_drop_samples() call. Or it may break LPHG (or may not)
				m_activations.hide_last_col();
				get_self().get_iMath().mrwMulByVec(m_activations, mask.data());
				m_activations.restore_last_col();
			} else {
				get_self().for_each_packed_layer([&mask](auto& l) {
					l.drop_samples(mask, false);
				});
			}
			if (bBiasesToo) {
				//applying bias mask to biases
				m_activations.copy_biases_from(mask.data());
			}
		}


	private:
		//support for ::boost::serialization
		friend class ::boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			get_self().for_each_packed_layer([&ar](auto& l) {
				ar & serialization::make_named_struct(l.get_layer_name_str().c_str(), l);
			});
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_horizontal
	// If you need to derive a new class, derive it from _layer_pack_horizontal (to make static polymorphism work)

	//to shorten class name to get rid of C4503
	template <typename ...PHLsT>
	class LPH final
		: public _layer_pack_horizontal < LPH<PHLsT...>, ::std::tuple<PHLsT...>>
	{
	public:
		~LPH() noexcept {};
		LPH(PHLsT&... phls) noexcept
			: _layer_pack_horizontal<LPH<PHLsT...>, ::std::tuple<PHLsT...>>(nullptr, ::std::make_tuple(phls...)) {};

		LPH(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal<LPH<PHLsT...>, ::std::tuple<PHLsT...>>(pCustomName, ::std::make_tuple(phls...)) {};
	};

	template <typename ..._T>
	using layer_pack_horizontal = typename LPH<_T...>;

	template <typename ...PHLsT> inline constexpr
	LPH <PHLsT...> make_layer_pack_horizontal(PHLsT&... phls) noexcept {
		return LPH<PHLsT...>(phls...);
	}
	template <typename ...PHLsT> inline constexpr
		LPH <PHLsT...> make_layer_pack_horizontal(const char* pCustomName, PHLsT&... phls) noexcept {
		return LPH<PHLsT...>(pCustomName, phls...);
	}

	//////////////////////////////////////////////////////////////////////////
	template <typename LossAddsTuple, typename ...PHLsT>
	class LPH_PA final
		: public _layer_pack_horizontal < LPH_PA<LossAddsTuple, PHLsT...>, ::std::tuple<PHLsT...>, LossAddsTuple>
	{
	public:
		static constexpr const char _defName[] = "lph_pa";

		~LPH_PA() noexcept {};
		LPH_PA(PHLsT&... phls) noexcept
			: _layer_pack_horizontal<LPH_PA<LossAddsTuple, PHLsT...>, ::std::tuple<PHLsT...>, LossAddsTuple>(nullptr, ::std::make_tuple(phls...)) {};

		LPH_PA(const char* pCustomName, PHLsT&... phls) noexcept
			: _layer_pack_horizontal<LPH_PA<LossAddsTuple, PHLsT...>, ::std::tuple<PHLsT...>, LossAddsTuple>(pCustomName, ::std::make_tuple(phls...)) {};
	};
}
