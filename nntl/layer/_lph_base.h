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
 
#include "_pack_.h"
#include "../utils.h"
#include "_activation_storage.h"

namespace nntl {
	namespace _impl {

		template<typename FinalPolymorphChild, typename PHLsTuple>
		class _LPH_base
			: public _act_stor<FinalPolymorphChild, typename ::std::remove_reference_t<::std::tuple_element_t<0, PHLsTuple>>::phl_original_t::interfaces_t>
		{
		private:
			typedef _act_stor<FinalPolymorphChild, typename ::std::remove_reference_t<::std::tuple_element_t<0, PHLsTuple>>::phl_original_t::interfaces_t> _base_class_t;

		protected:
			typedef _base_class_t _pre_LPH_base_class_t;

		public:
			//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
			typedef self_t LayerPack_t;

			typedef const PHLsTuple _phl_tuple;
			static_assert(tuple_utils::is_tuple<_phl_tuple>::value, "Must be a tuple!");

			static constexpr size_t phl_count = ::std::tuple_size<_phl_tuple>::value;
			static_assert(phl_count > 1, "For a pack with a single inner layer use that layer instead");

			template<typename T>
			struct _PHL_props : ::std::true_type {
				static_assert(::std::is_const<T>::value, "Must be a const");
				static_assert(!::std::is_reference<T>::value, "Must be an object");
				static_assert(is_PHL<T>::value, "Must be a PHL");
				typedef typename T::phl_original_t LT;
				static_assert(::std::is_same<LT, ::std::decay_t<LT>>::value, "wrong type of phl_original_t");
				static_assert(!::std::is_const<LT>::value, "mustn't be const");
				static_assert(!is_layer_input<LT>::value && !is_layer_output<LT>::value, "Inner layers mustn't be input or output layers!");
				static_assert(::std::is_base_of<_i_layer<real_t>, LT>::value, "must derive from _i_layer");
			};
			static_assert(tuple_utils::assert_each<_phl_tuple, _PHL_props>::value, "_LPH_base must be assembled from proper PHL objects!");

			typedef typename ::std::tuple_element_t<0, _phl_tuple>::phl_original_t first_layer_t;
			//typedef typename ::std::tuple_element_t<phl_count - 1, _phl_tuple>::phl_original_t last_layer_t;

		protected:
			_phl_tuple m_phl_tuple;

			realmtxdef_t m_innerdLdA, m_innerdLdAPrev;

			numel_cnt_t m_layers_max_dLdA_numel{ 0 };//max lid.max_dLdA_numel gathered from inner layers during init() phase. This value is used 
												// to allocate dLdA matricies to pass to layers bprop()

			// addresses at max m_max_fprop_batch_size elements to store a column of data of previous layer activation,
			// that will be replaced by biases during fprop()/bprop() phase.
			real_t* m_pTmpBiasStorage{ nullptr };

			//////////////////////////////////////////////////////////////////////////
		protected:
			//this is how we going to initialize layer indexes.
			friend class _preinit_layers;
			void _preinit_layer(init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
				//there should better be an exception, but we don't want exceptions at all.
				//anyway, there is nothing to help to those who'll try to abuse this API...
				NNTL_ASSERT(inc_neurons_cnt > 0);

				_base_class_t::_preinit_layer(ili, inc_neurons_cnt);

				_preinit_layers initializer(ili, inc_neurons_cnt);
				if (initializer.preparePHLCheck()) {
					tuple_utils::for_each_up(m_phl_tuple, initializer);
					if (!initializer.PHLCheck()) {
						NNTL_ASSERT(!"All lower layer activations must be covered by a set of inner layers of _LPH_base!");
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
			first_layer_t& first_layer() noexcept { return ::std::get<0>(m_phl_tuple).l; }
			//last_layer_t& last_layer() noexcept { return ::std::get<phl_count - 1>(m_phl_tuple).l; }

		private:
			static neurons_count_t _calcNeuronsCnt(const _phl_tuple& tupl) noexcept {
				neurons_count_t nc = 0;
				tuple_utils::for_each_up(tupl, [&nc](const auto& phl)noexcept {
					typedef ::std::remove_reference_t<decltype(phl)> PHL_t;
					static_assert(is_PHL<PHL_t>::value, "_LPH_base must be assembled from PHL objects!");
					static_assert(::std::is_const<PHL_t>::value, "Must be const!");

					typedef typename PHL_t::phl_original_t Layer_t;
					static_assert(!::std::is_const<Layer_t>::value, "PHL_t::phl_original_t must not be const!");
					static_assert(::std::is_same<Layer_t, ::std::decay_t<Layer_t>>::value, "wrong type of phl_original_t for a layer inside an PHL");
					static_assert(::std::is_same<real_t, typename Layer_t::real_t>::value, "Invalid real_t");

					static_assert(::std::is_base_of<_i_layer<real_t>, Layer_t>::value, "Each layer must derive from i_layer");

					static_assert(!is_layer_input<Layer_t>::value && !is_layer_output<Layer_t>::value
						, "Inner layers of _LPH_base mustn't be input or output layers!");

					nc += phl.l.get_neurons_cnt();
				});
				return nc;
			}

		public:
			~_LPH_base()noexcept {}
			_LPH_base(const char* pCustomName, const PHLsTuple& phls, const neurons_count_t addNeurons=0)noexcept 
				: _base_class_t(_calcNeuronsCnt(phls) + addNeurons, pCustomName), m_phl_tuple(phls)
			{}

			_LPH_base(const char* pCustomName, PHLsTuple&& phls, const neurons_count_t addNeurons = 0)noexcept
				: _base_class_t(_calcNeuronsCnt(phls) + addNeurons, pCustomName), m_phl_tuple(::std::move(phls))
			{}

			static constexpr const char _defName[] = "_lph_base";

			//////////////////////////////////////////////////////////////////////////
			//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
			template<typename _Func>
			void for_each_layer(_Func&& f)const noexcept {
				tuple_utils::for_each_up(m_phl_tuple, [&func{ f }](auto& phl)noexcept {
					//we shouldn't forward func here, because lambda might be called multiple times, therefore func should be lvalue
					call_F_for_each_layer(func, phl.l);
				});
			}

			template<typename _Func>
			void for_each_layer_down(_Func&& f)const noexcept {
				tuple_utils::for_each_down(m_phl_tuple, [&func{ f }](auto& phl)noexcept {
					call_F_for_each_layer_down(func, phl.l);
				});
			}

			//This will apply f to every layer, packed in tuple no matter whether it is a _pack_* kind of layer or no
			template<typename _Func>
			void for_each_packed_layer(_Func&& f)const noexcept {
				tuple_utils::for_each_up(m_phl_tuple, [&func{ f }](auto& phl)noexcept {
					func(phl.l);
				});
			}
			template<typename _Func>
			void for_each_packed_layer_down(_Func&& f)const noexcept {
				tuple_utils::for_each_down(m_phl_tuple, [&func{ f }](auto& phl)noexcept {
					func(phl.l);
				});
			}

			//////////////////////////////////////////////////////////////////////////
			//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
			bool hasLossAddendum()const noexcept {
				bool b = false;
				for_each_packed_layer([&b](auto& l) {
					NNTL_UNREF(l);
					b |= l.hasLossAddendum();
				});
				return b;
			}
			//returns a loss function summand, that's caused by this layer
			real_t lossAddendum()const noexcept {
				real_t la(.0);
				for_each_packed_layer([&la](auto& l) {
					NNTL_UNREF(l);
					la += l.lossAddendum();
				});
				return la;
			}

			//////////////////////////////////////////////////////////////////////////
			ErrorCode _init_phls(_layer_init_data_t& lid)noexcept {
				ErrorCode ec = ErrorCode::Success;
				layer_index_t failedLayerIdx = 0;
				auto initD = lid.dupe();
				neurons_count_t firstNeuronOfs = 0;
				for_each_packed_layer([&, &act = m_activations](auto& l)noexcept {
					if (ErrorCode::Success == ec) {
						initD.clean_using(lid);//we must propagate any IN flags set in the .lid variable to the layer being initialized.

						ec = l.init(initD, act.colDataAsVec(firstNeuronOfs));
						if (ErrorCode::Success == ec) {
							lid.update(initD);
							firstNeuronOfs += l.get_neurons_cnt();
						} else failedLayerIdx = l.get_layer_idx();
					}
				});
				NNTL_ASSERT(firstNeuronOfs + 1 == m_activations.cols());
				return ec;
			}

			ErrorCode _init_self(_layer_init_data_t& lid)noexcept { 
				NNTL_UNREF(lid);
				return ErrorCode::Success;
			}

			ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
				auto ec = _base_class_t::init(lid, pNewActivationStorage);
				if (ErrorCode::Success != ec) return ec;

				bool bSuccessfullyInitialized = false;
				utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
					if (!bSuccessfullyInitialized) get_self().deinit();
				});

				//now we must initialize encapsulated layers and that's a tricky part.
				// - we can't pass dLdA & dLdAPrev to these layers, because they could also be compound and may require
				//		a far bigger matrix sizes than dLdA allows. Therefore we must calculate the biggest max_dLdA_numel 
				//		for these layers (including dLdAPrev!!!) and allocate 2 corresponding matrices to be used as dLdA & dLdAPrev.
				//		However, ours .max_dLdA_numel should be set to a normal value for a simple layer [batchSize, neuronsCount]
				// - We should aggregate (by max()) layer's initMem() requirements and add them to ours
				NNTL_ASSERT(0 == lid.max_dLdA_numel && 0 == lid.maxMemFPropRequire && 0 == lid.maxMemTrainingRequire);

				neurons_count_t maxIncNeuronsCnt = 0;//we need maxIncNeuronsCnt to calculate the biggest possible internal dLdAPrev
				for_each_packed_layer([&maxIncNeuronsCnt](const auto& l)noexcept {
					maxIncNeuronsCnt = ::std::max(maxIncNeuronsCnt, l.get_incoming_neurons_cnt());
				});

				ec = get_self()._init_phls(lid);

				if (ErrorCode::Success == ec) {
					const auto biggestBatchSize = get_common_data().biggest_batch_size();
					//Now we should append our own training requirements (reqs of this (top-level) LPH layer)

					//we'll need a column-vector of length biggestBatchSize to store a data column of previous layer activation,
					// that will be substituted by biases for fprop() of one inner layer
					lid.maxMemFPropRequire += biggestBatchSize;

					if (get_common_data().is_training_possible()) {
						const auto _training_batch_size = get_common_data().training_batch_size();
						//adding backprop requirements
						// saving lid.max_dLdA_numel, gathered from inner layers and substituting it for this layer's max_dLdA_numel
						m_layers_max_dLdA_numel = ::std::max(lid.max_dLdA_numel
							, realmtx_t::sNumel(_training_batch_size, maxIncNeuronsCnt));
						//The first argument of max() - lid.max_dLdA_numel - describes the biggest dLdA size, the second
						//argument (realmtx_t::sNumel(training_batch_size(), maxIncNeuronsCnt)) describes the biggest dLdAPrev.

						//The biggest "outside" dLdA is limited to the ours activation matrix size (we can't send dLdA passed as bprop() argument
						//to encapsulated layer's bprop())
						lid.max_dLdA_numel = realmtx_t::sNumel(_training_batch_size, get_neurons_cnt());

						//reserving memory (additional to what encapsulated layers require) for two inner dLdA matrices
						// AND a column-vector of length biggestBatchSize to store a data column of previous layer activation
						lid.maxMemTrainingRequire += 2 * m_layers_max_dLdA_numel + biggestBatchSize;
					}

					ec = get_self()._init_self(lid);
				}

				if (ErrorCode::Success == ec) bSuccessfullyInitialized = true;

				//#TODO need some way to return failedLayerIdx
				return ec;
			}

			void deinit() noexcept {
				for_each_packed_layer([](auto& l) {l.deinit(); });
				m_layers_max_dLdA_numel = 0;
				m_innerdLdA.clear();
				m_innerdLdAPrev.clear();
				m_pTmpBiasStorage = nullptr;
				_base_class_t::deinit();
			}

			void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
				const auto _biggest_batch_size = static_cast<numel_cnt_t>(get_common_data().biggest_batch_size());
				NNTL_ASSERT(ptr && cnt >= _biggest_batch_size);
				m_pTmpBiasStorage = ptr;
				ptr += _biggest_batch_size;
				cnt -= _biggest_batch_size;

				if (get_common_data().is_training_possible()) {
					NNTL_ASSERT(cnt >= 2 * m_layers_max_dLdA_numel);
					m_innerdLdA.useExternalStorage(ptr, m_layers_max_dLdA_numel, false);
					ptr += m_layers_max_dLdA_numel;
					m_innerdLdAPrev.useExternalStorage(ptr, m_layers_max_dLdA_numel, false);
					ptr += m_layers_max_dLdA_numel;
					cnt -= 2 * m_layers_max_dLdA_numel;
				}

				for_each_packed_layer([=](auto& l) {l.initMem(ptr, cnt); });
			}

			void on_batch_size_change(const real_t learningRateScale, real_t*const pNewActivationStorage = nullptr)noexcept {
				_base_class_t::on_batch_size_change(learningRateScale, pNewActivationStorage);
				const vec_len_t batchSize = get_common_data().get_cur_batch_size();

				neurons_count_t firstNeuronOfs = 0;
				for_each_packed_layer([&act = m_activations, &firstNeuronOfs, learningRateScale](auto& lyr)noexcept {
					//we're just setting memory to store activation values of inner layers here.
					//there's no need to play with biases here.
					lyr.on_batch_size_change(learningRateScale, act.colDataAsVec(firstNeuronOfs));
					firstNeuronOfs += lyr.get_neurons_cnt();
				});
				NNTL_ASSERT(firstNeuronOfs + 1 == m_activations.cols());
			}

		protected:
			//support for ::boost::serialization
			friend class ::boost::serialization::access;
			template<class Archive> void serialize(Archive & ar, const unsigned int) {
				for_each_packed_layer([&ar](auto& l) {
					ar & serialization::make_named_struct(l.get_layer_name_str().c_str(), l);
				});
			}
		};
	}
}
