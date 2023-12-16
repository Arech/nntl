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

// defines common structs for layer_pack_* layers

#include <nntl/layer/_layer_base.h>
#include <nntl/layer/_pack_traits.h>

namespace nntl {

	struct PHL_coord {
		const neurons_count_t m_offset;//offset to a first neuron of previous layer activation that is sent to the LayerT input
		const neurons_count_t m_count;//total number of neurons of previous layer that is sent to the LayerT input

		PHL_coord(const neurons_count_t _ofs, const neurons_count_t _cnt)noexcept 
			: m_offset(_ofs), m_count(_cnt)
		{}
	};

	//////////////////////////////////////////////////////////////////////////
	template<typename LayerT>
	struct PHL {//"let me speak from my heart in English" (c): that's a Part of a Horizontal Layer :-D
		static_assert(::std::is_same<LayerT, ::std::decay_t<LayerT>>::value, "invalid type for PHL");
		static_assert(!::std::is_const<LayerT>::value, "PHL_t::phl_original_t must not be const!");

		//this typedef will also help to distinguish between PHL and other structs.
		typedef LayerT phl_original_t;

		phl_original_t& l;
		const PHL_coord coord;

		template<typename _L = phl_original_t>
		PHL(phl_original_t& _l, const neurons_count_t _ofs, const neurons_count_t _cnt
			, typename ::std::enable_if<::std::is_base_of<m_layer_autoneurons_cnt, _L>::value>::type* = 0)noexcept
			:l(_l), coord(_ofs,_cnt)
		{
			l._set_neurons_cnt(_cnt);
		}

		template<typename _L = phl_original_t >
		PHL(phl_original_t& _l, const neurons_count_t _ofs, const neurons_count_t _cnt
			, typename ::std::enable_if<!::std::is_base_of<m_layer_autoneurons_cnt, _L>::value>::type* = 0)noexcept
			:l(_l), coord(_ofs, _cnt)
		{ }
	};
	template<typename LayerT> inline constexpr
		PHL<LayerT> make_PHL(LayerT& l, const neurons_count_t _ofs, const neurons_count_t _cnt)noexcept
	{
		return PHL<LayerT>(l, _ofs, _cnt);
	}



	namespace _impl {
		// helper classes for layer_pack_* layers

		//////////////////////////////////////////////////////////////////////////
		// trainable_layer_wrapper wraps all activation matrix of a layer
		// PIMLT is the type you get by specializing m_propagate_markers<> with a corresponding layer type, m_propagate_markers<WrappedLayerT>
		// Don't use directly if you can, use the wrap_trainable_layer<> template below.
		// The reason for this whole burdensome machinery to get the same pseudo-layer type for different layer types and
		// prevent template code bloating
		template<typename PIMLT>
		class _trainable_layer_wrapper : public _i_layer_trainable<typename PIMLT::real_t>, public PIMLT {
			static_assert(_impl::is_m_prop_input_marker<PIMLT>::value, "Wrong PIMLT, did you use wrap_trainable_layer<>?");
		public:
			//this typedef helps to distinguish *_wrapper classes from other classes
			typedef PIMLT wrapped_layer_t;
			typedef typename PIMLT::real_t real_t;

		protected:
			const realmtx_t& m_act;

		public:
			~_trainable_layer_wrapper()noexcept {}

			template<typename LayerT>
			_trainable_layer_wrapper(const LayerT& L)noexcept
				: m_act(L.get_activations()) {}

			_trainable_layer_wrapper(const realmtx_t& underlyingLayerAct) noexcept : m_act(underlyingLayerAct) {}
			_trainable_layer_wrapper(const realmtxdef_t& underlyingLayerAct) noexcept 
				: m_act(static_cast<const realmtx_t&>(underlyingLayerAct)) {}

			const realmtx_t& get_activations()const noexcept { return m_act; }
			const realmtx_t* get_activations_storage()const noexcept { return &m_act; }
			//realmtx_t* get_activations_storage_mutable()noexcept { return &m_act; }
			//not sure it should expose get_activations_storage_mutable
			const mtx_size_t get_activations_size()const noexcept { return m_act.size(); }
		};

		// The reason for this whole burdensome machinery to get the same pseudo-layer type for different layer types and
		// prevent template code bloating
		template<typename LayerT>
		using wrap_trainable_layer = _trainable_layer_wrapper< m_propagate_markers<LayerT> >;


		//////////////////////////////////////////////////////////////////////////
		// trainable_partial_layer_wrapper wraps a subset of columns of activation matrix of a layer, defined by struct PHL
		
		// implementation of trainable_partial_layer_wrapper doesn't need the type of wrapped layer
		template <typename PIMLT>
		class _trainable_partial_layer_wrapper : public _i_layer_trainable<typename PIMLT::real_t>, public PIMLT {
			static_assert(_impl::is_m_prop_input_marker<PIMLT>::value, "Wrong PIMLT, did you use wrap_part_trainable_layer<>?");
			
		public:
			//this typedef helps to distinguish *_wrapper classes from other classes
			typedef PIMLT wrapped_layer_t;
			typedef typename PIMLT::real_t real_t;

		protected:
			realmtx_t m_act;
			real_t* m_pTmpBiasStor;

		public:
			~_trainable_partial_layer_wrapper()noexcept {
				//we must restore the original data back to bias column
				NNTL_ASSERT(!m_act.emulatesBiases() || m_act.test_biases_strict());
				if (m_pTmpBiasStor) {
					NNTL_ASSERT(m_act.emulatesBiases() && m_act.bBatchInColumn());
					memcpy(m_act.bias_column(), m_pTmpBiasStor, sizeof(*m_pTmpBiasStor)*m_act.rows());
				}
			}

		private:
			void _ctor(const realmtx_t& underlyingLayerAct, const PHL_coord& phl_coord, const bool bMakeBiases)noexcept {
				NNTL_ASSERT(underlyingLayerAct.test_biases_strict());
				NNTL_ASSERT(phl_coord.m_offset + phl_coord.m_count <= underlyingLayerAct.cols_no_bias());
				NNTL_ASSERT(underlyingLayerAct.bBatchInColumn() && m_act.bBatchInColumn());

				// activation matrix (m_act) are NOT expected to be changed from the outside, therefore trick with const_cast<> should do no harm.
				m_act.useExternalStorage(const_cast<real_t*>(underlyingLayerAct.colDataAsVec(phl_coord.m_offset))
					, underlyingLayerAct.rows(), phl_coord.m_count + bMakeBiases, bMakeBiases
					, bMakeBiases ? underlyingLayerAct.isHoleyBiases() : false);

				if (phl_coord.m_offset + phl_coord.m_count >= underlyingLayerAct.cols_no_bias()) {
					//biases must have already been set, because it's the end of the underlyingLayerAct matrix!
					NNTL_ASSERT(!bMakeBiases || m_act.test_biases_strict());
					m_pTmpBiasStor = nullptr;
				} else {
					if (bMakeBiases) {
						NNTL_ASSERT(m_pTmpBiasStor);
						//now we must save the real data under bias column and refill biases. On object destruction we must restore this data back
						memcpy(m_pTmpBiasStor, m_act.bias_column(), sizeof(*m_pTmpBiasStor)*m_act.rows());

						//we must use the bias column of original matrix, because biases might be holey --- no, they can't
						//m_act.copy_biases_from(underlyingLayerAct);
						m_act.set_biases();
					}else m_pTmpBiasStor = nullptr;
				}
			}

		public:
			template<typename LayerT>
			_trainable_partial_layer_wrapper(const LayerT& L, real_t*const pTmpBiasStor, const PHL_coord& phl_coord, const bool bMakeBiases = true)
				: m_pTmpBiasStor(pTmpBiasStor)
			{
				_ctor(L.get_activations(), phl_coord, bMakeBiases);
			}

			_trainable_partial_layer_wrapper(const realmtx_t& underlyingLayerAct
				, real_t*const pTmpBiasStor, const PHL_coord& phl_coord, const bool bMakeBiases = true)
				: m_pTmpBiasStor(pTmpBiasStor)
			{
				_ctor(underlyingLayerAct, phl_coord, bMakeBiases);
			}

			//constructor to use in *_optional layer. Assumes that underlyingLayerAct is a matrix
			//that might already has a valid bias column, if it's numel is less than or equal to a specified value.
			// However, if it has more than maxNumel elements than the bias column might be distorted and must be restored
			_trainable_partial_layer_wrapper(const realmtx_t& underlyingLayerAct, real_t*const pTmpBiasStor, const numel_cnt_t maxNumel)
				: m_pTmpBiasStor(pTmpBiasStor)
			{
				NNTL_ASSERT(underlyingLayerAct.emulatesBiases() && pTmpBiasStor);
				NNTL_ASSERT(underlyingLayerAct.bBatchInColumn() && m_act.bBatchInColumn());
				// activation matrix (m_act) are NOT expected to be changed from the outside, therefore trick with const_cast<> should do no harm.
				m_act.useExternalStorage(const_cast<real_t*>(underlyingLayerAct.data()), underlyingLayerAct.rows(), underlyingLayerAct.cols(), true);

				if (m_act.numel() <= maxNumel) {
					NNTL_ASSERT(m_act.test_biases_strict());
					m_pTmpBiasStor = nullptr;
				} else {
					//now we must save the real data under bias column and refill biases. On object destruction we must restore this data back
					memcpy(m_pTmpBiasStor, m_act.bias_column(), sizeof(*m_pTmpBiasStor)*m_act.rows());
					m_act.set_biases();
				}
			}

			const realmtx_t& get_activations()const noexcept { return m_act; }
			const realmtx_t* get_activations_storage()const noexcept { return &m_act; }
			//realmtx_t* get_activations_storage_mutable()noexcept { return &m_act; }
			//not sure it should expose get_activations_storage_mutable
			mtx_size_t get_activations_size()const noexcept { return m_act.size(); }
		};

		template<typename LayerT>
		using wrap_part_trainable_layer = _trainable_partial_layer_wrapper< m_propagate_markers<LayerT> >;
		

		//////////////////////////////////////////////////////////////////////////
		// Helper for the LPH to recognize if a LayerT class has .OuterLayerCustomFlag1Eval(const PhlsTupleT&) function to calculate bLPH_CustomFlag1 var
		template<class, class, class, class = ::std::void_t<>>
		struct layer_has_OuterLayerCustomFlag1Eval : ::std::false_type {};

		template<class LayerT, class PhlsTupleT, class LayerInitDataT>
		struct layer_has_OuterLayerCustomFlag1Eval<LayerT, PhlsTupleT, LayerInitDataT
			, ::std::void_t<decltype(::std::declval<LayerT>()
				.OuterLayerCustomFlag1Eval(::std::declval<const PhlsTupleT&>(),::std::declval<const LayerInitDataT&>()))>> : ::std::true_type {};
	}
}
