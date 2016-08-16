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

// defines common structs for layer_pack_* layers

#include "_layer_base.h"
#include "_pack_traits.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	template<typename LayerT>
	struct PHL {//"let me speak from my heart in English" (c): that's a Part of Horizontal Layer :-D

		//this typedef will also help to distinguish between PHL and other structs.
		typedef LayerT phl_original_t;

		phl_original_t& l;
		const neurons_count_t m_offset;//offset to a first neuron of previous layer activation that is sent to the LayerT input
		const neurons_count_t m_count;//total number of neurons of previous layer that is sent to the LayerT input

		template<typename _L = phl_original_t>
		PHL(phl_original_t& _l, const neurons_count_t o, const neurons_count_t c
			, typename std::enable_if<std::is_base_of<m_layer_autoneurons_cnt, _L>::value>::type* = 0)noexcept
			:l(_l), m_offset(o), m_count(c)
		{
			l._set_neurons_cnt(c);
		}

		template<typename _L = phl_original_t >
		PHL(phl_original_t& _l, const neurons_count_t o, const neurons_count_t c
			, typename std::enable_if<!std::is_base_of<m_layer_autoneurons_cnt, _L>::value>::type* = 0)noexcept
			:l(_l), m_offset(o), m_count(c)
		{ }
	};
	template<typename LayerT> inline
		PHL<LayerT> make_PHL(LayerT& l, const neurons_count_t o, const neurons_count_t c)noexcept
	{
		return PHL<LayerT>(l, o, c);
	}



	namespace _impl {
		// helper classes for layer_pack_* layers

		//////////////////////////////////////////////////////////////////////////
		// trainable_layer_wrapper wraps all activation matrix of a layer
		template<typename WrappedLayer>
		class trainable_layer_wrapper : public _i_layer_trainable<typename WrappedLayer::real_t>, public m_prop_input_marker<WrappedLayer> {
		public:
			//this typedef helps to distinguish *_wrapper classes from other classes
			typedef WrappedLayer wrapped_layer_t;

		protected:
			const realmtx_t& m_act;

		public:
			~trainable_layer_wrapper()noexcept {}
			trainable_layer_wrapper(const realmtx_t& underlyingLayerAct) : m_act(underlyingLayerAct) {}

			const realmtx_t& get_activations()const noexcept { return m_act; }
		};

		//////////////////////////////////////////////////////////////////////////
		// trainable_partial_layer_wrapper wraps a subset of columns of activation matrix of a layer, defined by struct PHL
		template<typename WrappedLayer>
		class trainable_partial_layer_wrapper : public _i_layer_trainable<typename WrappedLayer::real_t>, public m_prop_input_marker<WrappedLayer> {
		public:
			//this typedef helps to distinguish *_wrapper classes from other classes
			typedef WrappedLayer wrapped_layer_t;

		protected:
			realmtx_t m_act;
			real_t*const m_pTmpBiasStor;

		public:
			~trainable_partial_layer_wrapper()noexcept {
				//we must restore the original data back to bias column
				NNTL_ASSERT(m_act.test_biases_ok());
				memcpy(m_act.colDataAsVec(m_act.cols()-1), m_pTmpBiasStor, sizeof(*m_pTmpBiasStor)*m_act.rows());
			}

			template<typename PhlT>
			trainable_partial_layer_wrapper(const realmtx_t& underlyingLayerAct, real_t* pTmpBiasStor, const PhlT& phl)
				: m_pTmpBiasStor(pTmpBiasStor)
			{
				NNTL_ASSERT(phl.m_offset + phl.m_count <= underlyingLayerAct.cols_no_bias());
				m_act.useExternalStorage(
					const_cast<real_t*>(underlyingLayerAct.colDataAsVec(phl.m_offset)), underlyingLayerAct.rows(), phl.m_count + 1, true
					);
				//now we must save the real data under bias column and refill biases. On object destruction we must restore this data back
				memcpy(m_pTmpBiasStor, m_act.colDataAsVec(phl.m_count), sizeof(*m_pTmpBiasStor)*m_act.rows());
				m_act.set_biases();
			}

			const realmtx_t& get_activations()const noexcept { return m_act; }
		};


		//////////////////////////////////////////////////////////////////////////
		// Helper traits recognizer
		// primary template handles types that have no nested ::wrapped_layer_t member:
		template< class, class = std::void_t<> >
		struct is_layer_wrapper : std::false_type { };
		// specialization recognizes types that do have a nested ::wrapped_layer_t member:
		template< class T >
		struct is_layer_wrapper<T, std::void_t<typename T::wrapped_layer_t>> : std::true_type {};
	}
}
