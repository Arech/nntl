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

	struct PHL_coord {
		const neurons_count_t m_offset;//offset to a first neuron of previous layer activation that is sent to the LayerT input
		const neurons_count_t m_count;//total number of neurons of previous layer that is sent to the LayerT input

		PHL_coord(const neurons_count_t _ofs, const neurons_count_t _cnt)noexcept 
			: m_offset(_ofs), m_count(_cnt)
		{}
	};

	//////////////////////////////////////////////////////////////////////////
	template<typename LayerT>
	struct PHL {//"let me speak from my heart in English" (c): that's a Part of Horizontal Layer :-D

		//this typedef will also help to distinguish between PHL and other structs.
		typedef LayerT phl_original_t;

		phl_original_t& l;

		const PHL_coord coord;
		//const neurons_count_t m_offset;//offset to a first neuron of previous layer activation that is sent to the LayerT input
		//const neurons_count_t m_count;//total number of neurons of previous layer that is sent to the LayerT input

		template<typename _L = phl_original_t>
		PHL(phl_original_t& _l, const neurons_count_t _ofs, const neurons_count_t _cnt
			, typename ::std::enable_if<::std::is_base_of<m_layer_autoneurons_cnt, _L>::value>::type* = 0)noexcept
			:l(_l), coord(_ofs,_cnt) //m_offset(_ofs), m_count(_cnt)
		{
			l._set_neurons_cnt(_cnt);
		}

		template<typename _L = phl_original_t >
		PHL(phl_original_t& _l, const neurons_count_t _ofs, const neurons_count_t _cnt
			, typename ::std::enable_if<!::std::is_base_of<m_layer_autoneurons_cnt, _L>::value>::type* = 0)noexcept
			:l(_l), coord(_ofs, _cnt) //m_offset(_ofs), m_count(_cnt)
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
			const realmtx_t* get_activations_storage()const noexcept { return &m_act; }
			const mtx_size_t get_activations_size()const noexcept { return m_act.size(); }
		};



		//////////////////////////////////////////////////////////////////////////
		// trainable_partial_layer_wrapper wraps a subset of columns of activation matrix of a layer, defined by struct PHL
		
		// implementation of trainable_partial_layer_wrapper doesn't need the type of wrapped layer
		template <typename RealT>
		class _trainable_partial_layer_wrapper : public _i_layer_trainable<RealT> {
		protected:
			realmtx_t m_act;
			real_t* m_pTmpBiasStor;

		public:
			~_trainable_partial_layer_wrapper()noexcept {
				//we must restore the original data back to bias column
				NNTL_ASSERT(m_act.test_biases_ok());
				if (m_pTmpBiasStor) memcpy(m_act.colDataAsVec(m_act.cols() - 1), m_pTmpBiasStor, sizeof(*m_pTmpBiasStor)*m_act.rows());
			}

			//template<typename PhlT>
			_trainable_partial_layer_wrapper(const realmtx_t& underlyingLayerAct, real_t* pTmpBiasStor, const PHL_coord& phl_coord)
				: m_pTmpBiasStor(pTmpBiasStor)
			{
				NNTL_ASSERT(underlyingLayerAct.test_biases_ok());
				//don't test for underlyingLayerAct.emulatesBiases() here because if underlyingLayerAct belongs to a input_layer in
				//a minibatch mode, then this flag might be turned off while still there're biases set.
				// #todo: this flag should be turned ON for all layer activations INCLUDING input layer.

				NNTL_ASSERT(phl_coord.m_offset + phl_coord.m_count <= underlyingLayerAct.cols_no_bias());
				// activation matrix (m_act) are NOT expected to be changed from the outside, therefore trick with const_cast<> should do no harm.
				m_act.useExternalStorage(const_cast<real_t*>(underlyingLayerAct.colDataAsVec(phl_coord.m_offset))
					, underlyingLayerAct.rows(), phl_coord.m_count + 1, true, underlyingLayerAct.isHoleyBiases());

				if (phl_coord.m_offset + phl_coord.m_count >= underlyingLayerAct.cols_no_bias()) {
					//biases must have already been set, because it's the end of the underlyingLayerAct matrix!
					NNTL_ASSERT(m_act.test_biases_ok());
					m_pTmpBiasStor = nullptr;
				} else {
					//now we must save the real data under bias column and refill biases. On object destruction we must restore this data back
					memcpy(m_pTmpBiasStor, m_act.colDataAsVec(phl_coord.m_count), sizeof(*m_pTmpBiasStor)*m_act.rows());

					//m_act.set_biases();
					//we must use the bias column of original matrix, because biases might be holey
					m_act.copy_biases_from(underlyingLayerAct);
				}
			}

			const realmtx_t& get_activations()const noexcept { return m_act; }
			const realmtx_t* get_activations_storage()const noexcept { return &m_act; }
			const mtx_size_t get_activations_size()const noexcept { return m_act.size(); }
		};

		//#TODO seems like we don't need WrappedLayer parameter here. Better make it type-less and 
		//update LPH::bprop and other related code
		//final wrapper
		template<typename WrappedLayer>
		class trainable_partial_layer_wrapper 
			: public m_prop_input_marker<WrappedLayer>
			, public _trainable_partial_layer_wrapper<typename WrappedLayer::real_t>
		{
		private:
			typedef _trainable_partial_layer_wrapper<typename WrappedLayer::real_t> _base_class_t;
		public:
			typedef WrappedLayer wrapped_layer_t;//this typedef helps to distinguish *_wrapper classes from other classes
		public:
			~trainable_partial_layer_wrapper()noexcept {}
			trainable_partial_layer_wrapper(const realmtx_t& underlyingLayerAct, real_t* pTmpBiasStor, const PHL_coord& phl_coord)
				: _base_class_t(underlyingLayerAct, pTmpBiasStor, phl_coord)
			{}

			//we could generalize to the following constructor, however, it'll bring unnecessary dependence on the wrapped_layer_t type
			//that require to store layer object for some callers (see LPH::fprop()) and this might be overkill
// 			trainable_partial_layer_wrapper(const wrapped_layer_t& underlyingLayer, real_t* pTmpBiasStor, const PHL_coord& phl_coord)
// 				: _base_class_t(underlyingLayer.get_activations(), pTmpBiasStor, phl_coord)
// 			{}
		};


		//////////////////////////////////////////////////////////////////////////
		// Helper traits recognizer
		// primary template handles types that have no nested ::wrapped_layer_t member:
		template< class, class = ::std::void_t<> >
		struct is_layer_wrapper : ::std::false_type { };
		// specialization recognizes types that do have a nested ::wrapped_layer_t member:
		template< class T >
		struct is_layer_wrapper<T, ::std::void_t<typename T::wrapped_layer_t>> : ::std::true_type {};

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
