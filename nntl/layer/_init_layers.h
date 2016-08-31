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

#include <type_traits>
#include <algorithm>
#include "_pack_traits.h"
#include "../common_nn_data.h"

namespace nntl {

	//special marks for type checking of input and output layers
	struct m_layer_input {};
	struct m_layer_output {};

	//when a layer is derived from this class, it is expected to be used inside of some layer_pack_* objects and it
	// doesn't have a neurons count specified in constructor. Instead, compound layer (or it's support objects) specifies
	// the number of neurons during their construction via _set_neurons_cnt().
	// See layer_identity for an example
	struct m_layer_autoneurons_cnt {};

	namespace _impl {

		template< class, class = std::void_t<> >
		struct m_prop_input_marker { };
		// specialization recognizes types derived from m_layer_input
		template< class T >
		struct m_prop_input_marker<T, std::void_t<typename std::enable_if< std::is_base_of<m_layer_input,T>::value >::type > > : m_layer_input {};

		//////////////////////////////////////////////////////////////////////////
		//class to help propagate NN structure through layer stack during that stack creation
		class _preinit_layers {
		public:
			char* m_pPHLCheckStorage;//variable to hold a pointer to an array, that's used to check whether inner layers
							// of _layer_pack_horizontal cover all activation units of the range of underlying layer
							// Initialized to nullptr by default in constructors.

			neurons_count_t _incNeurons;
			layer_index_t _idx;			

			~_preinit_layers()noexcept {
				if(m_pPHLCheckStorage) delete[] m_pPHLCheckStorage;
			}
			_preinit_layers() noexcept : _idx(0), _incNeurons(0), m_pPHLCheckStorage(nullptr) {}
			_preinit_layers(const layer_index_t i, const neurons_count_t n) noexcept : _idx(i), _incNeurons(n), m_pPHLCheckStorage(nullptr) {
				NNTL_ASSERT(i && n);
			}

			//////////////////////////////////////////////////////////////////////////
			// variation to comply utils::for_eachwp_up() callback
			// To use with class layers{}
			template <typename LCur, typename LPrev>
			void operator()(LCur& lcur, LPrev& lprev, const bool bFirst)noexcept {
				static_assert(std::is_base_of<_i_layer<typename std::remove_reference<LCur>::type::real_t >
					, std::remove_reference<LCur>::type>::value, "Each layer must derive from i_layer");
				static_assert(std::is_base_of<_i_layer<typename std::remove_reference<LCur>::type::real_t >
					, std::remove_reference<LPrev>::type>::value, "Each layer must derive from i_layer");
				static_assert(std::is_same<LCur::iMath_t, LPrev::iMath_t>::value, "Math interface must be the same for all layers!");
				static_assert(std::is_same<LCur::iRng_t, LPrev::iRng_t>::value, "RNG interface must be the same for all layers!");

#ifdef NNTL_DEBUG
				layer_index_t curIdx = _idx;
#endif // NNTL_DEBUG

				if (bFirst) {
					lprev._preinit_layer(_idx, _incNeurons);
#ifdef NNTL_DEBUG
					NNTL_ASSERT(_idx > curIdx);
					curIdx = _idx;
#endif // NNTL_DEBUG
				}

				lcur._preinit_layer(_idx, lprev.get_neurons_cnt());
				NNTL_ASSERT(_idx > curIdx);
			}

			//////////////////////////////////////////////////////////////////////////
			// some machinery necessary for the layer_pack_horizontal class
			bool preparePHLCheck()noexcept {
				NNTL_ASSERT(_incNeurons && !m_pPHLCheckStorage);
				m_pPHLCheckStorage = new(std::nothrow) char[_incNeurons];
				if (m_pPHLCheckStorage) memset(m_pPHLCheckStorage, 0, _incNeurons);
				return nullptr != m_pPHLCheckStorage;
			}

			// variation to comply with utils::for_each_up() callback. For use with PHL structures in _layer_pack_horizontal
			template<typename PHLT>
			std::enable_if_t<is_PHL<PHLT>::value> operator()(PHLT& phl)noexcept {
				static_assert(std::is_base_of<_i_layer<PHLT::phl_original_t::real_t>, PHLT::phl_original_t>::value, "Each layer must derive from i_layer");
				static_assert(!std::is_base_of<m_layer_input, PHLT::phl_original_t>::value && !std::is_base_of<m_layer_output, PHLT::phl_original_t>::value,
					"No input/output layers is allowed within _layer_pack_horizontal");
				
				NNTL_ASSERT(m_pPHLCheckStorage && phl.m_count && phl.m_offset < _incNeurons && (phl.m_offset + phl.m_count) <= _incNeurons);
				const auto pBeg = m_pPHLCheckStorage + phl.m_offset;
				std::fill(pBeg, pBeg + phl.m_count, char(1));

#ifdef NNTL_DEBUG
				layer_index_t curIdx = _idx;
#endif // NNTL_DEBUG
				phl.l._preinit_layer(_idx, phl.m_count);
				NNTL_ASSERT(_idx > curIdx);
			}

			bool PHLCheck()noexcept {
				NNTL_ASSERT(m_pPHLCheckStorage);
				const bool r = std::all_of(m_pPHLCheckStorage, m_pPHLCheckStorage + _incNeurons, [](const char c)->bool {
					return c == char(1);
				});
				delete[] m_pPHLCheckStorage;
				m_pPHLCheckStorage = nullptr;
				return r;
			}

			//////////////////////////////////////////////////////////////////////////
			// variation to use in other case just to preinit single layer
			template<typename Layr>
			std::enable_if_t<!is_PHL<Layr>::value> operator()(Layr& layr)noexcept {
#ifdef NNTL_DEBUG
				layer_index_t curIdx = _idx;
#endif // NNTL_DEBUG

				layr._preinit_layer(_idx, _incNeurons);
#ifdef NNTL_DEBUG
				NNTL_ASSERT(_idx > curIdx);
#endif // NNTL_DEBUG
			}
		};


		//////////////////////////////////////////////////////////////////////////
		// This structure is passed to a _i_layer.init() during initialization phase.
		// OUT marks variables that should be filled/returned by a layer if applicable. Most of this variables are used to
		// find out how many shared memory real_t's should be allocated by a nnet object during initialization phase
		// and passed to the layer.initMem().
		template<typename CommonNnData>
		struct _layer_init_data : public math::simple_matrix_td {
			typedef CommonNnData common_data_t;

			// "IN" marks variables that are passed to init() function, "OUT" marks output from init()

			IN const common_data_t& commonData;

			//fprop and bprop may use different batch sizes during single training session (for example, fprop()/bprop()
			// uses small batch size during learning process, but whole data_x.rows() during fprop() for loss function
			// computation. Therefore to reduce memory consumption during evaluating and learning, we will demark fprop()
			// memory requirements from bprop() mem reqs.
			OUT numel_cnt_t maxMemFPropRequire;// total size of <real_t> array to be passed to layer.initMem() in order to compute fprop()
			OUT numel_cnt_t maxMemBPropRequire;// same for bprop - that's how much <real_t>s must be addressed by pointer passed to .initMem() in order to bprop() works
			OUT numel_cnt_t max_dLdA_numel;//Biggest size of dLdA matrix (numel) that could be passed into a layer for bprop()
			// We can always calculate this variable by knowing the batchSize and the layer's neurons count, aren't we?
			//NOOO!!! It's not the case for compound layers!!! We need this variable!
			OUT numel_cnt_t nParamsToLearn;//total number of parameters, that layer has to learn during training

			OUT bool bHasLossAddendum;//to be set by layer.init()

			_layer_init_data(const common_data_t& cd) noexcept : commonData(cd) {
				//clean(); //not necessary here because the struct is reused
			}

			//this function must be called on the object before it is passed to layer.init()
			//i.e. _i_layer.init() expects the object to be clean()'ed
			void clean()noexcept {
				maxMemFPropRequire = 0;
				maxMemBPropRequire = 0;
				max_dLdA_numel = 0;
				nParamsToLearn = 0;
				bHasLossAddendum = false;
			}

			//used by compound layers to gather data from layers encapsulated into them.
			void update(const _layer_init_data& o)noexcept {
				maxMemFPropRequire = std::max(maxMemFPropRequire, o.maxMemFPropRequire);
				maxMemBPropRequire = std::max(maxMemBPropRequire, o.maxMemBPropRequire);
				max_dLdA_numel = std::max(max_dLdA_numel, o.max_dLdA_numel);
				nParamsToLearn += o.nParamsToLearn;
				bHasLossAddendum |= o.bHasLossAddendum;
			}

			_layer_init_data dupe()const noexcept {
				return _layer_init_data(commonData);
			}
		};

		//////////////////////////////////////////////////////////////////////////
		// structure to be filled during layers.init() to return necessary data back to nnet object
		struct layers_mem_requirements : public math::simple_matrix_td {
			numel_cnt_t maxMemLayerTrainingRequire,//for nnet.train()
				maxMemLayersFPropRequire,//#todo for nnet.eval()
				maxSingledLdANumel,//the biggest dLdA matrix required for bprop()
				totalParamsToLearn;//The total parameters count the model has

			bool bHasLossAddendum;

			layers_mem_requirements() noexcept{
				zeros();
			}

			void zeros()noexcept {
				maxMemLayerTrainingRequire = 0;
				maxMemLayersFPropRequire = 0;
				maxSingledLdANumel = 0;//single! The biggest matrix.numel() to be used in a bprop()
				totalParamsToLearn = 0;
				bHasLossAddendum = false;
			}

			void updateLayerReq(const numel_cnt_t mmlF, const numel_cnt_t mmlB
				, const numel_cnt_t maxdLdA, const numel_cnt_t nLP, const bool _HasLossAddendum)noexcept
			{
				maxMemLayerTrainingRequire = std::max({ maxMemLayerTrainingRequire, mmlF, mmlB });
				maxMemLayersFPropRequire = std::max(maxMemLayersFPropRequire, mmlF);
				maxSingledLdANumel = std::max(maxSingledLdANumel, maxdLdA);
				totalParamsToLearn += nLP;
				bHasLossAddendum |= _HasLossAddendum;
			}
			/*template<typename _layer_init_data_t>
			void updateLayerReq(const _layer_init_data_t& lid)noexcept {
				maxMemLayerTrainingRequire = std::max({ maxMemLayerTrainingRequire, lid.maxMemFPropRequire, lid.maxMemBPropRequire });
				maxMemLayersFPropRequire = std::max(maxMemLayersFPropRequire, lid.maxMemFPropRequire);
				maxSingledLdANumel = std::max(maxSingledLdANumel, lid.max_dLdA_numel);
				totalParamsToLearn += lid.nParamsToLearn;
				bHasLossAddendum |= lid.bHasLossAddendum;
			}*/
			template<typename _layer_init_data_t>
			void updateLayerReq(const _layer_init_data_t& lid)noexcept {
				return updateLayerReq(lid.maxMemFPropRequire, lid.maxMemBPropRequire, lid.max_dLdA_numel
					, lid.nParamsToLearn, lid.bHasLossAddendum);
			}
		};
	}

}