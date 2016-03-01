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

namespace nntl {

	//special marks for type checking of input and output layers
	struct m_layer_input {};
	struct m_layer_output {};

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
			neurons_count_t _incNeurons;
			layer_index_t _idx;

			char* pPHLCheck;//variable to hold a pointer to an array, that's used to check whether inner layers of _layer_pack_horizontal
			//cover all activation units range of underlying layer
			//uninitialized by default in constructors

			~_preinit_layers()noexcept {
				if(pPHLCheck) delete[] pPHLCheck;
			}
			_preinit_layers() noexcept : _idx(0), _incNeurons(0), pPHLCheck(nullptr) {}
			_preinit_layers(const layer_index_t i, const neurons_count_t n) noexcept : _idx(i), _incNeurons(n), pPHLCheck(nullptr) {
				NNTL_ASSERT(i && n);
			}

			// variation to comply utils::for_eachwp_up() callback
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

			bool preparePHLCheck()noexcept {
				NNTL_ASSERT(_incNeurons && !pPHLCheck);
				pPHLCheck = new(std::nothrow) char[_incNeurons];
				if (pPHLCheck) memset(pPHLCheck, 0, _incNeurons);
				return nullptr != pPHLCheck;
			}

			// variation to comply utils::for_each_up() callback (used by _layer_pack_horizontal)
			template<typename PHLT>
			void operator()(PHLT& phl)noexcept {
				static_assert(std::is_base_of<_i_layer<PHLT::layer_t::real_t>, PHLT::layer_t>::value, "Each layer must derive from i_layer");
				static_assert(!std::is_base_of<m_layer_input, PHLT::layer_t>::value && !std::is_base_of<m_layer_output, PHLT::layer_t>::value,
					"No input/output layers is allowed within _layer_pack_horizontal");
				
				NNTL_ASSERT(pPHLCheck && phl.m_offset < _incNeurons && (phl.m_offset + phl.m_count) <= _incNeurons);
				const auto pBeg = pPHLCheck + phl.m_offset, pEnd = pBeg + phl.m_count;
				std::fill(pBeg, pEnd, char(1));

#ifdef NNTL_DEBUG
				layer_index_t curIdx = _idx;
#endif // NNTL_DEBUG
				phl.l._preinit_layer(_idx, phl.m_count);
				NNTL_ASSERT(_idx > curIdx);
			}

			bool PHLCheck()noexcept {
				NNTL_ASSERT(pPHLCheck);
				const bool r = std::all_of(pPHLCheck, pPHLCheck + _incNeurons, [](const char c)->bool {
					return c == char(1);
				});
				delete[] pPHLCheck;
				pPHLCheck = nullptr;
				return r;
			}
		};

		//////////////////////////////////////////////////////////////////////////
		// structure to be used in _i_layer.init()
		template<typename i_math_t_, typename i_rng_t_>
		struct _layer_init_data : public math::simple_matrix_typedefs {
			typedef i_math_t_ i_math_t;
			typedef i_rng_t_ i_rng_t;
			typedef typename i_math_t::real_t real_t;

			static_assert(std::is_base_of<math::_i_math<real_t>, i_math_t>::value, "i_math_t type should be derived from _i_math");
			static_assert(std::is_base_of<rng::_i_rng, i_rng_t>::value, "i_rng_t type should be derived from _i_rng");

			IN i_math_t& iMath;
			IN i_rng_t& iRng;

			//fprop and bprop may use different batch sizes during single training session (for example, fprop()/bprop() uses small batch size
			// during learning process, but whole data_x.rows() during fprop() for loss function computation. Therefore to reduce memory
			// consumption during learning, we will demark fprop() memory requirements and bprop() mem reqs.
			OUT numel_cnt_t maxMemFPropRequire;//this value should be set by layer.init()
			OUT numel_cnt_t maxMemBPropRequire;//this value should be set by layer.init()
			OUT numel_cnt_t max_dLdA_numel;//this value should be set by layer.init()
			OUT numel_cnt_t nParamsToLearn;//total number of parameters, that layer has to learn during training, to be set by layer.init()

			IN const vec_len_t max_fprop_batch_size;//usually this is data_x.rows()
			IN const vec_len_t training_batch_size;//usually this is batchSize

			OUT bool bHasLossAddendum;//to be set by layer.init()

			_layer_init_data(i_math_t& im, i_rng_t& ir, vec_len_t fbs, vec_len_t bbs) noexcept
				: iMath(im), iRng(ir), max_fprop_batch_size(fbs), training_batch_size(bbs)
			{
				NNTL_ASSERT(max_fprop_batch_size >= training_batch_size);//essential assumption
			}

			void clean()noexcept {
				maxMemFPropRequire = 0;
				maxMemBPropRequire = 0;
				max_dLdA_numel = 0;
				nParamsToLearn = 0;
				bHasLossAddendum = false;
			}

			void update(const _layer_init_data& o)noexcept {
				maxMemFPropRequire = std::max(maxMemFPropRequire, o.maxMemFPropRequire);
				maxMemBPropRequire = std::max(maxMemBPropRequire, o.maxMemBPropRequire);
				max_dLdA_numel = std::max(max_dLdA_numel, o.max_dLdA_numel);
				nParamsToLearn += o.nParamsToLearn;
				bHasLossAddendum |= o.bHasLossAddendum;
			}

			_layer_init_data dupe()const noexcept {
				return _layer_init_data(iMath, iRng, max_fprop_batch_size, training_batch_size);
			}
		};

		//////////////////////////////////////////////////////////////////////////
		// structure to be used in layers.init()
		template<typename RealT>
		struct layers_mem_requirements : public math::simple_matrix_typedefs {
			numel_cnt_t maxMemLayerTrainingRequire,//useful for nnet.train()
				maxMemLayersFPropRequire,//useful for nnet.eval()
				maxSingledLdANumel, totalParamsToLearn;

			bool bHasLossAddendum;

			void zeros()noexcept {
				maxMemLayerTrainingRequire = 0;
				maxMemLayersFPropRequire = 0;
				maxSingledLdANumel = 0;//single! biggest matrix numel() used in bprop()
				totalParamsToLearn = 0;
				bHasLossAddendum = false;
			}

			void updateLayerReq(numel_cnt_t mmlF, numel_cnt_t mmlB
				, numel_cnt_t maxdLdA, numel_cnt_t nLP, bool _HasLossAddendum)noexcept
			{
				maxMemLayerTrainingRequire = std::max({ maxMemLayerTrainingRequire, mmlF, mmlB });
				maxMemLayersFPropRequire = std::max(maxMemLayersFPropRequire, mmlF);
				maxSingledLdANumel = std::max(maxSingledLdANumel, maxdLdA);
				totalParamsToLearn += nLP;
				bHasLossAddendum |= _HasLossAddendum;
			}
			template<typename _layer_init_data_t>
			void updateLayerReq(const _layer_init_data_t& lid)noexcept {
				maxMemLayerTrainingRequire = std::max({ maxMemLayerTrainingRequire, lid.maxMemFPropRequire, lid.maxMemBPropRequire });
				maxMemLayersFPropRequire = std::max(maxMemLayersFPropRequire, lid.maxMemFPropRequire);
				maxSingledLdANumel = std::max(maxSingledLdANumel, lid.max_dLdA_numel);
				totalParamsToLearn += lid.nParamsToLearn;
				bHasLossAddendum |= lid.bHasLossAddendum;
			}
		};
	}

}