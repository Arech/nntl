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

#include "../_i_inspector.h"

namespace nntl {
namespace inspector {

	template<typename RealT>
	class stdcout : public _base<RealT>{
	public:
		typedef std::vector<std::string> layer_names_t;

	public:
		layer_names_t m_layerNames;
		size_t m_epochCount, m_layersCount, m_epochIdx;
		vec_len_t m_batchIdx, m_batchCount;

	public:
		~stdcout()noexcept {}
		stdcout()noexcept : m_epochIdx(-1), m_batchIdx(-1), m_epochCount(0), m_layersCount(0), m_batchCount(0) {}

		//////////////////////////////////////////////////////////////////////////
		//
		template<typename VarT> std::enable_if_t<!std::is_base_of<realmtx_t, VarT>::value>
		inspect(const VarT& v, const layer_index_t lIdx = 0, const char*const pVarName = nullptr)const noexcept
		{
			if (lIdx < m_layersCount) {
				STDCOUT("[" << m_layerNames[lIdx]);
			} else STDCOUT("[OutOfALayer");
			STDCOUTL("] @ " << m_epochIdx << "\\" << m_batchIdx << " variable \'" << (pVarName ? pVarName : "unk") << "\' = " << v);
		}
		template<typename VarT> std::enable_if_t<std::is_base_of<realmtx_t, VarT>::value>
		inspect(const VarT& v, const layer_index_t lIdx = 0, const char*const pVarName = nullptr)const noexcept
		{
			if (lIdx < m_layersCount) {
				STDCOUT("[" << m_layerNames[lIdx]);
			} else STDCOUT("[OutOfALayer");
			STDCOUTL("] @ " << m_epochIdx << "\\" << m_batchIdx << " matrix \'" << (pVarName ? pVarName : "unk") << "\' size = [" << v.rows() << "," << v.cols() << "]");
		}

		//////////////////////////////////////////////////////////////////////////

		//to notify about total layer, epoch and batches count
		void init_nnet(const size_t totalLayers, const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
			m_layersCount = totalLayers;
			m_epochCount = totalEpochs;
			m_batchCount = totalBatches;

			//#exceptions STL
			m_layerNames.resize(m_layersCount);
			m_layerNames.shrink_to_fit();
		}

		template<typename StrT>
		void init_layer(const layer_index_t lIdx, StrT&& LayerName)noexcept {
			NNTL_ASSERT(lIdx < m_layersCount);
			//#exceptions STL
			m_layerNames[lIdx].assign(std::forward<StrT>(LayerName));
			STDCOUTL("Layer " << m_layerNames[lIdx] << " is being initialized");
		};

		void train_epochBegin(const size_t epochIdx)noexcept {
			m_epochIdx = epochIdx;
		}
		void train_batchBegin(const vec_len_t batchIdx) noexcept {
			m_batchIdx = batchIdx;
		}

		void fprop_onEntry(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept {
			STDCOUT("fprop:" << (bTrainingMode?"training, ":"testing, "));
			inspect(prevAct, lIdx, "previous activations");
		}

	};


}
}