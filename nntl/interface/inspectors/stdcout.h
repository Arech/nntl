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

#include "../_i_inspector.h"

namespace nntl {
namespace inspector {

	template<typename RealT, size_t maxNnetDepth=10>
	class stdcout : public _impl::_base<RealT>{
	public:
		typedef ::std::vector<::std::string> layer_names_t;
		typedef stdcout<real_t, maxNnetDepth> self_t;

	protected:
		typedef utils::layer_idx_keeper<layer_index_t, _NoLayerIdxSpecified, maxNnetDepth> keeper_t;

	protected:
		layer_names_t m_layerNames;
		size_t m_epochCount, m_layersCount, m_epochIdx;
		vec_len_t m_batchIdx, m_batchCount;

		//layer_index_t m_curLayer;
		keeper_t m_curLayer;

		static constexpr char* _noLayerName = "[NoName]";

	protected:
		const char*const _layer_name(const layer_index_t lIdx)const noexcept {
			const auto _li = lIdx == _NoLayerIdxSpecified ? m_curLayer : lIdx;
			return _li < m_layersCount ? m_layerNames[_li].c_str() : _noLayerName;
		}

	public:
		~stdcout()noexcept {}
		stdcout()noexcept : m_epochIdx(::std::numeric_limits<decltype(m_epochIdx)>::max()),
			m_batchIdx(::std::numeric_limits<decltype(m_batchIdx)>::max())
			, m_epochCount(0), m_layersCount(0), m_batchCount(0) {}

		//////////////////////////////////////////////////////////////////////////
		//
		template<typename VarT> ::std::enable_if_t<!::std::is_base_of<realmtx_t, VarT>::value>
		inspect(const VarT& v, const char*const pVarName = nullptr, const layer_index_t lIdx = _NoLayerIdxSpecified)const noexcept
		{
			STDCOUT(_layer_name(lIdx));
			STDCOUTL("@" << m_epochIdx << "#" << m_batchIdx << " var \'" << (pVarName ? pVarName : "unk") << "\' = " << v);
		}
		template<typename VarT> ::std::enable_if_t<::std::is_base_of<realmtx_t, VarT>::value>
		inspect(const VarT& v, const char*const pVarName = nullptr, const layer_index_t lIdx = _NoLayerIdxSpecified)const noexcept
		{
			STDCOUT(_layer_name(lIdx));
			STDCOUTL("@" << m_epochIdx << "#" << m_batchIdx << " mtx \'" << (pVarName ? pVarName : "unk") << "\' size = [" << v.rows() << "," << v.cols() << "]");
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
		void init_layer(const layer_index_t lIdx, StrT&& LayerName, const layer_type_id_t layerTypeId)noexcept {
			NNTL_ASSERT(lIdx < m_layersCount);
			//#exceptions STL
			m_layerNames[lIdx].assign(::std::forward<StrT>(LayerName));
			STDCOUTL("Layer " << m_layerNames[lIdx] << " of type 0x" << ::std::hex << layerTypeId << ::std::dec << " is being initialized");
		};

		void train_epochBegin(const size_t epochIdx)noexcept {
			m_epochIdx = epochIdx;
		}
		void train_batchBegin(const vec_len_t batchIdx) noexcept {
			m_batchIdx = batchIdx;
		}

		//////////////////////////////////////////////////////////////////////////

		void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) noexcept {
			m_curLayer.push(lIdx);
			STDCOUT("fp:" << (bTrainingMode ? "tr, " : "t, "));
			inspect(prevAct, "previous activations");
		}
		void fprop_end(const realmtx_t& act) noexcept {
			inspect(act, "current activations");
			m_curLayer.pop();
		}
		void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) noexcept {
			m_curLayer.push(lIdx);
			STDCOUT("bp: ");
			inspect(dLdA, "dL/dA");
		}
		void bprop_end(const realmtx_t& dLdAPrev) noexcept {
			inspect(dLdAPrev, "dL/dA for a layer down the stack");
			m_curLayer.pop();
		}


		void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)const noexcept {
			NNTL_UNREF(prevAct);
			inspect(W, "Current weights");
		}

		void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept {
			inspect(W, "weights@apply_grad");
			inspect(WUpd, "Weights update@apply_grad");
		}
	};


}
}