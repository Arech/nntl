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

// This file provides a definition of inspector's interface and a dummy implementation of the interface
// that does nothing and effectively thrown away by a compiler (which results no run-time costs using the inspector's object).
// Other implementations may contain, for example, some data dumping facilities that provides a great help with
// debugging and baby-sitting a NN learning process.
//
// By default inspector's interface is defined as non-intrusive, however this isn't a strict requirement and
// could be overridden (but use that functionality on you own risk).
// 
// BTW, actually i_inspector API allows one to implement not only the inspection of values, but a full-scale control
// over the learning process including pausing/inspecting/modifying and so on. But that's a story for a future.

#include "math/smatrix.h"
#include "../utils/vector_conditions.h"
#include <stack>

namespace nntl {
namespace inspector {

	// interface is nowhere near a stable state, so expect changes.
	template<typename RealT>
	class _i_inspector : public math::smatrix_td {
		//!! copy constructor not needed
		_i_inspector(const _i_inspector& other)noexcept = delete;
		//!!assignment is not needed
		_i_inspector& operator=(const _i_inspector& rhs) noexcept = delete;

		//////////////////////////////////////////////////////////////////////////
	public:
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	protected:
		~_i_inspector()noexcept {}
		_i_inspector()noexcept {}

		static_assert(std::is_unsigned<layer_index_t>::value, "layer_index_t must be unsigned!");
		static constexpr layer_index_t _NoLayerIdxSpecified = layer_index_t(-1);

	public:
		//////////////////////////////////////////////////////////////////////////
		// generic functions
		template<typename VarT>
		nntl_interface void inspect(const VarT& v, const char*const pVarName = nullptr, const layer_index_t lIdx = _NoLayerIdxSpecified)const noexcept;

		//////////////////////////////////////////////////////////////////////////
		// specialized functions naming convention:
		// <phase>_<prefix><A/actionCamelCased><Suffix>()

		//to notify about total layer, epoch and batches count
		nntl_interface void init_nnet(const size_t totalLayers, const size_t totalEpochs, const vec_len_t totalBatches)const noexcept;

		//to notify about layer and it's name (for example, inspector can use this info to filter out calls from non-relevant layers later)
		//this call can cost something, but we don't care because it happens only during init phase
		template<typename StrT>
		nntl_interface void init_layer(const layer_index_t lIdx, StrT&& LayerName, const layer_type_id_t layerTypeId)const noexcept;

		nntl_interface void train_epochBegin(const size_t epochIdx)const noexcept;
		nntl_interface void train_epochEnd()const noexcept;

		//train_batch* functions are called during learning process only
		nntl_interface void train_batchBegin(const vec_len_t batchIdx)const noexcept;
		nntl_interface void train_batchEnd()const noexcept;

		//the following two functions are called during learning process only
		nntl_interface void train_preFprop(const realmtx_t& data_x)const noexcept;
		nntl_interface void train_preBprop(const realmtx_t& data_y)const noexcept;

		//the following 2 functions are called during learning process only
		nntl_interface void train_preCalcError(const bool bOnTrainSet)const noexcept;
		nntl_interface void train_postCalcError()const noexcept;

		//////////////////////////////////////////////////////////////////////////
		// FPROP
		//all calls between the following pair are guaranteed to be initiated be the same layer, however, nested calls are possible
		nntl_interface void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept;
		nntl_interface void fprop_end(const realmtx_t& Act) const noexcept;

		nntl_interface void fprop_preNesterovMomentum(const realmtx_t& vW, const real_t momentum, const realmtx_t& W)const noexcept;
		nntl_interface void fprop_postNesterovMomentum(const realmtx_t& vW, const realmtx_t& W)const noexcept;

		nntl_interface void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)const noexcept;
		nntl_interface void fprop_preactivations(const realmtx_t& Z)const noexcept;
		nntl_interface void fprop_activations(const realmtx_t& Act)const noexcept;

		//NB: we're using inverted dropout
		nntl_interface void fprop_preDropout(const realmtx_t& Act, const real_t dpa, const realmtx_t& dropoutMaskSrc)const noexcept;
		nntl_interface void fprop_postDropout(const realmtx_t& Act, const realmtx_t& dropoutMask)const noexcept;

		//////////////////////////////////////////////////////////////////////////
		//BPROP
		//all calls between the following pair are guaranteed to be initiated be the same layer, however, nested calls are possible
		nntl_interface void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) const noexcept;
		nntl_interface void bprop_end(const realmtx_t& dLdAPrev) const noexcept;

		nntl_interface void bprop_preCancelDropout(const realmtx_t& Act, const real_t dpa) const noexcept;
		nntl_interface void bprop_postCancelDropout(const realmtx_t& Act) const noexcept;
		
		nntl_interface void bprop_predLdZOut(const realmtx_t& Act, const realmtx_t& data_y) const noexcept;

		nntl_interface void bprop_predAdZ(const realmtx_t& Act) const noexcept;
		nntl_interface void bprop_dAdZ(const realmtx_t& dAdZ) const noexcept;
		nntl_interface void bprop_dLdZ(const realmtx_t& dLdZ) const noexcept;
		nntl_interface void bprop_postClampdLdZ(const realmtx_t& dLdZ,const real_t& Ub, const real_t& Lb) const noexcept;

		nntl_interface void apply_grad_begin(const realmtx_t& W, const realmtx_t& dLdW)const noexcept;
		nntl_interface void apply_grad_end(const realmtx_t& W)const noexcept;
		nntl_interface void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept;

		nntl_interface void apply_grad_preNesterovMomentum(const realmtx_t& vW, const realmtx_t& dLdW)const noexcept;
		nntl_interface void apply_grad_postNesterovMomentum(const realmtx_t& vW)const noexcept;
	};

	namespace _impl {
		//BTW: each and every _base's method must posses const function specifier to permit maximum optimizations
		//Derive from this class to have default function implementations
		template<typename RealT>
		class _base : public _i_inspector<RealT> {
		public:
			~_base()noexcept {}
			_base()noexcept {}

			//////////////////////////////////////////////////////////////////////////
			// generic functions
			template<typename VarT>
			void inspect(const VarT& v, const char*const pVarName = nullptr, const layer_index_t lIdx = _NoLayerIdxSpecified)const noexcept {}

			//////////////////////////////////////////////////////////////////////////
			// specialized functions naming convention:
			// <phase>_<prefix><A/actionCamelCased><Suffix>()

			//to notify about total layer, epoch and batches count
			void init_nnet(const size_t totalLayers, const size_t totalEpochs, const vec_len_t totalBatches)const noexcept {}

			//to notify about layer and it's name (for example, inspector can use this info to filter out calls from non-relevant layers later)
			//this call can cost something, but we don't care because it happens only during init phase
			template<typename StrT>
			void init_layer(const layer_index_t lIdx, StrT&& LayerName, const layer_type_id_t layerTypeId)const noexcept {};

			void train_epochBegin(const size_t epochIdx)const noexcept {}
			void train_epochEnd()const noexcept {}

			void train_batchBegin(const vec_len_t batchIdx)const noexcept {}
			void train_batchEnd()const noexcept {}

			//the following two functions are called during learning process only
			void train_preFprop(const realmtx_t& data_x)const noexcept {}
			void train_preBprop(const realmtx_t& data_y)const noexcept {}

			//the following 2 functions are called during learning process only
			void train_preCalcError(const bool bOnTrainSet)const noexcept {};
			void train_postCalcError()const noexcept {};

			//////////////////////////////////////////////////////////////////////////
			// FPROP
			void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept {}
			void fprop_end(const realmtx_t& Act) const noexcept {}

			void fprop_preNesterovMomentum(const realmtx_t& vW, const real_t momentum, const realmtx_t& W)const noexcept {}
			void fprop_postNesterovMomentum(const realmtx_t& vW, const realmtx_t& W)const noexcept {}

			void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)const noexcept {}
			void fprop_preactivations(const realmtx_t& Z)const noexcept {}
			void fprop_activations(const realmtx_t& Act)const noexcept {}

			void fprop_preDropout(const realmtx_t& Act, const real_t dpa, const realmtx_t& dropoutMaskSrc)const noexcept {}
			void fprop_postDropout(const realmtx_t& Act, const realmtx_t& dropoutMask)const noexcept {}

			//////////////////////////////////////////////////////////////////////////
			//BPROP
			void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) const noexcept {}
			void bprop_end(const realmtx_t& dLdAPrev) const noexcept {}

			void bprop_preCancelDropout(const realmtx_t& Act, const real_t dpa) const noexcept {}
			void bprop_postCancelDropout(const realmtx_t& Act) const noexcept {}

			void bprop_predLdZOut(const realmtx_t& Act, const realmtx_t& data_y) const noexcept{}
			void bprop_predAdZ(const realmtx_t& Act) const noexcept{}
			void bprop_dAdZ(const realmtx_t& dAdZ) const noexcept{}
			void bprop_dLdZ(const realmtx_t& dLdZ) const noexcept{}
			void bprop_postClampdLdZ(const realmtx_t& dLdZ, const real_t& Ub, const real_t& Lb) const noexcept{}

			void apply_grad_begin(const realmtx_t& W, const realmtx_t& dLdW)const noexcept {}
			void apply_grad_end(const realmtx_t& W)const noexcept {}

			void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept{}

			void apply_grad_preNesterovMomentum(const realmtx_t& vW, const realmtx_t& dLdW)const noexcept {}
			void apply_grad_postNesterovMomentum(const realmtx_t& vW)const noexcept {}
		};

		//////////////////////////////////////////////////////////////////////////
		//helper to store current layer index. Maximum depth is hardcoded into _maxDepth, but checked only during DEBUG builds
		template<typename IdxT, IdxT defaultVal, size_t _maxDepth>
		class layer_idx_keeper : private std::stack<IdxT, std::vector<IdxT>> {
		private:
			typedef std::stack<IdxT, std::vector<IdxT>> _base_class;
		public:
			typedef IdxT value_t;
			static constexpr size_t maxDepth = _maxDepth;
			static constexpr value_t default_value = defaultVal;

		public:
			~layer_idx_keeper()noexcept {
				NNTL_ASSERT(0 == size());
			}
			layer_idx_keeper()noexcept {
				c.reserve(maxDepth);
			}

			void push(const value_t& v)noexcept {
				NNTL_ASSERT(size() < maxDepth);
				_base_class::push(v);//#exceptions STL
			}
			void pop()noexcept {
				NNTL_ASSERT(size());
				_base_class::pop();
			}
			value_t top()const noexcept {
				return size() ? _base_class::top() : default_value;
			}
			operator value_t()const noexcept {
				return top();
			}

		};

		//////////////////////////////////////////////////////////////////////////
		//base class to support black&white lists of layers to enhance filtering while dumping variables (it could take
		// a lot of time for a large network)
		template<typename FinalChildT, typename RealT>
		class _bwlist : public _base<RealT> {
		public:
			typedef FinalChildT self_t;
			typedef FinalChildT& self_ref_t;
			typedef const FinalChildT& self_cref_t;
			typedef FinalChildT* self_ptr_t;

			typedef std::vector<layer_type_id_t> layer_types_list_t;
			typedef std::vector<layer_index_t> layer_idx_list_t;

		protected:
			vector_conditions m_dumpLayerCond;
			layer_types_list_t m_layerTypes;
			layer_idx_list_t m_layerIdxs;

			//placed the following bool here for packing reasons
			bool m_bDoDump;//it's 'protected' for some rare special unforeseen cases. Don't access in
						   // derived classes, use the bDoDump() or the CondDumpT


			bool m_bWhiteList;


		public:
			self_ref_t get_self() noexcept {
				static_assert(std::is_base_of<_bwlist, FinalChildT>::value, "FinalChildT must derive from _bwlist");
				return static_cast<self_ref_t>(*this);
			}
			self_cref_t get_self() const noexcept {
				static_assert(std::is_base_of<_bwlist, FinalChildT>::value, "FinalChildT must derive from _bwlist");
				return static_cast<self_cref_t>(*this);
			}

		private:
			self_ref_t _addToList(const layer_type_id_t& t)noexcept {
				m_layerTypes.push_back(t);
				return get_self();
			}
			self_ref_t _addToList(const layer_index_t& lidx)noexcept {
				m_layerIdxs.push_back(lidx);
				return get_self();
			}

			template<typename LtlT>
			std::enable_if_t< std::is_same<layer_type_id_t, std::remove_cv_t<typename LtlT::value_type>>::value, self_ref_t>
			_addToList(LtlT&& tl)noexcept {
				NNTL_ASSERT(!m_layerTypes.size());
				m_layerTypes = std::forward<LtlT>(tl);
				return get_self();
			}
			template<typename LilT>
			std::enable_if_t< std::is_same<layer_index_t, std::remove_cv_t<typename LilT::value_type>>::value, self_ref_t>
			_addToList(LilT&& lil)noexcept {
				NNTL_ASSERT(!m_layerIdxs.size());
				m_layerIdxs = std::forward<LilT>(lil);
				return get_self();
			}

			void _checksetWhitelist()noexcept {
				if (!isWhiteList()) {
					NNTL_ASSERT(!m_layerTypes.size());
					NNTL_ASSERT(!m_layerIdxs.size());
					setWhiteList();
				}
			}
			void _checksetBlacklist()noexcept {
				if (isWhiteList()) {
					NNTL_ASSERT(!m_layerTypes.size());
					NNTL_ASSERT(!m_layerIdxs.size());
					setBlackList();
				}
			}

		protected:
			void _bwlist_init(const size_t totalLayers)noexcept {
				m_dumpLayerCond.clear()
					.resize(totalLayers, !m_bWhiteList);
			}
			void _bwlist_updateLayer(const layer_index_t lIdx, const layer_type_id_t layerTypeId)noexcept {
				NNTL_ASSERT(lIdx < m_dumpLayerCond.size());

				if (
					(m_layerTypes.size() && std::any_of(m_layerTypes.begin(), m_layerTypes.end(), [layerTypeId](const layer_type_id_t& e)->bool {
						return e == layerTypeId;
					})) || (m_layerIdxs.size() && std::any_of(m_layerIdxs.begin(), m_layerIdxs.end(), [lIdx](const layer_index_t& e)->bool {
						return e == lIdx;
					}))
				){
					m_dumpLayerCond.set(lIdx, m_bWhiteList);
				}
			}

			const bool _bwlist_layerAllowed(const layer_index_t lIdx)const noexcept {
				return m_dumpLayerCond(lIdx);
			}

		public:
			~_bwlist()noexcept{}
			_bwlist(const bool bWhitelistByDefault = false)noexcept : m_bDoDump(false), m_bWhiteList(bWhitelistByDefault) {}

			self_ref_t resetLists()noexcept {
				m_dumpLayerCond.clear();
				m_layerTypes.clear();
				m_layerIdxs.clear();
				return get_self();
			}
			self_ref_t setMode(const bool bWhiteList)noexcept {
				m_bWhiteList = bWhiteList;
				return resetLists();
			}
			const bool getMode()const noexcept { return m_bWhiteList; }
			self_ref_t setWhiteList()noexcept { return setMode(true); }
			self_ref_t setBlackList()noexcept { return setMode(false); }
			const bool isWhiteList()const noexcept { return getMode(); }
			const bool isBlackList()const noexcept { return !getMode(); }

			self_ref_t whitelist(const layer_type_id_t& p)noexcept {
				_checksetWhitelist();
				return _addToList(p);
			}
			template<typename LtlT>
			std::enable_if_t< std::is_same<layer_type_id_t, std::remove_cv_t<typename LtlT::value_type>>::value, self_ref_t>
			whitelist(LtlT&& p)noexcept {
				_checksetWhitelist();
				return _addToList(std::forward<LtlT>(p));
			}
			self_ref_t whitelist(const layer_index_t& p)noexcept {
				_checksetWhitelist();
				return _addToList(p);
			}
			template<typename LilT>
			std::enable_if_t< std::is_same<layer_index_t, std::remove_cv_t<typename LilT::value_type>>::value, self_ref_t>
			whitelist(LilT&& p)noexcept {
				_checksetWhitelist();
				return _addToList(std::forward<LilT>(p));
			}

			self_ref_t blacklist(const layer_type_id_t& p)noexcept {
				_checksetBlacklist();
				return _addToList(p);
			}
			template<typename LtlT>
			std::enable_if_t< std::is_same<layer_type_id_t, std::remove_cv_t<typename LtlT::value_type>>::value, self_ref_t>
			blacklist(LtlT&& p)noexcept {
				_checksetBlacklist();
				return _addToList(std::forward<LtlT>(p));
			}
			self_ref_t blacklist(const layer_index_t& p)noexcept {
				_checksetBlacklist();
				return _addToList(p);
			}
			template<typename LilT>
			std::enable_if_t< std::is_same<layer_index_t, std::remove_cv_t<typename LilT::value_type>>::value, self_ref_t>
			blacklist(LilT&& p)noexcept {
				_checksetBlacklist();
				return _addToList(std::forward<LilT>(p));
			}
		};
	}

}
}
