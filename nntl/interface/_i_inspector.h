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
#include "../train_data/_i_train_data.h"
#include "../utils/layer_idx_keeper.h"

namespace nntl {
namespace inspector {

	template< class, class = ::std::void_t<> >
	struct is_gradcheck_inspector : ::std::false_type { };
	template< class T >
	struct is_gradcheck_inspector<T, ::std::void_t<typename T::gradcheck_inspector_t>> : ::std::true_type {};

	//template-less base class
	struct _i_inspector_base : public virtual DataSetsId {};
	
	// interface is nowhere near a stable state, so expect changes.
	template<typename RealT>
	class _i_inspector : public _i_inspector_base { //public math::smatrix_td {
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

		static_assert(::std::is_unsigned<layer_index_t>::value, "layer_index_t must be unsigned!");
		static constexpr layer_index_t _NoLayerIdxSpecified = layer_index_t(-1);

	public:
		//////////////////////////////////////////////////////////////////////////
		// generic functions
		template<typename VarT>
		nntl_interface void inspect(const VarT& v, const char*const pVarName = nullptr, const layer_index_t lIdx = _NoLayerIdxSpecified)const noexcept;

		//////////////////////////////////////////////////////////////////////////
		// specialized functions naming convention:
		// <phase>_<prefix><A/actionCamelCased><Suffix>()

		//to notify about total layers and epochs count
		nntl_interface void init_nnet(const size_t totalLayers, const numel_cnt_t totalEpochs)const noexcept;

		//to notify about layer and it's name (for example, inspector can use this info to filter out calls from non-relevant layers later)
		//this call can cost something, but we don't care because it happens only during init phase
		template<typename StrT>
		nntl_interface void init_layer(const layer_index_t lIdx, StrT&& LayerName, const layer_type_id_t layerTypeId)const noexcept;

		nntl_interface void train_epochBegin(const numel_cnt_t epochIdx)const noexcept;
		nntl_interface void train_epochEnd()const noexcept;

		//train_batch* functions are called during learning process only
		nntl_interface void train_batchBegin(const numel_cnt_t batchIdx)const noexcept;
		nntl_interface void train_batchEnd()const noexcept;

		//the following two functions are called during learning process only
		nntl_interface void train_preFprop(const realmtx_t& data_x)const noexcept;
		nntl_interface void train_preBprop(const realmtx_t& data_y)const noexcept;

		//the following 2 functions are called during learning process only
		nntl_interface void train_preCalcError(const data_set_id_t dataSetId)const noexcept;
		nntl_interface void train_postCalcError()const noexcept;

		//////////////////////////////////////////////////////////////////////////
		// FPROP
		//all calls between the following pair are guaranteed to be initiated be the same layer, however, nested calls are possible
		nntl_interface void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept;
		nntl_interface void fprop_end(const realmtx_t& Act) const noexcept;

		nntl_interface void fprop_preNesterovMomentum(const realmtx_t& vW, const real_t momentum, const realmtx_t& W)const noexcept;
		nntl_interface void fprop_postNesterovMomentum(const realmtx_t& vW, const realmtx_t& W)const noexcept;

		nntl_interface void fprop_preLRDropout4NesterovMomentum(const realmtx_t& vW, const real_t dpa, const realmtx_t& dropoutMask)const noexcept;
		nntl_interface void fprop_postLRDropout4NesterovMomentum(const realmtx_t& vW)const noexcept;

		//fprop_makePreActivations() has two forms - for layer that has params to learn
		nntl_interface void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)const noexcept;
		//and for layers without params to learn
		nntl_interface void fprop_makePreActivations(const realmtx_t& prevAct)const noexcept;
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

		//this function gets a final dL/dA that gets applied to the layer. It may contain added derivatives of auxiliary loss functions
		//based on activation values (such as DeCov loss)
		nntl_interface void bprop_finaldLdA(const realmtx_t& dLdA) const noexcept;

		nntl_interface void bprop_preCancelDropout(const realmtx_t& dLdA, const realmtx_t& Act, const real_t dpa) const noexcept;
		nntl_interface void bprop_postCancelDropout(const realmtx_t& dLdA, const realmtx_t& Act) const noexcept;
		
		nntl_interface void bprop_predLdZOut(const realmtx_t& Act, const realmtx_t& data_y) const noexcept;
		nntl_interface void bprop_predAdZ(const realmtx_t& Act) const noexcept;
		nntl_interface void bprop_dAdZ(const realmtx_t& dAdZ) const noexcept;
		nntl_interface void bprop_dLdZ(const realmtx_t& dLdZ) const noexcept;
		nntl_interface void bprop_postClampdLdZ(const realmtx_t& dLdZ,const real_t& Ub, const real_t& Lb) const noexcept;
		nntl_interface void bprop_dLdW(const realmtx_t& dLdZ, const realmtx_t& prevAct, const realmtx_t& dLdW) const noexcept;

		nntl_interface void apply_grad_begin(const realmtx_t& W, const realmtx_t& dLdW)const noexcept;
		nntl_interface void apply_grad_end(const realmtx_t& W)const noexcept;
		nntl_interface void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept;

		nntl_interface void apply_grad_preNesterovMomentum(const realmtx_t& vW, const realmtx_t& dLdW)const noexcept;
		nntl_interface void apply_grad_postNesterovMomentum(const realmtx_t& vW)const noexcept;

		nntl_interface void apply_grad_postOptimizer(const realmtx_t& dLdW, const realmtx_t& M1, const realmtx_t& M2
			, const real_t& beta1t, const real_t& beta2t) const noexcept;

		nntl_interface void apply_grad_preILR(const realmtx_t& dLdW, const realmtx_t& prevdLdW, const realmtx_t& Gain) const noexcept;
		nntl_interface void apply_grad_postILR(const realmtx_t& dLdW, const realmtx_t& Gain) const noexcept;

		nntl_interface void apply_grad_preLRDropout(const realmtx_t& dLdW, const real_t dpa, const realmtx_t& dropoutMask)const noexcept;
		nntl_interface void apply_grad_postLRDropout(const realmtx_t& dLdW)const noexcept;

		//to monitor dLdA addendums
		nntl_interface void dLossAddendumScaled(const realmtx_t& dLoss, const realmtx_t& dLossAdd, const real_t& scale, const char*const pLossName)const noexcept;
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
			void inspect(const VarT& v, const char*const pVarName = nullptr, const layer_index_t lIdx = _NoLayerIdxSpecified)const noexcept {
				NNTL_UNREF(v);				NNTL_UNREF(pVarName);				NNTL_UNREF(lIdx);
			}

			//////////////////////////////////////////////////////////////////////////
			// specialized functions naming convention:
			// <phase>_<prefix><A/actionCamelCased><Suffix>()

			//to notify about total layers and epochs count
			void init_nnet(const size_t totalLayers, const numel_cnt_t totalEpochs)const noexcept {
				NNTL_UNREF(totalLayers);				NNTL_UNREF(totalEpochs);
			}

			//to notify about layer and it's name (for example, inspector can use this info to filter out calls from non-relevant layers later)
			//this call can cost something, but we don't care because it happens only during init phase
			template<typename StrT>
			void init_layer(const layer_index_t lIdx, StrT&& LayerName, const layer_type_id_t layerTypeId)const noexcept {
				NNTL_UNREF(lIdx);				NNTL_UNREF(LayerName);				NNTL_UNREF(layerTypeId);
			};

			void train_epochBegin(const numel_cnt_t epochIdx)const noexcept { NNTL_UNREF(epochIdx); }
			void train_epochEnd()const noexcept {}

			void train_batchBegin(const numel_cnt_t batchIdx)const noexcept { NNTL_UNREF(batchIdx); }
			void train_batchEnd()const noexcept {}

			//the following two functions are called during learning process only
			void train_preFprop(const realmtx_t& data_x)const noexcept { NNTL_UNREF(data_x); }
			void train_preBprop(const realmtx_t& data_y)const noexcept { NNTL_UNREF(data_y); }

			//the following 2 functions are called during learning process only
			void train_preCalcError(const data_set_id_t dataSetId)const noexcept { NNTL_UNREF(dataSetId); };
			void train_postCalcError()const noexcept {};

			//////////////////////////////////////////////////////////////////////////
			// FPROP
			void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept {
				NNTL_UNREF(lIdx);				NNTL_UNREF(prevAct);				NNTL_UNREF(bTrainingMode);
			}
			void fprop_end(const realmtx_t& Act) const noexcept { NNTL_UNREF(Act); }

			void fprop_preNesterovMomentum(const realmtx_t& vW, const real_t momentum, const realmtx_t& W)const noexcept {
				NNTL_UNREF(vW);				NNTL_UNREF(momentum);				NNTL_UNREF(W);
			}
			void fprop_postNesterovMomentum(const realmtx_t& vW, const realmtx_t& W)const noexcept {
				NNTL_UNREF(vW);				NNTL_UNREF(W);
			}

			void fprop_preLRDropout4NesterovMomentum(const realmtx_t& vW, const real_t dpa, const realmtx_t& dropoutMask)const noexcept {
				NNTL_UNREF(vW); NNTL_UNREF(dpa); NNTL_UNREF(dropoutMask);
			}
			void fprop_postLRDropout4NesterovMomentum(const realmtx_t& vW)const noexcept {
				NNTL_UNREF(vW);
			}

			void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)const noexcept {
				NNTL_UNREF(W);				NNTL_UNREF(prevAct);
			}
			void fprop_makePreActivations(const realmtx_t& prevAct)const noexcept { NNTL_UNREF(prevAct); }
			void fprop_preactivations(const realmtx_t& Z)const noexcept { NNTL_UNREF(Z); }
			void fprop_activations(const realmtx_t& Act)const noexcept { NNTL_UNREF(Act); }

			void fprop_preDropout(const realmtx_t& Act, const real_t dpa, const realmtx_t& dropoutMaskSrc)const noexcept {
				NNTL_UNREF(Act);				NNTL_UNREF(dpa);				NNTL_UNREF(dropoutMaskSrc);
			}
			void fprop_postDropout(const realmtx_t& Act, const realmtx_t& dropoutMask)const noexcept {
				NNTL_UNREF(Act);				NNTL_UNREF(dropoutMask);
			}

			//////////////////////////////////////////////////////////////////////////
			//BPROP
			void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) const noexcept {
				NNTL_UNREF(lIdx);				NNTL_UNREF(dLdA);
			}
			void bprop_end(const realmtx_t& dLdAPrev) const noexcept { NNTL_UNREF(dLdAPrev); }

			void bprop_finaldLdA(const realmtx_t& dLdA) const noexcept { NNTL_UNREF(dLdA); }

			void bprop_preCancelDropout(const realmtx_t& dLdA, const realmtx_t& Act, const real_t dpa) const noexcept {
				NNTL_UNREF(dLdA); NNTL_UNREF(Act);				NNTL_UNREF(dpa);
			}
			void bprop_postCancelDropout(const realmtx_t& dLdA, const realmtx_t& Act) const noexcept { NNTL_UNREF(dLdA); NNTL_UNREF(Act); }

			void bprop_predLdZOut(const realmtx_t& Act, const realmtx_t& data_y) const noexcept{
				NNTL_UNREF(Act);				NNTL_UNREF(data_y);
			}
			void bprop_predAdZ(const realmtx_t& Act) const noexcept{ NNTL_UNREF(Act); }
			void bprop_dAdZ(const realmtx_t& dAdZ) const noexcept{ NNTL_UNREF(dAdZ); }
			void bprop_dLdZ(const realmtx_t& dLdZ) const noexcept{ NNTL_UNREF(dLdZ); }
			void bprop_postClampdLdZ(const realmtx_t& dLdZ, const real_t& Ub, const real_t& Lb) const noexcept{
				NNTL_UNREF(dLdZ);				NNTL_UNREF(Ub);				NNTL_UNREF(Lb);
			}
			void bprop_dLdW(const realmtx_t& dLdZ, const realmtx_t& prevAct, const realmtx_t& dLdW) const noexcept {
				NNTL_UNREF(dLdZ);				NNTL_UNREF(prevAct);				NNTL_UNREF(dLdW);
			}

			void apply_grad_begin(const realmtx_t& W, const realmtx_t& dLdW)const noexcept {
				NNTL_UNREF(W); NNTL_UNREF(dLdW);
			}
			void apply_grad_end(const realmtx_t& W)const noexcept { NNTL_UNREF(W); }

			void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept{
				NNTL_UNREF(W); NNTL_UNREF(WUpd);
			}

			void apply_grad_preNesterovMomentum(const realmtx_t& vW, const realmtx_t& dLdW)const noexcept {
				NNTL_UNREF(vW); NNTL_UNREF(dLdW);
			}
			void apply_grad_postNesterovMomentum(const realmtx_t& vW)const noexcept { NNTL_UNREF(vW); }

			void apply_grad_postOptimizer(const realmtx_t& dLdW, const realmtx_t& M1, const realmtx_t& M2
				, const real_t& beta1t, const real_t& beta2t) const noexcept
			{
				NNTL_UNREF(dLdW);				
				NNTL_UNREF(M1);	NNTL_UNREF(M2);
				NNTL_UNREF(beta1t); NNTL_UNREF(beta2t);
			}

			void apply_grad_preILR(const realmtx_t& dLdW, const realmtx_t& prevdLdW, const realmtx_t& Gain) const noexcept{
				NNTL_UNREF(dLdW); NNTL_UNREF(prevdLdW); NNTL_UNREF(Gain);
			}
			void apply_grad_postILR(const realmtx_t& dLdW, const realmtx_t& Gain) const noexcept{
				NNTL_UNREF(dLdW); NNTL_UNREF(Gain);
			}

			void apply_grad_preLRDropout(const realmtx_t& dLdW, const real_t dpa, const realmtx_t& dropoutMask)const noexcept {
				NNTL_UNREF(dLdW); NNTL_UNREF(dpa); NNTL_UNREF(dropoutMask);
			}
			void apply_grad_postLRDropout(const realmtx_t& dLdW)const noexcept {
				NNTL_UNREF(dLdW);
			}

			void dLossAddendumScaled(const realmtx_t& dLoss, const realmtx_t& dLossAdd, const real_t& scale, const char*const pLossName)const noexcept{
				NNTL_UNREF(dLoss); NNTL_UNREF(dLossAdd); NNTL_UNREF(scale); NNTL_UNREF(pLossName);
			}
		};

	}
}
}
