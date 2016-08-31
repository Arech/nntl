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

namespace nntl {
	
	//BTW: each and every i_inspector's method must posses const function specifier to permit maximum optimizations
	//Derive from this class to have default function implementations
	// 
	template<typename RealT>
	class i_inspector : public math::smatrix_td {
		//!! copy constructor not needed
		i_inspector(const i_inspector& other)noexcept = delete;
		//!!assignment is not needed
		i_inspector& operator=(const i_inspector& rhs) noexcept = delete;

		//////////////////////////////////////////////////////////////////////////
	public:
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	public:
		~i_inspector()noexcept {}
		i_inspector()noexcept {}

		//////////////////////////////////////////////////////////////////////////
		// generic functions
		template<typename VarT>
		void inspect(const VarT& v, const layer_index_t lIdx=0, const char*const pVarName=nullptr)const noexcept {}

		//////////////////////////////////////////////////////////////////////////
		// specialized functions naming convention:
		// <phase>_<prefix><A/actionCamelCased><Suffix>()

		//to notify about total layer, epoch and batches count
		void init_nnet(const size_t totalLayers, const size_t totalEpochs, const vec_len_t totalBatches)const noexcept {}

		//to notify about layer and it's name (for example, inspector can use this info to filter out calls from non-relevant layers later)
		//this call can cost something, but we don't care because it happens only during init phase
		template<typename StrT>
		void init_layer(const layer_index_t lIdx, StrT&& LayerName)const noexcept {};

		void train_epochBegin(const size_t epochIdx)const noexcept {}
		void train_batchBegin(const vec_len_t batchIdx)const noexcept {}

		void fprop_SourceData(const realmtx_t& data_x)const noexcept {}



		void train_batchEnd(const vec_len_t batchIdx)const noexcept {}
		void train_epochEnd(const size_t epochIdx)const noexcept {}
	};

	template< class, class = std::void_t<> >
	struct is_dummy_inspector : std::false_type { };
	template< class T >
	struct is_dummy_inspector<T, std::void_t<typename std::enable_if< std::is_same<i_inspector<typename T::real_t>, T>::value >::type > > : std::true_type {};


}
