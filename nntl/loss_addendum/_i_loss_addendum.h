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

#include "../_defs.h"
#include "../common.h"

// loss_addendums are small classes that implements some kind of (surprise!) addendum to a loss function.
// Typically it's kind of a penalty like the weigth-decay.
// 

namespace nntl {
namespace loss_addendum {

/*
	struct init_struct : public math::smatrix_td {
		//to initialize iMath internal memory storage a loss_addendum implementation must return maximum elements count for
		//fprop() only case and training case
		OUT numel_cnt_t maxIMathMemFPropRequire;
		OUT numel_cnt_t maxIMathMemTrainingRequire;
	};*/

	//////////////////////////////////////////////////////////////////////////
	// Each _i_loss_addendum derived class must also be:
	// - default constructible (it must be instantiate-able as other class's member)
	// - inactive by default
	template<typename RealT>
	class _i_loss_addendum : public math::smatrix_td {
	public:
		typedef RealT real_t;

		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;
		//typedef init_struct init_struct_t;

		//computes loss addendum for a given matrix of values (for weight-decay Vals parameter is a weight matrix)
		template <typename iMath>
		nntl_interface real_t lossAdd(const realmtx_t& Vals, iMath& iM) noexcept;

		template <typename iMath>
		nntl_interface void dLossAdd(const realmtx_t& Vals, realmtx_t& dLossdVals, iMath& iM) noexcept;

		// Performs initialization.
		//At mininum, it must call iMath.preinit()
		//override in derived class to suit needs
		template <typename iMath>
		static void init(const bool& bWillDoTraining, const realmtx_t& biggestMtx, iMath& iM) noexcept {
			iM.preinit(biggestMtx.numel_no_bias());
		}

		//Performs initialization.
		//At mininum, it must calculate how much temporary memory iMath object must allocate in order  to use lossAdd() or dLossAdd over a given matrix 
		//override in derived class to suit needs
		/*template <typename iMath>
		static void init(const realmtx_t& biggestMtx, init_struct_t& initStr, iMath& iM) noexcept {
			initStr.maxIMathMemFPropRequire = biggestMtx.numel_no_bias();
			initStr.maxIMathMemTrainingRequire = initStr.maxIMathMemFPropRequire;
		}*/
	};

	template<typename RealT>
	class _scaled_addendum : public _i_loss_addendum<RealT> {
	protected:
		real_t m_scale;

	public:
		_scaled_addendum()noexcept : m_scale(real_t(0.)) {}

		void scale(const real_t& s)noexcept {
			//NNTL_ASSERT(s >= real_t(0.));
			m_scale = s;
		}
		const real_t& scale()const noexcept { return m_scale; }
	};


}
}