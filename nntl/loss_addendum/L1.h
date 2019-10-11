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

#include "_i_loss_addendum.h"

namespace nntl {
namespace loss_addendum {

	//if bCalcOnFProp set to true, then it computes the necessary derivate during fprop step and stores it internally
	//until bprop() phase. This helps to deal with a dropout that modifies some activations and makes some loss_addendums
	// produce bogus results
	template<typename RealT, bool bCalcOnFProp = false, bool bAppendToNZGrad = false>
	class L1 : public _impl::scaled_addendum_with_mtx4fprop<RealT, bCalcOnFProp, bAppendToNZGrad, false>{
	public:
		static constexpr const char* getName()noexcept { return "L1"; }

		//computes loss addendum for a given matrix of values (for weight-decay Vals parameter is a weight matrix)
		template <typename CommonDataT>
		real_t lossAdd(const realmtx_t& Vals, const CommonDataT& CD) const noexcept {
			return m_scale* CD.iMath().vSumAbs(Vals);
		}

		template <typename CommonDataT, bool c = calcOnFprop>
		::std::enable_if_t<c> on_fprop(const realmtx_t& Vals, const CommonDataT& CD) noexcept {
			NNTL_ASSERT(!Vals.emulatesBiases());
			m_Mtx.deform_like(Vals);
			CD.iMath().evSign(m_Mtx, Vals);
		}

		template <typename CommonDataT, bool c = calcOnFprop>
		::std::enable_if_t<c> dLossAdd(const realmtx_t& Vals, realmtx_t& dLossdVals, const CommonDataT& CD) const noexcept {
			NNTL_ASSERT(m_Mtx.size() == Vals.size() && Vals.size() == dLossdVals.size());
			NNTL_ASSERT(!Vals.emulatesBiases() && !dLossdVals.emulatesBiases());
			//CD.iMath().evAddScaled_ip(dLossdVals, m_scale, m_Mtx);
			_appendGradient(CD.iMath(), dLossdVals, m_Mtx);
		}

		template <typename CommonDataT, bool c = calcOnFprop>
		::std::enable_if_t<!c> dLossAdd(const realmtx_t& Vals, realmtx_t& dLossdVals, const CommonDataT& CD) const noexcept {
			NNTL_ASSERT(!Vals.emulatesBiases() && !dLossdVals.emulatesBiases());
			//CD.iMath().evAddScaledSign_ip(dLossdVals, m_scale, Vals);
			_appendGradientSign(CD.iMath(), dLossdVals, Vals);
		}

	protected:
		template<typename iMathT, bool c = appendToNZGrad>
		::std::enable_if_t<c> _appendGradientSign(iMathT& iM, realmtx_t& dLossdVals, const realmtx_t& newGrad)const noexcept {
			iM.evNZAddScaledSign_ip(dLossdVals, m_scale, newGrad);
		}
		template<typename iMathT, bool c = appendToNZGrad>
		::std::enable_if_t<!c> _appendGradientSign(iMathT& iM, realmtx_t& dLossdVals, const realmtx_t& newGrad)const noexcept {
			iM.evAddScaledSign_ip(dLossdVals, m_scale, newGrad);
		}
	};

}
}