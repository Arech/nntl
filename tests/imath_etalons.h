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

#include "simple_math_etalons.h"


//////////////////////////////////////////////////////////////////////////
// declare etalon functions here

//TODO: STUPID implementation!!! Rewrite
template<typename iMath>
void evCMulSub_ET(iMath& iM, realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
	iM.evMulC_ip_st_naive(vW, momentum);
	iM.evSub_ip_st_naive(W, vW);
}

void ewBinarize_ip_ET(realmtx_t& A, const real_t frac)noexcept;
template<typename BaseDestT>
void ewBinarize_ET(nntl::math::smatrix<BaseDestT>& Dest, const realmtx_t& A, const real_t frac)noexcept {
	auto pA = A.data();
	auto pD = Dest.data();
	const auto pAE = pA + A.numel();
	while (pA != pAE) {
		*pD++ = *pA++ > frac ? BaseDestT(1.0) : BaseDestT(0.0);
	}
}

void softmax_parts_ET(const realmtx_t& act, const real_t* pMax, real_t* pDenominator, real_t* pNumerator)noexcept;
//pTmp is a vector of length at least act.numel() + 2*act.rows()
void softmax_ET(realmtxdef_t& act, real_t* pTmp)noexcept;
real_t loss_softmax_xentropy_ET(const realmtx_t& activations, const realmtx_t& data_y)noexcept;


void apply_momentum_ET(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept;
void apply_ILR_ET(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain, const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept;
void evAbs_ET(realmtx_t& dest, const realmtx_t& src)noexcept;
void evAdd_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept;
void evAddScaled_ip_ET(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept;
void evAddScaledSign_ip_ET(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept;
void evSquare_ET(realmtx_t& dest, const realmtx_t& src)noexcept;
void evSub_ET(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;
void evSub_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept;
real_t loss_sigm_xentropy_ET(const realmtx_t& activations, const realmtx_t& data_y)noexcept;
void make_dropout_ET(realmtx_t& act, const real_t dfrac, realmtx_t& dropoutMask)noexcept;

void ModProp_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept;
void RMSProp_Graves_ET(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept;
void RMSProp_Hinton_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept;
void RProp_ET(realmtx_t& dW, const real_t learningRate)noexcept;
void Adam_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
	const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept;
void AdaMax_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
	const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept;

real_t vSumAbs_ET(const realmtx_t& A)noexcept;
real_t vSumSquares_ET(const realmtx_t& A)noexcept;
real_t rowvecs_renorm_ET(realmtx_t& m, real_t* pTmp)noexcept;

void relu_ET(realmtx_t& f);
void drelu_ET(const realmtx_t& f, realmtx_t& df);
void leakyrelu_ET(realmtx_t& f, const real_t leak);
void dleakyrelu_ET(const realmtx_t& f, realmtx_t& df, const real_t leak);

void elu_ET(realmtx_t& f, const real_t alpha);
void delu_ET(const realmtx_t& f, realmtx_t& df, const real_t alpha);
void elu_unitalpha_ET(realmtx_t& f);
void delu_unitalpha_ET(const realmtx_t& f, realmtx_t& df);

void elogu_ET(const realmtx_t& x, realmtx_t& f, const real_t& alpha, const real_t& b);
void delogu_ET(const realmtx_t& x, realmtx_t& df, const real_t& alpha, const real_t& b);
void elogu_ua_ET(const realmtx_t& x, realmtx_t& f, const real_t& b);
void delogu_ua_ET(const realmtx_t& x, realmtx_t& df, const real_t& b);
void elogu_nb_ET(const realmtx_t& x, realmtx_t& f, const real_t& alpha);
void delogu_nb_ET(const realmtx_t& x, realmtx_t& df, const real_t& alpha);
void elogu_ua_nb_ET(const realmtx_t& x, realmtx_t& f);
void delogu_ua_nb_ET(const realmtx_t& x, realmtx_t& df);
