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

#include "stdafx.h"

#include <cmath>

#include "imath_etalons.h"

#include <algorithm>

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void ewBinarize_ip_ET(realmtx_t& A, const real_t frac)noexcept {
	auto pA = A.data();
	const auto pAE = pA + A.numel();
	while (pA != pAE) {
		const auto v = *pA;
		//NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
		*pA++ = v > frac ? real_t(1.0) : real_t(0.0);
	}
}

////////////////////////////////////////////////////////////////////////// 

void softmax_parts_ET(const realmtx_t& act, const real_t* pMax, real_t* pDenominator, real_t* pNumerator)noexcept {
	NNTL_ASSERT(pMax && pDenominator && act.numel() > 0);
	const auto rm = act.rows(), cm = act.cols();
	realmtx_t Numerator;
	Numerator.useExternalStorage(pNumerator, rm, cm);
	std::fill(pDenominator, pDenominator + rm, real_t(0.0));
	for (vec_len_t c = 0; c < cm; ++c) {
		for (vec_len_t r = 0; r < rm; ++r) {
			const auto num = std::exp(act.get(r, c) - pMax[r]);
			pDenominator[r] += num;
			Numerator.set(r, c, num);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//pTmp is a vector of length at least act.numel() + 2*act.rows()
void softmax_ET(realmtxdef_t& act, real_t* pTmp)noexcept {
	NNTL_ASSERT(pTmp && act.numel());
	const auto pNumerator = pTmp, pMax = pNumerator + act.numel(), pDenominator = pMax + act.rows();

	const auto bRestoreBiases = act.hide_biases();

	mrwMax_ET(act, pMax);
	softmax_parts_ET(act, pMax, pDenominator, pNumerator);
	memcpy(act.data(), pNumerator, act.byte_size());
	mrwDivideByVec_ET(act, pDenominator);

	if (bRestoreBiases) act.restore_biases();
}

// L = sum( -y*log(a) )/activations.rows()
real_t loss_softmax_xentropy_ET(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size());
	const auto pA = activations.data(), pY = data_y.data();
	const auto ne = activations.numel();
	real_t ret(0.0);
	for (numel_cnt_t i = 0; i < ne; ++i) {
		auto a = pA[i];
		const auto y = pY[i];
		NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));
		NNTL_ASSERT(y >= real_t(0.0) && y <= real_t(1.0));
		a = a > real_t(0.0) ? std::log(a) : nntl::math::real_t_limits<real_t>::log_almost_zero;
		ret -= y*a;
		NNTL_ASSERT(!isnan(ret));
	}
	return ret/ activations.rows();
}


void apply_momentum_ET(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
	NNTL_ASSERT(vW.size() == dW.size());
	NNTL_ASSERT(!vW.empty() && !dW.empty());

	const auto dataCnt = vW.numel();
	const auto pV = vW.data();
	const auto pdW = dW.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		pV[i] = momentum*pV[i] + pdW[i];
	}
}

//apply individual learning rate to dLdW
void apply_ILR_ET(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
	const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
{
	ASSERT_EQ(dLdW.size(), prevdLdW.size());
	ASSERT_EQ(dLdW.size(), ILRGain.size());
	ASSERT_TRUE(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

	const auto dataCnt = dLdW.numel();
	auto pdW = dLdW.data();
	const auto prevdW = prevdLdW.data();
	auto pGain = ILRGain.data();

	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto cond = pdW[i] * prevdW[i];
		auto gain = pGain[i];

		/*if (cond > 0) {
			gain *= incr;
			if (gain > capHigh)gain = capHigh;
		} else if (cond < 0) {
			gain *= decr;
			if (gain < capLow)gain = capLow;
		}*/

		if (cond > real_t(+0.)) {
			if (gain < capHigh) gain *= incr;
		} else if (cond < real_t(-0.)) {
			if (gain > capLow) gain *= decr;
		}

		pGain[i] = gain;
		pdW[i] *= gain;
	}
}

void evAbs_ET(realmtx_t& dest, const realmtx_t& src)noexcept {
	ASSERT_EQ(dest.size(), src.size());
	const auto pS = src.data();
	auto pD = dest.data();
	const auto dataCnt = src.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i)  pD[i] = abs(pS[i]);
}

void evAdd_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept {
	NNTL_ASSERT(A.size() == B.size());

	const auto dataCnt = A.numel();
	const auto pA = A.data();
	const auto pB = B.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += pB[i];
}

void evAddScaled_ip_ET(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
	NNTL_ASSERT(A.size() == B.size());

	const auto dataCnt = A.numel();
	const auto pA = A.data();
	const auto pB = B.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += c*pB[i];
}

void evAddScaledSign_ip_ET(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
	NNTL_ASSERT(A.size() == B.size());

	const auto dataCnt = A.numel();
	const auto pA = A.data();
	const auto pB = B.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += c*nntl::math::sign(pB[i]);
}

void evSquare_ET(realmtx_t& dest, const realmtx_t& src)noexcept {
	ASSERT_EQ(dest.size(), src.size());

	const auto pS = src.data();
	auto pD = dest.data();
	const auto dataCnt = src.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto s = pS[i];
		pD[i] = s*s;
	}
}

void evSub_ET(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
	NNTL_ASSERT(A.size() == B.size() && A.size() == C.size());

	const auto dataCnt = A.numel();
	const auto pA = A.data(), pB = B.data();
	const auto pC = C.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pC[i] = pA[i] - pB[i];
}

void evSub_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept {
	NNTL_ASSERT(A.size() == B.size());

	const auto dataCnt = A.numel();
	const auto pA = A.data();
	const auto pB = B.data();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] -= pB[i];
}

real_t loss_sigm_xentropy_ET(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
	NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
	const auto dataCnt = activations.numel();
	const auto ptrA = activations.data(), ptrY = data_y.data();
	constexpr auto log_zero = nntl::math::real_t_limits<real_t>::log_almost_zero;
	real_t ql = 0;
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto y = ptrY[i];
		auto a = ptrA[i];
		NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
		NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

		if (y > real_t(0.0)) {
			ql += (a == real_t(0.0) ? log_zero : log(a));
		} else {
			const auto oma = real_t(1.0) - a;
			ql += (oma == real_t(0.0) ? log_zero : log(oma));
		}
		NNTL_ASSERT(!isnan(ql));
	}
	return -ql / activations.rows();
}

//inverted dropout
void make_dropout_ET(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask)noexcept {
	const auto dataCnt = act.numel_no_bias();
	auto pDM = dropoutMask.data();
	const auto pA = act.data();
	const real_t dropPercActInv = real_t(1.) / dropPercAct;

	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		if (pDM[i] < dropPercAct) {
			pDM[i] = dropPercActInv;
			pA[i] *= dropPercActInv;
		} else {
			pDM[i] = real_t(0);
			pA[i] = real_t(0);
		}
	}
}

void ModProp_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept {
	ASSERT_EQ(dW.size(), rmsF.size());
	ASSERT_TRUE(emaDecay > 0 && emaDecay < 1);
	ASSERT_TRUE(numericStabilizer > 0 && numericStabilizer < 1);

	auto pdW = dW.data();
	auto prmsF = rmsF.data();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto dataCnt = dW.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		const auto rms = prmsF[i] * emaDecay + abs(pdW[i])*_1_emaDecay;
		prmsF[i] = rms;
		pdW[i] *= learningRate / (rms + numericStabilizer);
	}
}

void RMSProp_Graves_ET(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
	const real_t emaDecay, const real_t numericStabilizer)noexcept
{
	auto pdW = dW.data();
	auto prmsF = rmsF.data();
	auto prmsG = rmsG.data();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto dataCnt = dW.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		prmsF[i] = emaDecay*prmsF[i] + _1_emaDecay*pdW[i] * pdW[i];
		prmsG[i] = emaDecay*prmsG[i] + _1_emaDecay*pdW[i];
		pdW[i] *= learningRate / (sqrt(prmsF[i] - prmsG[i] * prmsG[i] + numericStabilizer));
	}
}

void RMSProp_Hinton_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
	const real_t emaDecay, const real_t numericStabilizer)noexcept
{
	auto pdW = dW.data();
	auto prmsF = rmsF.data();
	const auto _1_emaDecay = 1 - emaDecay;
	const auto dataCnt = dW.numel();
	for (numel_cnt_t i = 0; i < dataCnt; ++i) {
		prmsF[i] = emaDecay*prmsF[i] + _1_emaDecay*pdW[i] * pdW[i];
		pdW[i] *= learningRate / (sqrt(prmsF[i]) + numericStabilizer);
	}
}

void RProp_ET(realmtx_t& dW, const real_t learningRate)noexcept {
	auto p = dW.data();
	const auto im = dW.numel();
	for (numel_cnt_t i = 0; i < im; ++i) {
		const auto w = p[i];
		if (w > real_t(0)) {
			p[i] = learningRate;
		} else if (w < real_t(0)) {
			p[i] = -learningRate;
		} else p[i] = real_t(0);
	}
}

//////////////////////////////////////////////////////////////////////////

void Adam_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
	const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept 
{
	NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Vt.size());
	NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
	NNTL_ASSERT(real_t(0.) < beta1 && beta1 < real_t(1.));
	NNTL_ASSERT(real_t(0.) < beta2 && beta2 < real_t(1.));
	NNTL_ASSERT(real_t(0.) < numericStabilizer && numericStabilizer < real_t(1.));
	NNTL_ASSERT(real_t(0.) <= beta1t && beta1t <= real_t(1.));
	NNTL_ASSERT(real_t(0.) <= beta2t && beta2t <= real_t(1.));

	beta1t *= beta1;
	beta2t *= beta2;
	const real_t alphat = learningRate*sqrt(real_t(1.) - beta2t) / (real_t(1.) - beta1t);
	const real_t ombeta1 = real_t(1.) - beta1, ombeta2 = real_t(1.) - beta2;
	const auto ne = dW.numel();
	const auto pDw = dW.data(), pMt = Mt.data(), pVt = Vt.data();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		const auto g = pDw[i];
		pMt[i] = beta1*pMt[i] + ombeta1*g;
		pVt[i] = beta2*pVt[i] + ombeta2*g*g;
		pDw[i] = alphat*pMt[i] / (sqrt(pVt[i]) + numericStabilizer);
	}
}

void AdaMax_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
	const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept
{
	NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Ut.size());
	NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
	NNTL_ASSERT(real_t(0.) < beta1 && beta1 < real_t(1.));
	NNTL_ASSERT(real_t(0.) < beta2 && beta2 < real_t(1.));
	NNTL_ASSERT(real_t(0.) <= beta1t && beta1t <= real_t(1.));

	beta1t *= beta1;
	const real_t alphat = learningRate / (real_t(1.) - beta1t);
	const real_t ombeta1 = real_t(1.) - beta1;
	const auto ne = dW.numel();
	const auto pDw = dW.data(), pMt = Mt.data(), pUt = Ut.data();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		const auto g = pDw[i];
		pMt[i] = beta1*pMt[i] + ombeta1*g;
		pUt[i] = std::max({ beta2*pUt[i] ,abs(g) });
		pDw[i] = alphat*pMt[i] / (pUt[i] + numericStabilizer);
	}
}

//////////////////////////////////////////////////////////////////////////

real_t rowvecs_renorm_ET(realmtx_t& m, real_t* pTmp)noexcept {
	//calculate current norms of row-vectors into pTmp
	const auto mRows = m.rows(), mCols = m.cols();
	for (vec_len_t r = 0; r < mRows; ++r) {
		pTmp[r] = real_t(0.0);
		for (vec_len_t c = 0; c < mCols; ++c) {
			auto v = m.get(r, c);
			pTmp[r] += v*v;
		}
	}

	//finding average norm
	real_t meanNorm = static_cast<real_t>(std::accumulate(pTmp, pTmp + mRows, 0.0) / mRows);

	//test and renormalize
	//const real_t newNorm = meanNorm - sqrt(math::real_t_limits<real_t>::eps_lower_n(meanNorm, rowvecs_renorm_MULT));
	const real_t newNorm = meanNorm - 2*sqrt(nntl::math::real_t_limits<real_t>::eps_lower(meanNorm));
	for (vec_len_t r = 0; r < mRows; ++r) {
		if (pTmp[r] > meanNorm) {
			const real_t normCoeff = sqrt(newNorm / pTmp[r]);
			real_t nn = 0;
			for (vec_len_t c = 0; c < mCols; ++c) {
				const auto newV = m.get(r, c)*normCoeff;
				m.set(r, c, newV);
				nn += newV*newV;
			}
			EXPECT_TRUE(nn <= meanNorm);
		}
	}
	return meanNorm;
}

real_t vSumAbs_ET(const realmtx_t& A)noexcept {
	const auto dataCnt = A.numel();
	const auto p = A.data();
	real_t ret(0);
	for (numel_cnt_t i = 0; i < dataCnt; ++i) ret += abs(p[i]);
	return ret;
}

real_t vSumSquares_ET(const realmtx_t& A)noexcept {
	const auto dataCnt = A.numel();
	const auto p = A.data();
	real_t ret(0);
	for (numel_cnt_t i = 0; i < dataCnt; ++i) ret += p[i] * p[i];
	return ret;
}

//////////////////////////////////////////////////////////////////////////
void relu_ET(realmtx_t& f) {
	const auto p = f.data();
	const auto ne = f.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		if (p[i]<real_t(-0.)) p[i] = real_t(0.);
	}
}
void drelu_ET(const realmtx_t& f, realmtx_t& df) {
	const auto p = f.data();
	const auto pd = df.data();
	const auto ne = f.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		pd[i] = (p[i] < real_t(+0.)) ? real_t(0.) : real_t(1.);
	}
}
void leakyrelu_ET(realmtx_t& f, const real_t leak) {
	NNTL_ASSERT(leak > real_t(+0.));
	const auto p = f.data();
	const auto ne = f.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		if (p[i] < real_t(0.)) p[i] *= leak;
	}
}
void dleakyrelu_ET(const realmtx_t& f, realmtx_t& df, const real_t leak) {
	const auto p = f.data();
	const auto pd = df.data();
	const auto ne = f.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		pd[i] = (p[i] < real_t(0.)) ? leak : real_t(1.);
	}
}

void elu_ET(realmtx_t& f, const real_t alpha) {
	const auto p = f.data();
	const auto ne = f.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		if (p[i] < real_t(0.)) p[i] = alpha*(std::exp(p[i]) - real_t(1.));
	}
}
//#TODO: probably it's better to make df value out of plain x value instead of f(x). Update this and related functions and tests
void delu_ET(const realmtx_t& f, realmtx_t& df, const real_t alpha) {
	const auto p = f.data();
	const auto pd = df.data();
	const auto ne = f.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		pd[i] = (p[i] < real_t(0.)) ? (p[i] + alpha) : real_t(1.);
	}
}
void elu_unitalpha_ET(realmtx_t& f) { elu_ET(f, real_t(1.0)); }
void delu_unitalpha_ET(const realmtx_t& f, realmtx_t& df) { delu_ET(f, df, real_t(1.0)); }

void elogu_ET(const realmtx_t& x, realmtx_t& f, const real_t& alpha, const real_t& b) {
	NNTL_ASSERT(x.size() == f.size());
	const auto px = x.data();
	const auto dest = f.data();
	const auto ne = x.numel_no_bias();
	const auto ilb = real_t(1.) / log(b);
	for (numel_cnt_t i = 0; i < ne; ++i) {
		const auto xv = px[i];
		if (xv < real_t(0.)) {
			dest[i] = alpha*(std::exp(xv) - real_t(1.));
		} else {
			dest[i] = log(xv + real_t(1.))*ilb;
		}
	}
}
void delogu_ET(const realmtx_t& x, realmtx_t& df, const real_t& alpha, const real_t& b) {
	NNTL_ASSERT(df.size() == x.size_no_bias());
	const auto ilb = real_t(1.) / log(b);
	const auto px = x.data();
	const auto dest = df.data();
	const auto ne = x.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		const auto xv = px[i];
		if (xv < real_t(0.)) {
			dest[i] = alpha*std::exp(xv);
		} else {
			dest[i] = ilb / (xv + real_t(1.));
		}
	}
}
void elogu_ua_ET(const realmtx_t& x, realmtx_t& f, const real_t& b) { elogu_ET(x, f, real_t(1.), b); }
void delogu_ua_ET(const realmtx_t& x, realmtx_t& df, const real_t& b) { delogu_ET(x, df, real_t(1.), b); }
void elogu_nb_ET(const realmtx_t& x, realmtx_t& f, const real_t& alpha) { elogu_ET(x, f, alpha, real_t(M_E)); }
void delogu_nb_ET(const realmtx_t& x, realmtx_t& df, const real_t& alpha) { delogu_ET(x, df, alpha, real_t(M_E)); }
void elogu_ua_nb_ET(const realmtx_t& x, realmtx_t& f) { elogu_ET(x, f, real_t(1.), real_t(M_E)); }
void delogu_ua_nb_ET(const realmtx_t& x, realmtx_t& df) { delogu_ET(x, df, real_t(1.), real_t(M_E)); }