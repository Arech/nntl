/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
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

namespace nntl {
namespace math_etalons {

	// #supportsBatchInRow
	template<typename T>
	void evMulC_ip_ET(smtx<T>& A, const T cv, const bool bIgnoreBias = false)noexcept {
		ASSERT_TRUE(bIgnoreBias || !A.emulatesBiases());
		const auto rm = A.rows(bIgnoreBias), cm = A.cols(bIgnoreBias);
		for (vec_len_t c = 0; c < cm; ++c) {
			for (vec_len_t r = 0; r < rm; ++r) {
				A.get(r, c) *= cv;
			}
		}
	}

	// #supportsBatchInRow
	template<typename T>
	void evAddC_ip_ET(smtx<T>& A, const T cv, const bool bIgnoreBias = false)noexcept {
		ASSERT_TRUE(bIgnoreBias || !A.emulatesBiases());
		const auto rm = A.rows(bIgnoreBias), cm = A.cols(bIgnoreBias);
		for (vec_len_t c = 0; c < cm; ++c) {
			for (vec_len_t r = 0; r < rm; ++r) {
				A.get(r, c) += cv;
			}
		}
	}
	// #supportsBatchInRow
	template<typename T>
	void evMulCAddC_ip_ET(smtx<T>& A, const T mulC, const T addC, const bool bIgnoreBias = false)noexcept {
		ASSERT_TRUE(bIgnoreBias || !A.emulatesBiases());
		const auto rm = A.rows(bIgnoreBias), cm = A.cols(bIgnoreBias);
		for (vec_len_t c = 0; c < cm; ++c) {
			for (vec_len_t r = 0; r < rm; ++r) {
				A.set(r, c, A.get(r, c)*mulC + addC);
			}
		}
	}

	//TODO: STUPID implementation!!! Rewrite
	template<typename iMath>
	void evCMulSub_ET(iMath& iM, smtx<typename iMath::real_t>& vW, const typename iMath::real_t momentum
		, smtx<typename iMath::real_t>& W)noexcept
	{
		evMulC_ip_ET(vW, momentum, false);
		iM.evSub_ip_st_naive(W, vW);
	}

	template<typename T>
	void ewBinarize_ip_ET(T* pA, const numel_cnt_t n, const T frac)noexcept {
		const auto pAE = pA + n;
		while (pA != pAE) {
			const auto v = *pA;
			*pA++ = v > frac ? T(1.0) : T(0.0);
		}
	}

	template<typename T>
	void ewBinarize_ip_ET(smtx<T>& A, const T frac)noexcept {
		ewBinarize_ip_ET(A.data(), A.numel(), frac);
	}


	template<typename BaseDestT, typename T>
	void ewBinarize_ET(smtx<BaseDestT>& Dest, const smtx<T>& A, const T frac)noexcept {
		NNTL_ASSERT(Dest.size_no_bias() == A.size_no_bias());
		auto pA = A.data();
		auto pD = Dest.data();
		const auto pAE = pA + A.numel_no_bias();
		while (pA != pAE) {
			*pD++ = *pA++ > frac ? BaseDestT(1.0) : BaseDestT(0.0);
		}
	}

	template<typename T>
	void softmax_parts_ET(const smtx<T>& act, const T* pMax, T* pDenominator, T* pNumerator)noexcept {
		NNTL_ASSERT(pMax && pDenominator && act.numel() > 0);
		const auto rm = act.rows(), cm = act.cols();
		realmtx_t Numerator;
		Numerator.useExternalStorage(pNumerator, rm, cm);
		::std::fill(pDenominator, pDenominator + rm, T(0.0));
		for (vec_len_t c = 0; c < cm; ++c) {
			for (vec_len_t r = 0; r < rm; ++r) {
				const auto num = ::std::exp(act.get(r, c) - pMax[r]);
				pDenominator[r] += num;
				Numerator.set(r, c, num);
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	//pTmp is a vector of length at least act.numel() + 2*act.rows()
	template<typename T>
	void softmax_ET(sdefmtx<T>& act, T* pTmp)noexcept {
		NNTL_ASSERT(pTmp && act.numel());
		const auto pNumerator = pTmp, pMax = pNumerator + act.numel(), pDenominator = pMax + act.rows();

		const auto bRestoreBiases = act.hide_biases();

		mrwMax_ET(act, pMax);
		softmax_parts_ET(act, pMax, pDenominator, pNumerator);
		memcpy(act.data(), pNumerator, act.byte_size());
		mrwDivideByVec_ET(act, pDenominator);

		act.restore_biases(bRestoreBiases);
	}

	// L = sum( -y*log(a) )/activations.rows()
	template<typename T>
	T loss_softmax_xentropy_ET(const smtx<T>& activations, const smtx<T>& data_y)noexcept {
		NNTL_ASSERT(activations.size() == data_y.size());
		const auto pA = activations.data(), pY = data_y.data();
		const auto ne = activations.numel();
		T ret(0.0);
		for (numel_cnt_t i = 0; i < ne; ++i) {
			auto a = pA[i];
			const auto y = pY[i];
			NNTL_ASSERT(a >= T(0.0) && a <= T(1.0));
			NNTL_ASSERT(y >= T(0.0) && y <= T(1.0));
			a = a > T(0.0) ? ::std::log(a) : nntl::math::real_t_limits<T>::log_almost_zero;
			ret -= y*a;
			NNTL_ASSERT(!isnan(ret));
		}
		return ret/*/ activations.rows()*/;
	}

	template<typename T>
	void dSigmQuadLoss_dZ_ET(const smtx<T>& data_y, smtx<T>& act_dLdZ)noexcept {
		NNTL_ASSERT(data_y.size() == act_dLdZ.size());
		const auto pA = act_dLdZ.data();
		const auto pY = data_y.data();
		const auto ne = act_dLdZ.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto a = pA[i];
			pA[i] = (a - pY[i])*a*(T(1.0) - a);
		}
	}

	template<typename WlT, typename T>
	void dLoss_dZ_ET(const smtx<T>& data_y, smtx<T>& act_dLdZ)noexcept {
		NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
		NNTL_ASSERT(act_dLdZ.size() == data_y.size());

		const auto pY = data_y.data();
		const auto pA = act_dLdZ.data();
		const numel_cnt_t ne = act_dLdZ.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			pA[i] = WlT::dLdZ(pY[i], pA[i]);
		}
	}

	template<typename T>
	void apply_momentum_ET(smtx<T>& vW, const T momentum, const smtx<T>& dW)noexcept {
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
	template<typename T>
	void apply_ILR_ET(smtx<T>& dLdW, const smtx<T>& prevdLdW, smtx<T>& ILRGain,
		const T decr, const T incr, const T capLow, const T capHigh)noexcept
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

			if (cond > T(+0.)) {
				if (gain < capHigh) gain *= incr;
			} else if (cond < T(-0.)) {
				if (gain > capLow) gain *= decr;
			}

			pGain[i] = gain;
			pdW[i] *= gain;
		}
	}

	template<typename T>
	void evAbs_ET(smtx<T>& dest, const smtx<T>& src)noexcept {
		ASSERT_EQ(dest.size(), src.size());
		const auto pS = src.data();
		auto pD = dest.data();
		const auto dataCnt = src.numel();
		for (numel_cnt_t i = 0; i < dataCnt; ++i)  pD[i] = ::std::abs(pS[i]);
	}

	template<typename T>
	void evAdd_ip_ET(smtx<T>& A, const smtx<T>& B)noexcept {
		NNTL_ASSERT(A.size() == B.size());

		const auto dataCnt = A.numel();
		const auto pA = A.data();
		const auto pB = B.data();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += pB[i];
	}

	template<typename T>
	void evAddScaled_ip_ET(smtx<T>& A, const T c, const smtx<T>& B)noexcept {
		NNTL_ASSERT(A.size() == B.size());

		const auto dataCnt = A.numel();
		const auto pA = A.data();
		const auto pB = B.data();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += c*pB[i];
	}

	template<typename T>
	void evAddScaledSign_ip_ET(smtx<T>& A, const T c, const smtx<T>& B)noexcept {
		NNTL_ASSERT(A.size() == B.size());

		const auto dataCnt = A.numel();
		const auto pA = A.data();
		const auto pB = B.data();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += c*nntl::math::sign(pB[i]);
	}

	template<typename T>
	void evSquare_ET(smtx<T>& dest, const smtx<T>& src)noexcept {
		ASSERT_EQ(dest.size(), src.size());

		const auto pS = src.data();
		auto pD = dest.data();
		const auto dataCnt = src.numel();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) {
			const auto s = pS[i];
			pD[i] = s*s;
		}
	}

	template<typename T>
	void evSub_ET(const smtx<T>& A, const smtx<T>& B, smtx<T>& C)noexcept {
		NNTL_ASSERT(A.size() == B.size() && A.size() == C.size());

		const auto dataCnt = A.numel();
		const auto pA = A.data(), pB = B.data();
		const auto pC = C.data();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) pC[i] = pA[i] - pB[i];
	}

	template<typename T>
	void evSub_ip_ET(smtx<T>& A, const smtx<T>& B)noexcept {
		NNTL_ASSERT(A.size() == B.size());

		const auto dataCnt = A.numel();
		const auto pA = A.data();
		const auto pB = B.data();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] -= pB[i];
	}

	template<typename T>
	T loss_xentropy_ET(const smtx<T>& activations, const smtx<T>& data_y)noexcept {
		NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
		const auto dataCnt = activations.numel();
		const auto ptrA = activations.data(), ptrY = data_y.data();
		constexpr auto log_zero = nntl::math::real_t_limits<T>::log_almost_zero;
		T ql = 0;
		for (numel_cnt_t i = 0; i < dataCnt; ++i) {
			const auto y = ptrY[i];
			auto a = ptrA[i];
			NNTL_ASSERT(y == T(0.0) || y == T(1.0));
			NNTL_ASSERT(a >= T(0.0) && a <= T(1.0));

			if (y > T(0.0)) {
				ql += (a == T(0.0) ? log_zero : ::std::log(a));
			} else {
				//const auto oma = T(1.0) - a;
				//ql += (oma == T(0.0) ? log_zero : log(oma));
				ql += (a == T(1.0) ? log_zero : nntl::math::log1p(-a));
			}
			NNTL_ASSERT(!isnan(ql));
		}
		return -ql /*/ activations.rows()*/;
	}


	//inverted dropout
	template<typename T>
	void make_dropout_ET(smtx<T>& act, const T dropPercAct, smtx<T>& dropoutMask)noexcept {
		const auto dataCnt = act.numel_no_bias();
		auto pDM = dropoutMask.data();
		const auto pA = act.data();
		const T dropPercActInv = T(1.) / dropPercAct;

		for (numel_cnt_t i = 0; i < dataCnt; ++i) {
			if (pDM[i] < dropPercAct) {
				pDM[i] = dropPercActInv;
				pA[i] *= dropPercActInv;
			} else {
				pDM[i] = T(0);
				pA[i] = T(0);
			}
		}
	}

	//simple direct dropout, doesn't change the mask
	template<typename T>
	void apply_dropout_mask_ET(smtx<T>& act, const T dropPercAct, const smtx<T>& dropoutMask)noexcept {
		const auto dataCnt = act.numel_no_bias();
		auto pDM = dropoutMask.data();
		const auto pA = act.data();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) {
			if (pDM[i] >= dropPercAct) pA[i] = T(0);
		}
	}

	template<typename T>
	void ModProp_ET(smtx<T>& dW, smtx<T>& rmsF, const T learningRate, const T emaDecay, const T numericStabilizer)noexcept {
		ASSERT_EQ(dW.size(), rmsF.size());
		ASSERT_TRUE(emaDecay > 0 && emaDecay < 1);
		ASSERT_TRUE(numericStabilizer > 0 && numericStabilizer < 1);

		auto pdW = dW.data();
		auto prmsF = rmsF.data();
		const auto _1_emaDecay = 1 - emaDecay;
		const auto dataCnt = dW.numel();
		for (numel_cnt_t i = 0; i < dataCnt; ++i) {
			const auto rms = prmsF[i] * emaDecay + ::std::abs(pdW[i])*_1_emaDecay;
			prmsF[i] = rms;
			pdW[i] *= learningRate / (rms + numericStabilizer);
		}
	}

	template<typename T>
	void RMSProp_Graves_ET(smtx<T>& dW, smtx<T>& rmsF, smtx<T>& rmsG, const T learningRate,
		const T emaDecay, const T numericStabilizer)noexcept
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

	template<typename T>
	void RMSProp_Hinton_ET(smtx<T>& dW, smtx<T>& rmsF, const T learningRate,
		const T emaDecay, const T numericStabilizer)noexcept
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

	template<typename T>
	void RProp_ET(smtx<T>& dW, const T learningRate)noexcept {
		auto p = dW.data();
		const auto im = dW.numel();
		for (numel_cnt_t i = 0; i < im; ++i) {
			const auto w = p[i];
			if (w > T(0)) {
				p[i] = learningRate;
			} else if (w < T(0)) {
				p[i] = -learningRate;
			} else p[i] = T(0);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void Adam_ET(smtx<T>& dW, smtx<T>& Mt, smtx<T>& Vt, T& beta1t, T& beta2t, const T learningRate,
		const T beta1, const T beta2, const T numericStabilizer)noexcept
	{
		NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Vt.size());
		NNTL_ASSERT(T(0.) < learningRate && learningRate < T(1.));
		NNTL_ASSERT(T(0.) < beta1 && beta1 < T(1.));
		NNTL_ASSERT(T(0.) < beta2 && beta2 < T(1.));
		NNTL_ASSERT(T(0.) < numericStabilizer && numericStabilizer < T(1.));
		NNTL_ASSERT(T(0.) <= beta1t && beta1t <= T(1.));
		NNTL_ASSERT(T(0.) <= beta2t && beta2t <= T(1.));

		beta1t *= beta1;
		beta2t *= beta2;
		const T alphat = learningRate*sqrt(T(1.) - beta2t) / (T(1.) - beta1t);
		const T ombeta1 = T(1.) - beta1, ombeta2 = T(1.) - beta2;
		const auto ne = dW.numel();
		const auto pDw = dW.data(), pMt = Mt.data(), pVt = Vt.data();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto g = pDw[i];
			pMt[i] = beta1*pMt[i] + ombeta1*g;
			pVt[i] = beta2*pVt[i] + ombeta2*(g*g);
			pDw[i] = alphat*pMt[i] / (sqrt(pVt[i]) + numericStabilizer);
		}
	}

	template<typename T>
	void AdaMax_ET(smtx<T>& dW, smtx<T>& Mt, smtx<T>& Ut, T& beta1t, const T learningRate,
		const T beta1, const T beta2, const T numericStabilizer)noexcept
	{
		NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Ut.size());
		NNTL_ASSERT(T(0.) < learningRate && learningRate < T(1.));
		NNTL_ASSERT(T(0.) < beta1 && beta1 < T(1.));
		NNTL_ASSERT(T(0.) < beta2 && beta2 < T(1.));
		NNTL_ASSERT(T(0.) <= beta1t && beta1t <= T(1.));

		beta1t *= beta1;
		const T alphat = learningRate / (T(1.) - beta1t);
		const T ombeta1 = T(1.) - beta1;
		const auto ne = dW.numel();
		const auto pDw = dW.data(), pMt = Mt.data(), pUt = Ut.data();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto g = pDw[i];
			pMt[i] = beta1*pMt[i] + ombeta1*g;
			pUt[i] = ::std::max({ beta2*pUt[i] ,::std::abs(g) });
			pDw[i] = alphat*pMt[i] / (pUt[i] + numericStabilizer);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void Nadam_ET(smtx<T>& dW, smtx<T>& Mt, smtx<T>& Vt, T& mu_pow_t, T& eta_pow_t, const T learningRate,
		const T mu, const T eta, const T numericStabilizer)noexcept
	{
		NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Vt.size());
		NNTL_ASSERT(T(0.) < learningRate && learningRate < T(1.));
		NNTL_ASSERT(T(0.) < mu && mu < T(1.));
		NNTL_ASSERT(T(0.) < eta && eta < T(1.));
		NNTL_ASSERT(T(0.) < numericStabilizer && numericStabilizer < T(1.));
		NNTL_ASSERT(T(0.) <= mu_pow_t && mu_pow_t <= T(1.));
		NNTL_ASSERT(T(0.) <= eta_pow_t && eta_pow_t <= T(1.));

		mu_pow_t *= mu;
		eta_pow_t *= eta;

		const T mu_t = (mu - mu_pow_t) / (T(1.) - mu_pow_t), eta_t = (eta - eta_pow_t) / (T(1.) - eta_pow_t);

		const auto o_m_mu_t = T(1.) - mu_t, o_m_eta_t = T(1.) - eta_t;

		const T mHat_c_mt = mu*((T(1.) - mu_pow_t) / (T(1) - mu*mu_pow_t));
		const T mHat_c_g = o_m_mu_t;


		const auto ne = dW.numel();
		const auto pDw = dW.data(), pMt = Mt.data(), pVt = Vt.data();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto g = pDw[i];
			pMt[i] = mu_t*pMt[i] + o_m_mu_t*g;
			pVt[i] = eta_t*pVt[i] + o_m_eta_t*(g*g);
			const auto mhat = mHat_c_mt * pMt[i] + mHat_c_g*g;
			pDw[i] = learningRate*mhat / (sqrt(pVt[i]) + numericStabilizer);
		}
	}

	template<typename T>
	void Radam_ET(smtx<T>& dW, smtx<T>& Mt, smtx<T>& Vt, T& mu_pow_t, T& eta_pow_t, const T learningRate,
		const T mu, const T eta, const T gamma, const T numericStabilizer)noexcept
	{
		NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Vt.size());
		NNTL_ASSERT(T(0.) < learningRate && learningRate < T(1.));
		NNTL_ASSERT(T(0.) < mu && mu < T(1.));
		NNTL_ASSERT(T(0.) < eta && eta < T(1.));
		NNTL_ASSERT(T(0.) < gamma && gamma < T(1.));
		NNTL_ASSERT(T(0.) < numericStabilizer && numericStabilizer < T(1.));
		NNTL_ASSERT(T(0.) <= mu_pow_t && mu_pow_t <= T(1.));
		NNTL_ASSERT(T(0.) <= eta_pow_t && eta_pow_t <= T(1.));

		mu_pow_t *= mu;
		eta_pow_t *= eta;

		const T mu_t = (mu - mu_pow_t) / (T(1.) - mu_pow_t), eta_t = (eta - eta_pow_t) / (T(1.) - eta_pow_t);

		const auto o_m_mu_t = T(1.) - mu_t, o_m_eta_t = T(1.) - eta_t;

		const T mHat_c_mt = T(1) - gamma;
		const T mHat_c_g = gamma;

		const auto ne = dW.numel();
		const auto pDw = dW.data(), pMt = Mt.data(), pVt = Vt.data();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto g = pDw[i];
			pMt[i] = mu_t*pMt[i] + o_m_mu_t*g;
			pVt[i] = eta_t*pVt[i] + o_m_eta_t*(g*g);
			const auto mhat = mHat_c_mt * pMt[i] + mHat_c_g*g;
			pDw[i] = learningRate*mhat / (sqrt(pVt[i]) + numericStabilizer);
		}
	}

	template<typename T>
	T rowvecs_renorm_ET(smtx<T>& m, const T newNormSq, const bool bNormIncludesBias, T* pTmp)noexcept {
		//calculate current norms of row-vectors into pTmp
		const auto mRows = m.rows(), mCols = m.cols(), cols4norm = mCols - (!bNormIncludesBias);
		for (vec_len_t r = 0; r < mRows; ++r) {
			pTmp[r] = T(0.0);
			for (vec_len_t c = 0; c < cols4norm; ++c) {
				auto v = m.get(r, c);
				pTmp[r] += v*v;
			}
		}

		//finding average norm
		T meanNorm = static_cast<T>(::std::accumulate(pTmp, pTmp + mRows, 0.0) / cols4norm);

		//test and renormalize
		//const T newNorm = meanNorm - sqrt(math::real_t_limits<T>::eps_lower_n(meanNorm, rowvecs_renorm_MULT));
		const T newNorm = newNormSq;// -2 * sqrt(nntl::math::real_t_limits<T>::eps_lower(newNormSq));
		for (vec_len_t r = 0; r < mRows; ++r) {
			if (pTmp[r] > newNormSq) {
				const T normCoeff = sqrt(newNorm / pTmp[r]);
				//T nn = 0;
				for (vec_len_t c = 0; c < mCols; ++c) {
					const auto newV = m.get(r, c)*normCoeff;
					m.set(r, c, newV);
					//nn += newV*newV;
				}
				//EXPECT_TRUE(nn <= newNormSq);
			}
		}
		return meanNorm;
	}

	template<typename T>
	T vSumAbs_ET(const smtx<T>& A)noexcept {
		const auto dataCnt = A.numel();
		const auto p = A.data();
		T ret(0), C(0.), Y, tT;
		for (numel_cnt_t i = 0; i < dataCnt; ++i) {
			Y = ::std::abs(p[i]) - C;
			tT = ret + Y;
			C = tT - ret - Y;
			ret = tT;
		}
		return ret;
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void sigm_ET(smtx<T>& X) {
		const auto p = X.data();
		const auto ne = X.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			p[i] = T(1.0) / (T(1.0) + ::std::exp(-p[i]));
		}
	}
	template<typename T>
	void dsigm_ET(smtx<T>& f_df) {
		const auto p = f_df.data();
		const auto ne = f_df.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto f = p[i];
			NNTL_ASSERT(f >= 0 && f <= 1);
			p[i] = f * (T(1.) - f);
		}
	}
	template<typename T>
	void relu_ET(smtx<T>& f) {
		const auto p = f.data();
		const auto ne = f.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			if (p[i] <= T(0.)) p[i] = T(0.);
		}
	}
	template<typename T>
	void drelu_ET(smtx<T>& f_df) {
		const auto p = f_df.data();
		const auto ne = f_df.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			p[i] = (p[i] <= T(+0.)) ? T(0.) : T(1.);
		}
	}
	template<typename T>
	void leakyrelu_ET(smtx<T>& f, const T leak) {
		NNTL_ASSERT(leak > T(+0.));
		const auto p = f.data();
		const auto ne = f.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			if (p[i] < T(0.)) p[i] *= leak;
		}
	}
	template<typename T>
	void dleakyrelu_ET(smtx<T>& f_df, const T leak) {
		const auto p = f_df.data();
		const auto ne = f_df.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			p[i] = (p[i] <= T(0.)) ? leak : T(1.);
		}
	}
	template<typename T>
	void elu_ET(smtx<T>& f, const T alpha) {
		const auto p = f.data();
		const auto ne = f.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			//if (p[i] < T(0.)) p[i] = alpha*(::std::exp(p[i]) - T(1.));
			if (p[i] < T(0.)) p[i] = alpha*nntl::math::expm1(p[i]);
		}
	}
	//#TODO: probably it's better to make df value out of plain x value instead of f(x). Update this and related functions and tests
	template<typename T>
	void delu_ET(smtx<T>& f_df, const T alpha) {
		const auto p = f_df.data();
		const auto ne = f_df.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			p[i] = (p[i] < T(0.)) ? (p[i] + alpha) : T(1.);
		}
	}
	template<typename T>
	void elu_unitalpha_ET(smtx<T>& f) { elu_ET(f, T(1.0)); }
	template<typename T>
	void delu_unitalpha_ET(smtx<T>& f_df) { delu_ET(f_df, T(1.0)); }

	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void selu_ET(smtx<T>& f, const T& alpha, const T& lambda) {
		const auto p = f.data();
		const auto ne = f.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			//if (p[i] < T(0.)) p[i] = alpha*(::std::exp(p[i]) - T(1.));
			if (p[i] < T(0.)) {
				p[i] = (lambda*alpha)*nntl::math::expm1(p[i]);
			} else {
				p[i] *= lambda;
			}
		}
	}
	template<typename T>
	void dselu_ET(smtx<T>& f_df, const T& alpha, const T& lambda) {
		const auto p = f_df.data();
		const auto ne = f_df.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			p[i] = (p[i] < T(0.)) ? (p[i] + alpha*lambda) : lambda;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void elogu_ET(const smtx<T>& x, smtx<T>& f, const T& alpha, const T& b) {
		NNTL_ASSERT(x.size() == f.size());
		const auto px = x.data();
		const auto dest = f.data();
		const auto ne = x.numel_no_bias();
		const auto ilb = T(1.) / ::std::log(b);
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			if (xv < T(0.)) {
				//dest[i] = alpha*(::std::exp(xv) - T(1.));
				dest[i] = alpha*nntl::math::expm1(xv);
			} else {
				//dest[i] = log(xv + T(1.))*ilb;
				dest[i] = nntl::math::log1p(xv)*ilb;
			}
		}
	}
	template<typename T>
	void delogu_ET(const smtx<T>& x, smtx<T>& df, const T& alpha, const T& b) {
		NNTL_ASSERT(df.size() == x.size_no_bias());
		const auto ilb = T(1.) / ::std::log(b);
		const auto px = x.data();
		const auto dest = df.data();
		const auto ne = x.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			if (xv < T(0.)) {
				dest[i] = alpha*::std::exp(xv);
			} else {
				dest[i] = ilb / (xv + T(1.));
			}
		}
	}
	template<typename T>
	void elogu_ua_ET(const smtx<T>& x, smtx<T>& f, const T& b) { elogu_ET(x, f, T(1.), b); }
	template<typename T>
	void delogu_ua_ET(const smtx<T>& x, smtx<T>& df, const T& b) { delogu_ET(x, df, T(1.), b); }
	template<typename T>
	void elogu_nb_ET(const smtx<T>& x, smtx<T>& f, const T& alpha) { elogu_ET(x, f, alpha, T(M_E)); }
	template<typename T>
	void delogu_nb_ET(const smtx<T>& x, smtx<T>& df, const T& alpha) { delogu_ET(x, df, alpha, T(M_E)); }
	template<typename T>
	void elogu_ua_nb_ET(const smtx<T>& x, smtx<T>& f) { elogu_ET(x, f, T(1.), T(M_E)); }
	template<typename T>
	void delogu_ua_nb_ET(const smtx<T>& x, smtx<T>& df) { delogu_ET(x, df, T(1.), T(M_E)); }

	template<typename T>
	void loglogu_ET(const smtx<T>& x, smtx<T>& f, const T& b_neg, const T& b_pos) {
		NNTL_ASSERT(x.size() == f.size());
		const auto px = x.data();
		const auto dest = f.data();
		const auto ne = x.numel_no_bias();

		const auto ilbpos = T(1.) / ::std::log(b_pos);
		const auto nilbneg = T(-1.) / ::std::log(b_neg);

		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			if (xv < T(0.)) {
				//dest[i] = log(T(1.) - xv)*nilbneg;
				dest[i] = nntl::math::log1p(-xv)*nilbneg;
			} else {
				//dest[i] = log(xv + T(1.))*ilbpos;
				dest[i] = nntl::math::log1p(xv)*ilbpos;
			}
		}
	}
	template<typename T>
	void dloglogu_ET(const smtx<T>& x, smtx<T>& df, const T& b_neg, const T& b_pos) {
		NNTL_ASSERT(df.size() == x.size_no_bias());
		const auto px = x.data();
		const auto dest = df.data();
		const auto ne = x.numel_no_bias();

		const auto ilbpos = T(1.) / ::std::log(b_pos);
		const auto ilbneg = T(1.) / ::std::log(b_neg);

		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			if (xv < T(0.)) {
				dest[i] = ilbneg / (T(1.) - xv);
			} else {
				dest[i] = ilbpos / (xv + T(1.));
			}
		}
	}
	template<typename T>
	void loglogu_nbn_ET(const smtx<T>& x, smtx<T>& f, const T& b_pos) { loglogu_ET(x, f, T(M_E), b_pos); }
	template<typename T>
	void dloglogu_nbn_ET(const smtx<T>& x, smtx<T>& df, const T& b_pos) { dloglogu_ET(x, df, T(M_E), b_pos); }

	template<typename T>
	void loglogu_nbp_ET(const smtx<T>& x, smtx<T>& f, const T& b_neg) { loglogu_ET(x, f, b_neg, T(M_E)); }
	template<typename T>
	void dloglogu_nbp_ET(const smtx<T>& x, smtx<T>& df, const T& b_neg) { dloglogu_ET(x, df, b_neg, T(M_E)); }

	template<typename T>
	void loglogu_nbn_nbp_ET(const smtx<T>& x, smtx<T>& f) { loglogu_ET(x, f, T(M_E), T(M_E)); }
	template<typename T>
	void dloglogu_nbn_nbp_ET(const smtx<T>& x, smtx<T>& df) { dloglogu_ET(x, df, T(M_E), T(M_E)); }

	template<typename T>
	void softsign_ET(const smtx<T>& x, smtx<T>& f, const T a, const T c) {
		NNTL_ASSERT(x.size() == f.size());
		NNTL_ASSERT(a > 0 && c > 0);
		const auto px = x.data();
		const auto dest = f.data();
		const auto ne = x.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			dest[i] = (c*xv) / (a + ::std::abs(xv));
		}
	}
	template<typename T>
	void dsoftsign_ET(const smtx<T>& x, smtx<T>& df, const T a, const T c) {
		NNTL_ASSERT(df.size() == x.size_no_bias());
		NNTL_ASSERT(a > 0 && c > 0);

		const auto px = x.data();
		const auto dest = df.data();
		const auto ne = x.numel_no_bias();

		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			const auto d = a + ::std::abs(xv);
			dest[i] = (c*a) / (d*d);
		}
	}

	template<typename T>
	void softsigm_ET(const smtx<T>& x, smtx<T>& f, const T& a) {
		NNTL_ASSERT(x.size() == f.size());
		NNTL_ASSERT(a > 0);
		const auto px = x.data();
		const auto dest = f.data();
		const auto ne = x.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			dest[i] = T(.5) + xv / (T(2.)*(a + ::std::abs(xv)));
		}
	}
	template<typename T>
	void dsoftsigm_ET(const smtx<T>& x, smtx<T>& df, const T& a) {
		NNTL_ASSERT(df.size() == x.size_no_bias());
		const auto px = x.data();
		const auto dest = df.data();
		const auto ne = x.numel_no_bias();

		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto xv = px[i];
			const auto d = a + ::std::abs(xv);
			dest[i] = a / (T(2.)* d*d);
		}
	}

	//////////////////////////////////////////////////////////////////////////

	template<typename iMathT>
	void mColumnsCov_ET(const smtx<typename iMathT::real_t>& A, smtx<typename iMathT::real_t>& C, iMathT& iM) {
		NNTL_UNREF(iM);
	#pragma warning(disable:4459)
		typedef typename iMathT::real_t real_t;
	#pragma warning(default:4459)
		iM.mScaledMulAtB_C(real_t(1.) / real_t(A.rows()), A, A, C);
	}

	template<bool bLowerTriangl, typename iMathT>
	auto loss_deCov_ET(const smtx<typename iMathT::real_t>& A, smtx<typename iMathT::real_t>& tDM
		, smtx<typename iMathT::real_t>& tCov, ::std::vector<typename iMathT::real_t>& vMean, iMathT& iM)noexcept
	{
		NNTL_ASSERT(!tDM.emulatesBiases() && !tCov.emulatesBiases());
		NNTL_ASSERT(A.cols_no_bias() == tDM.cols() && A.rows() == tDM.rows());
		NNTL_ASSERT(tDM.cols() == tCov.cols() && tCov.cols() == tCov.rows());
		NNTL_ASSERT(vMean.size() == tDM.cols());

		const auto b = A.copy_data_skip_bias(tDM);
		NNTL_ASSERT(b);

		mcwMean_ET(tDM, &vMean[0]);
		mcwSub_ip_ET(tDM, &vMean[0]);
		mColumnsCov_ET(tDM, tCov, iM);

		//return ewSumSquaresTriang_ET<bLowerTriangl>(tCov) /*/ A.rows()*/;
		return static_cast<real_t>(static_cast<ext_real_t>(ewSumSquaresTriang_ET<bLowerTriangl>(tCov)) / ext_real_t(A.cols_no_bias() - 1));
	}

	template<bool bLowerTriangl, typename iMathT>
	void dLoss_deCov_ET(const smtx<typename iMathT::real_t>& A
		, smtx<typename iMathT::real_t>& dL
		, smtx<typename iMathT::real_t>& tDM
		, smtx<typename iMathT::real_t>& tCov, ::std::vector<typename iMathT::real_t>& vMean, iMathT& iM)noexcept
	{
		NNTL_ASSERT(!tDM.emulatesBiases() && !tCov.emulatesBiases() && !dL.emulatesBiases());
		NNTL_ASSERT(A.cols_no_bias() == tDM.cols() && A.rows() == tDM.rows() && tDM.size() == dL.size());
		NNTL_ASSERT(tDM.cols() == tCov.cols() && tCov.cols() == tCov.rows());
		NNTL_ASSERT(vMean.size() == tDM.cols());

		const auto b = A.copy_data_skip_bias(tDM);
		NNTL_ASSERT(b);

		mcwMean_ET(tDM, &vMean[0]);
		mcwSub_ip_ET(tDM, &vMean[0]);
		mColumnsCov_ET(tDM, tCov, iM);

		const auto N = A.rows(), actCnt = A.cols_no_bias();
		const ext_real_t ne = static_cast<ext_real_t>(A.numel_no_bias() - N);
		for (vec_len_t m = 0; m < N; ++m) {
			for (vec_len_t a = 0; a < actCnt; ++a) {
				typename iMathT::func_SUM<typename iMathT::real_t, true> F;
				for (vec_len_t j = 0; j < actCnt; ++j) {
					if (a != j) {
						//v += tCov.get(a, j) * tDM.get(m, j);
						F.op(tCov.get(a, j) * tDM.get(m, j));
					}
				}
				//dL.set(m, a, F.result() * 2 / N);
				dL.set(m, a, static_cast<real_t>(ext_real_t(F.result()) * 2 / ne));
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void make_alphaDropout_ET(smtx<T>& act, const T dropPercAct
		, const T a_dmKeepVal, const T b_mbKeepVal, const T mbDropVal
		, smtx<T>& dropoutMask) noexcept
	{
		NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
		NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
		NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);

		const auto pDM = dropoutMask.data();
		const auto pA = act.data();
		const auto _ne = dropoutMask.numel();
		for (numel_cnt_t i = 0; i < _ne; ++i) {
			const auto v = pDM[i];
			NNTL_ASSERT(v >= T(0.0) && v <= T(1.0));
			const auto bKeep = v < dropPercAct;
			const T dmV = bKeep ? a_dmKeepVal : T(0.);
			const T bV = bKeep ? b_mbKeepVal : mbDropVal;
			pDM[i] = dmV;

			pA[i] = pA[i] * dmV + bV;
		}
	}

	template<typename T>
	void evSubMtxMulC_ip_nb_ET(smtx<T>& A, const smtx<T>& M, const T c)noexcept {
		NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
		NNTL_ASSERT(!M.empty() && A.size_no_bias() == M.size());
		NNTL_ASSERT(c);

		const auto pA = A.data();
		const auto pM = M.data();
		const auto ne = M.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			pA[i] = (pA[i] - pM[i])*c;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	void evMul_ip_ET(smtx<T>& A, const smtx<T>& B)noexcept {
		NNTL_ASSERT(A.size() == B.size());
		const auto pA = A.data();
		const auto pB = B.data();
		const auto ne = A.numel();
		for (numel_cnt_t i = 0; i < ne; ++i) pA[i] *= pB[i];
	}

	template<typename T>
	T loss_quadratic_ET(const ::nntl::math::smatrix<T>& A, const ::nntl::math::smatrix<T>& Y)noexcept {
		NNTL_ASSERT(A.size() == Y.size());
		auto ptrEtA = A.data(), ptrEtY = Y.data();
		const auto dataSize = A.numel();
		T etQuadLoss = 0;

		for (unsigned i = 0; i < dataSize; ++i) {
			const T v = ptrEtA[i] - ptrEtY[i];
			etQuadLoss += v*v;
		}
		return etQuadLoss / (2 /** A.rows()*/);
	}

	template<typename T>
	size_t vCountNonZeros_ET(const T*const pVec, const size_t n)noexcept {
		NNTL_ASSERT(pVec && n);
		size_t a = 0;
		for (size_t i = 0; i < n; ++i) a += !(pVec[i] == T(0) || pVec[i] == T(+0.) || pVec[i] == T(-0.));
		return a;
	}

	template<typename T>
	size_t vCountNonZeros_naive(const T*const pVec, const size_t ne)noexcept {
		NNTL_ASSERT(pVec && ne);
		size_t a = 0;
		for (size_t i = 0; i < ne; ++i) {
			a += pVec[i] != T(0.);
		}
		return a;
	}


	template<typename T>
	void evOneCompl_ET(const ::nntl::math::smatrix<T>& gate, ::nntl::math::smatrix<T>& gcompl)noexcept {
		static_assert(::std::is_floating_point<T>::value, "");
		NNTL_ASSERT(gate.size() == gcompl.size());
		const auto pG = gate.data();
		const auto pGc = gcompl.data();
		const auto ne = gate.numel_no_bias();
		for (numel_cnt_t i = 0; i < ne; ++i) {
			const auto g = pG[i];
			NNTL_ASSERT(g == T(1.) || g == T(0.));
			const auto gc = T(1.) - g;
			pGc[i] = gc;
		}
	}

	template<typename T, typename SeqIt>
	void mExtractRows_ET(const ::nntl::math::smatrix<T>& src, const SeqIt& ridxsItBegin, ::nntl::math::smatrix<T>& dest)noexcept {
		NNTL_ASSERT(!dest.empty() && !src.empty());
		src.assert_storage_does_not_intersect(dest);
		const numel_cnt_t destRows = dest.rows(), srcRows = src.rows();
		NNTL_ASSERT(dest.cols() == src.cols() && destRows <= srcRows && !(src.emulatesBiases() ^ dest.emulatesBiases()));

		if (src.emulatesBiases()) {
			dest.holey_biases(src.isHoleyBiases());
		}

		const auto nCols = src.cols();

		for (numel_cnt_t dr = 0; dr < destRows; ++dr) {
			const auto curSrcRow = static_cast<numel_cnt_t>(ridxsItBegin[dr]);
			NNTL_ASSERT(curSrcRow < srcRows);
			auto pS = src.data() + curSrcRow;
			auto pD = dest.data() + dr;
			for (vec_len_t c = 0; c < nCols; ++c) {
				NNTL_ASSERT(pS <= src.end());
				NNTL_ASSERT(pD <= dest.end());
				*pD = *pS;
				pD += destRows;
				pS += srcRows;
			}
		}
	}

	template<typename T>
	void mExtractRowsByMask_ET(const ::nntl::math::smatrix<T>& src, const T*const pMask, ::nntl::math::smatrix_deform<T>& dest) noexcept {
		NNTL_ASSERT(!src.empty() && pMask && !dest.empty() && src.cols() == dest.cols());

		const size_t nzCnt = vCountNonZeros_naive(pMask, src.rows());
		NNTL_ASSERT(nzCnt);
		dest.deform_rows(static_cast<::nntl::neurons_count_t>(nzCnt));

		const size_t r = src.rows(), c = src.cols_no_bias();
		auto pD = dest.data();
		for (size_t ci = 0; ci < c; ++ci) {
			const auto pS = src.colDataAsVec(static_cast<vec_len_t>(ci));
			NNTL_ASSERT(pD == dest.colDataAsVec(static_cast<vec_len_t>(ci)));
			for (size_t i = 0; i < r; ++i) {
				NNTL_ASSERT(pMask[i] == T(0) || pMask[i] == T(1.));
				if (pMask[i] != T(0.)) {
					*pD++ = pS[i];
				}
			}
			NNTL_ASSERT(ci == c - 1 || pD == dest.colDataAsVec(static_cast<vec_len_t>(ci + 1)));
		}
	}


	template<typename T>
	void mFillRowsByMask_ET(const ::nntl::math::smatrix<T>& src, const T*const pMask, ::nntl::math::smatrix<T>& dest) noexcept {
		NNTL_ASSERT(!src.empty() && pMask && !dest.empty() && src.cols() == dest.cols());

		const size_t nzCnt = vCountNonZeros_naive(pMask, dest.rows());
		NNTL_ASSERT(nzCnt == src.rows());

		const size_t r = dest.rows(), c = dest.cols_no_bias();
		auto pS = src.data();
		for (size_t ci = 0; ci < c; ++ci) {
			const auto pD = dest.colDataAsVec(static_cast<vec_len_t>(ci));
			NNTL_ASSERT(pS == src.colDataAsVec(static_cast<vec_len_t>(ci)));
			for (size_t i = 0; i < r; ++i) {
				const auto m = pMask[i];
				NNTL_ASSERT(m == T(0) || m == T(1.));
				pD[i] = m != T(0.) ? *pS++ : T(0);
			}
			NNTL_ASSERT(ci == c - 1 || pS == src.colDataAsVec(static_cast<vec_len_t>(ci + 1)));
		}
	}

	// #supportsBatchInRow
	template<typename T>
	void mTranspose_ET(const ::nntl::math::smatrix<T>& src, ::nntl::math::smatrix<T>& dest, const bool bIgnoreBias = false) noexcept {
		NNTL_ASSERT(src.bBatchInRow() == !dest.bBatchInRow());
		NNTL_ASSERT(src.rows(bIgnoreBias) == dest.cols(bIgnoreBias) && src.cols(bIgnoreBias) == dest.rows(bIgnoreBias));
		NNTL_ASSERT(src.if_biases_test_strict() && dest.if_biases_test_strict());
		const auto sRows = src.rows(bIgnoreBias), sCols = src.cols(bIgnoreBias);
		for (vec_len_t r = 0; r < sRows; ++r) {
			for (vec_len_t c = 0; c < sCols; ++c) {
				dest.set(c, r, src.get(r, c));
			}
		}
		NNTL_ASSERT(dest.if_biases_test_strict());
	}

	// #supportsBatchInRow
	template<typename T>
	void mTranspose_ignore_bias_ET(const ::nntl::math::smatrix<T>& src, ::nntl::math::smatrix<T>& dest) noexcept {
		mTranspose_ET(src, dest, true);
	}

	template<typename T, typename SeqIt>
	void mExtractCols_ET(const smtx<T>& src, const SeqIt& cidxs, smtx<T>& dest) {
		NNTL_ASSERT(src.bBatchInRow() == dest.bBatchInRow());
		NNTL_ASSERT(src.rows_no_bias() == dest.rows_no_bias());
		NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());

		const vec_len_t totC = dest.cols_no_bias();
		const vec_len_t totR = dest.rows_no_bias();
		for (vec_len_t ci = 0; ci < totC; ++ci) {
			const vec_len_t srcIdx = cidxs[ci];

			for (vec_len_t ri = 0; ri < totR; ++ri) {
				dest.set(ri, ci, src.get(ri, srcIdx));
			}
		}

		NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
	}

}
}
