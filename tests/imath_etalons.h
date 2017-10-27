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
	iM.evMulC_ip_st(vW, momentum);
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

void dSigmQuadLoss_dZ_ET(const realmtx_t& data_y, realmtx_t& act_dLdZ)noexcept;

template<typename WlT>
void dLoss_dZ_ET(const realmtx_t& data_y, realmtx_t& act_dLdZ)noexcept {
	NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
	NNTL_ASSERT(act_dLdZ.size() == data_y.size());

	const auto pY = data_y.data();
	const auto pA = act_dLdZ.data();
	const numel_cnt_t ne = act_dLdZ.numel();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		pA[i] = WlT::dLdZ(pY[i], pA[i]);
	}
}

void apply_momentum_ET(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept;
void apply_ILR_ET(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain, const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept;
void evAbs_ET(realmtx_t& dest, const realmtx_t& src)noexcept;
void evAdd_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept;
void evAddScaled_ip_ET(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept;
void evAddScaledSign_ip_ET(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept;
void evSquare_ET(realmtx_t& dest, const realmtx_t& src)noexcept;
void evSub_ET(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;
void evSub_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept;
real_t loss_xentropy_ET(const realmtx_t& activations, const realmtx_t& data_y)noexcept;

void make_dropout_ET(realmtx_t& act, const real_t dfrac, realmtx_t& dropoutMask)noexcept;

void ModProp_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept;
void RMSProp_Graves_ET(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept;
void RMSProp_Hinton_ET(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate, const real_t emaDecay, const real_t numericStabilizer)noexcept;
void RProp_ET(realmtx_t& dW, const real_t learningRate)noexcept;
void Adam_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
	const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept;
void AdaMax_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
	const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept;

void Nadam_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& mu_pow_t, real_t& eta_pow_t, const real_t learningRate,
	const real_t mu, const real_t eta, const real_t numericStabilizer)noexcept;
void Radam_ET(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& mu_pow_t, real_t& eta_pow_t, const real_t learningRate,
	const real_t mu, const real_t eta, const real_t gamma, const real_t numericStabilizer)noexcept;


real_t vSumAbs_ET(const realmtx_t& A)noexcept;
real_t rowvecs_renorm_ET(realmtx_t& m, const real_t newNormSq, const bool bNormIncludesBias, real_t* pTmp)noexcept;

void sigm_ET(realmtx_t& x);
void dsigm_ET(realmtx_t& f_df);

void relu_ET(realmtx_t& f);
void drelu_ET(realmtx_t& f_df);
void leakyrelu_ET(realmtx_t& f, const real_t leak);
void dleakyrelu_ET(realmtx_t& f_df, const real_t leak);

void elu_ET(realmtx_t& f, const real_t alpha);
void delu_ET(realmtx_t& f_df, const real_t alpha);
void elu_unitalpha_ET(realmtx_t& f);
void delu_unitalpha_ET(realmtx_t& f_df);

void selu_ET(realmtx_t& f, const real_t& alpha, const real_t& lambda);
void dselu_ET(realmtx_t& f_df, const real_t& alpha, const real_t& lambda);

void elogu_ET(const realmtx_t& x, realmtx_t& f, const real_t& alpha, const real_t& b);
void delogu_ET(const realmtx_t& x, realmtx_t& df, const real_t& alpha, const real_t& b);
void elogu_ua_ET(const realmtx_t& x, realmtx_t& f, const real_t& b);
void delogu_ua_ET(const realmtx_t& x, realmtx_t& df, const real_t& b);
void elogu_nb_ET(const realmtx_t& x, realmtx_t& f, const real_t& alpha);
void delogu_nb_ET(const realmtx_t& x, realmtx_t& df, const real_t& alpha);
void elogu_ua_nb_ET(const realmtx_t& x, realmtx_t& f);
void delogu_ua_nb_ET(const realmtx_t& x, realmtx_t& df);

void loglogu_ET(const realmtx_t& x, realmtx_t& f, const real_t& b_neg, const real_t& b_pos);
void dloglogu_ET(const realmtx_t& x, realmtx_t& df, const real_t& b_neg, const real_t& b_pos);

void loglogu_nbn_ET(const realmtx_t& x, realmtx_t& f, const real_t& b_pos);
void dloglogu_nbn_ET(const realmtx_t& x, realmtx_t& df, const real_t& b_pos);

void loglogu_nbp_ET(const realmtx_t& x, realmtx_t& f, const real_t& b_neg);
void dloglogu_nbp_ET(const realmtx_t& x, realmtx_t& df, const real_t& b_neg);

void loglogu_nbn_nbp_ET(const realmtx_t& x, realmtx_t& f);
void dloglogu_nbn_nbp_ET(const realmtx_t& x, realmtx_t& df);

void softsign_ET(const realmtx_t& x, realmtx_t& f, const real_t a, const real_t c);
void dsoftsign_ET(const realmtx_t& x, realmtx_t& df, const real_t a, const real_t c);

void softsigm_ET(const realmtx_t& x, realmtx_t& f, const real_t& a);
void dsoftsigm_ET(const realmtx_t& x, realmtx_t& df, const real_t& a);

template<typename iMathT>
void mColumnsCov_ET(const nntl::math::smatrix<typename iMathT::real_t>& A, nntl::math::smatrix<typename iMathT::real_t>& C, iMathT& iM){
	NNTL_UNREF(iM);
#pragma warning(disable:4459)
	typedef typename iMathT::real_t real_t;
#pragma warning(default:4459)
	iM.mScaledMulAtB_C(real_t(1.) / real_t(A.rows()), A, A, C);
}

template<bool bLowerTriangl, typename iMathT>
auto loss_deCov_ET(const nntl::math::smatrix<typename iMathT::real_t>& A, nntl::math::smatrix<typename iMathT::real_t>& tDM
	, nntl::math::smatrix<typename iMathT::real_t>& tCov, ::std::vector<typename iMathT::real_t>& vMean, iMathT& iM)noexcept
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
	//return ewSumSquaresTriang_ET<bLowerTriangl>(tCov) / (static_cast<numel_cnt_t>(A.rows())*tDM.cols()*(tDM.cols() - 1));
	return ewSumSquaresTriang_ET<bLowerTriangl>(tCov) /*/ A.rows()*/;
}

template<bool bLowerTriangl, typename iMathT>
void dLoss_deCov_ET(const nntl::math::smatrix<typename iMathT::real_t>& A
	, nntl::math::smatrix<typename iMathT::real_t>& dL
	, nntl::math::smatrix<typename iMathT::real_t>& tDM
	, nntl::math::smatrix<typename iMathT::real_t>& tCov, ::std::vector<typename iMathT::real_t>& vMean, iMathT& iM)noexcept
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
	for (vec_len_t m = 0; m < N; ++m) {
		for (vec_len_t a = 0; a < actCnt; ++a) {
			//real_t v(real_t(0));
			typename iMathT::func_SUM<real_t, true> F;
			for (vec_len_t j = 0; j < actCnt; ++j) {
				if (a!=j) {
					//v += tCov.get(a, j) * tDM.get(m, j);
					F.op(tCov.get(a, j) * tDM.get(m, j));
				}
			}
			dL.set(m, a, F.result() * 2 / N);
			//dL.set(m, a, F.result() * 2 / (static_cast<numel_cnt_t>(N)*actCnt*(actCnt - 1)));
		}
	}
}

void make_alphaDropout_ET(realmtx_t& act, const real_t dropPercAct
							  , const real_t a_dmKeepVal, const real_t b_mbKeepVal, const real_t mbDropVal
							  , realmtx_t& dropoutMask) noexcept;

void evSubMtxMulC_ip_nb_ET(realmtx_t& A, const realmtx_t& M, const real_t c)noexcept;

void evMul_ip_ET(realmtx_t& A, const realmtx_t& B)noexcept;

template<typename real_t>
real_t loss_quadratic_ET(const ::nntl::math::smatrix<real_t>& A, const ::nntl::math::smatrix<real_t>& Y)noexcept {
	NNTL_ASSERT(A.size() == Y.size());
	auto ptrEtA = A.data(), ptrEtY = Y.data();
	const auto dataSize = A.numel();
	real_t etQuadLoss = 0;

	for (unsigned i = 0; i < dataSize; ++i) {
		const real_t v = ptrEtA[i] - ptrEtY[i];
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
void evOneCompl_ET(const ::nntl::math::smatrix<T>& gate, ::nntl::math::smatrix<T>& gcompl)noexcept{
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

template<typename real_t>
void mExtractRowsByMask_ET(const ::nntl::math::smatrix<real_t>& src, const real_t*const pMask, ::nntl::math::smatrix_deform<real_t>& dest) noexcept{
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
			const auto m = pMask[i];
			NNTL_ASSERT(m == real_t(0) || m == real_t(1.));
			if (pMask[i] != real_t(0.)) {
				*pD++ = pS[i];
			}
		}
		NNTL_ASSERT(ci == c - 1 || pD == dest.colDataAsVec(static_cast<vec_len_t>(ci + 1)));
	}
}


template<typename real_t>
void mFillRowsByMask_ET(const ::nntl::math::smatrix<real_t>& src, const real_t*const pMask, ::nntl::math::smatrix<real_t>& dest) noexcept {
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
			NNTL_ASSERT(m == real_t(0) || m == real_t(1.));
			pD[i] = m != real_t(0.) ? *pS++ : real_t(0);
		}
		NNTL_ASSERT(ci == c - 1 || pS == src.colDataAsVec(static_cast<vec_len_t>(ci + 1)));
	}
}