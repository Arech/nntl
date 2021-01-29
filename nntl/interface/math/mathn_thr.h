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

//This file define a single/multi-threading code thresholds for MathN class
//Substitute for your own if you want to utilize mt/st branching code of MathN
// Or just use MathN_mt for reasonably large data sizes

#include "smath_thr.h"
#include "../../activations/_loss_parts.h"

namespace nntl {
namespace math {

namespace _impl {

	template <typename real_t> struct MATHN_THR : public SMATH_THR<real_t> {};

	//It is better, than nothing. But in future this pron should be substituted by a run-time function profiling over real-task data

	template <> struct MATHN_THR<double> : public SMATH_THR<double> {
		static constexpr numel_cnt_t ewBinarize_ip = 132000;
		static constexpr numel_cnt_t ewBinarize = 11000;

		static constexpr numel_cnt_t ewBinarizeBatch = 11000;

		static constexpr numel_cnt_t mExtractCols = 6000;//nt
		static constexpr numel_cnt_t mExtractRows = 5000;//nt
		static constexpr vec_len_t mExtractRowsByMask = 100;//NT

		static constexpr numel_cnt_t mExtractRowsSeq = 6000;//nt

		static constexpr numel_cnt_t mTransposeTrsh = 90000/2;

		static constexpr vec_len_t mFillRowsByMask = 100;//nt

		static constexpr numel_cnt_t mrwL2NormSquared = 124000;
		static constexpr vec_len_t mrwL2NormSquared_mt_cw_ColsPerThread = 3;

		static constexpr numel_cnt_t mCheck_normalize_rows = 124000;
		static constexpr numel_cnt_t evClamp = 9000;
		static constexpr numel_cnt_t make_dropout = 7000;
		static constexpr numel_cnt_t apply_dropout_mask = 3500;
		static constexpr numel_cnt_t make_alphaDropout = 5000;

		static constexpr numel_cnt_t apply_ILR_st_vec = 2620/2;
		static constexpr numel_cnt_t apply_ILR_mt = 9000/2;
		static constexpr numel_cnt_t apply_ILR_mt_vec = 120000/2;
		static constexpr numel_cnt_t apply_ILR_mt_vec2 = 261000/2;

		static constexpr numel_cnt_t apply_momentum = 21300;

		static constexpr numel_cnt_t evMulCAddC_ip = 120000 / 2;
		static constexpr numel_cnt_t evAddC_ip = 135000/2;

		static constexpr numel_cnt_t evMulC_ip = 64000;
		//static constexpr numel_cnt_t evMulC_ip_nb = 64000;
		static constexpr numel_cnt_t evMul_ip = 36700;
		//static constexpr numel_cnt_t evMul_ip_Anb = 36700;
		static constexpr numel_cnt_t evAdd_ip = 21000;
		
		static constexpr numel_cnt_t evAddScaled_ip = 17000;//not tested
		static constexpr numel_cnt_t evNZAddScaled_ip = 7000;//not tested

		static constexpr numel_cnt_t evAddScaledSign_ip = 15000;//not tested
		static constexpr numel_cnt_t evNZAddScaledSign_ip = 4000; //nt

		static constexpr numel_cnt_t evSign = 15000;//not tested

		static constexpr numel_cnt_t evOneCompl = 15000;//nt

		static constexpr numel_cnt_t evSub_ip = 22000;
		static constexpr numel_cnt_t evSub = 12600;
		static constexpr numel_cnt_t evMulC_ip_Sub_ip = 15000;

		static constexpr numel_cnt_t evSubMtxMulC_ip_nb = 10000;

		static constexpr numel_cnt_t evSquare = 24400;
		//static constexpr numel_cnt_t vSumSquares = 20000;//not tested
		static constexpr numel_cnt_t evAbs = 20300;
		static constexpr numel_cnt_t vSumAbs = 18000;//not tested		

		//static constexpr numel_cnt_t dIdentity = 30000;//nt
		//static constexpr numel_cnt_t dIdentityQuadLoss_dZ = 15000;//nt
		static constexpr numel_cnt_t dIdentityXEntropyLoss_dZ = 10000;//nt

		static constexpr numel_cnt_t sigm = 1300;
		static constexpr numel_cnt_t dsigm = 24500;
		static constexpr numel_cnt_t dSigmQuadLoss_dZ = 14900;

		static constexpr numel_cnt_t relu = 108000/2;
		static constexpr numel_cnt_t drelu = 26500/2;
		static constexpr numel_cnt_t leakyrelu = 106000/2;
		static constexpr numel_cnt_t dleakyrelu = 26500/2;

		static constexpr numel_cnt_t elu = 1500;
		static constexpr numel_cnt_t delu = 25000;
		static constexpr numel_cnt_t elu_unitalpha = 1550;
		static constexpr numel_cnt_t delu_unitalpha = 25600;

		static constexpr numel_cnt_t elogu = 1100;
		static constexpr numel_cnt_t delogu = 1700;
		static constexpr numel_cnt_t elogu_ua = 1100;
		static constexpr numel_cnt_t delogu_ua = 1800;
		static constexpr numel_cnt_t elogu_nb = 1100;
		static constexpr numel_cnt_t delogu_nb = 1800;
		static constexpr numel_cnt_t elogu_ua_nb = 1150;
		static constexpr numel_cnt_t delogu_ua_nb = 1850;

		static constexpr numel_cnt_t loglogu = 1030;
		static constexpr numel_cnt_t dloglogu = 970;
		static constexpr numel_cnt_t loglogu_nbn = 1040;
		static constexpr numel_cnt_t dloglogu_nbn = 1000;
		static constexpr numel_cnt_t loglogu_nbp = 1040;
		static constexpr numel_cnt_t dloglogu_nbp = 980;
		static constexpr numel_cnt_t loglogu_nbn_nbp = 1020;
		static constexpr numel_cnt_t dloglogu_nbn_nbp = 3000;

		static constexpr numel_cnt_t softsign = 15000;
		static constexpr numel_cnt_t softsign_uc = 15000;
		static constexpr numel_cnt_t dsoftsign = 20000;
		static constexpr numel_cnt_t dsoftsign_ua_uc = 21000;
		static constexpr numel_cnt_t softsigm = 15000;
		static constexpr numel_cnt_t dsoftsigm = 15000;

		static constexpr numel_cnt_t dSoftSigmQuadLoss_dZ = 10000;
		static constexpr numel_cnt_t dSoftSigmXEntropyLoss_dZ = 5000;

		static constexpr numel_cnt_t selu = 1500;
		static constexpr numel_cnt_t dselu = 25000;

		static constexpr numel_cnt_t step = 132000;

		static constexpr numel_cnt_t softmax_parts = 1900;
		static constexpr numel_cnt_t softmax_parts_mt_cw_ColsPerThread = 3;
		static constexpr numel_cnt_t softmax_parts_mt_rows = 1000;
		static constexpr numel_cnt_t softmax = 3000;//not tested

		static constexpr numel_cnt_t loss_quadratic = 23600;
		static constexpr numel_cnt_t loss_quadratic_ns = 20000;
		static constexpr numel_cnt_t loss_xentropy = 1000;// 800;
		static constexpr numel_cnt_t loss_xentropy_ns = 1000;
		static constexpr numel_cnt_t loss_softmax_xentropy = 1100;

		static constexpr numel_cnt_t RMSProp_Hinton = 2940;
		static constexpr numel_cnt_t RMSProp_Graves = 2970;
		static constexpr numel_cnt_t RProp = 5220;
		static constexpr numel_cnt_t ModProp = 4170;
		static constexpr numel_cnt_t Adam = 3000;
		static constexpr numel_cnt_t AdaMax = 1200;

		static constexpr numel_cnt_t RNadam = 3000;

		//////////////////////////////////////////////////////////////////////////
		template<typename WlT> struct dLoss_dZ {};
		template<> struct dLoss_dZ<activation::tag_Linear_Loss_quadWeighted_FP> { static constexpr numel_cnt_t thr = 10000; };

		template<typename WlT> struct compute_loss {};
		template<> struct compute_loss<activation::tag_Loss_quadratic> { static constexpr numel_cnt_t thr = 11000; };
		template<> struct compute_loss<activation::tag_Loss_quadWeighted_FP> { static constexpr numel_cnt_t thr = 8100; };
	};

	template <> struct MATHN_THR<float> : public SMATH_THR<float> {
		static constexpr numel_cnt_t ewBinarize_ip = 13000;
		static constexpr numel_cnt_t ewBinarize = 9200;

		static constexpr numel_cnt_t ewBinarizeBatch = 9200;

		static constexpr numel_cnt_t mExtractCols = 10000;
		static constexpr numel_cnt_t mExtractRows = 7000;

		static constexpr numel_cnt_t mExtractRowsSeq = 10000;

		static constexpr vec_len_t mExtractRowsByMask = 12000;//*
		static constexpr vec_len_t mFillRowsByMask = 10000;//*

		static constexpr numel_cnt_t mTransposeTrsh = 90000;

		static constexpr numel_cnt_t mrwL2NormSquared = 250000;
		static constexpr vec_len_t mrwL2NormSquared_mt_cw_ColsPerThread = 3;

		static constexpr numel_cnt_t mCheck_normalize_rows = 250000;
		static constexpr numel_cnt_t evClamp = 14000;
		static constexpr numel_cnt_t make_dropout = 7500;//* for 0.5
		static constexpr numel_cnt_t apply_dropout_mask = 4000;
		static constexpr numel_cnt_t make_alphaDropout = 7500;//* for 0.9

		static constexpr numel_cnt_t apply_ILR_st_vec = 2620; //*
		static constexpr numel_cnt_t apply_ILR_mt = 9000; //*
		static constexpr numel_cnt_t apply_ILR_mt_vec = 120000; //*
		static constexpr numel_cnt_t apply_ILR_mt_vec2 = 261000; //*

		static constexpr numel_cnt_t apply_momentum = 49000;

		static constexpr numel_cnt_t evMulCAddC_ip = 20000;//nt
		static constexpr numel_cnt_t evAddC_ip = 30000;//nt
		static constexpr numel_cnt_t evMulC_ip = 25000;//nt

		static constexpr numel_cnt_t evMul_ip = 9000;
		//static constexpr numel_cnt_t evMul_ip_Anb = 73400;
		static constexpr numel_cnt_t evAdd_ip = 12000;

		static constexpr numel_cnt_t evAddScaled_ip = 10000;//*
		static constexpr numel_cnt_t evNZAddScaled_ip = 11000;//*
		
		static constexpr numel_cnt_t evAddScaledSign_ip = 15000;//not tested
		static constexpr numel_cnt_t evNZAddScaledSign_ip = 6000; //nt

		static constexpr numel_cnt_t evSign = 30000;//not tested

		static constexpr numel_cnt_t evOneCompl = 40000;//*

		static constexpr numel_cnt_t evSub_ip = 12000;//*
		static constexpr numel_cnt_t evSub = 28000;
		static constexpr numel_cnt_t evMulC_ip_Sub_ip = 30000;

		static constexpr numel_cnt_t evSubMtxMulC_ip_nb = 9000;//*

		static constexpr numel_cnt_t evSquare = 49000;
		//static constexpr numel_cnt_t vSumSquares = 40000; //not tested
		static constexpr numel_cnt_t evAbs = 43000;
		static constexpr numel_cnt_t vSumAbs = 36000;//not tested

		//static constexpr numel_cnt_t dIdentity = 30000;//nt
		//static constexpr numel_cnt_t dIdentityQuadLoss_dZ = 15000;//nt
		static constexpr numel_cnt_t dIdentityXEntropyLoss_dZ = 10000;//nt

		static constexpr numel_cnt_t sigm = 4300;//*
		static constexpr numel_cnt_t dsigm = 12000;//*
		static constexpr numel_cnt_t dSigmQuadLoss_dZ = 9200;//*

		static constexpr numel_cnt_t relu = 13000;//*
		static constexpr numel_cnt_t drelu = 14000;//*
		static constexpr numel_cnt_t leakyrelu = 13000;//*
		static constexpr numel_cnt_t dleakyrelu = 14000;//*

		static constexpr numel_cnt_t elu = 4300;//*
		static constexpr numel_cnt_t delu = 14500;//*
		static constexpr numel_cnt_t elu_unitalpha = 4300;//*
		static constexpr numel_cnt_t delu_unitalpha = 10000;//*

		static constexpr numel_cnt_t elogu = 1400;//*
		static constexpr numel_cnt_t delogu = 4300;//*
		static constexpr numel_cnt_t elogu_ua = 1400;//*
		static constexpr numel_cnt_t delogu_ua = 4300;//*
		static constexpr numel_cnt_t elogu_nb = 1400;//*
		static constexpr numel_cnt_t delogu_nb = 4300;//*
		static constexpr numel_cnt_t elogu_ua_nb = 1400;//*
		static constexpr numel_cnt_t delogu_ua_nb = 4300;//*

		static constexpr numel_cnt_t loglogu = 3600;//*
		static constexpr numel_cnt_t dloglogu = 4200;//*
		static constexpr numel_cnt_t loglogu_nbn = 1000;//*
		static constexpr numel_cnt_t dloglogu_nbn = 4300;//*
		static constexpr numel_cnt_t loglogu_nbp = 950;//*
		static constexpr numel_cnt_t dloglogu_nbp = 4800;//*
		static constexpr numel_cnt_t loglogu_nbn_nbp = 1000;//*
		static constexpr numel_cnt_t dloglogu_nbn_nbp = 4900;//*

		static constexpr numel_cnt_t softsign = 8900;//*
		static constexpr numel_cnt_t softsign_uc = 9300;//*
		static constexpr numel_cnt_t dsoftsign = 14000;//*
		static constexpr numel_cnt_t dsoftsign_ua_uc = 14000;//*
		static constexpr numel_cnt_t softsigm = 9000;//*
		static constexpr numel_cnt_t dsoftsigm = 11000;//*

		static constexpr numel_cnt_t dSoftSigmQuadLoss_dZ = 8700;
		static constexpr numel_cnt_t dSoftSigmXEntropyLoss_dZ = 7100;

		static constexpr numel_cnt_t selu = 4200;//*
		static constexpr numel_cnt_t dselu = 15000;//*

		static constexpr numel_cnt_t step = 500000;

		static constexpr numel_cnt_t softmax_parts = 3200;
		static constexpr numel_cnt_t softmax_parts_mt_cw_ColsPerThread = 3;
		static constexpr numel_cnt_t softmax_parts_mt_rows = 6000;
		static constexpr numel_cnt_t softmax = 5500;//not tested

		static constexpr numel_cnt_t loss_quadratic = 11000;//*
		static constexpr numel_cnt_t loss_quadratic_ns = 11000;
		static constexpr numel_cnt_t loss_xentropy = 850;//750;
		static constexpr numel_cnt_t loss_xentropy_ns = 850;//750;
		static constexpr numel_cnt_t loss_softmax_xentropy = 1100;

		static constexpr numel_cnt_t RMSProp_Hinton = 8100;
		static constexpr numel_cnt_t RMSProp_Graves = 8000;
		static constexpr numel_cnt_t RProp = 12000;
		static constexpr numel_cnt_t ModProp = 9300;
		static constexpr numel_cnt_t Adam = 7300;//*
		static constexpr numel_cnt_t AdaMax = 2900;//*

		static constexpr numel_cnt_t RNadam = 7400;

		//////////////////////////////////////////////////////////////////////////
		template<typename WlT> struct dLoss_dZ {};
		template<> struct dLoss_dZ<activation::tag_Linear_Loss_quadWeighted_FP> { static constexpr numel_cnt_t thr = 8100; };//*

		template<typename WlT> struct compute_loss {};
		template<> struct compute_loss<activation::tag_Loss_quadratic> { static constexpr numel_cnt_t thr = 11000; };//*
		template<> struct compute_loss<activation::tag_Loss_quadWeighted_FP> { static constexpr numel_cnt_t thr = 8100; };//*
		
	};

}

}
}
