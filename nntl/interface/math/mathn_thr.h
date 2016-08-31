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

//This file define a single/multi-threading code thresholds for iMath_basic class
//Substitute for your own if you want to utilize mt/st branching code of imath_basic
// Or just use imath_basic_mt for reasonably large data sizes

#include "smath_thr.h"

namespace nntl {
namespace math {

namespace _impl {

	template <typename real_t> struct IMATH_BASIC_THR : public SIMPLE_MATH_THR<real_t> {};

	//It is better, than nothing. But in future this pron should be substituted by a run-time function profiling over real-task data

	template <> struct IMATH_BASIC_THR<double> : public SIMPLE_MATH_THR<double> {
		static constexpr size_t ewBinarize_ip = 132000;
		static constexpr size_t ewBinarize = 11000;

		static constexpr size_t mExtractRows = 2000;
		static constexpr size_t mCheck_normalize_rows = 124000;
		static constexpr size_t evClamp = 9000;
		static constexpr size_t make_dropout = 7000;

		static constexpr size_t apply_ILR_st = 3035;
		static constexpr size_t apply_ILR_mt_lo = 131000;
		static constexpr size_t apply_ILR_mt_hi = 332000;

		static constexpr size_t apply_momentum = 21300;

		static constexpr size_t evMulC_ip = 64000;
		static constexpr size_t evMulC_ip_Anb = 64000;
		static constexpr size_t evMul_ip = 36700;
		static constexpr size_t evMul_ip_Anb = 36700;
		static constexpr size_t evAdd_ip = 21000;
		static constexpr size_t evAddScaled_ip = 17000;//not tested
		static constexpr size_t evAddScaledSign_ip = 15000;//not tested

		static constexpr size_t evSub_ip = 22000;
		static constexpr size_t evSub = 12600;
		static constexpr size_t evMulC_ip_Sub_ip = 15000;

		static constexpr size_t evSquare = 24400;
		static constexpr size_t vSumSquares = 20000;//not tested
		static constexpr size_t evAbs = 20300;
		static constexpr size_t vSumAbs = 18000;//not tested		

		static constexpr size_t sigm = 1000;
		static constexpr size_t dsigm = 28000;
		static constexpr size_t dSigmQuadLoss_dZ = 14900;

		static constexpr size_t relu = 6300;
		static constexpr size_t drelu = 4700;
		static constexpr size_t leakyrelu = 6000;
		static constexpr size_t dleakyrelu = 4700;

		static constexpr size_t elu = 1400;
		static constexpr size_t delu = 4700;
		static constexpr size_t elu_unitalpha = 1500;
		static constexpr size_t delu_unitalpha = 4700;

		static constexpr size_t softmax_parts = 1900;
		static constexpr size_t softmax_parts_mt_cw_ColsPerThread = 3;
		static constexpr size_t softmax_parts_mt_rows = 1000;
		static constexpr size_t softmax = 3000;//not tested

		static constexpr size_t loss_quadratic = 24400;
		static constexpr size_t loss_sigm_xentropy = 800;
		static constexpr size_t loss_softmax_xentropy = 1100;

		static constexpr size_t RMSProp_Hinton = 2940;
		static constexpr size_t RMSProp_Graves = 2970;
		static constexpr size_t RProp = 5220;
		static constexpr size_t ModProp = 4170;
		static constexpr size_t Adam = 3000;
		static constexpr size_t AdaMax = 1200;
	};

	template <> struct IMATH_BASIC_THR<float> : public SIMPLE_MATH_THR<float> {
		static constexpr size_t ewBinarize_ip = 500000;
		static constexpr size_t ewBinarize = 1100*19;

		static constexpr size_t mExtractRows = 3100;
		static constexpr size_t mCheck_normalize_rows = 250000;
		static constexpr size_t evClamp = 14000;
		static constexpr size_t make_dropout = 13000;//*

		static constexpr size_t apply_ILR_st = 3100;
		static constexpr size_t apply_ILR_mt_lo = 240000;
		static constexpr size_t apply_ILR_mt_hi = 1250000;

		static constexpr size_t apply_momentum = 49000;

		static constexpr size_t evMulC_ip = 131000;
		static constexpr size_t evMulC_ip_Anb = 131000;
		static constexpr size_t evMul_ip = 60000;
		static constexpr size_t evMul_ip_Anb = 73400;
		static constexpr size_t evAdd_ip = 41200;
		static constexpr size_t evAddScaled_ip = 31000;//not tested
		static constexpr size_t evAddScaledSign_ip = 28000;//not tested

		static constexpr size_t evSub_ip = 46000;
		static constexpr size_t evSub = 28000;
		static constexpr size_t evMulC_ip_Sub_ip = 30000;

		static constexpr size_t evSquare = 49000;
		static constexpr size_t vSumSquares = 40000; //not tested
		static constexpr size_t evAbs = 43000;
		static constexpr size_t vSumAbs = 36000;//not tested

		static constexpr size_t sigm = 35000;
		static constexpr size_t dsigm = 52000;
		static constexpr size_t dSigmQuadLoss_dZ = 32000;

		static constexpr size_t relu = 5500;
		static constexpr size_t drelu = 4700;
		static constexpr size_t leakyrelu = 5400;//*
		static constexpr size_t dleakyrelu = 4700;//*

		static constexpr size_t elu = 1400;//*
		static constexpr size_t delu = 4700;//*
		static constexpr size_t elu_unitalpha = 1500;//*
		static constexpr size_t delu_unitalpha = 4700;//*

		static constexpr size_t softmax_parts = 3200;
		static constexpr size_t softmax_parts_mt_cw_ColsPerThread = 3;
		static constexpr size_t softmax_parts_mt_rows = 6000;
		static constexpr size_t softmax = 5000;//not tested

		static constexpr size_t loss_quadratic = 49600;
		static constexpr size_t loss_sigm_xentropy = 750;
		static constexpr size_t loss_softmax_xentropy = 1000;

		static constexpr size_t RMSProp_Hinton = 8100;
		static constexpr size_t RMSProp_Graves = 8000;
		static constexpr size_t RProp = 12000;
		static constexpr size_t ModProp = 9300;
		static constexpr size_t Adam = 8000;//*
		static constexpr size_t AdaMax = 2900;//*
	};

}

}
}
