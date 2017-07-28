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

namespace nntl {

	enum class ADCorr {
		no,
		correctDoVal,
		correctVar,
		correctDoAndVar
	};

	namespace _impl {

		template<typename RealT, int64_t Alpha1e9 = 0, int64_t Lambda1e9 = 0, int fpMean1e6 = 0, int fpVar1e6 = 1000000, ADCorr corrType = ADCorr::no>
		struct SNN_td {
			typedef RealT real_t;

			static constexpr int64_t _TP_alpha = Alpha1e9;
			static constexpr int64_t _TP_lambda = Lambda1e9;
			static constexpr int _TP_fpMean = fpMean1e6;
			static constexpr int _TP_fpVar = fpVar1e6;
			static constexpr ADCorr _TP_corrType = corrType;


			static constexpr ext_real_t AlphaExt = Alpha1e9 ? ext_real_t(Alpha1e9) / ext_real_t(1e9) : ext_real_t(1.6732632423543772848170429916717);
			static constexpr ext_real_t LambdaExt = Lambda1e9 ? ext_real_t(Lambda1e9) / ext_real_t(1e9) : ext_real_t(1.0507009873554804934193349852946);

			static constexpr ext_real_t AlphaExt_t_LambdaExt = AlphaExt*LambdaExt;
			static constexpr ext_real_t Neg_AlphaExt_t_LambdaExt = -AlphaExt_t_LambdaExt;

			static constexpr real_t Alpha = real_t(AlphaExt);
			static constexpr real_t Lambda = real_t(LambdaExt);
			static constexpr real_t Alpha_t_Lambda = real_t(AlphaExt*LambdaExt);
			static constexpr real_t Neg_Alpha_t_Lambda = -Alpha_t_Lambda;

			static constexpr ext_real_t FixedPointMeanExt = ext_real_t(fpMean1e6) / ext_real_t(1e6);
			static constexpr ext_real_t FixedPointVarianceExt = ext_real_t(fpVar1e6) / ext_real_t(1e6);

			static constexpr real_t FixedPointMean = real_t(FixedPointMeanExt);
			static constexpr real_t FixedPointVariance = real_t(FixedPointVarianceExt);

		};

			

	}

}