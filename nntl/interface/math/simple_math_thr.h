/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015, Arech (aradvert@gmail.com; https://github.com/Arech)
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

namespace nntl {
namespace math {

	namespace _impl {
		using vec_len_t = simple_matrix<math_types::real_ty>::vec_len_t;

		template <typename real_t> struct SIMPLE_MATH_THR {};

		//It is better, than nothing. But in future this pron should be substituted by a run-time function profiling over real-task data

		template <> struct SIMPLE_MATH_THR<double> {
			static constexpr size_t ewSumProd = 8800;

			static constexpr size_t mrwDivideByVec_rw = 10000;
			static constexpr size_t mrwDivideByVec_mt_rows = 5000;
			static constexpr size_t mrwDivideByVec = 5000;

			static constexpr size_t mrwMulByVec_st_rows = 11000;
			static constexpr size_t mrwMulByVec_mt_rows = 80000;
			static constexpr size_t mrwMulByVec = 20000;

			static constexpr size_t mrwIdxsOfMax = 3700;
			static constexpr size_t mrwIdxsOfMax_st_rows = 20;//not tested
			static constexpr size_t mrwIdxsOfMax_mt_rows = 20;//not tested
			static constexpr size_t mrwIdxsOfMax_ColsPerThread = 4;//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			
			static constexpr size_t mrwMax = 5200;
			static constexpr size_t mrwMax_mt_cw_ColsPerThread = 4;//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!

			static constexpr size_t mrwSum_ip_st_cols = 11;
			static constexpr size_t mrwSum_ip_st_rows = 500;

			static constexpr size_t mrwSum = 5000;
			static constexpr size_t mrwSum_st = 24000;
			static constexpr vec_len_t mrwSum_mt_cw_colsPerThread = 3;
		};

		template <> struct SIMPLE_MATH_THR<float> {
			static constexpr size_t ewSumProd = 880*19;

			static constexpr size_t mrwDivideByVec_rw = 20000;
			static constexpr size_t mrwDivideByVec = 30000;
			static constexpr size_t mrwDivideByVec_mt_rows = 10000;

			static constexpr size_t mrwMulByVec_st_rows = 20000;
			static constexpr size_t mrwMulByVec_mt_rows = 150000;
			static constexpr size_t mrwMulByVec = 30000;


			static constexpr size_t mrwIdxsOfMax = 5000;
			static constexpr size_t mrwIdxsOfMax_st_rows = 30;//not tested
			static constexpr size_t mrwIdxsOfMax_mt_rows = 30;//not tested
			static constexpr size_t mrwIdxsOfMax_ColsPerThread = 4;//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!

			static constexpr size_t mrwMax = 8000;
			static constexpr size_t mrwMax_mt_cw_ColsPerThread = 4;//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!

			static constexpr size_t mrwSum_ip_st_cols = 11;
			static constexpr size_t mrwSum_ip_st_rows = 500;

			static constexpr size_t mrwSum = 10000;
			static constexpr size_t mrwSum_st = 24000*19/10;
			static constexpr vec_len_t mrwSum_mt_cw_colsPerThread = 3;

		};

	}

}
}

