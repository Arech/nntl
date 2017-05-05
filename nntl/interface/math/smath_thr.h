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

//This file define a single/multi-threading code thresholds for SMath class
//Substitute for your own if you want to utilize mt/st branching code of SMath

namespace nntl {
namespace math {

	namespace _impl {
		using vec_len_t = smatrix_td::vec_len_t;

		template <typename real_t> struct SMATH_THR {};

		//It is better, than nothing. But in future this pron should be substituted by a run-time function profiling over real-task data

		template <> struct SMATH_THR<double> {
			static constexpr size_t ewSumProd = 8800;
			static constexpr size_t ewSumSquares = 8800;//nt
			static constexpr size_t ewSumSquares_ns = 8800;//nt

			template <bool bLowerTriangl, bool bNumStab> struct ewSumSquaresTriang {};
			template<>struct ewSumSquaresTriang<false, true> { static constexpr vec_len_t v = 90; };//*
			template<>struct ewSumSquaresTriang<false, false> { static constexpr vec_len_t v = 176; };//*
			template<>struct ewSumSquaresTriang<true, true> { static constexpr vec_len_t v = 90; };//*
			template<>struct ewSumSquaresTriang<true, false> { static constexpr vec_len_t v = 165; };//*

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

			static constexpr size_t mrwBinaryOR = 5000;
			//static constexpr size_t mrwBinaryOR_st = 24000;
			static constexpr vec_len_t mrwBinaryOR_mt_cw_colsPerThread = 3;

			template <bool bNumStab> struct mcwMean {};
			template<>struct mcwMean<true> { static constexpr size_t v = 10000; };
			template<>struct mcwMean<false> { static constexpr size_t v = 10000; };

			static constexpr size_t mcwSub_ip = 30000;
			static constexpr size_t mcwMulDiag_ip = 20000;//*

			static constexpr vec_len_t mCloneCols = 2;
			static constexpr vec_len_t mCloneCol = 2;

			static constexpr size_t mTilingRoll = 23000;
			static constexpr vec_len_t mTilingRoll_mt_cols = 3;

			static constexpr size_t mTilingUnroll = 23000;
			static constexpr vec_len_t mTilingUnroll_mt_cols = 3;
		};

		template <> struct SMATH_THR<float> {
			static constexpr size_t ewSumProd = 880*19;
			static constexpr size_t ewSumSquares = 18000;
			static constexpr size_t ewSumSquares_ns = 17000;

			template <bool bLowerTriangl,bool bNumStab> struct ewSumSquaresTriang {};
			template<>struct ewSumSquaresTriang<false, true> { static constexpr vec_len_t v = 90; };
			template<>struct ewSumSquaresTriang<false, false> { static constexpr vec_len_t v = 176; };
			template<>struct ewSumSquaresTriang<true, true> { static constexpr vec_len_t v = 90; };
			template<>struct ewSumSquaresTriang<true, false> { static constexpr vec_len_t v = 165; };

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

			template <bool bNumStab> struct mcwMean {};
			template<>struct mcwMean<true> { static constexpr size_t v = 5000; };
			template<>struct mcwMean<false> { static constexpr size_t v = 17000; };
			
			static constexpr size_t mcwSub_ip = 30000;

			static constexpr size_t mcwMulDiag_ip = 24500;

			static constexpr size_t mrwBinaryOR = 10000;
			//static constexpr size_t mrwBinaryOR_st = 24000 * 19 / 10;
			static constexpr vec_len_t mrwBinaryOR_mt_cw_colsPerThread = 3;

			static constexpr vec_len_t mCloneCols = 2;
			static constexpr vec_len_t mCloneCol = 2;

			static constexpr size_t mTilingRoll = 23000*19/10;
			static constexpr vec_len_t mTilingRoll_mt_cols = 3;

			static constexpr size_t mTilingUnroll = 23000*19/10;
			static constexpr vec_len_t mTilingUnroll_mt_cols = 3;
		};

	}

}
}

