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
namespace activation {

	namespace _impl {
		template<typename RealT>
		struct L_normalize_as_quadratic {
			typedef RealT real_t;

			static constexpr real_t normalize(const real_t lossVal, const math::smatrix_td::vec_len_t batchSiz)noexcept {
				//return lossVal / (2 * batchSiz);
				NNTL_UNREF(batchSiz);
				return lossVal / 2;
			}
		};
	}

	struct tag_Loss_quadratic {};

	template<typename real_t>
	struct Loss_quadratic : public _impl::L_normalize_as_quadratic<real_t> {
		typedef tag_Loss_quadratic tag_loss;

		static real_t loss(const real_t act, const real_t targetY)noexcept {
			const real_t e = act - targetY;
			return e*e;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// The following loss functions are experimental and mostly ill-formed (could have discontinuities). Use with care
	// 
	struct tag_Loss_quadWeighted_FP{};

	template<typename real_t, unsigned int _FPWeightSqrt1e6 = 1414214, unsigned int _YBnd1e6 = 0>
	struct Loss_quadWeighted_FP : public _impl::L_normalize_as_quadratic<real_t> {
		typedef tag_Loss_quadWeighted_FP tag_loss;

		static constexpr real_t FPWeightSqrt = _FPWeightSqrt1e6 / real_t(1e6);
		static constexpr real_t FPWeight = FPWeightSqrt*FPWeightSqrt;
		static constexpr real_t YBoundary = _YBnd1e6 / real_t(1e6);

		static real_t loss(const real_t a, const real_t y)noexcept {
			const real_t e = ((y < YBoundary) & (a > y)) ? FPWeightSqrt*(a - y) : (a - y);
			//using the shortcutting && instead of binary & prevents vectorization. We can safely use binary & here,
			// because the comparison results are always binary.
			// However, probably due to the e*e statement below it still doesn't get vectorized.
			return e*e;
		}
	};

	template<typename real_t, unsigned int _FPWeightSqrt1e6 = 1414214, unsigned int _OtherWeightSqrt1e6 = 500000, unsigned int _YBnd1e6 = 0>
	struct Loss_quadWeighted2_FP : public _impl::L_normalize_as_quadratic<real_t> {
		typedef tag_Loss_quadWeighted_FP tag_loss;

		static constexpr real_t FPWeightSqrt = _FPWeightSqrt1e6 / real_t(1e6);
		static constexpr real_t FPWeight = FPWeightSqrt*FPWeightSqrt;
		static constexpr real_t OtherWeightSqrt = _OtherWeightSqrt1e6 / real_t(1e6);
		static constexpr real_t OtherWeight = OtherWeightSqrt*OtherWeightSqrt;

		static constexpr real_t YBoundary = _YBnd1e6 / real_t(1e6);

		static real_t loss(const real_t a, const real_t y)noexcept {
			const real_t e = (a - y)*(((y < YBoundary) & (a > y)) ? FPWeightSqrt : OtherWeightSqrt);
			//using the shortcutting && instead of binary & prevents vectorization. We can safely use binary & here,
			// because the comparison results are always binary
			// However, probably due to the e*e statement below it still doesn't get vectorized.
			return e*e;
		}
	};

	//this loss apply heavier penalty to a difference (a-YBoundary) when (y < YBoundary) & (a > YBoundary)
	template<typename real_t, unsigned int _FPWeightSqrt1e6 = 1414214, unsigned int _YBnd1e6 = 0>
	struct Loss_quadWeighted_res_FP : public _impl::L_normalize_as_quadratic<real_t> {
		typedef tag_Loss_quadWeighted_FP tag_loss;

		static constexpr real_t FPWeightSqrt = _FPWeightSqrt1e6 / real_t(1e6);
		static constexpr real_t FPWeight = FPWeightSqrt*FPWeightSqrt;
		static constexpr real_t YBoundary = _YBnd1e6 / real_t(1e6);

		static constexpr real_t addVSqrt = YBoundary*(real_t(1.) - FPWeightSqrt);
		static constexpr real_t addV = FPWeightSqrt*addVSqrt;

		//L 4fp = (FPWeightSqrt*(a-YBoundary) + YBoundary-y)^2 = (FPWeightSqrt*a + YBoundary(1-FPWeightSqrt) - y)^2 =
		//		= (FPWeightSqrt*a + addVSqrt - y)^2
		// dL/dZ 4FP = FPWeightSqrt(FPWeightSqrt*a + addVSqrt - y) = FPWeight*a + FPWeightSqrt*addVSqrt - FPWeightSqrt*y =
		//		= FPWeight*a + addV - FPWeightSqrt*y
		static real_t loss(const real_t a, const real_t y)noexcept {
			const real_t e = ((y < YBoundary) & (a > y)) ? (FPWeightSqrt*a + addVSqrt - y) : (a - y);
			//using the shortcutting && instead of binary & prevents vectorization. We can safely use binary & here,
			// because the comparison results are always binary
			// However, probably due to the e*e statement below it still doesn't get vectorized.
			return e*e;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	struct tag_Linear_Loss_quadWeighted_FP {};

	//Linear_Loss_quadWeighted_FP is a loss function that assigns more weight to false positive errors
	//default _FPWeightSqrt1e6 == sqrt(2)*1e6
	template<typename real_t, unsigned int _FPWeightSqrt1e6 = 1414214, unsigned int _YBnd1e6 = 0>
	struct Linear_Loss_quadWeighted_FP : public Loss_quadWeighted_FP<real_t, _FPWeightSqrt1e6, _YBnd1e6> {
		typedef tag_Linear_Loss_quadWeighted_FP tag_dLdZ;

		static constexpr real_t dLdZ(const real_t y, const real_t a)noexcept {
			return ((y < YBoundary) & (a > y)) ? FPWeight*(a - y) : (a - y);
			//using the shortcutting && instead of binary & prevents vectorization. We can safely use binary & here,
			// because the comparison results are always binary
		}
	};

	template<typename real_t, unsigned int _FPWeightSqrt1e6 = 1414214, unsigned int _OtherWeightSqrt1e6 = 500000, unsigned int _YBnd1e6 = 0>
	struct Linear_Loss_quadWeighted2_FP : public Loss_quadWeighted2_FP<real_t, _FPWeightSqrt1e6, _OtherWeightSqrt1e6, _YBnd1e6> {
		typedef tag_Linear_Loss_quadWeighted_FP tag_dLdZ;

		static constexpr real_t dLdZ(const real_t y, const real_t a)noexcept {
			return ((y < YBoundary) & (a > y)) ? FPWeight*(a - y) : OtherWeight*(a - y);
			//using the shortcutting && instead of binary & prevents vectorization. We can safely use binary & here,
			// because the comparison results are always binary
		}
	};

	template<typename real_t, unsigned int _FPWeightSqrt1e6 = 1414214, unsigned int _YBnd1e6 = 0>
	struct Linear_Loss_quadWeighted_res_FP : public Loss_quadWeighted_res_FP<real_t, _FPWeightSqrt1e6, _YBnd1e6> {
		typedef tag_Linear_Loss_quadWeighted_FP tag_dLdZ;
		//L 4fp = (FPWeightSqrt*(a-YBoundary) + YBoundary-y)^2 = (FPWeightSqrt*a + YBoundary(1-FPWeightSqrt) - y)^2 =
		//		= (FPWeightSqrt*a + addVSqrt - y)^2
		// dL/dZ 4FP = FPWeightSqrt(FPWeightSqrt*a + addVSqrt - y) = FPWeight*a + FPWeightSqrt*addVSqrt - FPWeightSqrt*y =
		//		= FPWeight*a + addV - FPWeightSqrt*y
		static constexpr real_t dLdZ(const real_t y, const real_t a)noexcept {
			return ((y < YBoundary) & (a > y)) ? (FPWeight*a + addV - FPWeightSqrt*y) : (a - y);
			//using the shortcutting && instead of binary & prevents vectorization. We can safely use binary & here,
			// because the comparison results are always binary
		}
	};
}
}