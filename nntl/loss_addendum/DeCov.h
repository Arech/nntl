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

//This loss addendum is designed to be used with activation values. It penalizes activations of neurons that is
//too correlated (actually, covariated), i.e. it penalizes linear dependency between layer activations values.
//For layers that have a mix of different activation functions (i.e. some neurons use relu and some sigm)
// DeCorr<> loss addendum is probably more suitable.
// 
// Implements DeCov regularizer from the paper "Reducing Overfitting in Deep Neural Networks by Decorrelating Representations", 2015, ArXiv:1511.06068
// (similar to “Discovering Hidden Factors of Variation in Deep Networks”, ArXiv:1412.6583)
// BTW, there's wrong derivative presented in "Reducing Overfitting...". Actual derivative must be 2 times greater, than printed in paper.

#include "_i_loss_addendum.h"

namespace nntl {
	namespace loss_addendum {

		// BE AWARE that the amount of scaling applied to this regularized is not stable w.r.t. the batch size and layer neurons count.
		// I.e. if you found some good scaling value for DeCov and then changed the batch size or the layer neurons count, then you'd
		// have to repeat the search for the proper scaling again!
		template<typename RealT, bool bNumStab = false, bool bLowerTriangl = false>
		class DeCov : public _scaled_addendum<RealT> {
		public:
			static constexpr const char* getName()noexcept { return "DeCov"; }

			
			template <typename iMathT>
			static void init(const bool& bWillDoTraining, const realmtx_t& biggestMtx, iMathT& iM) noexcept {
				//we'll need additional biggestMtx.numel_no_bias() elements to store temporary dLoss mtx
				iM.preinit(iM.loss_DeCov_tempMemReqs(bWillDoTraining, biggestMtx) + bWillDoTraining* biggestMtx.numel_no_bias());
			}

			//computes loss addendum for a given matrix of values (for weight-decay Vals parameter is a weight matrix)
			// DeCov loss is L=0.5*\sum_{i\neq j}C_{ij}, where C - covariance matrix
			// C_{ij}=\frac{1}{N} \sum_{n=1}^N (h_i^n - \mu_i)(h_j^n - \mu_j),
			// and \mu_i=\frac{1}{N} \sum_{n=1}^N h_i^n - columnwise mean of matrix H (or Vals; it has N rows (batchsize)).
			template <typename iMathT>
			real_t lossAdd(const realmtx_t& Vals, iMathT& iM) const noexcept {
				NNTL_ASSERT(!Vals.emulatesBiases());
				return m_scale* iM.loss_deCov<bLowerTriangl, bNumStab>(Vals);
			}
			// \frac{\partial L}{\partial h_a^m} = \frac{2}{N} \sum_{j\neq a} C_{aj} (h_j^m - \mu_j)
			// Contrary to what's posted in the paper, correct derivative has 2 in the numerator (instead of 1), because C is symmetrical matrix.
			template <typename iMathT, typename iInspectT>
			void dLossAdd(const realmtx_t& Vals, realmtx_t& dLossdVals, iMathT& iM, iInspectT& iI) const noexcept {
				NNTL_ASSERT(!Vals.emulatesBiases() && !dLossdVals.emulatesBiases());

				const auto dLNumel = dLossdVals.numel();
				real_t*const pDL = iM._istor_alloc(dLNumel);
				realmtx_t dL(pDL, dLossdVals.size());

				iM.dLoss_deCov<bLowerTriangl, bNumStab>(Vals, dL);

				iI.dLossAddendumScaled(dL, m_scale, getName());

				iM.evAddScaled_ip(dLossdVals, m_scale, dL);

				iM._istor_free(pDL, dLNumel);
			}
		};

	}
}