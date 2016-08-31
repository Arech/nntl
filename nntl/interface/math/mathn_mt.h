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

// imath_basic version with multithreaded only functions (no mt/st branching code)
// It's at max about 0.5-1.0% slower, than tuned version on a sufficiently large data sizes with double

#include "imath_basic.h"

namespace nntl {
namespace math {

	template <typename RealT, typename iThreadsT, typename ThresholdsT = _impl::IMATH_BASIC_THR<RealT>>
	class iMath_basic_mt final : public _iMath_basic<RealT, iThreadsT, ThresholdsT, iMath_basic_mt<RealT, iThreadsT, ThresholdsT>> {
	public:
		typedef _iMath_basic<RealT, iThreadsT, ThresholdsT, iMath_basic_mt<RealT, iThreadsT, ThresholdsT>> base_class_t;

		~iMath_basic_mt()noexcept {};
		iMath_basic_mt() noexcept : base_class_t(){}

		//////////////////////////////////////////////////////////////////////////
		// i_math interface implementation

		//////////////////////////////////////////////////////////////////////////
		// Contnr dest is a std::vector-like container of vec_len_t, sized to m.rows(). Will contain for each row column index
		//of greatest element in a row.
// 		template<typename Contnr>
// 		void mFindIdxsOfMaxRowwise(const realmtx_t& m, Contnr& dest)noexcept {
// 			mFindIdxsOfMaxRowwise_mt_naive(m, dest);
// 		}

		void mrwIdxsOfMax(const realmtx_t& m, vec_len_t* pDest)noexcept {
			//shouldn't just run _mt version
			base_class_t::mrwIdxsOfMax(m, pDest);
		}

		//not a part of _i_math
// 		void mrwMax(const realmtx_t& m, real_t* pMax)noexcept {
// 			//shouldn't just run _mt version
// 			base_class_t::mrwMax(m, pMax);
// 		}
		
		//////////////////////////////////////////////////////////////////////////
		//extract rows with indexes specified by Contnr ridxs into dest.
		template<typename SeqIt>
		void mExtractRows(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept {
			mExtractRows_mt_naive(src, ridxsItBegin, ridxsCnt, dest);
		}
		
		//////////////////////////////////////////////////////////////////////////
		//binarize elements of real-valued matrix according to their relaion to frac
		void ewBinarize_ip(realmtx_t& A, const real_t frac)noexcept {
			base_class_t::ewBinarize_ip(A, frac);//shouldn't just run _mt version
		}
		//binarize elements of real-valued matrix according to their relaion to frac into other matrix
		template<typename DestContainerT>
		void ewBinarize(DestContainerT& Dest, const realmtx_t& A, const real_t frac)noexcept {
			base_class_t::ewBinarize(Dest, A, frac);//shouldn't just run _mt version
		}

		//////////////////////////////////////////////////////////////////////////
		// clone matrix columns to another more wide matrix
		// srcCols - matrix with source columns to be copied into dest
		// dest - destination matrix, must have the same rows count as srcCols
		// colSpec - array of vec_len_t of size colSpecCnt== srcCols.cols(). Each element specifies how many
		//		copies of a corresponding srcCols column must be made. For example, if colSpec is {2,3,4}, then
		//		the srcCols must have 3 columns. The first column is copied 2 times into first 2 columns of dest,
		//		the second - 3, the third - 4. Therefore, dest must contain 2+3+4=9 columns.		
		void mCloneCols(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec)noexcept {
			base_class_t::mCloneCols(srcCols, dest, pColSpec);
		}

		// clone a matrix column to another more wide matrix dest.cols() number of times
		// (optimized version of mCloneCols where srcCols.cols()==1 )
		void mCloneCol(const realmtx_t& srcCol, realmtx_t& dest)noexcept {
			base_class_t::mCloneCol(srcCol, dest);
		}

		//////////////////////////////////////////////////////////////////////////
		// Transforms a data matrix to be used by tiled layer. For a data with biases it looks like this:
		//																	|x1_1...x1_n 1|		:transformed data_x
		//																	|........... 1|		:to be fed to the layer
		//	data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	===>	|xi_1...xi_n 1|
		//																	|........... 1|
		//																	|xk_1...xk_n 1|
		// For a data without biases the same, just drop all the ones in the picture.
		// If src is biased matrix, then src must be a matrix of size [m, k*n+1], dest - [k*m, n+1], also biased.
		//		Last column of dest is reserved to contain biases and must be preinitialized to 1s
		// If src doesn't have biases, then it's size must be equal to [m, k*n], dest.size() == [k*m, n]
		void mTilingRoll(const realmtx_t& src, realmtx_t& dest)noexcept {
			base_class_t::mTilingRoll(src, dest);
		}
		
		// Transforms a data matrix from a tiled layer format back to normal. For a data with biases it looks like this:
		//																	|x1_1...x1_n 1|		:transformed data_x
		//																	|........... 1|		:to be fed to the layer
		//	data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	<===	|xi_1...xi_n 1|
		//																	|........... 1|
		//																	|xk_1...xk_n 1|
		// For a data without biases the same, just drop all the ones in the picture.
		// If src is biased matrix, then src must be a matrix of size [k*m, n+1], dest - [m, k*n+1], also biased.
		//		Last column of dest is reserved to contain biases and must be preinitialized to 1s
		// If src doesn't have biases, then it's size must be equal to [k*m, n], dest.size() == [k*m, n]
		void mTilingUnroll(const realmtx_t& src, realmtx_t& dest)noexcept {
			base_class_t::mTilingUnroll(src, dest);
		}

		//////////////////////////////////////////////////////////////////////////
		// treat matrix as a set of row-vectors (matrices in col-major mode!). For each row-vector check, whether
		// its length/norm is not longer, than predefined value. If it's longer, than rescale vector to this max length
		// (for use in max-norm weights regularization)
		void mCheck_normalize_rows(realmtx_t& A, const real_t maxNormSquared)noexcept {
			mCheck_normalize_rows_mt(A, maxNormSquared);
		}

		//////////////////////////////////////////////////////////////////////////
		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		template<typename Contnr>
		size_t vCountSame(const Contnr& A, const Contnr& B)noexcept {
			return vCountSame_st_naive(A, B);
			// 			if (A.size()<=50000) {
			// 				return vCountSame_st_naive(A, B);
			// 			}else return vCountSame_mt_naive(A, B);
		}

		//////////////////////////////////////////////////////////////////////////
		//clamps vector values into range
		void evClamp(realmtx_t& m, real_t lo, real_t hi)noexcept {
			evClamp_mt(m, lo, hi);
		}

		//////////////////////////////////////////////////////////////////////////
		//on entry dropoutMask must be filled with random values in [0,1]
		//binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// act must be used in "no_bias" mode
		// Actually, the function must implement so called "inverted Dropout", see http://cs231n.github.io/neural-networks-2/
		void make_dropout(realmtx_t& act, real_t dropPercAct, realmtx_t& dropoutMask)noexcept {
			make_dropout_mt(act, dropPercAct, dropoutMask);
		}
		//////////////////////////////////////////////////////////////////////////
		//apply individual learning rate to dLdW
		void apply_ILR(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			apply_ILR_mt_naive(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
		}

		//////////////////////////////////////////////////////////////////////////
		//apply momentum vW = momentum.*vW + dW
		void apply_momentum(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			apply_momentum_mt(vW, momentum, dW);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = b.*A
		void evMulC_ip(realmtx_t& A, const real_t b)noexcept {
			evMulC_ip_mt_naive(A, b);
		}

		//inplace elementwise multiplication A(no_bias) = b.*A(no_bias)
		void evMulC_ip_Anb(realmtx_t& A, const real_t b)noexcept {
			evMulC_ip_Anb_mt_naive(A, b);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = A.*B
		void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			evMul_ip_mt_naive(A, B);
		}
		
		//inplace elementwise multiplication A(no_bias) = A(no_bias).*B, - A is taken in no_bias mode
		void evMul_ip_Anb(realmtx_t& A, const realmtx_t& B)noexcept {
			evMul_ip_Anb_mt_naive(A, B);
		}
		
		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition A = A+B
		void evAdd_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			evAdd_ip_mt(A, B);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise adding of scaled vector: A = A + c*B;
		void evAddScaled_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			evAddScaled_ip_mt(A, c, B);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition of scaled signum: A = A + c*sign(B);
		//(L1 regularization, dLdW update step)
		void evAddScaledSign_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			evAddScaledSign_ip_mt(A, c, B);
		}
		
		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise subtraction A = A-B
		void evSub_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			evSub_ip_mt_naive(A, B);
		}
		
		//////////////////////////////////////////////////////////////////////////
		//elementwise subtraction C = A-B
		void evSub(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			evSub_mt_naive(A, B, C);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise scaling and subtracting: vW = momentum.*vW, W = W-vW;
		//(it's pre-fprop step of Nesterov Momentum method)
		void evMulC_ip_Sub_ip(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
			evMulC_ip_Sub_ip_mt(vW, momentum, W);
		}

		//////////////////////////////////////////////////////////////////////////
		//elementwise squaring dest = src.^2;
		void evSquare(realmtx_t& dest, const realmtx_t& src)noexcept {
			evSquare_mt(dest, src);
		}

		//////////////////////////////////////////////////////////////////////////
		//finds sum of squares of elements (squared L2 norm): return sum( A.^2 )
		real_t vSumSquares(const realmtx_t& A)noexcept {
			return vSumSquares_mt(A);
		}
		
		//////////////////////////////////////////////////////////////////////////
		//finding elementwise absolute values dest = .abs(src);
		void evAbs(realmtx_t& dest, const realmtx_t& src)noexcept {
			evAbs_mt(dest, src);
		}

		//////////////////////////////////////////////////////////////////////////
		//finds sum of abs values (L1 norm): return sum( abs(A) );
		real_t vSumAbs(const realmtx_t& A)noexcept {
			return vSumAbs_mt(A);
		}
		
		//////////////////////////////////////////////////////////////////////////
		// divide each matrix A row by corresponding vector d element, A(i,:) = A(i,:) / d(i)
		//not a part of _i_math
// 		void mrwDivideByVec(realmtx_t& A, const real_t* pDiv)noexcept {
// 			//TODO: justify selection
// 			base_class_t::mrwDivideByVec(A, pDiv);
// 		}

		//////////////////////////////////////////////////////////////////////////
		// sigmoid function
		//////////////////////////////////////////////////////////////////////////
		void sigm(realmtx_t& srcdest) noexcept {
			sigm_mt(srcdest);
		}

		//////////////////////////////////////////////////////////////////////////
		// d(sigm)/d(arg) - sigmoid derivative df = f.*(1-f), where fValue is activation value (used in no_bias version)
		void dsigm(const realmtx_t& fValue, realmtx_t& df) noexcept {
			dsigm_mt(fValue, df);
		}

		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of quadratic loss function for sigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		//dL/dZ = (err===a-y)*a*(1-a)
		// because activations comes from the output layer, expecting no biases there
		void dSigmQuadLoss_dZ(const realmtx_t& activations, const realmtx_t& data_y, realmtx_t& dLdZ) {
			dSigmQuadLoss_dZ_mt_naive(activations, data_y, dLdZ);
		}
	
		//////////////////////////////////////////////////////////////////////////
		//ReLU
		void relu(realmtx_t& srcdest) noexcept {
			relu_mt_naive(srcdest);
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ReLU)/dZ
		void drelu(const realmtx_t& fValue, realmtx_t& df) noexcept {
			 drelu_mt_naive(fValue, df);
		}

		void leakyrelu(realmtx_t& srcdest, const real_t leak) noexcept {
			leakyrelu_mt(srcdest, leak);
		}
		void dleakyrelu(const realmtx_t& fValue, realmtx_t& df, const real_t leak) noexcept {
			dleakyrelu_mt(fValue, df, leak);
		}

		void elu(realmtx_t& srcdest, const real_t alpha) noexcept {
			elu_mt(srcdest, alpha);
		}
		void delu(const realmtx_t& fValue, realmtx_t& df, const real_t alpha) noexcept {
			delu_mt(fValue, df, alpha);
		}

		void elu_unitalpha(realmtx_t& srcdest) noexcept {
			elu_unitalpha_mt(srcdest);
		}
		void delu_unitalpha(const realmtx_t& fValue, realmtx_t& df) noexcept {
			delu_unitalpha_mt(fValue, df);
		}

		//////////////////////////////////////////////////////////////////////////
		//SoftMax
		// MUST ignore biases!
		void softmax(realmtxdef_t& srcdest) noexcept {
			softmax_mt(srcdest);
		}

		//////////////////////////////////////////////////////////////////////////
		//loss functions
		//////////////////////////////////////////////////////////////////////////
		real_t loss_quadratic(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			return loss_quadratic_mt_naive(activations, data_y);
		}

		// cross entropy function for sigmoid (applicable ONLY for binary data_y and sigmoid activation function)
		// L = -y*log(a)-(1-y)log(1-a), dL/dz = dL/dA * dA/dZ = (a-y)
		real_t loss_sigm_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			return loss_sigm_xentropy_mt(activations, data_y);
		}

		real_t loss_softmax_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			return loss_softmax_xentropy_mt(activations, data_y);
		}

		//////////////////////////////////////////////////////////////////////////
		//gradient application procedures
		void RMSProp_Hinton(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			RMSProp_Hinton_mt(dW, rmsF, learningRate, emaDecay, numericStabilizer);
		}

		//////////////////////////////////////////////////////////////////////////
		void RMSProp_Graves(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			RMSProp_Graves_mt(dW, rmsF, rmsG, learningRate, emaDecay, numericStabilizer);
		}

		//////////////////////////////////////////////////////////////////////////
		void RProp(realmtx_t& dW, const real_t learningRate)noexcept {
			RProp_mt(dW, learningRate);
		}

		//////////////////////////////////////////////////////////////////////////
		// ModProp - like RMSProp, but devide dW by abs( ema(dW) ), instead of
		//		sqrt(ema(dW ^ 2)).Seems no significant changes, but faster. And sometimes works when other doesn't
		void ModProp(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			ModProp_mt(dW, rmsF, learningRate, emaDecay, numericStabilizer);
		}

		void Adam(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept
		{
			get_self().Adam_mt(dW, Mt, Vt, beta1t, beta2t, learningRate, beta1, beta2, numericStabilizer);
		}

		void AdaMax(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept
		{
			get_self().AdaMax_mt(dW, Mt, Ut, beta1t, learningRate, beta1, beta2, numericStabilizer);
		}
	};

}
}