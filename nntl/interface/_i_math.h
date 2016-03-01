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

#include "../math.h"
#include "../common.h"

namespace nntl {
namespace math {
	
	//this class defines a common interface to an underlying mathematical subroutines (such as blas).
	//The point of _i_math is to gather all necessary mathematical computations in one single place in order to have an opportunity to
	// optimize them without thinking of NN specific. _i_math just defines necessary functions to compute NN and it is its successors
	// job to implement them as good and fast as possible.
	
	template<typename RealT>
	class _i_math {
		//!! copy constructor not needed
		_i_math(const _i_math& other)noexcept = delete;
		//!!assignment is not needed
		_i_math& operator=(const _i_math& rhs) noexcept = delete;

	protected:
		_i_math()noexcept {};
		~_i_math()noexcept {};

	public:
		typedef RealT real_t;
		//typedef math_types::realmtx_ty realmtx_t;
		//typedef math_types::realmtxdef_ty realmtxdef_t;
		//typedef realmtx_t::value_type real_t;
		typedef simple_matrix<real_t> realmtx_t;
		typedef simple_matrix_deformable<real_t> realmtxdef_t;
		typedef typename realmtx_t::vec_len_t vec_len_t;
		typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

		//last operation succeded
		//nntl_interface bool succeded()const noexcept;
		
		//math preinitialization, should be called from each NN layer. n - maximum data length (in real_t), that this layer will use in calls
		//to math interface. Used to calculate max necessary temporary storage length.
		nntl_interface void preinit(const numel_cnt_t n)noexcept;
		//real math initialization, used to allocate necessary temporary storage of size max(preinit::n)
		nntl_interface bool init()noexcept;
		//frees temporary resources, allocated by init()
		nntl_interface void deinit()noexcept;

		//////////////////////////////////////////////////////////////////////////

		//fills array pDest of size m.rows() with column indexes of greatest element in each row of m
		nntl_interface void mrwIdxsOfMax(const realmtx_t& m, vec_len_t* pDest)noexcept;

		//fills array pMax of size m.rows() with maximum element in each row of m
		//nntl_interface void mrwMax(const realmtx_t& m, real_t* pMax)noexcept;

		//extract ridxsCnt rows with indexes specified by sequential iterator ridxsItBegin into dest matrix.
		template<typename SeqIt>
		nntl_interface void mExtractRows(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept;

		//binarize elements of real-valued matrix according to their relaion to frac
		nntl_interface void ewBinarize_ip(realmtx_t& A, const real_t frac)noexcept;

		//binarize elements of real-valued matrix according to their relaion to frac into other matrix
		template<typename DestContainerT>
		nntl_interface void ewBinarize(DestContainerT& Dest, const realmtx_t& A, const real_t frac)noexcept;

		// treat matrix as a set of row-vectors (matrices in col-major mode!). For each row-vector check, whether
		// its length/norm is not longer, than predefined value. If it's longer, than rescale vector to this max length
		// (for use in max-norm weights regularization)
		nntl_interface void mCheck_normalize_rows(realmtx_t& A, const real_t maxNormSquared)noexcept;

		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		template<typename Contnr>
		nntl_interface size_t vCountSame(const Contnr& A, const Contnr& B)noexcept;

		//clamps matrix values into range
		nntl_interface void evClamp(realmtx_t& m, real_t lo, real_t hi)noexcept;

		//on entry dropoutMask must be filled with random values in [0,1]
		//binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// act must be used in "no_bias" mode
		nntl_interface void make_dropout(realmtx_t& act, real_t dfrac, realmtx_t& dropoutMask)noexcept;

		//apply individual learning rate to dLdW
		nntl_interface void apply_ILR(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept;

		//apply momentum vW = momentum.*vW + dW
		nntl_interface void apply_momentum(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept;

		//////////////////////////////////////////////////////////////////////////
 		//inplace elementwise multiplication A = b.*A
		nntl_interface void evMulC_ip(realmtx_t& A, const real_t b)noexcept;

		//inplace elementwise multiplication A(no_bias) = b.*A(no_bias)
		nntl_interface void evMulC_ip_Anb(realmtx_t& A, const real_t b)noexcept;
		
		//inplace elementwise multiplication A = A.*B
		nntl_interface void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept;

		//inplace elementwise multiplication A(no_bias) = A(no_bias).*B, - A is taken in no_bias mode
		nntl_interface void evMul_ip_Anb(realmtx_t& A, const realmtx_t& B)noexcept;

		//inplace elementwise addition A = A+B
		nntl_interface void evAdd_ip(realmtx_t& A, const realmtx_t& B)noexcept;
		//inplace elementwise addition *pA = *pA + *pB
		nntl_interface void vAdd_ip(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt)noexcept;

		//inplace elementwise adding of scaled vector: A = A + c*B;
		nntl_interface void evAddScaled_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept;

		//inplace elementwise addition of scaled signum: A = A + c*sign(B);
		//(L1 regularization, dLdW update step)
		nntl_interface void evAddScaledSign_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept;

		//inplace elementwise subtraction A = A-B
		nntl_interface void evSub_ip(realmtx_t& A, const realmtx_t& B)noexcept;
		//elementwise subtraction C = A-B
		nntl_interface void evSub(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;

		//inplace elementwise scaling and subtracting: vW = momentum.*vW, W = W-vW;
		//(it's pre-fprop step of Nesterov Momentum method)
		nntl_interface void evMulC_ip_Sub_ip(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept;

		//////////////////////////////////////////////////////////////////////////

		//elementwise squaring dest = src.^2;
		nntl_interface void evSquare(realmtx_t& dest, const realmtx_t& src)noexcept;

		//finds a sum of squares of elements (squared L2 norm): return sum( A.^2 )
		nntl_interface real_t vSumSquares(const realmtx_t& A)noexcept;

		//finding elementwise absolute values dest = abs(src);
		nntl_interface void evAbs(realmtx_t& dest, const realmtx_t& src)noexcept;

		//finds a sum of abs values (L1 norm): return sum( abs(A) );
		nntl_interface real_t vSumAbs(const realmtx_t& A)noexcept;
		
		//finds a sum of elementwise products return sum( A.*B );
		nntl_interface real_t ewSumProd(const realmtx_t& A, const realmtx_t& B)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//C = A * B, - matrix multiplication
		nntl_interface void mMulAB_C(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;
		//matrix multiplication C(no bias) = A * B` (B transposed). C could have emulated biases (they will be left untouched)
		nntl_interface void mMulABt_Cnb(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;
		//C = a*(A` * B) - matrix multiplication of transposed A times B with result normalization
		nntl_interface void mScaledMulAtB_C(real_t alpha, const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;

		//////////////////////////////////////////////////////////////////////////
		// divide each matrix A row by corresponding vector d element, A(i,:) = A(i,:) / d(i)
		//nntl_interface void mrwDivideByVec(realmtx_t& A, const real_t* pDiv)noexcept;

		//////////////////////////////////////////////////////////////////////////
		// sigm activation.
		// Remember to ignore biases for activation function calculations!
		nntl_interface void sigm(realmtx_t& srcdest) noexcept;
		nntl_interface void dsigm(const realmtx_t& fValue, realmtx_t& df) noexcept;
		//calculates derivative of quadratic loss function for sigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		nntl_interface void dSigmQuadLoss_dZ(const realmtx_t& activations, const realmtx_t& data_y, realmtx_t& dLdZ);

		//////////////////////////////////////////////////////////////////////////
		//ReLU
		// MUST ignore biases!
		nntl_interface void relu(realmtx_t& srcdest) noexcept;
		nntl_interface void drelu(const realmtx_t& fValue, realmtx_t& df) noexcept;

		//////////////////////////////////////////////////////////////////////////
		//SoftMax
		// helper function that return the amount of temporary memory (in real_t) needed to process by softmax()
		// a matrix of size act.size()
		nntl_interface numel_cnt_t softmax_needTempMem(const realmtx_t& act)noexcept;
		// MUST ignore biases!
		nntl_interface void softmax(realmtxdef_t& srcdest) noexcept;


		//////////////////////////////////////////////////////////////////////////
		//loss functions
		// quadratic loss == SUM_OVER_ALL((activations-data_y)^2)/(2*activations.rows())
		nntl_interface real_t loss_quadratic(const realmtx_t& activations, const realmtx_t& data_y)noexcept;
		
		// cross entropy function for sigmoid (applicable ONLY for binary data_y)
		// L = sum( -y*log(a)-(1-y)log(1-a) )/activations.rows(), dL/dz=a-y
		nntl_interface real_t loss_sigm_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept;

		// cross entropy function for softmax (applicable for data_y in range [0,1])
		// L = sum( -y*log(a) )/activations.rows(), dL/dz=a-y
		nntl_interface real_t loss_softmax_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//gradient application procedures
		nntl_interface void RMSProp_Hinton(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept;
		nntl_interface void RMSProp_Graves(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept;
		nntl_interface void RProp(realmtx_t& dW, const real_t learningRate)noexcept;
		nntl_interface void ModProp(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept;
	};

}
}
