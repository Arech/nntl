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

#include "../math.h"
#include "../../common.h"

namespace nntl {
namespace math {
	
	//this class defines a common interface to an underlying mathematical subroutines (such as blas).
	//The point of _i_math is to gather all necessary mathematical computations in one single place in order to have an opportunity to
	// optimize them without thinking of NN specific. _i_math just defines necessary functions to compute NN and it is its successors
	// job to implement them as good and fast as possible.
	// 
	class _i_math {
		//!! copy constructor not needed
		_i_math(const _i_math& other)noexcept = delete;
		//!!assignment is not needed
		_i_math& operator=(const _i_math& rhs) noexcept = delete;

	protected:
		_i_math()noexcept {};
		~_i_math()noexcept {};

	public:
		//typedef math_types::float_ty float_t_;
		typedef math_types::floatmtx_ty floatmtx_t;
		typedef floatmtx_t::value_type float_t_;
		typedef floatmtx_t::vec_len_t vec_len_t;
		typedef floatmtx_t::numel_cnt_t numel_cnt_t;

		//last operation succeded
		//nntl_interface bool succeded()const noexcept;
		
		//math preinitialization, should be called from each NN layer. n - maximum data length (in float_t_), that this layer will use in calls
		//to math interface. Used to calculate max necessary temporary storage length.
		nntl_interface void preinit(const numel_cnt_t n)noexcept;
		//real math initialization, used to allocate necessary temporary storage of size max(preinit::n)
		nntl_interface bool init()noexcept;
		//frees temporary resources, allocated by init()
		nntl_interface void deinit()noexcept;

		// Contnr dest is a std::vector-like container of vec_len_t, sized to m.rows(). Will contain for each row column index
		//of greatest element in a row.
		template<typename Contnr>
		nntl_interface void mFindIdxsOfMaxRowwise(const floatmtx_t& m, Contnr& dest)noexcept;

		//extract ridxsCnt rows with indexes specified by sequential iterator ridxsItBegin into dest matrix.
		template<typename SeqIt>
		nntl_interface void mExtractRows(const floatmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, floatmtx_t& dest)noexcept;

		//binarize real-valued matrix with values in [0,1] according to 0<=frac<=1
		nntl_interface void mBinarize(floatmtx_t& A, const float_t_ frac)noexcept;

		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		template<typename Contnr>
		nntl_interface size_t vCountSame(const Contnr& A, const Contnr& B)noexcept;

		//clamps matrix values into range
		nntl_interface void evClamp(floatmtx_t& m, float_t_ lo, float_t_ hi)noexcept;

		//on entry dropoutMask must be filled with random values in [0,1]
		//binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// act must be used in "no_bias" mode
		nntl_interface void make_dropout(floatmtx_t& act, float_t_ dfrac, floatmtx_t& dropoutMask)noexcept;

		//apply individual learning rate to dLdW
		nntl_interface void apply_ILR(floatmtx_t& dLdW, const floatmtx_t& prevdLdW, floatmtx_t& ILRGain,
			const float_t_ decr, const float_t_ incr, const float_t_ capLow, const float_t_ capHigh)noexcept;

		//apply momentum vW = momentum.*vW + dW
		nntl_interface void apply_momentum(floatmtx_t& vW, const float_t_ momentum, const floatmtx_t& dW)noexcept;

		//////////////////////////////////////////////////////////////////////////
		// elementwise substraction C=A-B. C is expected to be different from A and B
// 		nntl_interface void evSubtract(const floatmtx_t&A, const floatmtx_t& B, floatmtx_t& C) noexcept;
// 		// B=1-A elementwise subtraction. B must be different from A.
// 		nntl_interface void evOneMinusA(const floatmtx_t&A, floatmtx_t& B) noexcept;

 		//inplace elementwise multiplication A = b.*A
		nntl_interface void evMulC_ip(floatmtx_t& A, const float_t_ b)noexcept;

		//inplace elementwise multiplication A(no_bias) = b.*A(no_bias)
		nntl_interface void evMulC_ip_Anb(floatmtx_t& A, const float_t_ b)noexcept;
		
		//inplace elementwise multiplication A = A.*B
		nntl_interface void evMul_ip(floatmtx_t& A, const floatmtx_t& B)noexcept;

		//inplace elementwise multiplication A(no_bias) = A(no_bias).*B, - A is taken in no_bias mode
		nntl_interface void evMul_ip_Anb(floatmtx_t& A, const floatmtx_t& B)noexcept;

		//inplace elementwise subtraction A = A-B
		nntl_interface void evSub_ip(floatmtx_t& A, const floatmtx_t& B)noexcept;
		//elementwise subtraction C = A-B
		nntl_interface void evSub(const floatmtx_t& A, const floatmtx_t& B, floatmtx_t& C)noexcept;

		//elementwise squaring dest = src.^2;
		nntl_interface void evSquare(floatmtx_t& dest, const floatmtx_t& src)noexcept;

		//finding elementwise absolute values dest = .abs(src);
		nntl_interface void evAbs(floatmtx_t& dest, const floatmtx_t& src)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//C = A * B, - matrix multiplication
		nntl_interface void mMulAB_C(const floatmtx_t& A, const floatmtx_t& B, floatmtx_t& C)noexcept;
		//matrix multiplication C(no bias) = A * B` (B transposed). C could have emulated biases (they will be left untouched)
		nntl_interface void mMulABt_Cnb(const floatmtx_t& A, const floatmtx_t& B, floatmtx_t& C)noexcept;
		//C = a*(A` * B) - matrix multiplication of transposed A times B with result normalization
		nntl_interface void mScaledMulAtB_C(float_t_ alpha, const floatmtx_t& A, const floatmtx_t& B, floatmtx_t& C)noexcept;


		//////////////////////////////////////////////////////////////////////////
		// sigm activation.
		// Remember to ignore biases for activation function calculations!
		nntl_interface void sigm(floatmtx_t& srcdest) noexcept;
		nntl_interface void dsigm(floatmtx_t& srcdest) noexcept;
		//calculates derivative of quadratic loss function for sigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		nntl_interface void dSigmQuadLoss_dZ(const floatmtx_t& activations, const floatmtx_t& data_y, floatmtx_t& dLdZ);

		//////////////////////////////////////////////////////////////////////////
		//ReLU
		nntl_interface void relu(floatmtx_t& srcdest) noexcept;
		nntl_interface void drelu(floatmtx_t& srcdest) noexcept;

		//////////////////////////////////////////////////////////////////////////
		//loss functions
		// quadratic loss == SUM_OVER_ALL((activations-data_y)^2)/(2*activations.rows())
		nntl_interface float_t_ loss_quadratic(const floatmtx_t& activations, const floatmtx_t& data_y)noexcept;
		
		// cross entropy function for sigmoid (applicable ONLY for binary data_y)
		// L = -y*log(a)-(1-y)log(1-a), dL/dz=a-y
		nntl_interface float_t_ loss_sigm_xentropy(const floatmtx_t& activations, const floatmtx_t& data_y)noexcept;


		//////////////////////////////////////////////////////////////////////////
		//gradient application procedures
		nntl_interface void RMSProp_Hinton(floatmtx_t& dW, floatmtx_t& rmsF, const float_t_ learningRate,
			const float_t_ emaDecay, const float_t_ numericStabilizer)noexcept;
		nntl_interface void RMSProp_Graves(floatmtx_t& dW, floatmtx_t& rmsF, floatmtx_t& rmsG, const float_t_ learningRate,
			const float_t_ emaDecay, const float_t_ numericStabilizer)noexcept;
		nntl_interface void RProp(floatmtx_t& dW, const float_t_ learningRate)noexcept;
		nntl_interface void ModProp(floatmtx_t& dW, floatmtx_t& rmsF, const float_t_ learningRate,
			const float_t_ emaDecay, const float_t_ numericStabilizer)noexcept;
	};

}
}
