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
		typedef smatrix<real_t> realmtx_t;
		typedef smatrix_deform<real_t> realmtxdef_t;
		//typedef typename realmtx_t::vec_len_t vec_len_t;
		//typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

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
		nntl_interface void mExtractRows(const realmtx_t& src, const SeqIt& ridxsItBegin, realmtx_t& dest)noexcept;

		//binarize elements of real-valued matrix according to their relaion to frac
		nntl_interface void ewBinarize_ip(realmtx_t& A, const real_t& frac, const real_t& lBnd = real_t(0.), const real_t& uBnd = real_t(1.))noexcept;

		//binarize elements of real-valued matrix according to their relaion to frac into other matrix
		template<typename DestContainerT>
		nntl_interface void ewBinarize(DestContainerT& Dest, const realmtx_t& A, const real_t frac)noexcept;
		
		// clone matrix columns to another more wide matrix
		// srcCols - matrix with source columns to be copied into dest
		// dest - destination matrix, must have the same rows count as srcCols
		// colSpec - array of vec_len_t of size colSpecCnt==srcCols.cols(). Each element specifies how many
		//		copies of a corresponding srcCols column must be made. For example, if colSpec is {2,3,4}, then
		//		the srcCols must have 3 columns. The first column is copied 2 times into first 2 columns of dest,
		//		the second - 3, the third - 4. Therefore, dest must contain 2+3+4=9 columns.		
		//nntl_interface void mCloneCols(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec)noexcept;
		//#todo we need this definition in interface, however when it is uncommented there is an ambiguity of the symbol arises,
		//because mCloneCol() is already defined in _simpleMath class. Probably, I should refactor the interface definition.
		

		// clone a matrix column to another more wide matrix dest.cols() number of times
		// (optimized version of mCloneCols where srcCols.cols()==1 )
		//nntl_interface void mCloneCol(const realmtx_t& srcCol, realmtx_t& dest)noexcept;
		//#todo we need this definition in interface, however when it is uncommented there is an ambiguity of the symbol arises,
		//because mCloneCol() is already defined in _simpleMath class. Probably, I should refactor the interface definition.


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
		//nntl_interface void mTilingUnroll(const realmtx_t& src, realmtx_t& dest)noexcept;
		//#todo we need this definition in interface, however when it is uncommented there is an ambiguity of the symbol arises,

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
		//nntl_interface void mTilingRoll(const realmtx_t& src, realmtx_t& dest)noexcept;
		//#todo we need this definition in interface, however when it is uncommented there is an ambiguity of the symbol arises,

		// treat matrix as a set of row-vectors (matrices in col-major mode!). For each row-vector check, whether
		// its length/norm is not longer, than predefined value. If it's longer, than rescale vector to this max length
		// (for use in max-norm weights regularization)
		nntl_interface void mCheck_normalize_rows(realmtxdef_t& W, const real_t& maxL2NormSquared, const bool bNormIncludesBias)noexcept;

		nntl_interface void mrwL2NormSquared(const realmtx_t& A, real_t*const pNormsVec)noexcept;
		//nntl_interface void apply_max_norm(realmtxdef_t& W, const real_t& maxL2NormSquared, const bool bNormIncludesBias)noexcept;

		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		template<typename Contnr>
		nntl_interface size_t vCountSame(const Contnr& A, const Contnr& B)noexcept;

		//clamps matrix values into range
		nntl_interface void evClamp(realmtx_t& m, real_t lo, real_t hi)noexcept;

		//on entry the dropoutMask must be filled with random values in range [0,1]
		//Function binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// act must be used in "no_bias" mode.
		// dropPercAct - probability of keeping unit active
		// Actually, the function implements so called "Inverse Dropout", see http://cs231n.github.io/neural-networks-2/
		nntl_interface void make_dropout(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask)noexcept;
		//same as make_dropout(), but direct dropout used and dropoutMask must not be changed
		nntl_interface void apply_dropout_mask(realmtx_t& act, const real_t dropPercAct, const realmtx_t& dropoutMask)noexcept;

		//apply individual learning rate to dLdW
		nntl_interface void apply_ILR(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept;

		//apply momentum vW = momentum.*vW + dW
		nntl_interface void apply_momentum(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept;

		//////////////////////////////////////////////////////////////////////////
 		//inplace elementwise multiplication A = b.*A
		//nntl_interface void evMulC_ip(realmtx_t& A, const real_t b)noexcept;

		//inplace elementwise multiplication A(no_bias) = b.*A(no_bias)
		//nntl_interface void evMulC_ip_nb(realmtx_t& A, const real_t b)noexcept;
		
		//inplace elementwise multiplication A = A.*B
		nntl_interface void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept;

		//inplace elementwise multiplication A(no_bias) = A(no_bias).*B, - A is taken in no_bias mode
		nntl_interface void evMul_ip_Anb(realmtx_t& A, const realmtx_t& B)noexcept;

		//inplace elementwise addition A = A+B
		nntl_interface void evAdd_ip(realmtx_t& A, const realmtx_t& B)noexcept;
		//inplace elementwise addition *pA = *pA + *pB
		nntl_interface void vAdd_ip(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt)noexcept;

		//inplace elementwise adding of scaled vector: A = A + c*B;
		nntl_interface void evAddScaled_ip(realmtx_t& A, const real_t& c, const realmtx_t& B)noexcept;

		//inplace elementwise addition of scaled signum: A = A + c*sign(B);
		//(L1 regularization, dLdW update step)
		nntl_interface void evAddScaledSign_ip(realmtx_t& A, const real_t& c, const realmtx_t& B)noexcept;

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
		//nntl_interface real_t vSumSquares(const realmtx_t& A)noexcept;
		nntl_interface real_t ewSumSquares(const realmtx_t& A)noexcept;
		nntl_interface real_t ewSumSquares_ns(const realmtx_t& A)noexcept;

		//finding elementwise absolute values dest = abs(src);
		nntl_interface void evAbs(realmtx_t& dest, const realmtx_t& src)noexcept;

		//finds a sum of abs values (L1 norm): return sum( abs(A) );
		nntl_interface real_t vSumAbs(const realmtx_t& A)noexcept;
		
		//finds a sum of elementwise products return sum( A.*B );
		nntl_interface real_t ewSumProd(const realmtx_t& A, const realmtx_t& B)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//C = A * B, - matrix multiplication
		//////////////////////////////////////////////////////////////////////////
		nntl_interface void mMulAB_C(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;
		//matrix multiplication C(no bias) = A * B` (B transposed). C could have emulated biases (they will be left untouched)
		nntl_interface void mMulABt_Cnb(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;
		//C = a*(A` * B) - matrix multiplication of transposed A times B with result normalization
		nntl_interface void mScaledMulAtB_C(real_t alpha, const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept;

		//////////////////////////////////////////////////////////////////////////
		//Elements of SVD (singular value decomposition)
		//////////////////////////////////////////////////////////////////////////
		// mSVD_Orthogonalize_ss(A) performs SVD of m*n matrix A and returns in A same sized corresponding orthogonal matrix of singular vectors
		//		returns true if SVD was successful
		//		Restrictions: MUST NOT use the math object's local storage (the function is intended to be used during
		//			the weight initialization phase when the local storage is generally not initialized yet)
		nntl_interface bool mSVD_Orthogonalize_ss(realmtx_t& A)noexcept;
		// This function checks whether A is orthogonal, i.e. A'*A is identity matrix.
		// Could be not optimized, use FOR DEBUG PURPOSES ONLY!
		nntl_interface bool _mIsOrthogonal(const realmtx_t& A, bool bFirstTransposed, const real_t epsV)noexcept;

		//////////////////////////////////////////////////////////////////////////
		// divide each matrix A row by corresponding vector d element, A(i,:) = A(i,:) / d(i)
		//nntl_interface void mrwDivideByVec(realmtx_t& A, const real_t* pDiv)noexcept;

		//nntl_interface void dIdentity(realmtx_t& f_df)noexcept;
		//nntl_interface void dIdentityQuadLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ)noexcept;
		nntl_interface void dIdentityXEntropyLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ)noexcept;

		//////////////////////////////////////////////////////////////////////////
		// sigm activation.
		// Remember to ignore biases for activation function calculations!
		nntl_interface void sigm(realmtx_t& srcdest) noexcept;
		nntl_interface void dsigm(realmtx_t& f_df) noexcept;
		//calculates derivative of quadratic loss function for sigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		nntl_interface void dSigmQuadLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ);

		//////////////////////////////////////////////////////////////////////////
		//ReLU
		// MUST ignore biases!
		nntl_interface void relu(realmtx_t& srcdest) noexcept;
		nntl_interface void drelu(realmtx_t& f_df) noexcept;

		nntl_interface void leakyrelu(realmtx_t& srcdest, const real_t leak) noexcept;
		nntl_interface void dleakyrelu(realmtx_t& f_df, const real_t leak) noexcept;

		nntl_interface void elu(realmtx_t& srcdest, const real_t alpha) noexcept;
		nntl_interface void delu(realmtx_t& f_df, const real_t alpha) noexcept;
		nntl_interface void elu_unitalpha(realmtx_t& srcdest) noexcept;
		nntl_interface void delu_unitalpha(realmtx_t& f_df) noexcept;

		nntl_interface void elogu(realmtx_t& srcdest, const real_t& alpha, const real_t& b) noexcept;
		nntl_interface void delogu(realmtx_t& f_df, const real_t& alpha, const real_t& b) noexcept;
		nntl_interface void elogu_ua(realmtx_t& srcdest, const real_t& b) noexcept;
		nntl_interface void delogu_ua(realmtx_t& f_df, const real_t& b) noexcept;
		nntl_interface void elogu_nb(realmtx_t& srcdest, const real_t& alpha) noexcept;
		nntl_interface void delogu_nb(realmtx_t& f_df, const real_t& alpha) noexcept;
		nntl_interface void elogu_ua_nb(realmtx_t& srcdest) noexcept;
		nntl_interface void delogu_ua_nb(realmtx_t& f_df) noexcept;

		nntl_interface void softsign(realmtx_t& srcdest, const real_t a, const real_t c) noexcept;
		nntl_interface void softsign_uc(realmtx_t& srcdest, const real_t a) noexcept;
		nntl_interface void dsoftsign(realmtx_t& f_df, const real_t a) noexcept;
		nntl_interface void dsoftsign_ua_uc(realmtx_t& f_df) noexcept;
		nntl_interface void softsigm(realmtx_t& srcdest, const real_t a) noexcept;
		nntl_interface void dsoftsigm(realmtx_t& f_df, const real_t a) noexcept;

		nntl_interface void dSoftSigmQuadLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a)noexcept;
		nntl_interface void dSoftSigmXEntropyLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a)noexcept;

		nntl_interface void step(realmtx_t& srcdest) noexcept;

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
		nntl_interface real_t loss_quadratic_ns(const realmtx_t& activations, const realmtx_t& data_y)noexcept;
		
		// cross entropy function for sigmoid (applicable ONLY for binary data_y)
		// L = sum( -y*log(a)-(1-y)log(1-a) )/activations.rows(), dL/dz=a-y
		nntl_interface real_t loss_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept;
		nntl_interface real_t loss_xentropy_ns(const realmtx_t& activations, const realmtx_t& data_y)noexcept;

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

		nntl_interface void Adam(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept;
		nntl_interface void AdaMax(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept;

		nntl_interface void RNadam(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Nt, real_t& mu_pow_t, real_t& eta_pow_t
			, const real_t& learningRate, const real_t& mu, const real_t& eta, const real_t& gamma, const real_t& numericStabilizer)noexcept;
	};

}
}
