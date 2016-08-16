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

#include "../nntl/nntl.h"

#define MNIST_FILE_DEBUG "../data/mnist200_100.bin"
#define MNIST_FILE_RELEASE  "../data/mnist60000.bin"

#if defined(TESTS_SKIP_NNET_LONGRUNNING)
//ALWAYS run debug build with similar relations of data sizes:
// if release will run in minibatches - make sure, there will be at least 2 minibatches)
// if release will use different data sizes for train/test - make sure, debug will also run on different datasizes
#define MNIST_FILE MNIST_FILE_DEBUG
#else
#define MNIST_FILE MNIST_FILE_RELEASE
#endif // _DEBUG

void seqFillMtx(realmtx_t& m);

void readTd(nntl::train_data<real_t>& td, const char* pFile = MNIST_FILE);

void _allowMask(const realmtx_t& srcMask, realmtx_t& mask, const realmtx_t& data_y, const vec_len_t c);

void _maskStat(const realmtx_t& m);


template<typename iRngT, typename iMathT>
void makeDataXForGatedSetup(const realmtx_t& data_x, const realmtx_t& data_y, iRngT& iR, iMathT& iM, const bool bBinarize, realmtx_t& new_x) {
	SCOPED_TRACE("makeDataXForGatedSetup");

	realmtx_t realMask(data_x.rows(), 1, false);
	ASSERT_TRUE(!realMask.isAllocationFailed());
	iR.gen_matrix_norm(realMask);
	if (bBinarize) iM.ewBinarize_ip(realMask, real_t(.5));

	_maskStat(realMask);

	const auto gateIdx = data_x.cols_no_bias() / 2;

	new_x.will_emulate_biases();
	ASSERT_TRUE(new_x.resize(data_x.rows(), data_x.cols_no_bias() + 1)) << "Failed to resize new_x";

	memcpy(new_x.data(), data_x.data(), data_x.byte_size_no_bias() / 2);
	memcpy(new_x.colDataAsVec(gateIdx), realMask.data(), realMask.byte_size());
	memcpy(new_x.colDataAsVec(gateIdx + 1), data_x.colDataAsVec(gateIdx), data_x.byte_size_no_bias() / 2);
}

template<typename iRngT, typename iMathT>
void makeDataXForGatedSetup(const realmtx_t& data_x, const realmtx_t& data_y, iRngT& iR, iMathT& iM,
	const bool bBinarize, const vec_len_t gatesCnt, realmtx_t& new_x)
{
	SCOPED_TRACE("makeDataXForGatedSetup");
	const auto xWidth = data_x.cols_no_bias();

	realmtx_t realMask(data_x.rows(), gatesCnt, false);
	ASSERT_TRUE(!realMask.isAllocationFailed());
	iR.gen_matrix_norm(realMask);
	if (bBinarize) iM.ewBinarize_ip(realMask, real_t(.5));

	_maskStat(realMask);

	const auto gateIdx = xWidth / 4;

	new_x.will_emulate_biases();
	ASSERT_TRUE(new_x.resize(data_x.rows(), data_x.cols_no_bias() + gatesCnt)) << "Failed to resize new_x";
	
	memcpy(new_x.data(), data_x.data(), sizeof(real_t)*(data_x.colDataAsVec(gateIdx)-data_x.data()));
	memcpy(new_x.colDataAsVec(gateIdx), realMask.data(), realMask.byte_size());
	memcpy(new_x.colDataAsVec(gateIdx + gatesCnt), data_x.colDataAsVec(gateIdx),
		sizeof(real_t)*(data_x.colDataAsVec(xWidth) - data_x.colDataAsVec(gateIdx)));
}

void makeTdForGatedSetup(const nntl::train_data<real_t>& td, nntl::train_data<real_t>& tdGated, const uint64_t seedV
	, const bool bBinarize, const vec_len_t gatesCnt=1);


template<typename _I>
struct modify_layer_set_RMSProp_and_NM {
	template<typename _L>
	std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l)noexcept {
		l.m_gradientWorks.set_type(decltype(l.m_gradientWorks)::RMSProp_Hinton).set_nesterov_momentum(_I::nesterovMomentum);
		//	.set_L2(l2).set_max_norm(0).set_ILR(.91, 1.05, .0001, 10000);
	}

	template<typename _L>
	std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L& l)noexcept {}
};
