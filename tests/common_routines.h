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

#include "../nntl/nntl.h"
#include "../nntl/_supp/io/binfile.h"

#define MNIST_FILE_DEBUG "../../nntl/data/mnist200_100.bin"
#define MNIST_FILE_RELEASE  "../../nntl/data/mnist60000.bin"

#if defined(TESTS_SKIP_NNET_LONGRUNNING)
//ALWAYS run debug build with similar relations of data sizes:
// if release will run in minibatches - make sure, there will be at least 2 minibatches)
// if release will use different data sizes for train/test - make sure, debug will also run on different datasizes
#define MNIST_FILE MNIST_FILE_DEBUG
#else
#define MNIST_FILE MNIST_FILE_RELEASE
#endif // _DEBUG

//#todo better use ::std::iota()
template<typename real_t>
void seqFillMtx(::nntl::math::smatrix<real_t>& m) {
	NNTL_ASSERT(!m.empty() && m.numel_no_bias());
	const auto p = m.data();
	const auto ne = m.numel_no_bias();
	for (numel_cnt_t i = 0; i < ne; ++i) {
		p[i] = static_cast<real_t>(i + 1);
	}
}

template<typename real_t>
void readTd(::nntl::inmem_train_data_stor<real_t>& td, const char* pFile = MNIST_FILE) {
	typedef nntl_supp::binfile reader_t;

	SCOPED_TRACE("readTd");
	reader_t reader;

	STDCOUTL("Reading datafile '" << pFile << "'...");
	reader_t::ErrorCode rec = reader.read(NNTL_STRING(pFile), td);
	ASSERT_EQ(reader_t::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());
}

template<typename _I>
struct modify_layer_set_RMSProp_and_NM {
	template<typename _L>
	::std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l)noexcept {
		l.get_gradWorks().set_type(::std::decay_t<decltype(l.get_gradWorks())>::RMSProp_Hinton).nesterov_momentum(_I::nesterovMomentum);
		//	.L2(l2).max_norm2(0).set_ILR(.91, 1.05, .0001, 10000);
	}

	template<typename _L>
	::std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L&)noexcept {}
};
