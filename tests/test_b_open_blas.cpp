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
#include "stdafx.h"

#include "../nntl/interface/math/bindings/b_open_blas.h"

#include "../nntl/interface/math.h"
#include "../nntl/common.h"
#include "../nntl/_supp/jsonreader.h"

using namespace nntl;

/*
TEST(TestBOpenBLAS, Gemm) {
	//read source matrices from files, multiply them and check with hardcoded result

	using namespace nntl_supp;
	using ErrCode = jsonreader::ErrorCode;
	using mtx_t = train_data::mtx_t;
	using mtx_size_t = train_data::mtx_t::mtx_size_t;
	using real_t = train_data::mtx_t::value_type;

	train_data td1,td2,td3;
	jsonreader reader;

	ErrCode ec = reader.read(NNTL_STRING("./test_data/mtxmul_m1.json"), td1,false);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();

	ec = reader.read(NNTL_STRING("./test_data/mtxmul_m2.json"), td2, false);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();

	ec = reader.read(NNTL_STRING("./test_data/mtxmul_m3.json"), td3, false);
	ASSERT_EQ(ErrCode::Success, ec) << "Error code description: " << reader.get_last_error_string();

	const mtx_t& A=td1.train_x(), &B=td2.train_x(), &hcC = td3.train_x();
	mtx_t C(true, 3, 2);
	C.zeros();
	const real_t alpha = 1, beta = 0;

	ASSERT_EQ(C.rows(), A.rows());
	ASSERT_EQ(C.cols(), B.cols());
	ASSERT_EQ(A.cols(), B.rows());
	ASSERT_EQ(C.size(), hcC.size());

	math::b_OpenBLAS::gemm(false, false,
		A.rows(), B.cols(), A.cols(),
		alpha, A.dataAsVec(true), A.rows(),
		B.dataAsVec(true), B.rows(),
		beta, C.dataAsVec(true), C.rows());
	ASSERT_TRUE(C == hcC);
}
*/

/*C.zeros();
math::iACML::gemm( CblasNoTrans, CblasNoTrans,
A.rows(), B.cols(), A.cols(),
alpha, A.dataAsVec(true), A.rows(),
B.dataAsVec(true), B.rows(),
beta, C.dataAsVec(true), C.rows());
ASSERT_TRUE(C == hcC);*/