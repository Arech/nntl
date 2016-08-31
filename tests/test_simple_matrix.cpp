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
#include "stdafx.h"

#include "../nntl/math.h"
#include "../nntl/common.h"

//#include "../nntl/interface/rng/std.h"
//#include "../nntl/utils/chrono.h"

using namespace nntl;

TEST(TestSimpleMatrix, Basics) {
	math::smatrix<float> m(1, 2);
	math::smatrix<float> t(std::move(m));

	EXPECT_TRUE(m.empty());

	EXPECT_EQ(t.rows(), 1);
	EXPECT_EQ(t.cols(), 2);
	EXPECT_TRUE(!t.empty());

	math::smatrix<float> k = std::move(t);

	EXPECT_TRUE(t.empty());

	EXPECT_EQ(k.rows(), 1);
	EXPECT_EQ(k.cols(), 2);
	EXPECT_TRUE(!k.empty());
}

TEST(TestSimpleMatrix, ColMajorOrder) {
	typedef math::smatrix<int> mtx;

	mtx m(2, 3);
	ASSERT_FALSE(m.isAllocationFailed());

	m.set(0, 0, 0);
	m.set(1, 0, 1);
	m.set(0, 1, 2);
	m.set(1, 1, 3);
	m.set(0, 2, 4);
	m.set(1, 2, 5);

	mtx::cvalue_ptr_t ptr = m.data();
	for (int i = 0; i < 6;++i) {
		EXPECT_EQ(i, ptr[i]) << "Wrong element at offset " << i;
	}
}


/*
TEST(TestSimpleMatrix, ExtractRowsFromColMajor) {
	typedef math::smatrix<int> mtx;

	mtx src(5, 2),destCM(3,2),destRM(3,2);
	for (mtx::numel_cnt_t i = 0, im = src.numel(); i < im; ++i) src.data()[i] = static_cast<int>(i);

	std::vector<mtx::vec_len_t> v(3);
	v[0]=1; v[1]=2; v[2]=4;
	
	src.extractRows(v.begin(), 3, destCM);
	//src.extractRows(v.begin(), 3, destRM);

	EXPECT_EQ(destCM.get(0, 0), 1);
	EXPECT_EQ(destCM.get(0, 1), 6);
	EXPECT_EQ(destCM.get(1, 0), 2);
	EXPECT_EQ(destCM.get(1, 1), 7);
	EXPECT_EQ(destCM.get(2, 0), 4);
	EXPECT_EQ(destCM.get(2, 1), 9);

// 	EXPECT_EQ(destRM.get(0, 0), 1);
// 	EXPECT_EQ(destRM.get(0, 1), 6);
// 	EXPECT_EQ(destRM.get(1, 0), 2);
// 	EXPECT_EQ(destRM.get(1, 1), 7);
// 	EXPECT_EQ(destRM.get(2, 0), 4);
// 	EXPECT_EQ(destRM.get(2, 1), 9);
}*/

TEST(TestSimpleMatrix, ColMajorWithBiases) {
	typedef math::smatrix<int> mtx;

	mtx m(2, 3,true);
	ASSERT_FALSE(m.isAllocationFailed());
	ASSERT_EQ(2, m.rows());
	ASSERT_EQ(4, m.cols());
	
	m.set(0, 0, 0);
	m.set(1, 0, 1);
	m.set(0, 1, 2);
	m.set(1, 1, 3);
	m.set(0, 2, 4);
	m.set(1, 2, 5);

	mtx::cvalue_ptr_t ptr = m.data();
	for (int i = 0; i < 6; ++i) {
		EXPECT_EQ(i, ptr[i]) << "Wrong element at offset " << i;
	}
	for (int i = 6; i < 8; ++i) {
		EXPECT_EQ(1, ptr[i]) << "Wrong element at offset " << i;
	}
}


