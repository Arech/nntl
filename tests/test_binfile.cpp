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

#include "../nntl/interface/math.h"
#include "../nntl/common.h"

#include "../nntl/_supp/binfile.h"

#include <memory>
#include <array>

using namespace nntl;

TEST(TestBinFile, ReadMatrix) {
	typedef math_types::floatmtx_ty floatmtx_t;
	typedef floatmtx_t::vec_len_t vec_len_t;
	typedef nntl_supp::binfile binfile;
	typedef binfile::ErrorCode ErrorCode;

	const strchar_t* mtx_fname = "./test_data/mtx.bin";
	const double mtx_content[4]{1,4,3,2};
	
	floatmtx_t m;
	binfile r;

	auto ec = r.read(NNTL_STRING(mtx_fname), m);
	ASSERT_EQ(ErrorCode::Success, ec) << r.get_last_error_str();
	ASSERT_EQ(floatmtx_t::mtx_size_t(2, 2), m.size());
	
	const auto p = m.dataAsVec();
	const auto im = m.numel();
	for (floatmtx_t::numel_cnt_t i = 0; i < im; ++i) ASSERT_EQ(mtx_content[i], p[i]);

}

TEST(TestBinFile, ReadTrainData) {
	typedef train_data::mtx_t floatmtx_t;
	typedef floatmtx_t::vec_len_t vec_len_t;
	using mtx_size_t = floatmtx_t::mtx_size_t;
	using float_t_ = floatmtx_t::value_type;

	typedef nntl_supp::binfile binfile;
	typedef binfile::ErrorCode ErrorCode;

	const strchar_t* mtx_fname = "./test_data/td.bin";
	const double mtx_content[4]{ 1,4,3,2 };

	train_data td;
	binfile r;

	auto ec = r.read(NNTL_STRING(mtx_fname), td);
	ASSERT_EQ(ErrorCode::Success, ec) << r.get_last_error_str();

	//biased (default) version:
	const std::array<std::array<float_t_, 3>, 5> train_x_data{ 77,80,19,49,45,65,71,75,28,68,66,16 ,1,1,1 };
	const std::array<std::array<float_t_, 3>, 1> train_y_data{ 12,50,96 };
	const std::array<std::array<float_t_, 2>, 5> test_x_data{ 34,59,22,75,26,51,70,89 ,1,1 };
	const std::array<std::array<float_t_, 2>, 1> test_y_data{ 96,55 };

	mtx_size_t train_x_size(static_cast<vec_len_t>(train_x_data[0].size()), static_cast<vec_len_t>(train_x_data.size())),
		train_y_size(static_cast<vec_len_t>(train_y_data[0].size()), static_cast<vec_len_t>(train_y_data.size())),
		test_x_size(static_cast<vec_len_t>(test_x_data[0].size()), static_cast<vec_len_t>(test_x_data.size())),
		test_y_size(static_cast<vec_len_t>(test_y_data[0].size()), static_cast<vec_len_t>(test_y_data.size()));

	ASSERT_EQ(train_x_size, td.train_x().size()) << "td.train_x size differs from expected";
	ASSERT_EQ(train_y_size, td.train_y().size()) << "td.train_y size differs from expected";
	ASSERT_EQ(test_x_size, td.test_x().size()) << "td.test_x size differs from expected";
	ASSERT_EQ(test_y_size, td.test_y().size()) << "td.test_y size differs from expected";

	for (int i = 0; i < train_x_data.size(); i++) {
		for (int j = 0; j < train_x_data[0].size(); j++)
			EXPECT_EQ(train_x_data[i][j], td.train_x().get(j, i));
	}
	for (int i = 0; i < train_y_data.size(); i++) {
		for (int j = 0; j < train_y_data[0].size(); j++)
			EXPECT_EQ(train_y_data[i][j], td.train_y().get(j, i));
	}
	for (int i = 0; i < test_x_data.size(); i++) {
		for (int j = 0; j < test_x_data[0].size(); j++)
			EXPECT_EQ(test_x_data[i][j], td.test_x().get(j, i));
	}
	for (int i = 0; i < test_y_data.size(); i++) {
		for (int j = 0; j < test_y_data[0].size(); j++)
			EXPECT_EQ(test_y_data[i][j], td.test_y().get(j, i));
	}

}
