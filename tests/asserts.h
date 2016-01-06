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

#include "../nntl/math.h"
#include "../nntl/common.h"

#include <numeric>

//using realmtx_t = nntl::math_types::realmtx_ty;
using real_t = nntl::math_types::real_ty;
typedef nntl::math::simple_matrix<real_t> realmtx_t;
typedef nntl::math::simple_matrix_deformable<real_t> realmtxdef_t;
typedef typename realmtx_t::vec_len_t vec_len_t;
typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

#define MTXSIZE_SCOPED_TRACE(_r,_c,_descr) constexpr unsigned _scopeMsgLen = 128; \
char _scopeMsg[_scopeMsgLen]; \
sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements)", (_descr), (_r), (_c), realmtx_t::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

#define MTXSIZE_SCOPED_TRACE1(_r,_c,_descr, fparam) constexpr unsigned _scopeMsgLen = 128; \
char _scopeMsg[_scopeMsgLen]; \
sprintf_s(_scopeMsg, "%s%f: data size is %dx%d (%lld elements)", (_descr), (fparam),(_r), (_c), realmtx_t::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

inline void _ASSERT_REALMTX_NEAR(const realmtx_t& c1, const realmtx_t& c2, const char* descr, const double eps) noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_NEAR(p1[i], p2[i], eps) << "Mismatches element #" << i << " @ " << descr;
	}
}
#define ASSERT_REALMTX_NEAR(c1,c2,descr,eps) ASSERT_NO_FATAL_FAILURE(_ASSERT_REALMTX_NEAR(c1,c2,descr,eps));

template<typename BaseT>
//void _ASSERT_MTX_EQ(const nntl::math::simple_matrix<BaseT>& c1, const nntl::math::simple_matrix<BaseT>& c2, const char* descr = "") noexcept {
void _ASSERT_MTX_EQ(const BaseT& c1, const BaseT& c2, const char* descr = "") noexcept {
	static_assert(false, "WTF");
}

template<typename BaseT>
std::enable_if_t<std::is_integral<BaseT>::value>
_ASSERT_MTX_EQ(const nntl::math::simple_matrix<BaseT>& c1, const nntl::math::simple_matrix<BaseT>& c2, const char* descr = "") noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}

template<>
inline void _ASSERT_MTX_EQ(const nntl::math::simple_matrix<float>& c1, const nntl::math::simple_matrix<float>& c2, const char* descr) noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_FLOAT_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}
template<>
inline void _ASSERT_MTX_EQ(const nntl::math::simple_matrix<double>& c1, const nntl::math::simple_matrix<double>& c2, const char* descr) noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_DOUBLE_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}


#define ASSERT_MTX_EQ(c1,c2,descr) ASSERT_NO_FATAL_FAILURE(_ASSERT_MTX_EQ(c1,c2,descr));

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename BaseT>
void _ASSERT_VECTOR_NEAR(const std::vector<BaseT>& v1, const std::vector<BaseT>& v2, const char* descr, const double eps) noexcept {
	ASSERT_EQ(v1.size(), v2.size()) << descr;
	const auto im = v1.size();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_NEAR(v1[i], v2[i], eps) << "Mismatches element #" << i << " @ " << descr;
	}
}
#define ASSERT_VECTOR_NEAR(c1,c2,descr,eps) ASSERT_NO_FATAL_FAILURE(_ASSERT_VECTOR_NEAR(c1,c2,descr,eps));

template<typename BaseT>
void _ASSERT_VECTOR_EQ(const std::vector<BaseT>& v1, const std::vector<BaseT>& v2, const char* descr) noexcept {
	static_assert(false, "WTF");
}
template<>
inline void _ASSERT_VECTOR_EQ(const std::vector<double>& v1, const std::vector<double>& v2, const char* descr) noexcept {
	ASSERT_EQ(v1.size(), v2.size()) << descr;
	if (!descr) descr = "";
	const auto im = v1.size();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_DOUBLE_EQ(v1[i], v2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}
template<>
inline void _ASSERT_VECTOR_EQ(const std::vector<float>& v1, const std::vector<float>& v2, const char* descr) noexcept {
	ASSERT_EQ(v1.size(), v2.size()) << descr;
	if (!descr) descr = "";
	const auto im = v1.size();
	for (numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_FLOAT_EQ(v1[i], v2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}
#define ASSERT_VECTOR_EQ(c1,c2,descr) ASSERT_NO_FATAL_FAILURE(_ASSERT_VECTOR_EQ(c1,c2,descr));