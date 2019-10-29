/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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
#include "../nntl/interfaces.h"

#include <numeric>

//#BUGBUG there must be no global typedefs!!!!
//typedef nntl::d_interfaces::real_t real_t;
//typedef nntl::math::smatrix<real_t> realmtx_t;
//typedef nntl::math::smatrix_deform<real_t> realmtxdef_t;
//typedef typename ::nntl::math::smatrix_td::vec_len_t vec_len_t;
//typedef typename ::nntl::math::smatrix_td::numel_cnt_t numel_cnt_t;
//typedef ::nntl::vec_len_t vec_len_t;
//typedef ::nntl::numel_cnt_t numel_cnt_t;

#define MTXSIZE_SCOPED_TRACE(_r,_c,_descr) constexpr unsigned _scopeMsgLen = 128; \
char _scopeMsg[_scopeMsgLen]; \
sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements)", (_descr), (_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

#define MTXSIZE_SCOPED_TRACE1(_r,_c,_descr, fparam) constexpr unsigned _scopeMsgLen = 128; \
char _scopeMsg[_scopeMsgLen]; \
sprintf_s(_scopeMsg, "%s%f: data size is %dx%d (%lld elements)", (_descr), (fparam),(_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

template<typename _T>
inline void _ASSERT_REALMTX_NEAR(const nntl::math::smatrix<_T>& c1, const nntl::math::smatrix<_T>& c2, const char* descr, const double eps) noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_NEAR(p1[i], p2[i], eps) << "Mismatches element #" << i << "(" << (i%c1.rows()) << "," << (i / c1.rows())
			<< ") of [" << c1.rows() << "," << c1.cols() << "] @ " << descr;
	}
}
#define ASSERT_REALMTX_NEAR(c1,c2,descr,eps) ASSERT_NO_FATAL_FAILURE(_ASSERT_REALMTX_NEAR(c1,c2,descr,eps));

template<typename BaseT>
//void _ASSERT_MTX_EQ(const nntl::math::smatrix<BaseT>& c1, const nntl::math::smatrix<BaseT>& c2, const char* descr = "") noexcept {
void _ASSERT_MTX_EQ(const BaseT& c1, const BaseT& c2, const char* descr = "") noexcept {
	static_assert(false, "WTF");
}

template<typename BaseT>
::std::enable_if_t<::std::is_integral<BaseT>::value>
	_ASSERT_MTX_EQ(const nntl::math::smatrix<BaseT>& c1, const nntl::math::smatrix<BaseT>& c2, const char* descr = "") noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}

template<>
inline void _ASSERT_MTX_EQ(const nntl::math::smatrix<float>& c1, const nntl::math::smatrix<float>& c2, const char* descr) noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_FLOAT_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}
template<>
inline void _ASSERT_MTX_EQ(const nntl::math::smatrix<double>& c1, const nntl::math::smatrix<double>& c2, const char* descr) noexcept {
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_DOUBLE_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}


#define ASSERT_MTX_EQ(c1,c2,descr) ASSERT_NO_FATAL_FAILURE(_ASSERT_MTX_EQ(c1,c2,descr));

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename BaseT>
void _ASSERT_VECTOR_NEAR(const ::std::vector<BaseT>& v1, const ::std::vector<BaseT>& v2, const char* descr, const double eps) noexcept {
	ASSERT_EQ(v1.size(), v2.size()) << descr;
	const auto im = ::nntl::conform_sign(v1.size());
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_NEAR(v1[i], v2[i], eps) << "Mismatches element #" << i << " @ " << descr;
	}
}
#define ASSERT_VECTOR_NEAR(c1,c2,descr,eps) ASSERT_NO_FATAL_FAILURE(_ASSERT_VECTOR_NEAR(c1,c2,descr,eps));

template<typename BaseT>
void _ASSERT_VECTOR_EQ(const ::std::vector<BaseT>& v1, const ::std::vector<BaseT>& v2, const char* descr) noexcept {
	static_assert(false, "WTF");
}
template<>
inline void _ASSERT_VECTOR_EQ(const ::std::vector<double>& v1, const ::std::vector<double>& v2, const char* descr) noexcept {
	ASSERT_EQ(v1.size(), v2.size()) << descr;
	if (!descr) descr = "";
	const auto im = ::nntl::conform_sign(v1.size());
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_DOUBLE_EQ(v1[i], v2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}
template<>
inline void _ASSERT_VECTOR_EQ(const ::std::vector<float>& v1, const ::std::vector<float>& v2, const char* descr) noexcept {
	ASSERT_EQ(v1.size(), v2.size()) << descr;
	if (!descr) descr = "";
	const auto im = ::nntl::conform_sign(v1.size());
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_FLOAT_EQ(v1[i], v2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}
#define ASSERT_VECTOR_EQ(c1,c2,descr) ASSERT_NO_FATAL_FAILURE(_ASSERT_VECTOR_EQ(c1,c2,descr));
