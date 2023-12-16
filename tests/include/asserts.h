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

#include <nntl/math_details.h>
#include <nntl/common.h>
#include <nntl/interfaces.h>

#include <numeric>

#define ASSERT_SUPPORTED_REAL_T(T) static_assert(::std::is_same<T,float>::value || ::std::is_same<T,double>::value, \
"Only float or double supported for type=" #T);

#define MTXSIZE_SCOPED_TRACE(_r,_c,_descr) constexpr unsigned _scopeMsgLen = 128; \
char _scopeMsg[_scopeMsgLen]; \
sprintf_s(_scopeMsg, "%s: data size is %dx%d (%lld elements)", (_descr), (_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

#define MTXSIZE_SCOPED_TRACE1(_r,_c,_descr, fparam) constexpr unsigned _scopeMsgLen = 128; \
char _scopeMsg[_scopeMsgLen]; \
sprintf_s(_scopeMsg, "%s%f: data size is %dx%d (%lld elements)", (_descr), (fparam),(_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

#define MTXSIZE_SCOPED_TRACE1d2f(_r,_c,_descr, decParam, fparam) char _scopeMsg[256]; \
sprintf_s(_scopeMsg, "%s=(%d,%f): data size is %dx%d (%lld elements)", (_descr), (fparam),(_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

//////////////////////////////////////////////////////////////////////////

#define MTXSIZE_SCOPED_TRACE_TYPED(_r,_c,_descr) constexpr unsigned _scopeMsgLen = 256; \
char _scopeMsg[_scopeMsgLen]; \
ASSERT_SUPPORTED_REAL_T(real_t) \
sprintf_s(_scopeMsg, "%s: real_t=%s, data size is %dx%d (%lld elements)", (_descr) \
, ::std::is_same<real_t,float>::value ? "float" : "double" , (_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

#define MTXSIZE_SCOPED_TRACE_TYPED_1d(_r,_c,_descr, _d) constexpr unsigned _scopeMsgLen = 256; \
char _scopeMsg[_scopeMsgLen]; \
ASSERT_SUPPORTED_REAL_T(real_t) \
sprintf_s(_scopeMsg, "%s%d: real_t=%s, data size is %dx%d (%lld elements)", (_descr), (_d) \
, ::std::is_same<real_t,float>::value ? "float" : "double" , (_r), (_c), ::nntl::math::smatrix_td::sNumel((_r), (_c))); \
SCOPED_TRACE(_scopeMsg);

//////////////////////////////////////////////////////////////////////////

template<typename _T>
inline void _ASSERT_REALMTX_NEAR(const nntl::math::smatrix<_T>& c1, const nntl::math::smatrix<_T>& c2, const char* descr
	, const double eps, const ::nntl::thread_id_t ti = -1) noexcept
{
	if (ti >= 0) {
		ASSERT_EQ(c1.bBatchInRow(), c2.bBatchInRow()) << "(threads cnt " << ti << ")" << descr;
		ASSERT_EQ(c1.size(), c2.size()) << "(threads cnt " << ti << ")" << descr;
		ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << "(threads cnt " << ti << ")" << descr;
	} else {
		ASSERT_EQ(c1.bBatchInRow(), c2.bBatchInRow()) << descr;
		ASSERT_EQ(c1.size(), c2.size()) << descr;
		ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	}
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	if (ti >= 0) {
		for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
			ASSERT_NEAR(p1[i], p2[i], eps) << "Mismatches element #" << i << "(" << (i%c1.rows()) << "," << (i / c1.rows())
				<< ") of [" << c1.rows() << "," << c1.cols() << "] @ " << "(threads cnt " << ti << ")" << descr;
		}
	} else {
		for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
			ASSERT_NEAR(p1[i], p2[i], eps) << "Mismatches element #" << i << "(" << (i%c1.rows()) << "," << (i / c1.rows())
				<< ") of [" << c1.rows() << "," << c1.cols() << "] @ " << descr;
		}
	}
}
#define ASSERT_REALMTX_NEAR(c1,c2,descr,eps) ASSERT_NO_FATAL_FAILURE(_ASSERT_REALMTX_NEAR(c1,c2,descr,eps));
#define ASSERT_REALMTX_NEAR_THRD(c1, c2, descr, eps, nTrd) ASSERT_NO_FATAL_FAILURE(_ASSERT_REALMTX_NEAR(c1, c2, descr, eps, nTrd));

template<typename BaseT>
//void _ASSERT_MTX_EQ(const nntl::math::smatrix<BaseT>& c1, const nntl::math::smatrix<BaseT>& c2, const char* descr = "") noexcept {
void _ASSERT_MTX_EQ(const BaseT& c1, const BaseT& c2, const char* descr = "") noexcept {
	static_assert(false, "WTF");
}

template<typename BaseT>
::std::enable_if_t<::std::is_integral<BaseT>::value>
	_ASSERT_MTX_EQ(const nntl::math::smatrix<BaseT>& c1, const nntl::math::smatrix<BaseT>& c2, const char* descr = "") noexcept
{
	ASSERT_EQ(c1.bBatchInRow(), c2.bBatchInRow()) << descr;
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
	ASSERT_EQ(c1.bBatchInRow(), c2.bBatchInRow()) << descr;
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
	ASSERT_EQ(c1.bBatchInRow(), c2.bBatchInRow()) << descr;
	ASSERT_EQ(c1.size(), c2.size()) << descr;
	ASSERT_EQ(c1.emulatesBiases(), c2.emulatesBiases()) << descr;
	const auto p1 = c1.data(), p2 = c2.data();
	const auto im = c1.numel();
	for (::nntl::numel_cnt_t i = 0; i < im; ++i) {
		ASSERT_DOUBLE_EQ(p1[i], p2[i]) << "Mismatches element #" << i << " @ " << descr;
	}
}


#define ASSERT_MTX_EQ(c1,c2,descr) ASSERT_NO_FATAL_FAILURE(_ASSERT_MTX_EQ(c1,c2,descr));
#define ASSERT_MTX_EQt(T, c1,c2,descr) ASSERT_NO_FATAL_FAILURE(_ASSERT_MTX_EQ<T>(c1,c2,descr));

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

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename T, bool ce = ::std::is_floating_point<T>::value>
inline ::std::enable_if_t<ce> dbg_show_matrix(const ::nntl::math::smatrix<T>& m, const char* pName=nullptr, int wid=6, int prec=2) {
	const auto mr = m.rows(), mc = m.cols();
	STDCOUT((pName ? pName : "matrix") << " = ");
	for (vec_len_t r = 0; r < mr; ++r) {
		::std::cout << ::std::endl << "[" << ::std::setw(2) << r << "]";
		for (vec_len_t c = 0; c < mc; ++c) {
			STDCOUT(std::setprecision(prec) << ::std::setw(wid) << m.get(r, c));
		}
	}
	::std::cout << ::std::endl;
}

template<typename T, bool ce = ::std::is_floating_point<T>::value>
inline ::std::enable_if_t<!ce> dbg_show_matrix(const ::nntl::math::smatrix<T>& m, const char* pName = nullptr, int wid = 6) {
	const auto mr = m.rows(), mc = m.cols();
	STDCOUT((pName ? pName : "matrix") << " = ");
	for (vec_len_t r = 0; r < mr; ++r) {
		::std::cout << ::std::endl << "[" << ::std::setw(2) << r << "]";
		for (vec_len_t c = 0; c < mc; ++c) {
			STDCOUT(::std::setw(wid) << m.get(r, c));
		}
	}
	::std::cout << ::std::endl;
}