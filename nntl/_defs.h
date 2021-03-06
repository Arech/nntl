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

//TODO consider using #define _HAS_EXCEPTIONS 0 to drop exceptions from STL and then turn off c++ expections completely
//with a compiler switch.

//Microsoft compiler only definition
//#define nntl_deprecated(msg) __declspec(deprecated(msg))
//This def should work cross-platform
#define nntl_deprecated(msg) [[deprecated(msg)]]
//use of #pragma deprecated () is more standard, but makes all uses of symbol even in derived classes deprecated
// - clearly not what we need

//if it breaks compilation because of non-standard definition of nntl_deprecated, just make it blank
// (and remove subsequent pragma) and make sure there is no code for protected by this macro definitions
#define nntl_interface nntl_deprecated("is used for an interface definition only. Never call it directly!")

#define nntl_static_warning(msg) nntl_deprecated(msg)

//#pragma warning(error:996)
#pragma warning( default : 996 )

//__forceinline is MS-specific. Should branch on a compiler here
#define nntl_force_inline __forceinline

//this is for testing whether the forceinline actually helps or hurts
#define nntl_probably_force_inline __forceinline

//alignment specifier. C++11 has standard alignas construct, but here is the method to quickly change it to
// something else (like  __declspec(align(#)) ) if needed
#define nntl_align(n) alignas(n)

#ifndef nntl_pragma_macro
#define nntl_pragma_macro __pragma
#endif // !nntl_pragma_macro

//can be used to wrap strings
#define NNTL_STRING(s) s

#define NNTL_STRINGIZE(s) #s

//#define NNTL_UNREF(P)          (P)
#define NNTL_UNREF(P)     (P)

//////////////////////////////////////////////////////////////////////////
// debugging-specific tools
// set to __func__ to conform ANSI C
#define NNTL_FUNCTION __FUNCSIG__
//its helpfull to debug code flow with statements like ::std::cout << "** in " << NNTL_FUNCTION << ::std::endl;

#if defined(_DEBUG) || defined(DEBUG)

#define NNTL_DEBUG
#define NNTL_ASSERT _ASSERTE

#define NNTL_DEBUG_DECLARE(v) v
#define NNTL_DEBUG_ARG(p) ,p

#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
#define NNTL_ASSERT_MTX_NO_NANS(mtx) _ASSERTE(mtx.test_noNaNs())
#else //NNTL_AGGRESSIVE_NANS_DBG_CHECK
#define NNTL_ASSERT_MTX_NO_NANS(mtx) ((void)(0))
#endif //NNTL_AGGRESSIVE_NANS_DBG_CHECK

#else //defined(_DEBUG) || defined(DEBUG)

#ifdef NNTL_RELEASE_WITH_DEBUG
//if you encounter crashes with NNTL_RELEASE_WITH_DEBUG, make sure it is defined _before_ including any other nntl file
// (put it on the top of stdafx.h)

#define NNTL_DEBUG
#define NNTL_ASSERT(a) nntl_pragma_macro(warning(push)) nntl_pragma_macro(warning(disable:4127)) if(!(a)) __debugbreak(); nntl_pragma_macro(warning(pop))
#define NNTL_DEBUG_DECLARE(v) v
#define NNTL_DEBUG_ARG(p) ,p

#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
#define NNTL_ASSERT_MTX_NO_NANS(mtx) nntl_pragma_macro(warning(push)) nntl_pragma_macro(warning(disable:4127)) if(!(mtx.test_noNaNs())) __debugbreak(); nntl_pragma_macro(warning(pop))
#else //NNTL_AGGRESSIVE_NANS_DBG_CHECK
#define NNTL_ASSERT_MTX_NO_NANS(mtx) ((void)(0))
#endif //NNTL_AGGRESSIVE_NANS_DBG_CHECK

#else //NNTL_RELEASE_WITH_DEBUG

#define NNTL_ASSERT(a) ((void)(0))
#define NNTL_DEBUG_ARG(p)
#define NNTL_DEBUG_DECLARE(v)

#define NNTL_ASSERT_MTX_NO_NANS(mtx)

#endif // !NNTL_RELEASE_WITH_DEBUG

#endif // defined(_DEBUG) || defined(DEBUG)

#ifndef IN
#define IN
#endif
#ifndef OUT
#define OUT
#endif

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

#define NNTL_TYPEDEFS_SELF() typedef self_t& self_ref_t; \
typedef const self_t& self_cref_t;

//shortcuts for get_self() methods
//NB: ensure the self_t is defined and macro is placed under a correct visibility scope
#define NNTL_METHODS_SELF_CHECKED(cond, msg) \
NNTL_TYPEDEFS_SELF() \
self_ref_t get_self() noexcept { \
  static_assert(cond, msg); \
  return static_cast<self_ref_t>(*this); \
} \
self_cref_t get_self() const noexcept { \
  static_assert(cond, msg); \
  return static_cast<self_cref_t>(*this); \
}

#define NNTL_METHODS_SELF() \
NNTL_TYPEDEFS_SELF() \
self_ref_t get_self() noexcept { return static_cast<self_ref_t>(*this); } \
self_cref_t get_self() const noexcept {  return static_cast<self_cref_t>(*this); }

template<typename r> class always_false : public ::std::false_type {};
//helper to show type T
template<typename T> static constexpr int static_show_type() {	
	static_assert(always_false<T>::value, "you've requested type of T, see below");
	return 0;
};
//static constexpr int v = static_show_type<TypeToCheck>();

template<typename T> static constexpr int static_show_type(T&& v) {
	static_assert(always_false<T>::value, "you've requested type of T, see below");
	return 0;
};

