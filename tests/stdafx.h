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
// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

//to make ::rand_s() available
#define _CRT_RAND_S

//min() and max() triggers very weird compiler crash while doing ::std::numeric_limits<vec_len_t>::max()
#define NOMINMAX

//\vc\include\yvals.h(112): error C2338: /RTCc rejects conformant code, so it isn't supported by the C++ Standard Library.
//Either remove this compiler option, or define _ALLOW_RTCc_IN_STL to acknowledge that you have received this warning.
#define _ALLOW_RTCc_IN_STL

#define _USE_MATH_DEFINES // for C++ math constants

#include "targetver.h"

#include <stdio.h>

//dont need obsolete tchar machinery
//#include <tchar.h>

#include <iostream>

//#include <array>

#define STDCOUT(args) std::cout << args
#define STDCOUTL(args) STDCOUT(args) << std::endl

//////////////////////////////////////////////////////////////////////////
// special externals, necessary only for tests, but not nntl

#pragma warning(disable: 28182)
#include <gtest/gtest.h>
#pragma warning(default: 28182)

#ifdef _DEBUG
#define _GTEST_LIB "gtestd.lib"
#else
#define _GTEST_LIB "gtest.lib"
#endif // _DEBUG
#pragma comment(lib,_GTEST_LIB)

//////////////////////////////////////////////////////////////////////////

#define NNTL_MATLAB_AVAILABLE 1

//////////////////////////////////////////////////////////////////////////

#ifdef _DEBUG
#define TESTS_SKIP_LONGRUNNING
#define TESTS_SKIP_NNET_LONGRUNNING
#endif

//#define TESTS_SKIP_LONGRUNNING
//#define TESTS_SKIP_NNET_LONGRUNNING

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

#include "../nntl/_test/test.h"

#ifdef TESTS_SKIP_LONGRUNNING
#define NNTL_RUN_TEST4(aroundNumel,maxPercDiff,percDiffStep,numelMult) _NNTL_RUN_TEST( (aroundNumel/10),0,1,numelMult,__COUNTER__)
#define NNTL_RUN_TEST2(aroundNumel,numelMult) _NNTL_RUN_TEST((aroundNumel/10),0,1,numelMult,__COUNTER__)
#else
#define NNTL_RUN_TEST4(aroundNumel,maxPercDiff,percDiffStep,numelMult) _NNTL_RUN_TEST(aroundNumel,maxPercDiff,percDiffStep,numelMult,__COUNTER__)
#define NNTL_RUN_TEST2(aroundNumel,numelMult) _NNTL_RUN_TEST(aroundNumel,4,2,numelMult,__COUNTER__)
#endif
