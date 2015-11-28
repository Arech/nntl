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
//this file provides a sample of math interface customization.
//It's not included anywhere in nntl core files, but is required for compilation.
//#include it before any nntl core file 

//redefine if needed, but don't use anywhere else
//MUST be the same in all compilation units, DEFINE ON PROJECT-LEVEL
#ifndef NNTL_CFG_DEFAULT_TYPE
#define NNTL_CFG_DEFAULT_TYPE double
#endif

//set to 0 to use row major order
//MUST be the same in all compilation units, DEFINE ON PROJECT-LEVEL
//#ifndef NNTL_CFG_DEFAULT_COL_MAJOR_ORDER
//#define NNTL_CFG_DEFAULT_COL_MAJOR_ORDER 1
//#endif

//about NNTL_CFG_DEFAULT_TYPE and NNTL_CFG_DEFAULT_COL_MAJOR_ORDER
// It looked like a good idea that should allow different compilation units to have different versions of API based on
// this settings. However it turned out, it just won't work in MSVC2015. Either I'm doing something terribly wrong, or
// it's really a compiler bug. But you can't have this settings different in different cpp files. Or, more precisely,
// you can, but nobody knows which version really be used then. I got some weird results when tried to compile one cpp with
// float/colmajor and another with double/rowmajor (variants of test_jsonreader.cpp) - it happend that despite the
// different settings some pieces of latter code still have used float/colmajor instead of double/rowmajor, but at the
// same time, typedefs derived from this code on the very upper level yielded correct double/rowmajor types.
// Also, I tried to make templated version of _basic_types, but got the same results...
// Therefore, you would better never use different values of this settings in a single project.


//about BLAS and similar math libraries
//Clearly, all math stuff should be offloaded to external fast-as-hell math library.
// Which one, provided that there are huge number of options? I evaluated some of them to fit just 2 simple
// criteria: be as optimized and fast as possible and work under Win x64. Here are the results:
// a. ATLAS - very interesting, but huge problems (possibly performance too) on Win x64. Very tricky installation.
//		Therefore decided not to waste time on it.
// **b. AMD ACML - it's rumored that it's not very well optimized. And it is abandoned. But I probably will use it as a baseline
// c. Boost uBLAS - known to be slow and almost abandoned
// *d. Intel MKL - very promising as it's rumored as one of the best and fastest libs available. But it's non-free.
//		May be later I'll be able to qualify for copy as open-source developer.
// *e. Yeppp! - very promising, but doesn't support matrices, only vectors.
// **f. BLIS - very interesting, but lack of normal support of Win x64 make me cry...
// **g. ulmBLAS - interesting project, but hard to believe it will last long...
// ***h. OpenBLAS - Primary target.
// **i. Blaze - looks good, but requires working BLAS install.
// **j. Eigen - looks good, should try. But there are controversies about its real performance
// k. clBLAS - might give it a try to quickstart GPU usage
// *l. Armadillo - it's rumored that Armadillo is slow. And it requires BLAS too. Though, may be try it later.

// There might be a need to link to several BLAS libraries in a single project (for benchmarking for example).
// Doing it straight is a recipe for disaster
// because of linker symbols collision. One possible workaround is to load libraries dynamically and call them via function
// pointers. But this requires some additional (cross-platform) work to be done on dynamic library loading. I failed to find
// a well implemented code that does this job (only QLibrary, but it's a hell to include it in project), therefore probably
// had to do it myself.


//TODO: there are faster implementations of stl vectors available. Find, evaluate and use them instead of std::vector
//#include <vector>

#include "../_defs.h"
#include "math/simple_matrix.h"

namespace nntl {
namespace math {

	//class to store various math type definitions. May need to be rewritten to support different math libs.
	//NNTL basic types is defined with _ty suffix. Aliases (typedefs) to this types defines with _t or _t_ (if collision 
	// with standard types possible) suffix
	struct _basic_types {
		//basic data type for nn inputs, weights and so on.
		//change to any suitable type
		typedef NNTL_CFG_DEFAULT_TYPE float_ty;

		//matrix of float_ty, colMajor order, because many math libs uses it by default
		//TODO: specialization for the case of 1D data (such as binary classification results) may provide some performance gain
		typedef simple_matrix<float_ty> floatmtx_ty;

		typedef simple_matrix_deformable<float_ty> floatmtxdef_ty;
		static_assert(std::is_base_of<floatmtx_ty, floatmtxdef_ty>::value, "floatmtxdef_ty must be derived from floatmtx_ty!");
		
		

		//////////////////////////////////////////////////////////////////////////
		//definitions below is to be corrected

		//generic type for matrices with model data (train-test samples)
		//typedef simple_matrix<float_ty> rawdata_mtx_ty;

		
	};

	//thanks to http://stackoverflow.com/a/4609795
	template <typename T> int sign(T val) {
		return (T(+0.0) < val) - (val < T(-0.0));
	}

	template <typename _T> struct float_ty_limits {};
	template <> struct float_ty_limits<double> {
		//natural log of closest to zero but non zero (realmin) value
		static constexpr double log_almost_zero = double(-708.3964185322642);

		//returns minimum value greater than zero, that can be added to v and the result be represented by double
		static double eps_greater(double v)noexcept {
			return std::nextafter(v, std::numeric_limits<double>::infinity()) - v;
		}
		static double eps_greater_n(double v,double n)noexcept {
			return n*eps_greater(v);
		}

		//returns minimum value greater than zero, that can be subtracted from v and the result be represented by double
		static double eps_lower(double v)noexcept {
			return v - std::nextafter(v, -std::numeric_limits<double>::infinity());
		}
		static double eps_lower_n(double v, double n)noexcept {
			return n*eps_lower(v);
		}
	};
	template <> struct float_ty_limits<float> {
		//natural log of closest to zero but non zero (realmin) value
		static constexpr float log_almost_zero = float(-87.336544750402);

		//returns minimum value greater than zero, that can be added to v and the result be represented by float
		static float eps_greater(float v)noexcept {
			return std::nextafter(v, std::numeric_limits<float>::infinity()) - v;
		}
		static float eps_greater_n(float v, float n)noexcept {
			return n*eps_greater(v);
		}

		//returns minimum value greater than zero, that can be subtracted from v and the result be represented by float
		static float eps_lower(float v)noexcept {
			return v - std::nextafter(v, -std::numeric_limits<float>::infinity());
		}
		static float eps_lower_n(float v, float n)noexcept {
			return n*eps_lower(v);
		}
	};

	//choose necessary types right after inclusion of this file (or similar with basic_types definition) with 
	// code:
	// namespace nntl{ using math_types = math::basic_types; }

}
}
