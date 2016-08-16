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
#pragma once
//this file provides a sample of math interface customization.
//It's not included anywhere in nntl core files, but is required for compilation.
//#include it before any nntl core file 

//redefine if needed, but don't use anywhere else
//MUST be the same in all compilation units, DEFINE ON PROJECT-LEVEL
#ifndef NNTL_CFG_DEFAULT_TYPE
//#define NNTL_CFG_DEFAULT_TYPE float
#define NNTL_CFG_DEFAULT_TYPE double
#endif

//TODO: there are faster implementations of stl vectors available. Find, evaluate and use them instead of std::vector
//#include <vector>

#include "_defs.h"
#include "interface/math/simple_matrix.h"

namespace nntl {
namespace math {

	//////////////////////////////////////////////////////////////////////////
	// _basic_types is DEPRECATED and will be removed in future. Use correct template parameters where applicable.
	// 
	//class to store various math type definitions. May need to be rewritten to support different math libs.
	//NNTL basic types is defined with _ty suffix. Aliases (typedefs) to this types defines with _t or _t_ (if collision 
	// with standard types possible) suffix
	struct _basic_types {
		//basic data type for nn inputs, weights and so on.
		//change to any suitable type
		typedef NNTL_CFG_DEFAULT_TYPE real_ty;
		static constexpr char* real_ty_name = NNTL_STRINGIZE(NNTL_CFG_DEFAULT_TYPE);

		//matrix of real_ty, colMajor order, because many math libs uses it by default
		//TODO: specialization for the case of 1D data (such as binary classification results) may provide some performance gain
		//DEPRECATED typedef. Use simple_matrix class directly
		typedef simple_matrix<real_ty> realmtx_ty;

		//DEPRECATED typedef. Use simple_matrix_deformable class directly
		typedef simple_matrix_deformable<real_ty> realmtxdef_ty;
		static_assert(std::is_base_of<realmtx_ty, realmtxdef_ty>::value, "realmtxdef_ty must be derived from realmtx_ty!");
		
		

		//////////////////////////////////////////////////////////////////////////
		//definitions below is to be corrected

		//generic type for matrices with model data (train-test samples)
		//typedef simple_matrix<real_ty> rawdata_mtx_ty;

		
	};

	//thanks to http://stackoverflow.com/a/4609795
	template <typename T> int sign(T val) {
		return (T(+0.0) < val) - (val < T(-0.0));
	}
	
	template <typename _T> struct real_ty_limits {};
	template <> struct real_ty_limits<double> {
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
	template <> struct real_ty_limits<float> {
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
}
}
