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
//this file provides a sample of math interface customization.
//It's not included anywhere in nntl core files, but is required for compilation.
//#include it before any nntl core file 

//redefine if needed, but don't use anywhere else
//MUST be the same in all compilation units, DEFINE ON PROJECT-LEVEL
#ifndef NNTL_CFG_DEFAULT_TYPE
#define NNTL_CFG_DEFAULT_TYPE float
//#define NNTL_CFG_DEFAULT_TYPE double
#endif

//SSE require at least 16. 32 is probably safer for AVX enabled procs. Or even 64 for AVX2
#ifndef NNTL_CFG_DEFAULT_FP_PTR_ALIGN
#define NNTL_CFG_DEFAULT_FP_PTR_ALIGN 32
#endif
//and BTW, due to extensive use of smatrix.useExternalStorage() through the lib, we may occasionally fail to obey
//some strict alignment requirements... This shouldn't happen frequently, but still.. I didn't thought about
// it when designed everything... Will deal with it when it bites (if it will bite actually... may be not)

//////////////////////////////////////////////////////////////////////////
//most of the time you would not want to have denormals, but they may occur during even a
// normal neural net training (not counting most of the cases with wrong/ill metaparameters).
// If you have to rely on them - you're probably doing something very wrong.
#if !defined(NNTL_DENORMALS2ZERO) || NNTL_DENORMALS2ZERO!=0
#define NNTL_DENORMALS2ZERO 1
#endif

//::std::exp() with large negative argument may produce -nan(ind) when compiler vectorizes and it into intinsic.
//To prevent this behaviour while maintaining the same code performance helps setting floating point roundind towards zero
#if !defined(NNTL_FP_ROUND_TO_ZERO)
#define NNTL_FP_ROUND_TO_ZERO 1
#else
#if NNTL_FP_ROUND_TO_ZERO!=0
#undef NNTL_FP_ROUND_TO_ZERO
#define NNTL_FP_ROUND_TO_ZERO 1
#endif
#endif

//////////////////////////////////////////////////////////////////////////
#if NNTL_FP_ROUND_TO_ZERO
#define NNTL_FPU_SET_ROUND_TO_ZERO unsigned int current_word = 0; _controlfp_s(&current_word, _RC_CHOP, _MCW_RC);
//_controlfp_s(&current_word, _RC_NEAR, _MCW_RC);//--default mode breaks intrinsic of ::std::exp
#define NNTL_FPU_RESTORE_ROUND_TO_ZERO unsigned int current_word = 0; _controlfp_s(&current_word, _CW_DEFAULT, _MCW_RC);
#else
#define NNTL_FPU_SET_ROUND_TO_ZERO
#define NNTL_FPU_RESTORE_ROUND_TO_ZERO
#endif
namespace nntl {
	namespace utils {
		template<typename T>
		inline constexpr size_t mem_align_for() noexcept {
			return ::std::is_floating_point<T>::value ? NNTL_CFG_DEFAULT_FP_PTR_ALIGN : alignof(T);
		}

		template<typename T>
		inline bool is_ptr_aligned(const T*const p) noexcept {
			auto i = reinterpret_cast<::std::uintptr_t>(p);
			return !(i % static_cast<::std::uintptr_t>(mem_align_for<T>()));
		}


		//helper class to restore FPU config to conventional before calling external code and then revert back
		struct _scoped_restore_FPU {
			~_scoped_restore_FPU()noexcept {
				NNTL_FPU_SET_ROUND_TO_ZERO
			}
			_scoped_restore_FPU()noexcept {
				NNTL_FPU_RESTORE_ROUND_TO_ZERO
			}
		};
	}
}

//////////////////////////////////////////////////////////////////////////

#if NNTL_DENORMALS2ZERO
//#pragma message("NNTL_DENORMALS2ZERO: denormalized floats WILL BE flushed to zero in global_denormalized_floats_mode()" )

// the following defs is for debugging purposes.
#if !defined(NNTL_DEBUGBREAK_ON_DENORMALS)
#define NNTL_DEBUGBREAK_ON_DENORMALS 0
#endif

#if !defined(NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS)
#define NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS 0
#endif

#if NNTL_DEBUGBREAK_ON_DENORMALS && !(NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS)
#pragma message("NNTL_DEBUGBREAK_ON_DENORMALS is ON, turning on NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS")
#undef NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
#define NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS 1
#endif

#if !defined(NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH)
#define NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH 0
#endif

#else //NNTL_DENORMALS2ZERO
//#pragma message("NNTL_DENORMALS2ZERO: global_denormalized_floats_mode() will leave handling of denormalized floats as it is" )

#define NNTL_DEBUGBREAK_ON_DENORMALS 0
#define NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS 0
#define NNTL_DEBUG_CHECK_DENORMALS_ON_EACH_EPOCH 0

#endif //NNTL_DENORMALS2ZERO


//////////////////////////////////////////////////////////////////////////
//if NNTL_CFG_CAREFULL_LOG_EXP is specified and set, nntl uses special ::std::log1p() and ::std::expm1() functions where possible
// to gain some numeric stability (therefore, more precise calculation) at the cost of
// about 40% slowdown (measured for loglogu at my HW with /fp:precise)
#if !defined(NNTL_CFG_CAREFULL_LOG_EXP) || 1!=NNTL_CFG_CAREFULL_LOG_EXP
#define NNTL_CFG_CAREFULL_LOG_EXP 0
#endif // !NNTL_CFG_CAREFULL_LOG_EXP

#if NNTL_CFG_CAREFULL_LOG_EXP
//#pragma message("NNTL_CFG_CAREFULL_LOG_EXP: will use log1p()/expm1() to favor their precision over speed of log()/exp()")
#else
//#pragma message("NNTL_CFG_CAREFULL_LOG_EXP: will NOT use log1p()/expm1() in favor of speed of log()/exp()")
#endif

#ifndef NNTL_NO_AGGRESSIVE_NANS_DBG_CHECK
#define NNTL_AGGRESSIVE_NANS_DBG_CHECK
#endif

//////////////////////////////////////////////////////////////////////////

#include "_defs.h"
#include "interface/math/_base.h"
#include "interface/math/smatrix.h"

