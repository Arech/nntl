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

//this file includes necessary files to use matlab API from the app

// You must have Matlab installed and NNTL_MATLAB_AVAILABLE defined to nonzero value to make this code work.
// In order to complile, update project VC++ Directories property page to point to correct 
// Matlab's includes (something like C:\Program Files\MATLAB\R2014a\extern\include )
// and libraries (C:\Program Files\MATLAB\R2014a\extern\lib\win64\microsoft)
// "Environments" propery of Debugging property page should set correct path to Matlab binaries
// (set this property to something like PATH=%PATH%;c:\Program Files\MATLAB\R2014a\bin\win64\ and 
// be sure to turn on "Merge Environment" switch)

#ifndef NNTL_MATLAB_AVAILABLE
#define NNTL_MATLAB_AVAILABLE 0
#endif

#if NNTL_MATLAB_AVAILABLE

#include <mclmcrrt.h>//matlab rt proxy to eliminate platform specifics
#include <mat.h>

#pragma comment(lib,"libmat.lib")
#pragma comment(lib,"libmx.lib")
#pragma comment(lib,"mclmcrrt.lib")

namespace nntl {
namespace utils {
namespace matlab {
	
	template<typename BaseT> struct type2id {};

	template<> struct type2id<bool> { static constexpr mxClassID id = mxLOGICAL_CLASS; };

	template<> struct type2id<::std::int8_t> { static constexpr mxClassID id = mxINT8_CLASS; };
	template<> struct type2id<::std::uint8_t> { static constexpr  mxClassID id = mxUINT8_CLASS; };

	template<> struct type2id<::std::int32_t> { static constexpr  mxClassID id = mxINT32_CLASS; };
	template<> struct type2id<::std::uint32_t> { static constexpr  mxClassID id = mxUINT32_CLASS; };

	template<> struct type2id<::std::int64_t> { static constexpr  mxClassID id = mxINT64_CLASS; };
	template<> struct type2id<::std::uint64_t> { static constexpr  mxClassID id = mxUINT64_CLASS; };

	template<> struct type2id<float> { static constexpr mxClassID id = mxSINGLE_CLASS; };
	template<> struct type2id<double> { static constexpr  mxClassID id = mxDOUBLE_CLASS; };

	//converts matlab data to smatrix<DestT>
	template<typename DestT>
	inline bool fill_data_no_bias(math::smatrix<DestT>& d, mxArray*const pVar)noexcept {
		const auto matlabType = mxGetClassID(pVar);
		bool ret = true;
		const void*const pSrc = mxGetData(pVar);
		if (type2id<DestT>::id == matlabType) {
			d.fill_from_array_no_bias(static_cast<const DestT*>(pSrc));
		} else {
			switch (matlabType) {//#todo: table lookup would be better here
			case mxDOUBLE_CLASS:
				d.fill_from_array_no_bias(static_cast<const double*>(pSrc));
				break;
			case mxSINGLE_CLASS:
				d.fill_from_array_no_bias(static_cast<const float*>(pSrc));
				break;
			case mxINT64_CLASS:
			case mxUINT64_CLASS:
				d.fill_from_array_no_bias(static_cast<const ::std::uint64_t*>(pSrc));
				break;
			case mxINT32_CLASS:
			case mxUINT32_CLASS:
				d.fill_from_array_no_bias(static_cast<const ::std::uint32_t*>(pSrc));
				break;
			case mxINT8_CLASS:
			case mxUINT8_CLASS:
				d.fill_from_array_no_bias(static_cast<const ::std::uint8_t*>(pSrc));
				break;
			case mxLOGICAL_CLASS:
				d.fill_from_array_no_bias(static_cast<const bool*>(pSrc));
				break;

			default:
				ret = false;
				break;
			}
		}
		return ret;
	}

}
}
}

#endif
