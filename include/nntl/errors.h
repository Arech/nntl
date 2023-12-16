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

#include <nntl/_defs.h>
#include <nntl/common.h>

namespace nntl {

	/*
	 *Here is the sample of minimal interface of ErrStruct template parameter of _has_last_error<> class
	
	struct jsonreader_err {
		enum ErrorCode {
			Success = 0
		};

		//TODO: table lookup would be better here.
		static const nntl::error_char_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case ErrorCode::Success: return NNTL_STRING("No error / success.");

			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}
		
		//Also, derived class may define a function with the following interface to provide customized messages
		/ *::std::string get_last_error_string()const noexcept {
			::std::string les(get_last_error_str());

			if (ErrorCode::FailedToParseJson == get_last_error()) {
				les = les + " Rapidjson: " + rapidjson::GetParseError_En(get_parse_error())
					+ " Offset: " + ::std::to_string(get_parse_error_offset());
			}
			return les;
		}* /

	};*/

	template <typename ErrStruct>
	class _has_last_error {
	public:
		//type has to be defined in derived class and has Success member
		typedef enum ErrStruct::ErrorCode ErrorCode;

	private:
		ErrorCode m_le;

	protected:
		_has_last_error()noexcept : m_le(ErrorCode::Success){}
		~_has_last_error()noexcept {}

	public:
		const ErrorCode get_last_error()const noexcept { return m_le; }
		const strchar_t* get_last_error_str()const noexcept { return get_error_str(m_le); }
		static const strchar_t* get_error_str(const ErrorCode ec) noexcept { return ErrStruct::get_error_str(ec); }

	protected:
		ErrorCode _set_last_error(ErrorCode ec)noexcept { m_le = ec; return ec; }
	};
}