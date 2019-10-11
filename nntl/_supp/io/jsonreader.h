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

// I use the following MATLAB struct to describe training data in my MATLAB research code:
//		.train_x (double N*D)
//		.train_y (double N*C)
//		.test_x (double M*D)
//		.test_y (double M*C)
//	where: N is a number of training samples D dimensions each.
//		M - number of test samples
//		C - number of classes
//		Sample data is stored rowwise.
//		Column major ordering (default for Matlab)
//
// You may serialize this struct to json via ./matlab/td2json_file.m (uses https://github.com/christianpanton/matlab-json/).
// Matrices serialized as "vector of vectors". Due to a bug in matlab-json, we can't tell the difference between vector
// orientation (be it 1*N or N*1). Therefore, all vectors read as being in bColMajor order. This means, that we don't support
// TD structures with only one training/testing sample. Not really big drawback, I think.
// 
// json2td is a class to read json file into object of train_data class
// 
// USAGE:
// be sure to include correct math interface definition before this file, such as ../math.h
// 

#include <limits>

#include "../../errors.h"
#include "../../train_data.h"

//define necessary RAPIDJSON_ macros before including this file
#define RAPIDJSON_SSE2
#define RAPIDJSON_HAS_STDSTRING 1
#include "../../../_extern/rapidjson/include/rapidjson/document.h"
#include "../../../_extern/rapidjson/include/rapidjson/filereadstream.h"
#include "../../../_extern/rapidjson/include/rapidjson/error/en.h"

namespace nntl_supp {

	struct _jsonreader_errs {
		enum ErrorCode {
			Success = 0,
			FailedToOpenFile,
			FailedToParseJson,
			RootIsNotAnObject,

			NoTrainX,//ATTN: it's required that order of any trainx, trainy, testx and testy members remains constant
			NoTrainY,
			NoTestX,
			NoTestY,

			InvalidTrainX,
			InvalidTrainY,
			InvalidTestX,
			InvalidTestY,

			MismatchingDataLength,
			MemoryAllocationFailed
		};

		//TODO: table lookup would be better here. But it's not essential
		static const nntl::strchar_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case Success: return NNTL_STRING("No error / success.");
			case FailedToOpenFile: return NNTL_STRING("Failed to open file.");
			case FailedToParseJson: return NNTL_STRING("Failed to parse json file.");
			case RootIsNotAnObject: return NNTL_STRING("Parsed JSON: Root is not an object.");

			case NoTrainX: return NNTL_STRING("There is no required 'train_x' field.");
			case NoTrainY: return NNTL_STRING("There is no required 'train_y' field.");
			case NoTestX: return NNTL_STRING("There is no required 'test_x' field.");
			case NoTestY: return NNTL_STRING("There is no required 'test_y' field.");

			case InvalidTrainX: return NNTL_STRING("Invalid 'train_x' field.");
			case InvalidTrainY: return NNTL_STRING("Invalid 'train_y' field.");
			case InvalidTestX: return NNTL_STRING("Invalid 'test_x' field.");
			case InvalidTestY: return NNTL_STRING("Invalid 'test_y' field.");

			case MismatchingDataLength: return NNTL_STRING("Invalid data length");
			case MemoryAllocationFailed: return NNTL_STRING("Not Enough Memory");

			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}
	};

	class jsonreader : public nntl::_has_last_error<_jsonreader_errs>, protected nntl::math::smatrix_td {
	protected:
// 		typedef nntl::train_data::realmtx_t realmtx_t;
// 		typedef realmtx_t::value_type mtx_value_t;
// 		typedef realmtx_t::vec_len_t vec_len_t;

		template<typename T_> using smatrix = nntl::math::smatrix<T_>;
		template<typename T_> using train_data = nntl::train_data<T_>;

		//////////////////////////////////////////////////////////////////////////
		//members
	public:
		::std::size_t m_readBufSize;

	protected:
		::std::size_t m_parseErrorOffset;
		rapidjson::ParseErrorCode m_parseError;

	public:
		jsonreader () noexcept	: nntl::_has_last_error<_jsonreader_errs>()
			, m_readBufSize(1024*1024*4), m_parseError(rapidjson::kParseErrorNone)
			, m_parseErrorOffset(0){}
		~jsonreader () noexcept {}
		
		// read fname, parses it as jsonized struct TD into dest var, which can be either nntl::train_data or nntl::train_data::mtx_t
		// If readInto_t == nntl::train_data, then all X data will be created with emulateBiases() feature and bMakeMtxBiased param will be ignored
		template <typename readInto_t>
		const ErrorCode read(const char* fname, readInto_t& dest, const bool bMakeMtxBiased=false) {
			static_assert(::std::is_same<train_data<typename readInto_t::value_type>, readInto_t>::value
				|| ::std::is_same<smatrix<typename readInto_t::value_type>, readInto_t>::value,
				"Only nntl::train_data or nntl::train_data::mtx_t is supported as readInto_t template parameter");
			//bMakeMtxBiased is ignored for nntl::train_data and should be set as false by default
			NNTL_ASSERT( (!::std::is_same<train_data<typename readInto_t::value_type>, readInto_t>::value || !bMakeMtxBiased));

			::std::FILE* fp = nullptr;

			//TODO: there may be a conflict if fname is not char -derived type
			if (fopen_s(&fp, fname, "rbS") || nullptr==fp) { return _set_last_error(ErrorCode::FailedToOpenFile); }

			rapidjson::Document d;
			::std::vector<char> readbuf(m_readBufSize);
			rapidjson::FileReadStream is(fp, &readbuf[0], readbuf.size());
			d.ParseStream<rapidjson::kParseFullPrecisionFlag>(is);
			readbuf.clear();
			fclose(fp);

			m_parseError = d.GetParseError();
			m_parseErrorOffset = d.GetErrorOffset();
			if (d.HasParseError()) { return _set_last_error(ErrorCode::FailedToParseJson); }

			return _parse_json_doc(d, dest, bMakeMtxBiased);
		}
	protected:
		enum _root_members {
			train_x = ((int)ErrorCode::NoTrainX - (int)ErrorCode::NoTrainX),
			train_y = ((int)ErrorCode::NoTrainY - (int)ErrorCode::NoTrainX),
			test_x = ((int)ErrorCode::NoTestX - (int)ErrorCode::NoTrainX),
			test_y = ((int)ErrorCode::NoTestY - (int)ErrorCode::NoTrainX)
		};

		//TODO: table lookup would be better here. Such as:
		//static const char* kTypeNames[] = { "Null", "False", "True", "Object", "Array", "String", "Number" };
		inline static const char* _get_root_member_str(const _root_members m)noexcept {
			switch (m) {
			case train_x:return NNTL_STRING("train_x");
			case train_y:return NNTL_STRING("train_y");
			case test_x:return NNTL_STRING("test_x");
			case test_y:return NNTL_STRING("test_y");
			default:
				NNTL_ASSERT(!"Cant be here!");
				abort();
			}
		}

		template <typename MembersEnumId>
		const ErrorCode _members2ErrorCode(const ErrorCode b, const MembersEnumId rm)noexcept {
			NNTL_UNREF(b);
			const ErrorCode e = static_cast<ErrorCode>((int)ErrorCode::NoTrainX + (int)rm);
			return _set_last_error(e);
		}

		template <typename MembersEnumId, typename T_>
		const ErrorCode _parse_mtx(const rapidjson::Document& o, const MembersEnumId memberId, smatrix<T_>& dest)noexcept {
			auto mn = _get_root_member_str(memberId);

			auto mIt = o.FindMember(mn);
			if (o.MemberEnd() == mIt) { return _members2ErrorCode(ErrorCode::NoTrainX, memberId); }

			const rapidjson::Document::ValueType& vec = mIt->value;
			if (!vec.IsArray() || vec.Begin() == vec.End()) { return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId); }

			// higher level array (vec) could be either a simple array (then, it's implied that the other matrix dimension is 1), or
			// an array of arrays. Deal with it here
			return vec.Begin()->IsArray() 
				? _parse_as_mtx(vec, memberId, dest)
				: _parse_as_vector(vec, memberId, dest);
		}

		template <typename MembersEnumId, typename T_>
		const ErrorCode _parse_as_mtx(const rapidjson::Document::ValueType& vec, const MembersEnumId memberId, smatrix<T_>& dest)noexcept {
			if (vec.Begin()->Size() > ::std::numeric_limits<vec_len_t>::max()) { return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId); }
			
			const vec_len_t inrd = static_cast<vec_len_t>(vec.Begin()->Size());
			if (0 == inrd) { return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId); }

			if (!dest.resize(inrd, vec.Size())) { return _set_last_error(ErrorCode::MemoryAllocationFailed); }

			vec_len_t i = 0;
			rapidjson::Document::ValueType::ConstValueIterator itrend = vec.End();
			for (rapidjson::Document::ValueType::ConstValueIterator itr = vec.Begin(); itr != itrend; ++itr, ++i) {
				if (!itr->IsArray() || inrd != itr->Size()) {
					dest.clear();
					return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId);
				}

				vec_len_t j = 0;
				rapidjson::Document::ValueType::ConstValueIterator eiend = itr->End();
				for (rapidjson::Document::ValueType::ConstValueIterator ei = itr->Begin(); ei != eiend; ++ei, ++j) {
					if (!ei->IsNumber()) {
						dest.clear();
						return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId);
					}
					//const mtx_value_t v = ei->GetDouble();
					const T_ v = _extract_value<T_>(ei);
					dest.set(j, i, v);
				}
			}
			return ErrorCode::Success;
		}

		template <typename MembersEnumId, typename T_>
		const ErrorCode _parse_as_vector(const rapidjson::Document::ValueType& vec, const MembersEnumId memberId, smatrix<T_>& dest)noexcept {
			if (vec.Size() > ::std::numeric_limits<vec_len_t>::max()) { return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId); }

			if (!dest.resize(vec.Size(), 1)) { return _set_last_error(ErrorCode::MemoryAllocationFailed); }

			vec_len_t i = 0;
			rapidjson::Document::ValueType::ConstValueIterator itrend = vec.End();
			for (rapidjson::Document::ValueType::ConstValueIterator itr = vec.Begin(); itr != itrend; ++itr, ++i) {
				if (!itr->IsNumber()) {
					dest.clear();
					return _members2ErrorCode(ErrorCode::InvalidTrainX, memberId);
				}

				//const mtx_value_t v = itr->GetDouble();
				const T_ v = _extract_value<T_>(itr);
				dest.set(i, 0, v);
			}
			return ErrorCode::Success;
		}
		
		template <typename mvt>
		typename ::std::enable_if_t< ::std::is_floating_point<mvt>::value, mvt > _extract_value(rapidjson::Document::ValueType::ConstValueIterator & itr)const noexcept {
			return static_cast<mvt>(itr->GetDouble());
		}
		template <typename mvt>
		typename ::std::enable_if_t< ::std::is_integral<mvt>::value && ::std::is_signed<mvt>::value, mvt > _extract_value(rapidjson::Document::ValueType::ConstValueIterator & itr)const noexcept {
			return static_cast<mvt>(itr->GetInt64());
		}
		template <typename mvt>
		typename ::std::enable_if_t< ::std::is_integral<mvt>::value && ::std::is_unsigned<mvt>::value, mvt > _extract_value(rapidjson::Document::ValueType::ConstValueIterator & itr)const noexcept {
			return static_cast<mvt>(itr->GetUint64());
		}

		/*template <typename readInto_t, typename T_ = readInto_t::value_type>
		const ErrorCode _parse_json_doc(const rapidjson::Document& d, readInto_t& dest, const bool bMakeMtxBiased)noexcept {
			static_assert(!"No function specialization for type readInto_t");
		}
*/

		template<typename T_>
		const ErrorCode _parse_json_doc(const rapidjson::Document& d, train_data<T_>& dest, const bool )noexcept {
			if (!d.IsObject()) { return _set_last_error(ErrorCode::RootIsNotAnObject); }

			smatrix<T_> tr_x,tr_y,t_x,t_y;
			//if (bMakeXDataBiased) {
				tr_x.will_emulate_biases();
				t_x.will_emulate_biases();
			//}

			if (ErrorCode::Success != _parse_mtx(d, train_x, tr_x)) { return get_last_error(); }
			if (ErrorCode::Success != _parse_mtx(d, train_y, tr_y)) { return get_last_error(); }
			if (ErrorCode::Success != _parse_mtx(d, test_x, t_x)) { return get_last_error(); }
			if (ErrorCode::Success != _parse_mtx(d, test_y, t_y)) { return get_last_error(); }
			
			if (!dest.absorb(::std::move(tr_x), ::std::move(tr_y), ::std::move(t_x), ::std::move(t_y))) { //, !bMakeXDataBiased)) {
				return _set_last_error(ErrorCode::MismatchingDataLength);
			}

			return _set_last_error(ErrorCode::Success);
		}

		template<typename T_>
		const ErrorCode _parse_json_doc(const rapidjson::Document& d, smatrix<T_>& dest, const bool bMakeMtxBiased)noexcept {
			if (!d.IsObject()) { return _set_last_error(ErrorCode::RootIsNotAnObject); }

			if (bMakeMtxBiased) dest.will_emulate_biases();
			
			if (ErrorCode::Success != _parse_mtx(d, train_x, dest)) { return get_last_error(); }
			
			return _set_last_error(ErrorCode::Success);
		}

	public:
		::std::string get_last_error_string()const noexcept { 
			::std::string les(get_last_error_str());

			if (ErrorCode::FailedToParseJson == get_last_error()) {
				les = les + " Rapidjson: " + rapidjson::GetParseError_En(get_parse_error())
					+ " Offset: " + ::std::to_string(get_parse_error_offset());
			}
			return les;
		}

		const rapidjson::ParseErrorCode get_parse_error() const noexcept { return m_parseError; }
		const ::std::size_t get_parse_error_offset() const noexcept { return m_parseErrorOffset; }
	};

}