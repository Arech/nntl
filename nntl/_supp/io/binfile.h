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

//this file might be absent on non-windows OSes. It is used only for DWORD/WORD/BYTE typedefs, remove it and define them accordingly
//#include <minwindef.h>

#include "../../errors.h"
#include "../../train_data.h"
#include "../../utils/scope_exit.h"


namespace nntl_supp {

	namespace bin_file {
		typedef uint32_t DWORD;
		typedef uint16_t WORD;
		typedef uint8_t  BYTE;

		enum DATA_TYPES {
			dt_double = 0,
			dt_float = 1
		};

#pragma pack(push, 1)
		struct HEADER {
			DWORD dwSignature;
			WORD wFieldsCount;

			static constexpr DWORD sSignature = 'ltnn';
		};
		static_assert(6 == sizeof(HEADER), "WTF??");

		struct FIELD_ENTRY{
			static constexpr unsigned sFieldNameTotalLength = 15;

			DWORD dwRows;
			DWORD dwCols;
			char szName[sFieldNameTotalLength];
			BYTE bDataType;//MUST go strictly after szName (used as zero-terminator)
#pragma warning(disable: 4200)
			BYTE rawData[];
#pragma warning(default: 4200)
		};
		static_assert(8 + 1 + FIELD_ENTRY::sFieldNameTotalLength == sizeof(FIELD_ENTRY), "WTF?");
#pragma pack(pop)

		template <typename DestDT> inline bool correct_data_type(BYTE dt)noexcept { return false; }
		template <> inline bool correct_data_type<double>(BYTE dt)noexcept { return dt_double == dt; }
		template <> inline bool correct_data_type<float>(BYTE dt)noexcept { return dt_float == dt; }
	}

	struct _binfile_errs {
		enum ErrorCode {
			Success = 0,
			FailedToOpenFile,
			FailedToReadHeader,
			WrongHeaderSignature,
			WrongElementsCount,
			FailedToReadFieldEntry,
			UnsupportedIncorrectDataType,
			FailedToAllocateMemoryForDataConvertion,
			InvalidDataSize,
			FailedToReadData,

			UnknownFieldName,
			FieldHasBeenRead,
			FailedToMakeTDOutOfReadData,
			
			MemoryAllocationFailed
		};

		//TODO: table lookup would be better here. But it's not essential
		static const nntl::strchar_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case Success: return NNTL_STRING("No error / success.");
			case FailedToOpenFile: return NNTL_STRING("Failed to open file.");
			case FailedToReadHeader: return NNTL_STRING("Failed to read header.");
			case WrongHeaderSignature: return NNTL_STRING("Wrong header signature.");
			case WrongElementsCount: return NNTL_STRING("File contains unsupported elements count");
			case FailedToReadFieldEntry: return NNTL_STRING("Failed to read field entry");
			case UnsupportedIncorrectDataType:  return NNTL_STRING("Unsupported or incorrect data type");
			case FailedToAllocateMemoryForDataConvertion: return NNTL_STRING("Failed to allocate memory necessary for data type convertion");
			case InvalidDataSize: return NNTL_STRING("Invalid data size");
			case FailedToReadData: return NNTL_STRING("Failed to read data");			
			case UnknownFieldName: return NNTL_STRING("Unknown field name found");
			case FieldHasBeenRead: return NNTL_STRING("Two fields with the same name found!");
			case FailedToMakeTDOutOfReadData: return NNTL_STRING("Failed to assemble train_data out of read data. Probably not all necessary data have been read!");
			
			case MemoryAllocationFailed: return NNTL_STRING("Not Enough Memory");

			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}
	};


	class binfile : public nntl::_has_last_error<_binfile_errs>, protected nntl::math::smatrix_td {
 	protected:
// 		typedef nntl::train_data::realmtx_t realmtx_t;
// 		typedef realmtx_t::value_type mtx_value_t;
// 		typedef realmtx_t::vec_len_t vec_len_t;

		template<typename T_> using smatrix = nntl::math::smatrix<T_>;
		template<typename T_> using train_data = nntl::train_data<T_>;

	public:
		~binfile()noexcept {}
		binfile()noexcept{}

		// read fname, parses it as jsonized struct TD into dest var, which can be either nntl::train_data or nntl::train_data::mtx_t
		// If readInto_t == nntl::train_data, then all X data will be created with emulateBiases() feature and bMakeMtxBiased param will be ignored
		template <typename readInto_t>
		const ErrorCode read(const char* fname, readInto_t& dest) {
			static_assert(::std::is_same<train_data<typename readInto_t::value_type>, readInto_t>::value
				|| ::std::is_same<smatrix<typename readInto_t::value_type>, readInto_t>::value,
				"Only nntl::train_data or nntl::train_data::mtx_t is supported as readInto_t template parameter");

			FILE* fp=nullptr;
			if (fopen_s(&fp, fname, NNTL_STRING("rbS")) || nullptr == fp) return _set_last_error(ErrorCode::FailedToOpenFile);
			nntl::utils::scope_exit on_exit([&fp]() {
				if (fp) {
					fclose(fp);
					fp = nullptr;
				}
			});

			{
				bin_file::HEADER hdr;

#pragma warning(disable:28020)
				//MSVC SAL goes "slightly" mad here
				if (1 != fread_s(&hdr, sizeof(hdr), sizeof(hdr), 1, fp)) return _set_last_error(ErrorCode::FailedToReadHeader);
#pragma warning(default:28020)
				if (hdr.sSignature != hdr.dwSignature) return _set_last_error(ErrorCode::WrongHeaderSignature);
				if (!elements_count_correct<readInto_t>(hdr.wFieldsCount)) return _set_last_error(ErrorCode::WrongElementsCount);
			}

			return _read_into(fp, dest);
		}

	protected:
		enum _root_members {
			train_x = 0,
			train_y,
			test_x,
			test_y,
			total_members
		};

		inline static const _root_members _name2id(const char* sz) noexcept {
			if (0 == strcmp("train_x", sz)) return _root_members::train_x;
			if (0 == strcmp("train_y", sz)) return _root_members::train_y;
			if (0 == strcmp("test_x", sz)) return _root_members::test_x;
			if (0 == strcmp("test_y", sz)) return _root_members::test_y;
			return _root_members::total_members;
		}

		/*template <typename readInto_t, typename T_= typename readInto_t::value_type>
		const ErrorCode _read_into(FILE* fp, readInto_t& dest)noexcept {
			static_assert(!"No function specialization for type readInto_t");
		}*/

		template<typename T_>
		const ErrorCode _read_into(FILE* fp, train_data<T_>& dest)noexcept {
			typedef smatrix<T_> mtx_t;
			mtx_t mtxs[total_members];

			for (unsigned nel = 0; nel < _root_members::total_members; ++nel) {
				mtx_t m;
				_root_members fieldId;
				const auto err = _read_field_entry(fp, m, fieldId, true);
				if (ErrorCode::Success != err) return err;
				if (!mtxs[fieldId].empty())  return _set_last_error(ErrorCode::FieldHasBeenRead);
				mtxs[fieldId] = ::std::move(m);
			}
			
			if (!dest.absorb(::std::move(mtxs[_root_members::train_x]), ::std::move(mtxs[_root_members::train_y])
				, ::std::move(mtxs[_root_members::test_x]), ::std::move(mtxs[_root_members::test_y])))
			{
				return _set_last_error(ErrorCode::FailedToMakeTDOutOfReadData);
			}

			return ErrorCode::Success;
		}
		template<typename T_>
		const ErrorCode _read_into(FILE* fp, smatrix<T_>& dest)noexcept {
			NNTL_ASSERT(!dest.bDontManageStorage());
			NNTL_ASSERT(dest.empty());

			_root_members f;
			return _read_field_entry(fp, dest, f, false);
		}

		template<typename T_>
		const ErrorCode _read_field_entry(FILE* fp, smatrix<T_>& m, _root_members& fieldId, const bool bReadTD=true)noexcept{
			if (!m.empty()) return _set_last_error(ErrorCode::FieldHasBeenRead);
#pragma warning(disable : 4815)
			bin_file::FIELD_ENTRY fe;
#pragma warning(default : 4815)

#pragma warning(disable:28020)
			//MSVC SAL goes "slightly" mad here
			if (1 != fread_s(&fe, sizeof(fe), sizeof(fe), 1, fp)) return _set_last_error(ErrorCode::FailedToReadFieldEntry);
#pragma warning(default:28020)

			const auto fieldDataType = fe.bDataType;
			const auto bSameTypes = bin_file::correct_data_type<T_>(fieldDataType);

			if (fe.dwRows <= 0 || fe.dwCols <= 0) return _set_last_error(ErrorCode::InvalidDataSize);

			fe.bDataType = 0;
			if (bReadTD) {
				fieldId = _name2id(fe.szName);
				if(_root_members::total_members == fieldId) return _set_last_error(ErrorCode::UnknownFieldName);
				if (_root_members::train_x == fieldId || _root_members::test_x == fieldId) {
					m.will_emulate_biases();
				} else m.dont_emulate_biases();
			}

			if (!m.resize(static_cast<vec_len_t>(fe.dwRows), static_cast<vec_len_t>(fe.dwCols))) return _set_last_error(ErrorCode::MemoryAllocationFailed);

			void* pReadTo = m.data();
			size_t readSize = m.byte_size_no_bias();
			if (!bSameTypes) {
				switch (fieldDataType) {
				case bin_file::dt_float:
					readSize = sizeof(float)*m.numel_no_bias();
					break;

				case bin_file::dt_double:
					readSize = sizeof(double)*m.numel_no_bias();
					break;

				default:
					return _set_last_error(ErrorCode::UnsupportedIncorrectDataType);
					break;
				}
				pReadTo = malloc(readSize);
				if (!pReadTo) return _set_last_error(ErrorCode::FailedToAllocateMemoryForDataConvertion);
			}

#pragma warning(disable:28020)
			//MSVC SAL goes "slightly" mad here
			if (1 != fread_s(pReadTo, readSize, readSize, 1, fp))
				return _set_last_error(ErrorCode::FailedToReadData);
#pragma warning(default:28020)

			if (!bSameTypes) {
				switch (fieldDataType) {
				case bin_file::dt_float:
					//_convert_data<T_, float>(m, static_cast<float*>(pReadTo));
					m.fill_from_array_no_bias(static_cast<const float*>(pReadTo));
					break;

				case bin_file::dt_double:
					//_convert_data<T_, double>(m, static_cast<double*>(pReadTo));
					m.fill_from_array_no_bias(static_cast<const double*>(pReadTo));
					break;
				}

				free(pReadTo);
			}

			return ErrorCode::Success;
		}

		/*template <typename dest_value_type, typename src_value_type>
		static void _convert_data(smatrix<dest_value_type>& dest, src_value_type* pSrc) noexcept {
			const auto pSrcE = pSrc + dest.numel_no_bias();
			auto pD = dest.data();
			while (pSrc != pSrcE) *pD++ = static_cast<dest_value_type>(*pSrc++);
		}*/

		template <typename readInto_t>
		typename ::std::enable_if_t< ::std::is_same<smatrix<typename readInto_t::value_type>, readInto_t>::value, bool> elements_count_correct(decltype(bin_file::HEADER::wFieldsCount) cnt)const noexcept {
			return 1 == cnt;
		}
		template <typename readInto_t>
		typename ::std::enable_if_t< ::std::is_same<train_data<typename readInto_t::value_type>, readInto_t>::value, bool> elements_count_correct(decltype(bin_file::HEADER::wFieldsCount) cnt)const noexcept {
			return 4 == cnt;
		}

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:


	};

}