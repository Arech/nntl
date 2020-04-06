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
			WORD wVersionNum; //format version number
			WORD wFieldsCount;//total count of all fields besides HEADER

			WORD wSeqClassCount;//if non zero, then expecting file to contain a set of sequences belonging to this amount of classes
			//there must be this count of CLASS_ENTRY structures immediately after HEADER

			static constexpr DWORD sSignature = 'ltnn';
			static constexpr WORD sLatestVersion = 0;
		};
		static_assert(10 == sizeof(HEADER), "WTF??");

		struct CLASS_ENTRY {
			WORD wTotalCountOfDifferentSequences;
		};
		static_assert(2 == sizeof(CLASS_ENTRY), "WTF??");

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
			FailedToReadSeqClassHeader,
			WrongHeaderSignature,
			UnsupportedFormatVersion,
			WrongElementsCount,
			InvalidSeqClassCount,
			InvalidSeqClassElement,
			FailedToReadFieldEntry,
			UnsupportedIncorrectDataType,
			FailedToAllocateMemoryForDataConvertion,
			InvalidDataSize,
			FailedToReadData,

			UnknownFieldName,
			FieldHasBeenRead,
			FailedToMakeTDOutOfReadData,
			
			FailedToPreallocateSeqData,
			FailedToChangeFilePtr,
			FailedToParseClassId,
			IncoherentSeqClassCount,
			IncoherentMetaAndContent,

			MemoryAllocationFailed
		};

		//TODO: table lookup would be better here. But it's not essential
		static const nntl::strchar_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case Success: return NNTL_STRING("No error / success.");
			case FailedToOpenFile: return NNTL_STRING("Failed to open file.");
			case FailedToReadHeader: return NNTL_STRING("Failed to read header.");
			case FailedToReadSeqClassHeader: return NNTL_STRING("Failed to read CLASS_ENTRY.");
			case WrongHeaderSignature: return NNTL_STRING("Wrong header signature.");
			case UnsupportedFormatVersion: return NNTL_STRING("Unknown or unsupported format version.");
			case WrongElementsCount: return NNTL_STRING("File contains unsupported elements count");
			case InvalidSeqClassCount: return NNTL_STRING("Invalid sequence class count");
			case InvalidSeqClassElement: return NNTL_STRING("Invalid sequence class element entry");
			case FailedToReadFieldEntry: return NNTL_STRING("Failed to read field entry");
			case UnsupportedIncorrectDataType:  return NNTL_STRING("Unsupported or incorrect data type");
			case FailedToAllocateMemoryForDataConvertion: return NNTL_STRING("Failed to allocate memory necessary for data type convertion");
			case InvalidDataSize: return NNTL_STRING("Invalid data size");
			case FailedToReadData: return NNTL_STRING("Failed to read data");			
			case UnknownFieldName: return NNTL_STRING("Unknown field name found");
			case FieldHasBeenRead: return NNTL_STRING("Two fields with the same name found!");
			case FailedToMakeTDOutOfReadData: return NNTL_STRING("Failed to assemble train_data out of read data. Probably not all necessary data have been read!");
			
			case FailedToPreallocateSeqData: return NNTL_STRING("Failed to preallocate sequence data");
			case FailedToChangeFilePtr: return NNTL_STRING("Failed to change file pointer");
			case FailedToParseClassId: return NNTL_STRING("Failed to parse class id");
			case IncoherentSeqClassCount: return NNTL_STRING("Incoherent sequence class count");
			case IncoherentMetaAndContent:return NNTL_STRING("Incoherent file header description and content");

			case MemoryAllocationFailed: return NNTL_STRING("Not Enough Memory");

			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}
	};


	class binfile : public nntl::_has_last_error<_binfile_errs>, protected nntl::math::smatrix_td {
 	protected:
		typedef ::nntl::vec_len_t vec_len_t;
		typedef ::nntl::numel_cnt_t numel_cnt_t;

		template<typename T_> using smatrix = ::nntl::math::smatrix<T_>;
		template<typename T_> using smatrix_deform = ::nntl::math::smatrix_deform<T_>;
		template<typename T_> using train_data = ::nntl::simple_train_data_stor<T_>;
		template<typename T_> using seq_data = ::nntl::seq_data<T_>;

	public:
		~binfile()noexcept {}
		binfile()noexcept{}

		// read fname, parses it as jsonized struct TD into dest var, which can be either nntl::train_data or nntl::train_data::mtx_t or mtxdef_t
		// If readInto_t == nntl::train_data, then all X data will be created with emulateBiases() feature and bMakeMtxBiased param will be ignored
		template <typename readInto_t>
		const ErrorCode read(const char* fname, readInto_t& dest) {
			static_assert(
				::std::is_base_of<train_data<typename readInto_t::value_type>, readInto_t>::value
				|| ::std::is_same<seq_data<typename readInto_t::value_type>, readInto_t>::value
				|| ::std::is_same<smatrix<typename readInto_t::value_type>, readInto_t>::value
				|| ::std::is_same<smatrix_deform<typename readInto_t::value_type>, readInto_t>::value,
				"Only ::nntl::_impl::simple_train_data_stor derived, or seq_data, math::smatrix or math::smatrix_deform is supported as readInto_t template parameter");

			FILE* fp=nullptr;
			if (fopen_s(&fp, fname, NNTL_STRING("rbS")) || nullptr == fp) return _set_last_error(ErrorCode::FailedToOpenFile);
			nntl::utils::scope_exit on_exit([&fp]() {
				if (fp) {
					fclose(fp);
					fp = nullptr;
				}
			});

			bin_file::HEADER hdr;

#pragma warning(disable:28020)
			//MSVC SAL goes "slightly" mad here
			if (1 != fread_s(&hdr, sizeof(hdr), sizeof(hdr), 1, fp)) return _set_last_error(ErrorCode::FailedToReadHeader);
#pragma warning(default:28020)
			if (hdr.sSignature != hdr.dwSignature) return _set_last_error(ErrorCode::WrongHeaderSignature);
			if (hdr.sLatestVersion != hdr.wVersionNum) return _set_last_error(ErrorCode::UnsupportedFormatVersion);

			return _read_into(fp, dest, static_cast<int>(hdr.wFieldsCount), static_cast<int>(hdr.wSeqClassCount));
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
		//////////////////////////////////////////////////////////////////////////
		template<typename T_>
		ErrorCode _read_into(FILE* fp, seq_data<T_>& dest, const int totalFieldsCount, const int totalClassCount)noexcept {
			if (totalFieldsCount <= totalClassCount) return _set_last_error(ErrorCode::WrongElementsCount);

			//reading count of sequences for each class (CLASS_ENTRY)
			::std::vector<vec_len_t> clsSeqCnt(totalClassCount);
			int totalElsToExpect = 0;
			for (int i = 0; i < totalClassCount; ++i) {
				bin_file::CLASS_ENTRY clsEntry;

			#pragma warning(disable:28020)
				//MSVC SAL goes "slightly" mad here
				if (1 != fread_s(&clsEntry, sizeof(clsEntry), sizeof(clsEntry), 1, fp)) return _set_last_error(ErrorCode::FailedToReadSeqClassHeader);
			#pragma warning(default:28020)
				const auto sc = static_cast<vec_len_t>(clsEntry.wTotalCountOfDifferentSequences);
				if (sc<=0) return _set_last_error(ErrorCode::InvalidSeqClassElement);
				clsSeqCnt[i] = sc;
				totalElsToExpect += static_cast<int>(sc);
			}
			if (totalElsToExpect + totalClassCount != totalFieldsCount)return _set_last_error(ErrorCode::WrongElementsCount);

			if (! dest._prealloc_classes(::std::move(clsSeqCnt))) return _set_last_error(ErrorCode::FailedToPreallocateSeqData);

			//reading FIELD_ENTRY and deciding a class of a sequence by parsing field name (to let store data in the file in any order)
			for (int i = 0; i < totalElsToExpect; ++i) {
				//reading entry name first (actually it's easier to read whole entry)
				const auto fBeg = _ftelli64(fp);

			#pragma warning(disable : 4815)
				bin_file::FIELD_ENTRY fe;
			#pragma warning(default : 4815)

			#pragma warning(disable:28020)
				//MSVC SAL goes "slightly" mad here
				if (1 != fread_s(&fe, sizeof(fe), sizeof(fe), 1, fp)) return _set_last_error(ErrorCode::FailedToReadFieldEntry);
			#pragma warning(default:28020)
				fe.bDataType = 0;
				
				int classId;
				if (1 != sscanf_s(fe.szName, "c%d", &classId) || classId < 1) return _set_last_error(ErrorCode::FailedToParseClassId);

				auto pMtx4Seq = dest._next_mtx_for_class(static_cast<size_t>(classId - 1));
				if (!pMtx4Seq) return _set_last_error(ErrorCode::IncoherentSeqClassCount);

				//reading whole entry
				if (0 != _fseeki64(fp, fBeg, SEEK_SET)) return _set_last_error(ErrorCode::FailedToChangeFilePtr);
				
				const auto ec = _read_field_entry(fp, *pMtx4Seq);
				if (ErrorCode::Success != ec) return ec;
			}

			if (dest.empty()) return _set_last_error(ErrorCode::IncoherentMetaAndContent);

			return ErrorCode::Success;
		}

		template<typename T_>
		ErrorCode _read_into(FILE* fp, train_data<T_>& dest, const int totalFieldsCount, const int totalClassCount)noexcept {
			if (totalFieldsCount != total_members) return _set_last_error(ErrorCode::WrongElementsCount);
			if (0 != totalClassCount) return _set_last_error(ErrorCode::InvalidSeqClassCount);

			typedef smatrix_deform<T_> mtxdef_t;
			mtxdef_t mtxs[total_members];

			for (unsigned nel = 0; nel < _root_members::total_members; ++nel) {
				mtxdef_t m;
				_root_members fieldId;
				const auto err = _read_field_entry(fp, m, &fieldId);
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
		ErrorCode _read_into(FILE* fp, smatrix<T_>& dest, const int totalFieldsCount, const int totalClassCount)noexcept {
			NNTL_ASSERT(!dest.bDontManageStorage());
			NNTL_ASSERT(dest.empty());
			if (totalFieldsCount != 1) return _set_last_error(ErrorCode::WrongElementsCount);
			if (0 != totalClassCount) return _set_last_error(ErrorCode::InvalidSeqClassCount);
			return _read_field_entry(fp, dest);
		}

		template<typename T_>
		ErrorCode _read_into(FILE* fp, smatrix_deform<T_>& dest, const int totalFieldsCount, const int totalClassCount)noexcept {
			NNTL_ASSERT(!dest.bDontManageStorage());
			NNTL_ASSERT(dest.empty());
			if (totalFieldsCount != 1) return _set_last_error(ErrorCode::WrongElementsCount);
			if (0 != totalClassCount) return _set_last_error(ErrorCode::InvalidSeqClassCount);
			return _read_field_entry(fp, dest);
		}

		template<typename T_>
		ErrorCode _read_field_entry(FILE* fp, smatrix<T_>& m, _root_members* pTdFieldId = nullptr)noexcept {
			return _impl_read_field_entry(fp, m, pTdFieldId);
		}
		template<typename T_>
		ErrorCode _read_field_entry(FILE* fp, smatrix_deform<T_>& m, _root_members* pTdFieldId = nullptr)noexcept {
			NNTL_ASSERT(m.empty());//must be empty or we will spoil it with update_on_hidden_resize()
			const auto ec = _impl_read_field_entry(fp, m, pTdFieldId);
			m.update_on_hidden_resize();
			return ec;
		}

		//////////////////////////////////////////////////////////////////////////
		template<typename T_>
		ErrorCode _impl_read_field_entry(FILE* fp, smatrix<T_>& m, _root_members* pTdFieldId = nullptr)noexcept {
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
			if (pTdFieldId) {
				const auto fieldId = _name2id(fe.szName);
				*pTdFieldId = fieldId;
				if(_root_members::total_members == fieldId) return _set_last_error(ErrorCode::UnknownFieldName);
				if (_root_members::train_x == fieldId || _root_members::test_x == fieldId) {
					m.will_emulate_biases();
				} else m.dont_emulate_biases();
			}

			if (!m.resize(static_cast<vec_len_t>(fe.dwRows), static_cast<vec_len_t>(fe.dwCols)))
				return _set_last_error(ErrorCode::MemoryAllocationFailed);

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

	};

}