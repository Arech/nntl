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

//this file contains implementation of ::boost::serialize Saving Archive Concept interface, that writes passed object into
//Matlab's .mat file using a Matlab's own libmat/libmx API. Restrictions: works only with nvp or named_struct objects!
// It's intended to create a data link between nntl and Matlab, but not to use a mat file as a permanent storage medium.
// Also it should be able to read train_data from .mat files

#include "../../utils/matlab.h"

#if NNTL_MATLAB_AVAILABLE

//#pragma warning(push)
//#pragma warning( disable : 4503 )
#include <vector>
#include <stack>
#include <bitset>
//#pragma warning(pop)
#include <cstdio>


#include "../../serialization/serialization.h"

#include "../../errors.h"
#include "../../interface/math/smatrix.h"

#include "../../utils/denormal_floats.h"
// #ATTENTION!
// MATLAB C API _MAY_ change how denormals are handled by a processor!
// Therefore for safety better call global_denormalized_floats_mode() after using MATLAB C API

namespace nntl_supp {

	struct _matfile_errs {
		enum ErrorCode {
			Success = 0,
			FailedToOpenFile,
			FileAlreadyOpened,
			InvalidOpenMode,
			ErrorClosingFile,
			NoFileOpened,
			NoStructToSave,
			WrongState_NameAlreadySet,
			WrongState_NoName,
			WrongState_NoStructAllocated,
			FailedToAllocateMxData,
			FailedToTransferVariable,
			FailedToCreateStructVariable,
			CantAddNewFieldToStruct,
			WrongVariableTypeHasBeenRead,
			EmptyVariableHasBeenRead,
			NoMemoryToCreateReadVariable,
			WrongStateStructUnsaved,
			WrongStateManualStructUnsaved,
			FailedToAssignDestinationVar //it can be used by calling code
		};

		//TODO: table lookup would be better here. But it's not essential
		static const nntl::strchar_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case Success: return NNTL_STRING("No error / success.");
			case FailedToOpenFile: return NNTL_STRING("Failed to open file");
			case FileAlreadyOpened: return NNTL_STRING("File has been already opened");
			case InvalidOpenMode: return NNTL_STRING("Invalid file open mode specified");
			case ErrorClosingFile: return NNTL_STRING("There was an error while closing file");
			case NoFileOpened: return NNTL_STRING("No file opened");
			case NoStructToSave: return NNTL_STRING("There's no struct to save. Probably the order of save_struct_begin/save_struct_end has been permutted.");
			case WrongState_NameAlreadySet: return NNTL_STRING("Wrong archive state, variable name already set. Looks like another variable saving in process!");
			case WrongState_NoName: return NNTL_STRING("No variable name available. Did you call archiver with nvp() or named_struct() object?");
			case WrongState_NoStructAllocated: return NNTL_STRING("Wrong state while trying to finish struct creation - no struct actually exist");
			case FailedToAllocateMxData: return NNTL_STRING("mx*() function failed to allocate data. Probably not enough memory");
			case FailedToTransferVariable: return NNTL_STRING("Failed to store/read variable to/from file");
			case FailedToCreateStructVariable: return NNTL_STRING("Failed to create struct variable");
			case CantAddNewFieldToStruct: return NNTL_STRING("Cant add a new field to current struct");
			case WrongVariableTypeHasBeenRead: return NNTL_STRING("Wrong (structure instead of numeric matrix) or inconvertable (complex instead of real) variable has been read");
			case EmptyVariableHasBeenRead: return NNTL_STRING("The variable been read is empty");
			case NoMemoryToCreateReadVariable: return NNTL_STRING("Variable has been read, but can't be created because of lack memory");
			case WrongStateStructUnsaved: return NNTL_STRING("Wrong state, structure stack is not empty");
			case WrongStateManualStructUnsaved: return NNTL_STRING("Wrong state, manual structure saving stack is not empty");
			case FailedToAssignDestinationVar: return NNTL_STRING("Failed to assign destination variable");

			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}
	};

	//just to get rid of unnecessary memory allocations during a work.
	//fixed array would fit even better, but ::std::array doesn't support necessary API, and it's not practical to implement it.
	template<typename BaseT, size_t reserveElements=4, typename ContT = ::std::vector<BaseT>>
	class stack_presized : public ::std::stack<BaseT, ContT> {
	public:
		stack_presized() : ::std::stack<BaseT,ContT>() {
			c.reserve(reserveElements);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template<typename ErrorsClassT = _matfile_errs>
	class _matfile_base 
		: public nntl::_has_last_error<ErrorsClassT>
		//, public nntl::serialization::_i_nntl_archive
		, protected nntl::math::smatrix_td 
	{
	protected:
		typedef stack_presized<mxArray*, 4> structure_stack_t;		

	protected:
		MATFile* m_matFile;
		const char* m_curVarName;
		structure_stack_t m_structureStack;

	public:
		typedef nntl::vec_len_t vec_len_t;
		typedef nntl::numel_cnt_t numel_cnt_t;

		enum FileOpenMode {
			Read,//read only mode
			WriteDelete,//write only, delete before opening
			UpdateKeepOld,//read&write, keep old file
			UpdateDelete//read&write, delete before opening
		};

		//#TODO probably should also check variable names lengths in a code
		static constexpr size_t MaxVarNameLength = 63;

	protected:

		//////////////////////////////////////////////////////////////////////////
		// Saving functions
		// update last_error only if an error occured!

		//type-less code prevents bloating
		ErrorCode _save_var(const vec_len_t r, const vec_len_t c, const void*const pData, const numel_cnt_t bytesCnt, const mxClassID dataId)noexcept {
			NNTL_ASSERT(r > 0 && c > 0 && pData && bytesCnt >= r* c);
			ErrorCode ec = ErrorCode::Success;
			if (m_matFile) {
				if (m_curVarName) {
					const auto pA = (mxLOGICAL_CLASS == dataId) ? mxCreateLogicalMatrix(r, c) : mxCreateNumericMatrix(r, c, dataId, mxREAL);
					if (pA) {
						memcpy(mxGetData(pA), pData, bytesCnt);
						ec = _save_to_file_or_struct(pA);
					} else ec = ErrorCode::FailedToAllocateMxData;
				} else ec = ErrorCode::WrongState_NoName;
			} else ec = ErrorCode::NoFileOpened;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			return ec;
		}

		// stores Matlab variable pA into either a file (if stack is empty; and it also frees memory and cleanup variable name)
		// or current struct variable (if non-empty)
		ErrorCode _save_to_file_or_struct(mxArray*const pA) {
			NNTL_ASSERT(m_matFile && m_curVarName);
			auto ec = ErrorCode::Success;
			if (m_structureStack.size() > 0) {
				const auto pStruct = m_structureStack.top();
				NNTL_ASSERT(mxIsStruct(pStruct));
				//we must check whether the field name exist and non empty. If so - free previously allocated value.
				auto fieldNum = mxGetFieldNumber(pStruct, m_curVarName);
				if (-1 == fieldNum) {
					//add new field
					fieldNum = mxAddField(pStruct, m_curVarName);
					if (-1 == fieldNum) ec = ErrorCode::CantAddNewFieldToStruct;
				} else {
					//check whether the old field contains value and free it if needed
					const auto oldVal = mxGetFieldByNumber(pStruct, 0, fieldNum);
					if (oldVal) mxDestroyArray(oldVal);
				}
				if (ErrorCode::Success == ec) {
					mxSetFieldByNumber(pStruct, 0, fieldNum, pA);
					m_curVarName = nullptr;
				}
				//we mustn't free pA here, it'll be automatically (?) deleted when the struct will be disposed
			} else {
				if (matPutVariable(m_matFile, m_curVarName, pA)) {
					ec = ErrorCode::FailedToTransferVariable;
				} else m_curVarName = nullptr;
				mxDestroyArray(pA);
			}
			return ec;
		}

		// loads variable with given in m_curVarName name from file or current struct field
		// returned pointer must be _free_loaded_var()
		mxArray*const _load_var(vec_len_t& r, vec_len_t& c, bool bExpectStruct=false) noexcept{
			mxArray* ret = nullptr;
			auto ec = ErrorCode::Success;
			if (m_matFile) {
				if (m_curVarName) {
					const bool bIsField = m_structureStack.size() > 0;
					if (bIsField) {
						//we currently reading a structure field, so returned pointer must not be mxDestroyArray()'ed
						ret = mxGetField(m_structureStack.top(), 0, m_curVarName);
						if (!ret) ec = ErrorCode::FailedToTransferVariable;
					} else {
						//reading top-level variable from file. Returned pointer must be mxDestroyArray()'ed
						ret = matGetVariable(m_matFile, m_curVarName);
						if (!ret) ec = ErrorCode::FailedToTransferVariable;
					}
					if (ret) {
						if ((bExpectStruct && mxIsStruct(ret)) || (!bExpectStruct && (mxIsNumeric(ret) || mxIsLogical(ret)))) {
							//obtaining dimensions
							r = static_cast<vec_len_t>(mxGetM(ret));
							c = static_cast<vec_len_t>(mxGetN(ret));
							if (r <= 0 || c <= 0) ec = ErrorCode::EmptyVariableHasBeenRead;
						}else { 
							ec = ErrorCode::WrongVariableTypeHasBeenRead;
							if (!bIsField)mxDestroyArray(ret);
							ret = nullptr;
						}
					}
				} else ec = ErrorCode::WrongState_NoName;
			} else ec = ErrorCode::NoFileOpened;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			return ret;
		}
		void _free_loaded_var(mxArray*const pA) noexcept{
			//must free only top-level variables read from file. Others will be destroyed as parts of loaded top-level structures
			if (m_structureStack.size() == 0 && pA) mxDestroyArray(pA);
		}
		
		//denormals safe
		ErrorCode _open(const char* fname, FileOpenMode fom) {
			NNTL_ASSERT(!m_matFile);
			if (m_matFile) return _set_last_error(ErrorCode::FileAlreadyOpened);
			const char* m;
			switch (fom) {
			case Read:
				m = "r";
				break;
			case WriteDelete:
				m = "w";
				break;
			case UpdateDelete:
				::std::remove(fname);
				{
					ErrorCode ec = ErrorCode::Success;
					const auto pt = matOpen(fname, "w");
					if (pt) {
						if (matClose(pt)) ec = ErrorCode::ErrorClosingFile;
					} else ec = ErrorCode::FailedToOpenFile;
					if (ErrorCode::Success != ec) return _set_last_error(ec);
				}
				m = "u";
				break;
			case UpdateKeepOld:
				m = "u";
				break;
			default:
				return _set_last_error(ErrorCode::InvalidOpenMode);
			}
			m_matFile = matOpen(fname, m);
			global_denormalized_floats_mode();
			return _set_last_error(m_matFile ? ErrorCode::Success : ErrorCode::FailedToOpenFile);
		}

	public:

		//denormals safe
		~_matfile_base()noexcept {
			close();
		}
		//denormals safe
		_matfile_base() : m_matFile(nullptr), m_curVarName(nullptr){
			mclmcrInitialize();//looks safe to call multiple times
			global_denormalized_floats_mode();
		}

		//denormals safe
		ErrorCode close()noexcept {
			NNTL_ASSERT(!m_curVarName);
			NNTL_ASSERT(!m_structureStack.size());

			ErrorCode ec = ErrorCode::Success;

			if (m_curVarName) ec = ErrorCode::WrongState_NameAlreadySet;
			m_curVarName = nullptr;

			//Trying to cleanup - however, this code doesn't guarantee a correct cleanup
			if (m_structureStack.size()) {
				ec = ErrorCode::WrongStateStructUnsaved;
				for (size_t i = m_structureStack.size() - 1; i >= 1; --i) {
					NNTL_ASSERT(m_structureStack.top());
					m_structureStack.pop();
				}
				NNTL_ASSERT(m_structureStack.top());
				mxDestroyArray(m_structureStack.top());
				m_structureStack.pop();
			}

			if (m_matFile) {
				if (matClose(m_matFile)) ec = ErrorCode::ErrorClosingFile;
				m_matFile = nullptr;
			}
			_set_last_error(ec);
			global_denormalized_floats_mode();
			return ec;
		}

		bool success()const noexcept {
			return ErrorCode::Success == get_last_error();
		}
		void mark_invalid_var()noexcept {
			_set_last_error(ErrorCode::FailedToAssignDestinationVar);
		}

		//////////////////////////////////////////////////////////////////////////
		// use the following function only when you know what you're doing
		void _drop_last_error()noexcept {
			m_curVarName = nullptr;
			_set_last_error(ErrorCode::Success);
		}
	};



	template<typename FinalChildT, bool bSavingArchive, typename ErrorsClassT = _matfile_errs>
	class _matfile : public _matfile_base<ErrorsClassT>, public nntl::serialization::simple_archive<FinalChildT, bSavingArchive> {
	protected:
		typedef nntl::serialization::simple_archive<FinalChildT, bSavingArchive> base_archive_t;

		template<typename BaseT>
		using type2id = nntl::utils::matlab::type2id<BaseT>;		

	protected:

	public:
		~_matfile()noexcept {}
		_matfile() {}

		//using base_archive_t::operator <<;
		//using base_archive_t::operator >>;

		//////////////////////////////////////////////////////////////////////////
		//denormals safe
		template<bool b = bSavingArchive>
		::std::enable_if_t<b, ErrorCode> open(const char* fname, FileOpenMode fom = FileOpenMode::WriteDelete)noexcept {
			if (FileOpenMode::Read == fom) return _set_last_error(ErrorCode::InvalidOpenMode);
			return _open(fname, fom);
		}
		template<bool b = bSavingArchive>
		::std::enable_if_t<b, ErrorCode> open(const ::std::string& fname, FileOpenMode fom = FileOpenMode::WriteDelete)noexcept {
			return open(fname.c_str(), fom);
		}
		template<bool b = bSavingArchive>
		::std::enable_if_t<!b, ErrorCode> open(const char* fname, FileOpenMode fom = FileOpenMode::Read)noexcept {
			if (FileOpenMode::WriteDelete == fom) return _set_last_error(ErrorCode::InvalidOpenMode);
			return _open(fname, fom);
		}
		template<bool b = bSavingArchive>
		::std::enable_if_t<!b, ErrorCode> open(const ::std::string& fname, FileOpenMode fom = FileOpenMode::Read)noexcept {
			return open(fname.c_str(), fom);
		}

		//////////////////////////////////////////////////////////////////////////
		// common operators
		
		//saving
		//denormals safe
		template<class T>
		self_ref_t operator<<(const ::boost::serialization::nvp< T > & t) {
			if (!success()) return get_self();

			if (m_matFile) {
				if (m_curVarName) {
					_set_last_error(ErrorCode::WrongState_NameAlreadySet);
				} else {
					m_curVarName = t.name();
					get_self() << t.const_value();
					NNTL_ASSERT(m_curVarName == nullptr);// m_curVarName must be cleaned by saver code if it succeded.
														 //DON'T _set_last_error here, or it'll overwrite the value set by get_self() << t.const_value();
				}
			} else _set_last_error(ErrorCode::NoFileOpened);
			global_denormalized_floats_mode();
			return get_self();
		}

		//denormals safe
		template<class T>
		self_ref_t operator<<(const nntl::serialization::named_struct< T > & t) {
			if (!success()) return get_self();

			auto ec = ErrorCode::Success;
			if (m_matFile) {
				if (m_curVarName) {
					ec = ErrorCode::WrongState_NameAlreadySet;
				} else {
					// create new struct variable and push it into the stack. After that all operations over t.const_value() must add new 
					// fields to the struct (not add variables to the file). Once done - write the struct to the file and pop the stack.
					auto pNewStruct = mxCreateStructMatrix(1, 1, 0, nullptr);
					if (pNewStruct) {
						m_structureStack.push(pNewStruct);
						get_self() << t.const_value();
						NNTL_ASSERT(m_curVarName == nullptr);// m_curVarName must be cleaned by saver code if it succeded.
															 //now save it to file

						//#BUGBUG: shouldn't save directly to file here, or we'll lose inner structs here
						//if (matPutVariable(m_matFile, t.name(), pNewStruct))  ec = ErrorCode::FailedToTransferVariable;
						//m_structureStack.pop();
						//mxDestroyArray(pNewStruct);

						m_structureStack.pop();
						m_curVarName = t.name();
						ec = _save_to_file_or_struct(pNewStruct);
						NNTL_ASSERT(m_curVarName == nullptr);

					} else ec = ErrorCode::FailedToCreateStructVariable;
				}
			} else ec = ErrorCode::NoFileOpened;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			global_denormalized_floats_mode();
			return get_self();
		}

		// loading
		//denormals safe
		template<class T>
		self_ref_t operator>>(::boost::serialization::nvp< T > & t) {
			if (!success()) return get_self();

			if (m_matFile) {
				if (m_curVarName) {
					_set_last_error(ErrorCode::WrongState_NameAlreadySet);
				} else {
					m_curVarName = t.name();
					get_self() >> t.value();
					NNTL_ASSERT(m_curVarName == nullptr || get_last_error() != ErrorCode::Success);// m_curVarName must be cleaned by saver code if it succeded.
														 //DON'T _set_last_error here, or it'll overwrite the value set by get_self() << t.const_value();
				}
			} else _set_last_error(ErrorCode::NoFileOpened);
			global_denormalized_floats_mode();
			return get_self();
		}
		template<class T>
		self_ref_t operator >> (::boost::serialization::nvp< T > && t) {
			return get_self().operator>> (t);
		}

		//denormals safe
		template<class T>
		self_ref_t operator>>(nntl::serialization::named_struct< T > & t) {
			if (!success()) return get_self();

			auto ec = ErrorCode::Success;
			if (m_matFile) {
				if (m_curVarName) {
					ec = ErrorCode::WrongState_NameAlreadySet;
				} else {
					// create new struct variable and push it into the stack. After that all operations over t.const_value() must add new 
					// fields to the struct (not add variables to the file). Once done - write the struct to the file and pop the stack.
					m_curVarName = t.name();
					vec_len_t r, c;
					auto pNewStruct = _load_var(r,c,true);
					m_curVarName = nullptr;
					if (pNewStruct) {
						if (1 == r && 1 == c) {
							m_structureStack.push(pNewStruct);
							get_self() >> t.value();
							NNTL_ASSERT(m_curVarName == nullptr || get_last_error() != ErrorCode::Success);// m_curVarName must be cleaned by saver code if it succeded.
																 //now save it to file
							m_structureStack.pop();
						} else  ec = ErrorCode::WrongVariableTypeHasBeenRead;
						_free_loaded_var(pNewStruct);
					}//else error has been set by _load_var
				}
			} else ec = ErrorCode::NoFileOpened;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			global_denormalized_floats_mode();
			return get_self();
		}

		//////////////////////////////////////////////////////////////////////////
		// Special saving functions
		// We will update last_error only if an error occured!

		//denormals safe
		template<typename BaseT>
		self_ref_t operator<<(const nntl::math::smatrix<BaseT>& t) {
			NNTL_ASSERT(!t.empty() && t.numel() > 0 );
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			_save_var(t.rows(), t.cols(), t.data(), t.byte_size(), type2id<BaseT>::id);
			global_denormalized_floats_mode();
			return get_self();
		}
		//smatrix_deform is expected to be in it's greatest possible size (or hidden data will be lost)
		//denormals safe
		template<typename BaseT>
		self_ref_t operator<<(const nntl::math::smatrix_deform<BaseT>& t) {
			return get_self().operator<<(static_cast<const nntl::math::smatrix<BaseT>&>(t));
		}

		//denormals safe
		template<size_t _Bits> self_ref_t operator<<(const ::std::bitset<_Bits>& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			//just a placeholder to make compiler happy. Don't need flags at the moment...
			//#todo: implement
			NNTL_UNREF(t);
			m_curVarName = nullptr;
			global_denormalized_floats_mode();
			return get_self();
		}

		//denormals safe
		template<typename T>
		::std::enable_if_t< ::std::is_arithmetic<T>::value, self_ref_t> operator<<(const T& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			_save_var(1, 1, &t, sizeof(T), type2id<T>::id);
			global_denormalized_floats_mode();
			return get_self();
		}
		//denormals safe
		template<typename T>
		::std::enable_if_t< ::std::is_enum<T>::value, self_ref_t> operator<<(const T& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			get_self() << static_cast<size_t>(t);
			return get_self();
		}
		//because we've just shadowed (const BaseT& t) signature, have to repeat default code here
		//denormals safe
		template<class T>
		::std::enable_if_t<!::std::is_arithmetic<T>::value && !::std::is_enum<T>::value, self_ref_t> operator<<(T const & t) {
			::boost::serialization::serialize_adl(get_self(), const_cast<T &>(t), ::boost::serialization::version< T >::value);
			return get_self();
		}
		
		

		//////////////////////////////////////////////////////////////////////////
		// Loading functions
		// We will update last_error only if an error occurred!
		// //denormals safe
		template<typename BaseT>
		self_ref_t operator>>(nntl::math::smatrix<BaseT>& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForLoad() and use nvp/named_struct to address data to read!");
			vec_len_t r, c;
			auto pVar = _load_var(r, c);
			if (pVar) {
				auto ec = ErrorCode::Success;
				t.clear();
				if (0 == strcmp("train_x", m_curVarName) || 0 == strcmp("test_x", m_curVarName)) {
					if (--c > 0) {
						t.will_emulate_biases();
					} else ec = ErrorCode::EmptyVariableHasBeenRead;
				} else t.dont_emulate_biases();
				if (ErrorCode::Success == ec) {
					if (t.resize(r,c)) {//checking type and doing a data copying
						if (utils::matlab::fill_data_no_bias(t, pVar)) {
							m_curVarName = nullptr;
						}else ec = ErrorCode::WrongVariableTypeHasBeenRead;
					} else ec = ErrorCode::NoMemoryToCreateReadVariable;
				}
				_free_loaded_var(pVar);
				if (ErrorCode::Success != ec) _set_last_error(ec);
			}//else error has already been updated
			global_denormalized_floats_mode();
			return get_self();
		}

		//denormals safe
		//greatest size of smatrix_deform will correspond to the size read.
		template<typename BaseT>
		self_ref_t operator>>(nntl::math::smatrix_deform<BaseT>& t) {
			get_self().operator>>(static_cast<nntl::math::smatrix<BaseT>&>(t));
			if (ErrorCode::Success == get_last_error()) t.update_on_hidden_resize();
			return get_self();
		}
		//denormals safe
		template<typename BaseT>
		::std::enable_if_t< ::std::is_arithmetic<BaseT>::value, self_ref_t> operator>>(BaseT& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForLoad() and use nvp/named_struct to address data to read!");
			vec_len_t r, c;
			auto pVar = _load_var(r, c);
			if (pVar) {
				if (1 == r && 1 == c && type2id<BaseT>::id == mxGetClassID(pVar)) {
					t = * static_cast<const BaseT*>(mxGetData(pVar));
					m_curVarName = nullptr;
				} else _set_last_error(ErrorCode::WrongVariableTypeHasBeenRead);
				_free_loaded_var(pVar);
			}//else error has already been updated
			global_denormalized_floats_mode();
			return get_self();
		}
		//denormals safe
		//because we've just shadowed (const BaseT& t) signature, have to repeat default code here
		template<class T>
		::std::enable_if_t<!::std::is_arithmetic<T>::value, self_ref_t> operator>>(T & t) {
			::boost::serialization::serialize_adl(get_self(), t, ::boost::serialization::version< T >::value);
			return get_self();
		}
	};
	
	//////////////////////////////////////////////////////////////////////////
	//extended API for saving archive. It is required for inspectors::dumper only at this moment
	template<typename FinalChildT, typename ErrorsClassT = _matfile_errs>
	class _matfile_savingEx : public _matfile<FinalChildT, true, ErrorsClassT> {
	private:
		typedef _matfile<FinalChildT, true, ErrorsClassT> _base_class;

	protected:
		//typedefs necessary for breaking nested calls to save_struct_begin()/save_struct_end() into parallel calls
		typedef ::std::pair<const ::std::string, mxArray*const> nested_structs_info_t;
		typedef stack_presized<nested_structs_info_t, 4> nested_structs_stack_t;

	protected:
		nested_structs_stack_t m_nestedStructs;


	public:
		//denormals safe
		~_matfile_savingEx()noexcept{
			close();
		}
		_matfile_savingEx()noexcept{}

		//denormals safe
		ErrorCode close()noexcept {
			//The code doesn't guarantee a correct cleanup, because it depends on where an error occured
			NNTL_ASSERT(!m_nestedStructs.size());
			NNTL_ASSERT(!m_curVarName);
			NNTL_ASSERT(!m_structureStack.size());

			ErrorCode ec = ErrorCode::Success;
			if (m_nestedStructs.size()) {
				ec = ErrorCode::WrongStateManualStructUnsaved;
				const auto s = m_nestedStructs.size();
				if (m_structureStack.size()) {
					for (size_t i = 0; i < s; ++i)  m_nestedStructs.pop();
				} else {
					for (size_t i = 0; i < s; ++i) {
						const auto& t = m_nestedStructs.top();
						if (t.second) mxDestroyArray(t.second);
						m_nestedStructs.pop();
					}
				}
			}
			const auto ec2 = _base_class::close();
			return ec == ErrorCode::Success ? ec2 : _set_last_error(ec);
		}

		//denormals safe
		//MUST be accompanied by save_struct_end(). Permits nested calls for a different structures
		// If the bUpdateIfExist is specified, then tries to load the struct from file first
		// If the bDontNest is specified, then the new struct will be created on the same level of nesting as the
		//		parental struct (the one, that is already presents on the top of m_structureStack)
		ErrorCode save_struct_begin(::std::string&& structName, const bool bUpdateIfExist, const bool bDontNest)noexcept {
			if (!success()) return get_last_error();

			auto ec = ErrorCode::Success;
			if (m_matFile) {
				if (m_curVarName) {
					ec = ErrorCode::WrongState_NameAlreadySet;
				} else {
					mxArray* pNewStruct = nullptr;
					mxArray* pOldHead;
					if (bDontNest && m_structureStack.size()) {
						pOldHead = m_structureStack.top();
						m_structureStack.pop();
					} else pOldHead = nullptr;

					if (bUpdateIfExist) {
						m_curVarName = structName.c_str();
						NNTL_ASSERT(m_curVarName);
						vec_len_t r, c;
						pNewStruct = _load_var(r, c, true);						
						m_curVarName = nullptr;
						if (pNewStruct) {
							if (1 != r || 1 != c) {
								// ec = ErrorCode::WrongVariableTypeHasBeenRead;
								_free_loaded_var(pNewStruct);
								pNewStruct = nullptr;
							}
						} else {
							//error has been set by the _load_var(), therefore overwriting it
							_set_last_error(ErrorCode::Success);
						}
					}
					if (!pNewStruct) {
						// create new struct variable and push it into the stack. After that all operations over t.const_value() must add new 
						// fields to the struct (not add variables to the file). Once done - write the struct to the file and pop the stack.
						pNewStruct = mxCreateStructMatrix(1, 1, 0, nullptr);
					}
					if (pNewStruct) {
						m_nestedStructs.push(::std::make_pair(::std::forward<::std::string>(structName), pOldHead));
						m_structureStack.push(pNewStruct);
					} else {
						if (ErrorCode::Success == ec) ec = ErrorCode::FailedToCreateStructVariable;
						if (pOldHead) m_structureStack.push(pOldHead);
					}
				}
			} else ec = ErrorCode::NoFileOpened;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			global_denormalized_floats_mode();
			return ec;
		}
		//denormals safe
		ErrorCode save_struct_end()noexcept {
			if (!success()) return get_last_error();
			auto ec = ErrorCode::Success;
			if (m_nestedStructs.size()) {
				if (m_matFile) {
					if (m_curVarName) {
						ec = ErrorCode::WrongState_NameAlreadySet;
					} else {
						if (m_structureStack.size() > 0) {
							//saving current struct
							auto pNewStruct = m_structureStack.top();
							NNTL_ASSERT(pNewStruct);
							m_structureStack.pop();
							const auto& curStructInfo = m_nestedStructs.top();
							m_curVarName = curStructInfo.first.c_str();
							ec = _save_to_file_or_struct(pNewStruct);
							NNTL_ASSERT(m_curVarName == nullptr);

							//if (bFree) _free_loaded_var(pNewStruct);
							//memory is freed by _save_to_file_or_struct() (either directly by calling to mxDestroyArray()
							// or indirectly by storing pNewStruct as a field of an enclosing struct)
							
							//restoring previously created struct
							if (curStructInfo.second) {
								m_structureStack.push(curStructInfo.second);
							}
							m_nestedStructs.pop();
						} else ec = ErrorCode::WrongState_NoStructAllocated;
					}
				} else ec = ErrorCode::NoFileOpened;
			} else ec = ErrorCode::NoStructToSave;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			global_denormalized_floats_mode();
			return ec;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	template<typename SerializationOptionsEnumT = serialization::CommonOptions>
	class omatfile final 
		: public _matfile<omatfile<SerializationOptionsEnumT>, true, _matfile_errs>
		, public nntl::utils::binary_options<SerializationOptionsEnumT>
	{
	public:
		~omatfile()  {}
		omatfile() {
			turn_on_all_options();
		}
	};

	template<typename SerializationOptionsEnumT = serialization::CommonOptions>
	class omatfileEx final
		: public _matfile_savingEx<omatfileEx<SerializationOptionsEnumT>, _matfile_errs>
		, public nntl::utils::binary_options<SerializationOptionsEnumT>
	{
	public:
		~omatfileEx() {}
		omatfileEx() {
			turn_on_all_options();
		}
	};

	template<typename SerializationOptionsEnumT = serialization::CommonOptions>
	class imatfile final 
		: public _matfile<imatfile<SerializationOptionsEnumT>, false, _matfile_errs>
		, public nntl::utils::binary_options<SerializationOptionsEnumT>
	{
	public:
		~imatfile() {}
		imatfile() {
			turn_on_all_options();
		}
	};
}

#endif