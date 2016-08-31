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

//this file contains implementation of boost::serialize Saving Archive Concept interface, that writes passed object into
//Matlab's .mat file using a Matlab's own libmat/libmx API. Restrictions: works only with nvp or named_struct objects!
// It's intended to create a data link between nntl and Matlab, but not to use a mat file as a permanent storage medium.
// Also it should be able to read train_data from .mat files

#ifdef NNTL_MATLAB_AVAILABLE

#include "../../utils/matlab.h"

#include "../../serialization/serialization.h"

#include "../../errors.h"
#include "../../interface/math/smatrix.h"

//#pragma warning(push)
//#pragma warning( disable : 4503 )
#include <vector>
#include <stack>
#include <bitset>
//#pragma warning(pop)

namespace nntl_supp {

	struct _matfile_errs {
		enum ErrorCode {
			Success = 0,
			FailedToOpenFile,
			ErrorClosingFile,
			NoFileOpened,
			WrongState_NameAlreadySet,
			WrongState_NoName,
			FailedToAllocateMxData,
			FailedToTransferVariable,
			FailedToCreateStructVariable,
			CantAddNewFieldToStruct,
			WrongVariableTypeHasBeenRead,
			EmptyVariableHasBeenRead,
			NoMemoryToCreateReadVariable

		};

		//TODO: table lookup would be better here. But it's not essential
		static const nntl::strchar_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case Success: return NNTL_STRING("No error / success.");
			case FailedToOpenFile: return NNTL_STRING("Failed to open file");
			case ErrorClosingFile: return NNTL_STRING("There was an error while closing file");
			case NoFileOpened: return NNTL_STRING("No file opened");
			case WrongState_NameAlreadySet: return NNTL_STRING("Wrong archive state, variable name already set. Looks like another variable saving in process!");
			case WrongState_NoName: return NNTL_STRING("No variable name available. Did you call archiver with nvp() or named_struct() object?");
			case FailedToAllocateMxData: return NNTL_STRING("mx*() function failed to allocate data. Probably not enough memory");
			case FailedToTransferVariable: return NNTL_STRING("Failed to store/read variable to/from file");
			case FailedToCreateStructVariable: return NNTL_STRING("Failed to create struct variable");
			case CantAddNewFieldToStruct: return NNTL_STRING("Cant add a new field to current struct");
			case WrongVariableTypeHasBeenRead: return NNTL_STRING("Wrong (structure instead of numeric matrix) or inconvertable (complex instead of real) variable has been read");
			case EmptyVariableHasBeenRead: return NNTL_STRING("The variable been read is empty");
			case NoMemoryToCreateReadVariable: return NNTL_STRING("Variable has been read, but can't be created because of lack memory");

			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}
	};

	//just to get rid of unnecessary memory allocations during a work.
	//fixed array would fit even better, but std::array doesn't support necessary API, and it's not practical to implement it.
	template<typename BaseT, size_t reserveElements=4, typename ContT = std::vector<BaseT>>
	class stack_presized : public std::stack<BaseT, ContT> {
	public:
		stack_presized() : std::stack<BaseT,ContT>() {
			c.reserve(reserveElements);
		}
	};

	template<typename ErrorsClassT = _matfile_errs>
	class _matfile_base : public nntl::_has_last_error<ErrorsClassT>, protected nntl::math::simple_matrix_td {
	protected:
		typedef stack_presized<mxArray*, 4> structure_stack_t;		

	protected:
		MATFile* m_matFile;
		const char* m_curVarName;
		structure_stack_t m_structureStack;

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

	public:
		~_matfile_base()noexcept {
			close();
		}
		_matfile_base() : m_matFile(nullptr), m_curVarName(nullptr){
			mclmcrInitialize();//looks safe to call multiple times
		}

		ErrorCode openForSave(const char* fname)noexcept {
			m_matFile = matOpen(fname, "w");
			return _set_last_error(m_matFile ? ErrorCode::Success : ErrorCode::FailedToOpenFile);
		}
		ErrorCode openForLoad(const char* fname)noexcept {
			m_matFile = matOpen(fname, "r");
			return _set_last_error(m_matFile ? ErrorCode::Success : ErrorCode::FailedToOpenFile);
		}
		ErrorCode openForSave(const std::string& fname)noexcept { return openForSave(fname.c_str()); }
		ErrorCode openForLoad(const std::string& fname)noexcept { return openForLoad(fname.c_str()); }

		ErrorCode close()noexcept {
			ErrorCode ec = ErrorCode::Success;
			if (m_matFile) {
				if (matClose(m_matFile)) ec = ErrorCode::ErrorClosingFile;
				m_matFile = nullptr;
			}// else ec = ErrorCode::NoFileOpened;
			 //if (ErrorCode::Success != ec) 
			_set_last_error(ec);
			return ec;
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

		using base_archive_t::operator <<;
		using base_archive_t::operator >>;

		//////////////////////////////////////////////////////////////////////////
		// common operators
		
		//saving
		template<class T>
		self_ref_t operator<<(const boost::serialization::nvp< T > & t) {
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
			return get_self();
		}
		template<class T>
		self_ref_t operator<<(const nntl::serialization::named_struct< T > & t) {
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
			return get_self();
		}

		// loading
		template<class T>
		self_ref_t operator>>(boost::serialization::nvp< T > & t) {
			if (m_matFile) {
				if (m_curVarName) {
					_set_last_error(ErrorCode::WrongState_NameAlreadySet);
				} else {
					m_curVarName = t.name();
					get_self() >> t.value();
					NNTL_ASSERT(m_curVarName == nullptr);// m_curVarName must be cleaned by saver code if it succeded.
														 //DON'T _set_last_error here, or it'll overwrite the value set by get_self() << t.const_value();
				}
			} else _set_last_error(ErrorCode::NoFileOpened);
			return get_self();
		}
		template<class T>
		self_ref_t operator>>(nntl::serialization::named_struct< T > & t) {
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
							NNTL_ASSERT(m_curVarName == nullptr);// m_curVarName must be cleaned by saver code if it succeded.
																 //now save it to file
							m_structureStack.pop();
						} else  ec = ErrorCode::WrongVariableTypeHasBeenRead;
						_free_loaded_var(pNewStruct);
					}
				}
			} else ec = ErrorCode::NoFileOpened;
			if (ErrorCode::Success != ec) _set_last_error(ec);
			return get_self();
		}

		//////////////////////////////////////////////////////////////////////////
		// Special saving functions
		// We will update last_error only if an error occured!

		template<typename BaseT>
		self_ref_t operator<<(const nntl::math::simple_matrix<BaseT>& t) {
			NNTL_ASSERT(!t.empty() && t.numel() > 0 );
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			_save_var(t.rows(), t.cols(), t.data(), t.byte_size(), type2id<BaseT>::id);
			return get_self();
		}
		//simple_matrix_deformable is expected to be in it's greatest possible size (or hidden data will be lost)
		template<typename BaseT>
		self_ref_t operator<<(const nntl::math::simple_matrix_deformable<BaseT>& t) {
			return get_self().operator<<(static_cast<const nntl::math::simple_matrix<BaseT>&>(t));
		}

		template<size_t _Bits> self_ref_t operator<<(const std::bitset<_Bits>& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			//just a placeholder to make compiler happy. Don't need flags at the moment...
			//#todo: implement
			m_curVarName = nullptr;
			return get_self();
		}

		template<typename T>
		std::enable_if_t< std::is_arithmetic<T>::value, self_ref_t> operator<<(const T& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			_save_var(1, 1, &t, sizeof(T), type2id<T>::id);
			return get_self();
		}
		template<typename T>
		std::enable_if_t< std::is_enum<T>::value, self_ref_t> operator<<(const T& t) {
			NNTL_ASSERT((m_matFile && m_curVarName) || !"Open file with openForSave() and use nvp/named_struct to pass data for saving!");
			get_self() << static_cast<size_t>(t);
			return get_self();
		}
		//because we've just shadowed (const BaseT& t) signature, have to repeat default code here
		template<class T>
		std::enable_if_t<!std::is_arithmetic<T>::value && !std::is_enum<T>::value, self_ref_t> operator<<(T const & t) {
			boost::serialization::serialize_adl(get_self(), const_cast<T &>(t), ::boost::serialization::version< T >::value);
			return get_self();
		}
		
		

		//////////////////////////////////////////////////////////////////////////
		// Loading functions
		// We will update last_error only if an error occurred!
		template<typename BaseT>
		self_ref_t operator>>(nntl::math::simple_matrix<BaseT>& t) {
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
			return get_self();
		}

		//greatest size of simple_matrix_deformable will correspond to the size read.
		template<typename BaseT>
		self_ref_t operator>>(nntl::math::simple_matrix_deformable<BaseT>& t) {
			get_self().operator>>(static_cast<nntl::math::simple_matrix<BaseT>&>(t));
			if (ErrorCode::Success == get_last_error()) t.update_on_hidden_resize();
			return get_self();
		}

		template<typename BaseT>
		std::enable_if_t< std::is_arithmetic<BaseT>::value, self_ref_t> operator>>(BaseT& t) {
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
			return get_self();
		}
		//because we've just shadowed (const BaseT& t) signature, have to repeat default code here
		template<class T>
		std::enable_if_t<!std::is_arithmetic<T>::value, self_ref_t> operator>>(T & t) {
			boost::serialization::serialize_adl(get_self(), t, ::boost::serialization::version< T >::value);
			return get_self();
		}
	};

	template<typename SerializationOptionsEnumT = serialization::CommonOptions>
	class omatfile final : public _matfile<omatfile<SerializationOptionsEnumT>,true>, public nntl::utils::binary_options<SerializationOptionsEnumT>{
	public:
		typedef _matfile<omatfile<SerializationOptionsEnumT>, true> base_archive_t;
		using base_archive_t::operator <<;
		using base_archive_t::operator&;
		~omatfile()  {}
		omatfile() {
			turn_on_all_options();
		}

	};

	template<typename SerializationOptionsEnumT = serialization::CommonOptions>
	class imatfile final : public _matfile<imatfile<SerializationOptionsEnumT>, false>, public nntl::utils::binary_options<SerializationOptionsEnumT> {
	public:
		typedef _matfile<imatfile<SerializationOptionsEnumT>, false> base_archive_t;
		using base_archive_t::operator >>;
		using base_archive_t::operator&;
		~imatfile() {}
		imatfile() {
			turn_on_all_options();
		}
	};
}

#endif