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

//This file contains implementation of some simplest generic-purpose(!) math algorithms over the data
// inside smatrix class object

#include "../_i_threads.h"
#include "smatrix.h"
#include "smath_thr.h"
#include <algorithm>
#include <numeric>
#include "../../utils/denormal_floats.h"

namespace nntl {
namespace math {

	//////////////////////////////////////////////////////////////////////////
	// _st versions of functions MUST NOT call generic and/or multithreaded function implementations (ONLY _st).
	//		(They may be used in future in some parallel algorithms)
	// _mt and generic function versions may use any suitable implementations.
	// generic and _st versions MUST accept any datasizes. However, _mt as well as specializations,
	//		such as _mt_cw MAY put restrictions on acceptable data sizes.
	//
	//////////////////////////////////////////////////////////////////////////
	// Notes on correctness testing:
	// - each public function (generic, _st, _mt and so on) MUST be tested for correctness using some easy to read etalon implementation
	// - test function must be parametrized by types used. At least every type used in computations must be tested
	// - matrices with&without bias column should be tested where appropriate
	// 
	// Notes on performance testing:
	// - yes, we need run-time profiling instead of hand-tuning thresholds. no time for it now. TBD
	// - char data type should specifically be tested due to possible strict aliasing issues
	// 

	// #todo drop <typename RealT> class template parameter in favor of corresponding function template parameter!
	template<typename RealT, typename iThreadsT, typename ThresholdsT, typename FinalPolymorphChild>
	class _SMath {
		static_assert(::std::is_base_of<threads::_i_threads<RealT, typename iThreadsT::range_t>, iThreadsT>::value, "iThreads must implement threads::_i_threads");

	public:
		typedef FinalPolymorphChild self_t;
		NNTL_METHODS_SELF_CHECKED( (::std::is_base_of<_SMath<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild>, FinalPolymorphChild>::value)
			, "FinalPolymorphChild must derive from _SMath<RealT, iThreadsT, FinalPolymorphChild>" );

		//OBSOLETE! Note that actually any math function should not depend on RealT/real_t type and dependent types,
		// but must be parametrized with it so we'd be able to call it with any suitable data type.
		// Don't use it in new code
		typedef RealT real_t;
		typedef smatrix<real_t> realmtx_t;
		typedef smatrix_deform<real_t> realmtxdef_t;

		typedef s_rowcol_range rowcol_range;
		typedef s_elems_range elms_range;
		typedef s_vec_range vec_range;
		//#WARNING use proxy variables instead of members of a structure, received by reference in performance sensitive code

		// here's small guide for using rowcol_range/elms_range :
		// Every function that should be multithreaded should have 3 (!!!) functions:
		//	.1 func_st() which does purely singlethreaded processing. If needed, it must be aware of other threads running and
		//			process data in a multithreading-safe way. This function is mostly a thunk to _ifunc_st().
		//	.2 _ifunc_st() actually implements single-threaded processing
		//	.3 func_mt() spawns multithreaded processing by means of calls to _ifunc_st()
		// _ifunc_st() must receive const rowcol_range/elms_range & rcr/er as a last parameter. This parameter
		//		describes a range of data to process.
		//		#WARNING use proxy variables instead of members of a structure, received by reference in performance sensitive code!
		// func_st() SHOULD receive const rowcol_range/elms_range *const pRCR/pER as a last parameter. It should help other
		// _st functions (called from _mt()) functions to call func_st() without knowing it's implementation details.


		typedef iThreadsT iThreads_t;
		typedef typename iThreads_t::range_t range_t;
		typedef typename iThreads_t::par_range_t par_range_t;
		typedef typename iThreads_t::reduce_data_t reduce_data_t;

		template<typename T>
		using converter_reduce_data_tpl = typename iThreads_t::template converter_reduce_data_t<T>;

		//static_assert(::std::is_same<typename realmtx_t::numel_cnt_t, typename iThreadsT::range_t>::value, "iThreads::range_t should be the same as realmtx_t::numel_cnt_t");
		static_assert(::std::is_same<numel_cnt_t, typename iThreadsT::range_t>::value, "iThreads::range_t should be the same as realmtx_t::numel_cnt_t");

		//ALL branching functions require refactoring
		typedef typename ThresholdsT Thresholds_t;

		//TODO: probably don't need this assert
		static_assert(::std::is_base_of<_impl::SMATH_THR<real_t>, Thresholds_t>::value, "Thresholds_t must be derived from _impl::SMATH_THR<real_t>");

	protected:
		typedef ::std::vector<real_t> thread_temp_storage_t;

		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		iThreads_t m_threads;

		numel_cnt_t m_minTempStorageSize, m_curStorElementsAllocated;
		thread_temp_storage_t m_threadTempRawStorage;
		
	public:
		~_SMath()noexcept {}
		_SMath()noexcept : m_minTempStorageSize(0), m_curStorElementsAllocated(0){
			global_denormalized_floats_mode();
		}

		// use with care, it's kind of "internal memory" of the class object. Don't know, if really 
		// should expose it into public (for some testing purposes only at this moment)
		// Always perform corresponding call to _istor_free() in LIFO (stack) order!
		// THREAD UNSAFE BY DESIGN! NEVER call from inside of _st() which is called from _mt()
		////////
		// NOTE THAT THIS INTERNAL MEMORY BUFFER IS AS AS WRONG AS IT COULD BE, BECAUSE here we're asking in real_t[count]
		// but almost elsewhere we're computing requirements as typename T[count]. #TODO
		real_t* _istor_alloc(const numel_cnt_t maxDataSize)noexcept {
			NNTL_ASSERT(m_minTempStorageSize >= maxDataSize + m_curStorElementsAllocated);
			NNTL_ASSERT(conform_sign(m_threadTempRawStorage.size()) >= m_minTempStorageSize);
			auto r = &m_threadTempRawStorage[m_curStorElementsAllocated];
			m_curStorElementsAllocated += maxDataSize;
			return r;
		}
		void _istor_free(real_t*const ptr, const numel_cnt_t maxDataSize)noexcept {
			NNTL_UNREF(ptr);
			NNTL_ASSERT(m_curStorElementsAllocated >= maxDataSize);
			m_curStorElementsAllocated -= maxDataSize;
			NNTL_ASSERT(ptr == &m_threadTempRawStorage[m_curStorElementsAllocated]);//if this assert triggered, then the wrong pointer was allocated/freed
		}

		iThreads_t& ithreads()noexcept { return m_threads; }

		//math preinitialization, should be called from each NN layer. n - maximum data length (in real_t), that this layer will use in calls
		//to math interface. Used to calculate max necessary temporary storage length.
		// #TODO some functions have weird or non-trivial memory requirements (see for example softmax_needTempMem()), so
		// it's better to employ some mechanism to specify which functions with which biggest datasizes code is going to use
		// and let the internals to do all the necessary tmp mem requrements calculations. Before that, make at least
		// a special function similar to noted softmax_needTempMem() for every implementation and call it to get correct
		// preinit() argument.
		void preinit(const numel_cnt_t n)noexcept {
			if (n > m_minTempStorageSize) m_minTempStorageSize = n;
		}

		//real math initialization, used to allocate necessary temporary storage of size max(preinit::n)
		// #note that init() as well as preinit() MUST allow subsequent calls without doing deinit() first.
		bool init()noexcept {
			if (conform_sign(m_threadTempRawStorage.size()) < m_minTempStorageSize) {
				//TODO: memory allocation exception handling here!
				m_threadTempRawStorage.resize(m_minTempStorageSize);
			}
			NNTL_ASSERT(m_curStorElementsAllocated == 0 || !"WTF?! Internal storage MUST NOT be in use at this moment!");
			m_curStorElementsAllocated = 0;
			return true;
		}
		void deinit()noexcept {
			NNTL_ASSERT(m_curStorElementsAllocated == 0 || !"WTF?! Internal storage MUST NOT be in use at this moment!");
			m_threadTempRawStorage.clear();
			m_threadTempRawStorage.shrink_to_fit();
			m_minTempStorageSize = 0;
			m_curStorElementsAllocated = 0;
		}


		//////////////////////////////////////////////////////////////////////////
		// Math Methods
	//protected:
		template<typename FuncT>
		static void _vec_apply_func(const typename ::std::remove_reference_t<FuncT>::value_type *const _ptr
			, const numel_cnt_t _cnt, FuncT&& F)noexcept
		{
			NNTL_ASSERT(_ptr && _cnt > 0);
			for (numel_cnt_t i = 0; i < _cnt; ++i) {
				//(::std::forward<FuncT>(F)).op(_ptr[i]);
				F.op(_ptr[i]);
				// There's no move happening here! ::std::forward<FuncT> is just a type cast that restores original variable type.
				// This matters a lot for ref-qualified operations! However, we have a loop here therefore if an 
				// rvalue-qualified operation exists, it will be called multiple times, but that is wrong because generally
				// rvalue-qualified operations are expected to be called only once (they are a matter of optimization on a
				// temporarily-soon-to-destruct objects and may "spoil" internal state of the object). Therefore
				// we're using universal reference in function definition to be able to get any 
				// kind of const FuncT&, FuncT& and FuncT&& as an argument, and nothing more. We'll use it as lvalue only.
			}
		}
		template <typename FunctorT>
		static typename FunctorT::value_type _vec_apply_func_get_result(const typename FunctorT::value_type* _ptr, const numel_cnt_t _cnt)noexcept {
			FunctorT f;
			_vec_apply_func(_ptr, _cnt, f);
			return f.result();
		}

		//////////////////////////////////////////////////////////////////////////
		template<typename _T, bool bNumStab>
		struct func_SUM {};

		template<typename _T>
		struct func_SUM<_T, false> {
			typedef _T value_type;

			value_type ret;

			func_SUM()noexcept:ret(value_type(0.)) {}
			void op(const value_type v)noexcept {
				ret += v;
			}
			value_type result()const noexcept { return ret; }
		};

		template<typename _T>
		struct func_SUM<_T, true> {
			typedef _T value_type;

			value_type ret, C;

			func_SUM()noexcept:ret(value_type(0.)), C(value_type(0.)) {}
			//#TODO: #pragma float_control(precise, on) wouldn't work here, but need some way to impose strict math here and elsewhere
			void op(const value_type v)noexcept {
				const auto Y = v - C;
				const auto T = ret + Y;
				C = T - ret - Y;
				ret = T;
			}
			value_type result()const noexcept { return ret; }
		};

		template<typename _T, bool bNumStab>
		struct func_SUM_squares : public func_SUM<_T, bNumStab> {
		private:
			typedef func_SUM<_T, bNumStab> _base_class_t;
		public:
			void op(const _T v)noexcept {
				_base_class_t::op(v*v);
			}
		};
		
		template<typename _T, bool bNumStab>
		struct func_SUMNZ {};

		template<typename _T>
		struct func_SUMNZ<_T, false> {
			typedef _T value_type;

			value_type ret;
			const value_type _centerVal;
			numel_cnt_t _cnt;
			
			func_SUMNZ(const value_type cv)noexcept: ret(value_type(0.)), _centerVal(cv), _cnt(0) {}
			void op(const value_type v)noexcept {
				const auto cv = _centerVal;
				const auto b = (v != cv);
				_cnt += b;
				ret += b*v;
			}
			value_type result()const noexcept { return ret; }
			numel_cnt_t count()const noexcept { return _cnt; }
		};

		template<typename _T>
		struct func_SUMNZ<_T, true> {
			typedef _T value_type;

			value_type ret, C;
			const value_type _centerVal;
			numel_cnt_t _cnt;

			func_SUMNZ(const value_type cv)noexcept: ret(value_type(0.)), C(value_type(0.)), _centerVal(cv), _cnt(0) {}
			//#TODO: #pragma float_control(precise, on) wouldn't work here, but need some way to impose strict math here and elsewhere
			void op(const value_type v)noexcept {
				const auto b = (v == _centerVal);
				_cnt += !b;

				const auto r = ret;
				const auto _C = C;

				const auto Y = v - _C;
				const auto T = r + Y;
				C = b ? _C :  T - r - Y;
				ret = b ? r : T;
			}
			value_type result()const noexcept { return ret; }
			numel_cnt_t count()const noexcept { return _cnt; }
		};
		
		/*template<typename _T, bool bNumStab>
		struct func_SUMNZ {};

		template<typename _T>
		struct func_SUMNZ<_T, false> {
			typedef _T value_type;

			numel_cnt_t _cnt;
			value_type ret;

			func_SUMNZ()noexcept: _cnt(0), ret(value_type(0.)) {}
			void op(const value_type v)noexcept {
				_cnt += !!v;
				ret += v;
			}
			value_type result()const noexcept { return ret; }
			numel_cnt_t count()const noexcept { return _cnt; }
		};

		template<typename _T>
		struct func_SUMNZ<_T, true> {
			typedef _T value_type;

			numel_cnt_t _cnt;
			value_type ret, C;

			func_SUMNZ()noexcept: _cnt(0), ret(value_type(0.)), C(value_type(0.)) {}
			void op(const value_type v)noexcept {
				_cnt += !!v;
				_base_class_t::updCounter(v);
				const auto Y = v - C;
				const auto T = ret + Y;
				C = T - ret - Y;
				ret = T;
			}
			value_type result()const noexcept { return ret; }
			numel_cnt_t count()const noexcept { return _cnt; }
		};*/

		//////////////////////////////////////////////////////////////////////////
		template <bool bNumStab, typename _T>
		static _T _vec_sum(const _T* _ptr, const numel_cnt_t _cnt)noexcept {
			return _vec_apply_func_get_result<func_SUM<_T, bNumStab>>(_ptr, _cnt);
		}
		template <bool bNumStab, typename _T>
		static _T _vec_sum_squares(const _T* _ptr, const numel_cnt_t _cnt)noexcept {
			return _vec_apply_func_get_result<func_SUM_squares<_T, bNumStab>>(_ptr, _cnt);
		}
		//////////////////////////////////////////////////////////////////////////

		template <typename T, bool bNumStab = false>
		static ::std::enable_if_t<!bNumStab, T> _reduce_vec_sum(const reduce_data_t*const _ptr, const numel_cnt_t _cnt)noexcept {
			typedef converter_reduce_data_tpl<T> converter_t;
			T r{ T(0) };
			for (numel_cnt_t i = 0; i < _cnt; ++i) {
				r += converter_t::from(_ptr[i]);
			}
			return r;
		}

		//#TODO: #pragma float_control(precise, on) wouldn't work here, but need some way to impose strict math here and elsewhere
		template <typename T, bool bNumStab = false>
		static ::std::enable_if_t<bNumStab, T> _reduce_vec_sum(const reduce_data_t*const _ptr, const numel_cnt_t _cnt)noexcept {
			typedef converter_reduce_data_tpl<T> converter_t;
			T r{ T(0) }, C{ T(0) }, Y, Tt;
			for (numel_cnt_t i = 0; i < _cnt; ++i) {
				Y = converter_t::from(_ptr[i]) - C;
				Tt = r + Y;
				C = Tt - r - Y;
				r = Tt;
			}
			return r;
		}

		//////////////////////////////////////////////////////////////////////////
		//contrary to a common matrix elements numeration, `er` var describes elements indexes in TRIANGULAR matrix (not a square matrix as usual)
		template<bool bLowerTriangl, typename FunctorT>
		static auto _triang_apply_func_get_result(const smatrix<typename FunctorT::value_type>& A, const elms_range& er)noexcept {
			NNTL_ASSERT(A.rows() > 1);
			NNTL_ASSERT(er.elmEnd <= A.numel_triangl());

			vec_len_t ri, ci, endRi, endCi;
			A.triangl_coords_from_idx<bLowerTriangl>(er.elmBegin, ri, ci);
			A.triangl_coords_from_idx<bLowerTriangl>(er.elmEnd, endRi, endCi);
			NNTL_ASSERT(ci < endCi || (ci == endCi && ri < endRi));

			const auto n = A.rows();
			NNTL_ASSERT(ri < n && endRi < n && (endCi < n || (endCi == n && endRi == 0)));
			const bool bFirstIsLastCol = ci == endCi;
			const auto _n = static_cast<ptrdiff_t>(n);
			FunctorT F;
			const auto* pCol = A.colDataAsVec(ci);

			//first column of the range (it has non-default beginning row index)
			if (bLowerTriangl) {
				NNTL_ASSERT(ri > 0);
				_vec_apply_func(pCol + ri, n - ri - bFirstIsLastCol*(n - endRi), F);
				ri = ci + 2;
			} else {
				_vec_apply_func(pCol + ri, ci - ri - bFirstIsLastCol*(ci - endRi), F);
			}
			++ci;
			pCol += _n;

			//intermediate columns
			for (; ci < endCi; ++ci) {
				if (bLowerTriangl) {
					NNTL_ASSERT(ri < n);
					_vec_apply_func(pCol + ri, n - ri, F);
					++ri;
				} else {
					_vec_apply_func(pCol, ci, F);
				}
				pCol += _n;
			}

			//final column (it has non-default ending row index)
			if (!bFirstIsLastCol) {
				if (bLowerTriangl) {
					if (endRi > endCi + 1) {
						NNTL_ASSERT(pCol < A.end());
						NNTL_ASSERT(ri < n && ri == endCi + 1);
						_vec_apply_func(pCol + ri, endRi - ri, F);
					}
				} else {
					if (endRi > 0) { //endRi==0 means just the begining of the next column
						NNTL_ASSERT(pCol < A.end());
						_vec_apply_func(pCol, endRi, F);
					}
				}
			}

			return F.result();
		}

		//////////////////////////////////////////////////////////////////////////
		//Copies pRCR->totalRows() elements from a column of A, starting at row=(pRCR->rowBegin) col=(pRCR->colBegin) element.
		template<typename T_>
		nntl_force_inline static void _memcpy_rowcol_range(T_* dest, const smatrix<T_>& A, const rowcol_range*const pRCR)noexcept {
			const T_* src;
			vec_len_t rm;
			if (pRCR) {
				NNTL_ASSERT(pRCR->can_apply(A));
				dest += pRCR->rowBegin;
				src = A.colDataAsVec(pRCR->colBegin) + pRCR->rowBegin;
				rm = pRCR->totalRows();
			} else {
				src = A.data();
				rm = A.rows();
			}
			memcpy(dest, src, sizeof(T_)*rm);
		}
		template<typename T_>
		nntl_force_inline static void _memset_rowrange(T_* dest, const T_ v, numel_cnt_t elems, const rowcol_range*const pRCR)noexcept {
			if (pRCR) {
				dest += pRCR->rowBegin;
				elems = pRCR->totalRows();
			}
			//#todo which one is really faster?
			//memset(dest, src, sizeof(T_)*elems);
			//::std::fill(dest, dest + elems, v); //doesn't get vectorized!
			for (numel_cnt_t i = 0; i < elems; ++i) dest[i] = v;
		}

		//////////////////////////////////////////////////////////////////////////
		enum _OperationType {
			mrw_cw,//processing rows of matrix columnwise
			mrw_rw //processing rows of matrix rowwise
		};

		//////////////////////////////////////////////////////////////////////////
		// operation helpers
	#pragma warning(push)
	#pragma warning(disable:4100)
		struct _mrwHlpr_rw_InitVecElmByVec {
			static constexpr vec_len_t rw_FirstColumnIdx = 0;

			//numel_cnt_t ldM - numel_cnt_t by intention!
			template<typename VecBaseT, typename MtxBaseT>
			static constexpr VecBaseT rw_initVecElm(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const numel_cnt_t ldM
				, const numel_cnt_t colBegin, const numel_cnt_t r)noexcept
			{ return vecElm; }
		};
		struct _mrwHlpr_rw_InitVecElmByMtxElm {
			static constexpr vec_len_t rw_FirstColumnIdx = 1;

			//numel_cnt_t ldM - numel_cnt_t by intention!
			template<typename VecBaseT, typename MtxBaseT>
			static VecBaseT rw_initVecElm(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const numel_cnt_t ldM
				, const numel_cnt_t colBegin, const numel_cnt_t r)noexcept
			{
				const auto v = *pFirstMtxElm;
				pFirstMtxElm += ldM;
				return v;
			}
		};
		struct _mrwHlpr_rw_InitVecElmByZero {
			static constexpr vec_len_t rw_FirstColumnIdx = 0;

			//numel_cnt_t ldM - numel_cnt_t by intention!
			template<typename VecBaseT, typename MtxBaseT>
			static constexpr VecBaseT rw_initVecElm(const VecBaseT& vecElm, const MtxBaseT*const & pFirstMtxElm, const numel_cnt_t ldM
				, const numel_cnt_t colBegin, const numel_cnt_t r)noexcept
			{
				return VecBaseT(0);
			}
		};
		struct _mrwHlpr_rw_Dont_UpdVecElm {
			template<typename BaseT>
			static constexpr void rw_updVecElm(BaseT& vecElm, BaseT& v, const numel_cnt_t r)noexcept {}
		};
		struct _mrwHlpr_rw_UpdVecElm {
			template<typename BaseT>
			static constexpr void rw_updVecElm(BaseT& vecElm, BaseT& v, const numel_cnt_t r)noexcept {
				vecElm = v;
			}
		};
		struct _mrwHlpr_simpleLoops {
			static constexpr void initOperation(const vec_len_t colBegin, const vec_len_t ldM)noexcept {};

			static constexpr void cw_toNextCol(const numel_cnt_t ldM)noexcept {};
		};

		//////////////////////////////////////////////////////////////////////////
		//operations
		/*struct _mrw_COUNT_Zeros : public _mrwHlpr_rw_InitVecElmByZero, public _mrwHlpr_rw_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const numel_cnt_t ldM)noexcept {
				vecElm += (mtxElm==real_t(0));
			}
		};*/

		struct _mrw_MUL_mtx_by_vec : public _mrwHlpr_rw_InitVecElmByVec, public _mrwHlpr_rw_Dont_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(BaseT& mtxElm, const BaseT vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				mtxElm *= vecElm;
			}
		};
		struct _mrw_DIV_mtx_by_vec : public _mrwHlpr_rw_InitVecElmByVec, public _mrwHlpr_rw_Dont_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(BaseT& mtxElm, const BaseT vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				mtxElm /= vecElm;
			}
		};
		struct _mrwFind_MAX : public _mrwHlpr_rw_InitVecElmByMtxElm, public _mrwHlpr_rw_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(const BaseT mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				if (mtxElm > vecElm) vecElm = mtxElm;
			}
		};
		template<bool bSaveMaxOnUpdate>
		struct _mrwFindIdxsOf_MAX : public _mrwHlpr_simpleLoops {
			vec_len_t*const pDest;
			numel_cnt_t _maxColumnIdx;
			_mrwFindIdxsOf_MAX(vec_len_t* pd)noexcept : pDest(pd) {}

			template<_OperationType OpType, typename BaseT>
			::std::enable_if_t<OpType == mrw_cw> op(const BaseT mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				if (mtxElm > vecElm) {
					vecElm = mtxElm;
					pDest[r] = static_cast<vec_len_t>(c);
				}
			}
			template<_OperationType OpType, typename BaseT>
			::std::enable_if_t<OpType == mrw_rw> op(const BaseT mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				if (mtxElm > vecElm) {
					vecElm = mtxElm;
					_maxColumnIdx = c;
				}
			}

			static constexpr vec_len_t rw_FirstColumnIdx = 1;

			template<typename VecBaseT, typename MtxBaseT>
			VecBaseT rw_initVecElm(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const numel_cnt_t ldM
				, const numel_cnt_t colBegin, const numel_cnt_t r)noexcept
			{
				_maxColumnIdx = colBegin;
				const auto v = *pFirstMtxElm;
				pFirstMtxElm += ldM;
				return v;
			}

			template<typename BaseT, bool B = bSaveMaxOnUpdate>
			::std::enable_if_t<!B> rw_updVecElm(BaseT& vecElm, BaseT& v, const numel_cnt_t r)noexcept {
				pDest[r] = static_cast<vec_len_t>(_maxColumnIdx);
			}
			template<typename BaseT, bool B = bSaveMaxOnUpdate>
			::std::enable_if_t<B> rw_updVecElm(BaseT& vecElm, BaseT& v, const numel_cnt_t r)noexcept {
				pDest[r] = static_cast<vec_len_t>(_maxColumnIdx);
				vecElm = v;
			}
		};
		struct _mrw_SUM : public _mrwHlpr_rw_InitVecElmByMtxElm, public _mrwHlpr_rw_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(const BaseT mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				vecElm += mtxElm;
			}
		};

		//Binary/bitwise OR
		struct _mrw_BinaryOR : public _mrwHlpr_rw_InitVecElmByMtxElm, public _mrwHlpr_rw_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(const BaseT mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t ldM)noexcept {
				vecElm |= mtxElm;
			}
		};
	#pragma warning(pop)
		//////////////////////////////////////////////////////////////////////////
		// Matrix/Vector elementwise operations
		//////////////////////////////////////////////////////////////////////////
		// Apply operation F.op to every element of matrix/vector A
		// #todo: make a wrapper to mate vector api (size(), begin() and so on) and matrix api (numel(), data() ...)
		/*template<typename ContainerT, typename ewOperationT>
		nntl_force_inline static void _ewOperation_st(ContainerT& A, const elms_range& er, ewOperationT&& F)noexcept {
			const auto pA = A.data();
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) {
				const auto p = pA + i;
				F.op(*p);
				//::std::forward<ewOperationT>(F).op(*p);
			}
		}
		template<typename ContainerT, typename ewOperationT>
		nntl_force_inline static void _ewOperation_st2(ContainerT& A, const elms_range& er, ewOperationT&& F)noexcept {
			auto pA = A.data() + er.elmBegin;
			const auto pAE = A.data() + er.elmEnd;
			while(pA!=pAE){
				//::std::forward<ewOperationT>(F).op(*pA);//dont do F.op(*pA++) or you'll get serious performance penalty
				F.op(*pA);
				pA++;
			}
		}*/

		//////////////////////////////////////////////////////////////////////////
		// Matrix rowwise operations
		//////////////////////////////////////////////////////////////////////////
		//apply operation F.op to each element of matrix A rows and corresponding element of row-vector pVec (must 
		// have at least A.rows() elements)
		// Columnwise
		// here and later passing functor by T&& helps to deal with statefullness while still allowing to use stateless functors.
		// No ref-qualified functor operations should be defined!
		template<typename MtxT, typename VecValueT, typename mrwOperationT>
		nntl_probably_force_inline static void _mrwVecOperation_st_cw(MtxT& A, VecValueT*const pVec, vec_len_t colBegin
			, const rowcol_range& RCR, mrwOperationT&& F, const bool bIgnoreBias = false)noexcept
		{
			static_assert(::std::is_same< smatrix<::std::remove_const_t<VecValueT>>, ::std::remove_const_t<MtxT> >::value, "Types mismatch");
			NNTL_UNREF(F);
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			NNTL_ASSERT(colBegin == 0 || colBegin == 1);
			const numel_cnt_t ldA = A.ldim();
			colBegin += RCR.colBegin;
			NNTL_ASSERT(colBegin <= RCR.colEnd);
			auto pA = A.colDataAsVec(colBegin);
			//::std::forward<mrwOperationT>(F).initOperation(colBegin, A.rows());
			F.initOperation(colBegin, A.rows(bIgnoreBias));
			const numel_cnt_t ce = RCR.colEnd, re = RCR.rowEnd;
			NNTL_ASSERT(re <= A.rows(bIgnoreBias));
			for (numel_cnt_t c = colBegin; c < ce; ++c) {
				for (numel_cnt_t r = RCR.rowBegin; r < re; ++r) {//FOR cycle with offset calculation is generally faster than WHILE,
					const auto pV = pVec + r;//because usually compiler can unfold a cycle into many instructions
					const auto pElm = pA + r;//In WHILE setup with mrwOperationT::op(*pA++, *pV++) it can't
					//and calculating offsets is faster than mrwOperationT::op(*pA++, *pV++);
					//static call-style mrwOperationT::op() vs. object call-style F.op() doesn't make any difference in asm 
					// code (at the moment of testing), but object call-style permits far more generic algorithms creation
					//::std::forward<mrwOperationT>(F).op<mrw_cw>(*pElm, *pV, r, c, rm);
					F.op<mrw_cw>(*pElm, *pV, r, c, ldA);
				}
				pA += ldA;
				//::std::forward<mrwOperationT>(F).cw_toNextCol(rm);
				F.cw_toNextCol(ldA);
			}
		}

		//Rowwise
		template<typename MtxT, typename VecValueT, typename mrwOperationT>
		nntl_probably_force_inline static void _mrwVecOperation_st_rw(MtxT& A, VecValueT*const pVec, const rowcol_range& RCR, mrwOperationT&& F)noexcept {
			static_assert(::std::is_same< smatrix<::std::remove_const_t<VecValueT>>, ::std::remove_const_t<MtxT> >::value, "Types mismatch");
			NNTL_UNREF(F);
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			const auto pA = A.colDataAsVec(RCR.colBegin); //A.data();
			const numel_cnt_t rm = A.rows();
			//::std::forward<mrwOperationT>(F).initOperation(RCR.colBegin, A.rows());
			F.initOperation(RCR.colBegin, A.rows());
			const numel_cnt_t ce = RCR.colEnd, re = RCR.rowEnd;
			for (numel_cnt_t r = RCR.rowBegin; r < re; ++r) {
				const auto pV = pVec + r;
				auto pElm = pA + r;
				//auto v = ::std::forward<mrwOperationT>(F).rw_initVecElm(*pV, pElm, rm, RCR.colBegin, r);
				auto v = F.rw_initVecElm(*pV, pElm, rm, RCR.colBegin, r);
				for (numel_cnt_t c = RCR.colBegin + mrwOperationT::rw_FirstColumnIdx; c < ce; ++c) {
					//::std::forward<mrwOperationT>(F).op<mrw_rw>(*pElm, v, r, c, rm);
					F.op<mrw_rw>(*pElm, v, r, c, rm);
					pElm += rm;
				}
				//::std::forward<mrwOperationT>(F).rw_updVecElm<VecValueT>(*pV, v, r);
				F.rw_updVecElm<VecValueT>(*pV, v, r);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// colwise matrix processing

		//functor to perform colwise vector element subtraction A = A - vec, where size(vec)==A.cols()
		struct _mcwSUB_ip {
			template<typename _T>
			static void op(_T& mtxElm, const _T vecElm)noexcept {
				mtxElm -= vecElm;
			}
		};

		template<typename MtxT, typename VecValueT, typename mcwOperationT>
		nntl_probably_force_inline static void _mcwVecOperation_st(MtxT& A, VecValueT* pVec, const rowcol_range& RCR, mcwOperationT&& F)noexcept {
			static_assert(::std::is_same< smatrix<::std::remove_const_t<VecValueT>>, ::std::remove_const_t<MtxT> >::value, "Types mismatch");
			NNTL_UNREF(F);
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			NNTL_ASSERT(RCR.rowBegin == 0 && RCR.rowEnd == A.rows());//we won't be using row information

			auto* pA = A.colDataAsVec(RCR.colBegin);
			auto*const pAE = A.colDataAsVec(RCR.colEnd);
			pVec += RCR.colBegin;
			const numel_cnt_t rc = A.rows();
			while (pA != pAE) {
				const auto pC = pA;
				pA += rc;
				const auto v = *pVec++;
				for (numel_cnt_t i = 0; i < rc; ++i) {
					//::std::forward<mcwOperationT>(F).op(pC[i], v);
					F.op(pC[i], v);
				}
			}
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// mt stuff
		thread_id_t _howMuchThreadsNeededForCols(const vec_len_t minColumnsPerThread, const vec_len_t columns)const noexcept {
			NNTL_ASSERT(columns > minColumnsPerThread);
			const auto minThreadsReq = static_cast<thread_id_t>(ceil(real_t(columns) / minColumnsPerThread));
			NNTL_ASSERT(minThreadsReq > 1);
			auto workersCnt = m_threads.cur_workers_count();
			if (minThreadsReq < workersCnt) workersCnt = minThreadsReq;
			return workersCnt;
		}
		//////////////////////////////////////////////////////////////////////////
		// colwise processing
		// Variation to work with the A without additional tmp vector
		//LambdaF is void(*F)(const rowcol_range& RCR)
		template<typename VT, typename LambdaF>
		nntl_probably_force_inline void _do_processMtx_cw(const smatrix<VT>& A, LambdaF&& Func, const vec_len_t colsNum)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			//TODO: for some algorithms and datasizes it may be highly beneficial to make smart partitioning, that takes into account
			//CPU cache size (will probably require more than workers_count() call to worker function, but each call will run significanly
			// faster, due to correct cache use)
			m_threads.run([&A, &F{ Func }](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				F(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())));// , pr.tid());
				//#note we mustn't forward back to LambdaF here, because current lambda is called by a several threads simultaneously
				//and if there are a rvalue-qualified operator() available, the first call to it might "spoil" the F object state
			}, colsNum);
		}		

		template<typename VT, typename LambdaF>
		nntl_probably_force_inline void _processMtx_cw(const smatrix<VT>& A, LambdaF&& Func, const bool bIgnoreBias=false)noexcept {
			get_self()._do_processMtx_cw(A, ::std::forward<LambdaF>(Func), A.cols(bIgnoreBias));
		}

		template<typename VT, typename LambdaF>
		nntl_probably_force_inline void _processMtx_cw_nb(const smatrix<VT>& A, LambdaF&& Func)noexcept {
			get_self()._processMtx_cw(A, ::std::forward<LambdaF>(Func), true);
		}

		//max mem required for _processMtx_cw() with 4+ args to process matrix A
		template<typename ScndVecType, typename VT>
		nntl_probably_force_inline numel_cnt_t _processMtx_cw_needTempMem(const smatrix<VT>& A)const noexcept {
			return _processMtx_cw_needTempMem<ScndVecType>(A.rows());
		}
		template<typename ScndVecType>
		nntl_probably_force_inline numel_cnt_t _processMtx_cw_needTempMem(const vec_len_t aRows)const noexcept {
			static_assert(::std::is_pod<ScndVecType>::value, "");
			const auto elemsCnt = smatrix_td::sNumel(aRows, m_threads.cur_workers_count());
			return elemsCnt + static_cast<numel_cnt_t>(ceil((ext_real_t(sizeof(ScndVecType)) / sizeof(real_t))*elemsCnt));
		}

		//Variation to make a vector out of const A
		//LambdaF is void(*Func)(const rowcol_range& RCR, real_t*const pVec), where pVec is temporary vector of length A.rows() to serve 
		// as F destination
		// LambdaFinal is void(*FinFunc)(realmtx_t& fin), there fin is a matrix of size A.rows()*threadsUsed that store results (columnwise) of
		// each F pVec computation
		template<typename VT, typename LambdaF, typename LambdaFinal>
		nntl_probably_force_inline void _processMtx_cw(const smatrix<VT>& A, const vec_len_t mt_cw_ColsPerThread
			, LambdaF&& Func, LambdaFinal&& FinFunc, VT*const pTVec = nullptr)noexcept
		{
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			const auto rm = A.rows(), cm = A.cols();
			NNTL_ASSERT(cm > mt_cw_ColsPerThread && mt_cw_ColsPerThread >= 3);//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			const auto threadsToUse = _howMuchThreadsNeededForCols(mt_cw_ColsPerThread, cm);
			static_assert(sizeof(VT) == sizeof(real_t),"Mismatching type sizes will lead to wrong malloc");

			numel_cnt_t elmsToAlloc(0);
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			const auto pTmpMem = pTVec ? pTVec : reinterpret_cast<VT*>(get_self()._istor_alloc(elmsToAlloc = smatrix_td::sNumel(rm, threadsToUse)));
			thread_id_t threadsUsed = 0;
			//TODO: for some algorithms and datasizes it may be highly beneficial to make smart partitioning, that takes into account
			//CPU cache size (will probably require more than workers_count() calls to worker function, but each call will run significanly
			// faster, due to correct cache use)
			//m_threads.run([&A, pTmpMem, rm, F{ ::std::forward<LambdaF>(Func) }](const par_range_t& pr) noexcept{
			m_threads.run([&A, pTmpMem, rm, &F{ Func }](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				auto pVec = pTmpMem + smatrix<VT>::sNumel(rm, pr.tid());
				//::std::forward<LambdaF>(F)(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pVec);
				F(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pVec);
			}, cm, threadsToUse, &threadsUsed);

			NNTL_ASSERT(threadsToUse == threadsUsed);
			smatrix<VT> fin;
			fin.useExternalStorage(pTmpMem, rm, threadsUsed);

			::std::forward<LambdaFinal>(FinFunc)(fin);//forwarding is OK here, because FinFunc is used only once
			if (!pTVec) {
				//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
				get_self()._istor_free(reinterpret_cast<real_t*>(pTmpMem), elmsToAlloc);
			}
		}

		//Variation to make a vector out of const A, using additional (second) temporary data vector
		//LambdaF is void(*Func)(const realmtx_t& Apart, real_t*const pVec, ScndVecType*const pScndVec), where pVec and pScndVec are temporary 
		// vectors of length A.rows() to serve as F destination
		// LambdaFinal is void(*FinFunc)(const realmtx_t& fin, ScndVecType*const pFullScndMtx), there fin and pFullScndMtx are matrices
		// of size A.rows()*threadsUsed that store results (columnwise) of each F pVec computation
		template<typename ScndVecType, typename VT, typename LambdaF, typename LambdaFinal>
		nntl_probably_force_inline void _processMtx_cw(const smatrix<VT>& A, const vec_len_t mt_cw_ColsPerThread
			, LambdaF&& Func, LambdaFinal&& FinFunc)noexcept
		{
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			const auto rm = A.rows(), cm = A.cols();
			NNTL_ASSERT(cm > mt_cw_ColsPerThread && mt_cw_ColsPerThread >= 3);//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			const auto threadsToUse = _howMuchThreadsNeededForCols(mt_cw_ColsPerThread, cm);

			static_assert(sizeof(VT) == sizeof(real_t), "Mismatching type sizes will lead to wrong malloc");
			const auto elemsCnt = smatrix_td::sNumel(rm, threadsToUse);
			const auto tmemSize = elemsCnt + static_cast<numel_cnt_t>(ceil((ext_real_t(sizeof(ScndVecType)) / sizeof(real_t))*elemsCnt));
			const auto pTmpMem = get_self()._istor_alloc(tmemSize);

			const auto pMainVec = pTmpMem;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			ScndVecType*const pScndVec = reinterpret_cast<ScndVecType*>(pTmpMem + elemsCnt);

			thread_id_t threadsUsed = 0;
			//TODO: for some algorithms and datasizes it may be highly beneficial to make smart partitioning, that takes into account
			//CPU cache size (will probably require more than workers_count() call to worker function, but each call will run significanly
			// faster, due to correct cache use)
			//m_threads.run([&A, pMainVec, pScndVec, rm, F{ ::std::forward<LambdaF>(Func) }](const par_range_t& pr) noexcept{
			m_threads.run([&A, pMainVec, pScndVec, rm, &F{ Func }](const par_range_t& pr) noexcept{
				const auto _tmpElmOffset = smatrix<VT>::sNumel(rm, pr.tid());
				auto pVec = pMainVec + _tmpElmOffset;
				auto pSVec = pScndVec + _tmpElmOffset;
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				//::std::forward<LambdaF>(F)(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pVec, pSVec);
				F(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pVec, pSVec);
			}, cm, threadsToUse, &threadsUsed);
			NNTL_ASSERT(threadsToUse == threadsUsed);

			smatrix<VT> fin;
			fin.useExternalStorage(pMainVec, rm, threadsUsed);

			::std::forward<LambdaFinal>(FinFunc)(fin, pScndVec);
			get_self()._istor_free(pTmpMem, tmemSize);
		}

		//////////////////////////////////////////////////////////////////////////
		// Rowwise processing
		//LambdaF is void (*Func) (const rowcol_range& RCR)
		template<typename MtxT, typename LambdaF>
		nntl_probably_force_inline void _processMtx_rw(MtxT& A, LambdaF&& Func, const bool bIgnoreBias = false)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			m_threads.run([&A, &F{ Func }](const par_range_t& pr) noexcept{
				const auto ofs = static_cast<vec_len_t>(pr.offset());
				F(rowcol_range(ofs, ofs + static_cast<vec_len_t>(pr.cnt()), A));
			}, A.rows(bIgnoreBias));
		}

	public:
		
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Element-wise Operations
		//////////////////////////////////////////////////////////////////////////
		// Cumulative Operations

		//finds a sum of elementwise products return sum( A.*B );
		real_t ewSumProd(const realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::ewSumProd) {
				return get_self().ewSumProd_st(A, B);
			} else return get_self().ewSumProd_mt(A, B);
		}
		real_t ewSumProd_st(const realmtx_t& A, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			return get_self()._iewSumProd_st(A, B, pER ? *pER : elms_range(A));
		}
		static real_t _iewSumProd_st(const realmtx_t& A, const realmtx_t& B, const elms_range& er)noexcept {
			NNTL_ASSERT(!A.empty() && !B.empty() && B.size() == A.size());
			const auto pA = A.data(), pB = B.data();
			real_t ret(0.0);
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i)  ret += pA[i]*pB[i];
			return ret;
		}
		real_t ewSumProd_mt(const realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(!A.empty() && !B.empty() && B.size() == A.size());
			return m_threads.reduce([&A, &B, this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iewSumProd_st(A, B, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t, true>, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//finds a sum of elementwise squares: return sum( A.*A );
		
		static real_t _iewSumSquares_st(const real_t* pA, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && er.totalElements() > 0);
			return _vec_sum_squares<false>(pA + er.elmBegin, er.totalElements());
		}
		static real_t _iewSumSquares_st_ns(const real_t* pA, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && er.totalElements() > 0);
			return _vec_sum_squares<true>(pA + er.elmBegin, er.totalElements());
		}
		//////////////////////////////////////////////////////////////////////////
		real_t ewSumSquares(const realmtx_t& A)noexcept {
			if (A.numel() < Thresholds_t::ewSumSquares) {
				return get_self().ewSumSquares_st(A);
			} else return get_self().ewSumSquares_mt(A);
		}
		real_t ewSumSquares_st(const realmtx_t& A, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!pER || pER->elmEnd <= A.numel());
			return get_self()._iewSumSquares_st(A.data(), pER ? *pER : elms_range(A));
		}
		real_t ewSumSquares_mt(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());
			return m_threads.reduce([pA = A.data(), this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iewSumSquares_st(pA, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t>, A.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		real_t ewSumSquares(const real_t* pA, const numel_cnt_t n)noexcept {
			if (n < Thresholds_t::ewSumSquares) {
				return get_self().ewSumSquares_st(pA, n);
			} else return get_self().ewSumSquares_mt(pA, n);
		}
		real_t ewSumSquares_st(const real_t* pA, const numel_cnt_t n, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!pER || pER->elmEnd <= n);
			return get_self()._iewSumSquares_st(pA, pER ? *pER : elms_range(0, n));
		}
		real_t ewSumSquares_mt(const real_t* pA, const numel_cnt_t n)noexcept {
			return m_threads.reduce([pA, this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iewSumSquares_st(pA, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t>, n);
		}
		//////////////////////////////////////////////////////////////////////////
		real_t ewSumSquares_ns(const realmtx_t& A)noexcept {
			if (A.numel() < Thresholds_t::ewSumSquares_ns) {
				return get_self().ewSumSquares_st_ns(A);
			} else return get_self().ewSumSquares_mt_ns(A);
		}
		real_t ewSumSquares_st_ns(const realmtx_t& A, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!pER || pER->elmEnd <= A.numel());
			return get_self()._iewSumSquares_st_ns(A.data(), pER ? *pER : elms_range(A));
		}
		real_t ewSumSquares_mt_ns(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());
			return m_threads.reduce([pA = A.data(), this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iewSumSquares_st_ns(pA, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t, true>, A.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		real_t ewSumSquares_ns(const real_t* pA, const numel_cnt_t n)noexcept {
			if (n < Thresholds_t::ewSumSquares_ns) {
				return get_self().ewSumSquares_st_ns(pA, n);
			} else return get_self().ewSumSquares_mt_ns(pA, n);
		}
		real_t ewSumSquares_st_ns(const real_t* pA, const numel_cnt_t n, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!pER || pER->elmEnd <= n);
			return get_self()._iewSumSquares_st_ns(pA, pER ? *pER : elms_range(0, n));
		}
		real_t ewSumSquares_mt_ns(const real_t* pA, const numel_cnt_t n)noexcept {
			return m_threads.reduce([pA, this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iewSumSquares_st_ns(pA, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t, true>, n);
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Sum of squares of elements that lies in the upper or the lower triangular part of a square matrix A
		// Main diagonal elements excluded.
		template<bool bLowerTriangl, bool bNumStab, typename _T>
		_T ewSumSquaresTriang(const smatrix<_T>& A)noexcept {
			NNTL_ASSERT(A.rows() == A.cols());
			if (A.rows() < Thresholds_t::ewSumSquaresTriang<bLowerTriangl, bNumStab>::v) {
				return get_self().ewSumSquaresTriang_st<bLowerTriangl, bNumStab>(A);
			} else return get_self().ewSumSquaresTriang_mt<bLowerTriangl, bNumStab>(A);
		}

		template<bool bLowerTriangl, bool bNumStab, typename _T>
		_T ewSumSquaresTriang_st(const smatrix<_T>& A, const elms_range*const pER = nullptr)noexcept {
			return get_self()._iewSumSquaresTriang_st<bLowerTriangl, bNumStab>(A, pER ? *pER : elms_range(0, A.numel_triangl()));
		}

		template<bool bLowerTriangl, bool bNumStab, typename _T>
		_T ewSumSquaresTriang_mt(const smatrix<_T>& A)noexcept {
			return m_threads.reduce([&A, this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iewSumSquaresTriang_st<bLowerTriangl, bNumStab>(A, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t, bNumStab>, A.numel_triangl());
		}

		//contrary to a common matrix elements numeration, `er` var describes elements indexes in TRIANGULAR matrix (not a square matrix as usual)
		template<bool bLowerTriangl, bool bNumStab, typename _T>
		static _T _iewSumSquaresTriang_st(const smatrix<_T>& A, const elms_range& er)noexcept{
			return _triang_apply_func_get_result<bLowerTriangl, func_SUM_squares<_T, bNumStab>>(A, er);
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Matrix RowWise operations
		//////////////////////////////////////////////////////////////////////////
		// divide each matrix A row by a corresponding vector d element, A(i,:) = A(i,:) / d(i)
		// #TODO: preprocess vector pDiv = 1/pDiv and then call mrwMulByVec
		void mrwDivideByVec(realmtx_t& A, const real_t*const pDiv)noexcept {
			if (A.numel() < Thresholds_t::mrwDivideByVec) {
				get_self().mrwDivideByVec_st(A, pDiv);
			} else get_self().mrwDivideByVec_mt(A, pDiv);
		}
		void mrwDivideByVec_st(realmtx_t& A, const real_t*const pDiv, const rowcol_range*const pRCR = nullptr)noexcept {
			//TODO: should be branched by rows/cols
			if (A.numel() < Thresholds_t::mrwDivideByVec_rw) {
				get_self().mrwDivideByVec_st_rw(A, pDiv, pRCR);
			} else get_self().mrwDivideByVec_st_cw(A, pDiv, pRCR);
		}
		void mrwDivideByVec_mt(realmtx_t& A, const real_t*const pDiv)noexcept {
			if (A.rows() < Thresholds_t::mrwDivideByVec_mt_rows) {
				get_self().mrwDivideByVec_mt_cw(A, pDiv);
			} else get_self().mrwDivideByVec_mt_rw(A, pDiv);
		}
		static void mrwDivideByVec_st_cw(realmtx_t& A, const real_t*const pDiv, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDiv);
			_mrwVecOperation_st_cw(A, pDiv, 0, pRCR ? *pRCR : rowcol_range(A), _mrw_DIV_mtx_by_vec());
		}
		static void mrwDivideByVec_st_rw(realmtx_t& A, const real_t*const pDiv, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDiv);
			_mrwVecOperation_st_rw(A, pDiv, pRCR ? *pRCR : rowcol_range(A), _mrw_DIV_mtx_by_vec());
		}		
		void mrwDivideByVec_mt_cw(realmtx_t& A, const real_t*const pDiv)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDiv);
			_processMtx_cw(A, [&A, pDiv, this](const rowcol_range& RCR)noexcept {//, const thread_id_t _tid) {
				get_self().mrwDivideByVec_st(A, pDiv, &RCR);
			});
		}
		void mrwDivideByVec_mt_rw(realmtx_t& A, const real_t*const pDiv)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDiv);
			_processMtx_rw(A, [&A, pDiv, this](const rowcol_range& RCR)noexcept {
				get_self().mrwDivideByVec_st(A, pDiv, &RCR);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// multiply each matrix A row by corresponding vector d element, A(i,:) = A(i,:) .* d(i)
		void mrwMulByVec(realmtx_t& A, const real_t*const pMul)noexcept {
			if (A.numel() < Thresholds_t::mrwMulByVec) {
				get_self().mrwMulByVec_st(A, pMul);
			} else get_self().mrwMulByVec_mt(A, pMul);
		}
		void mrwMulByVec_st(realmtx_t& A, const real_t*const pMul, const rowcol_range*const pRCR = nullptr)noexcept {
			//TODO: should be branched by rows/cols
			if (A.rows() < Thresholds_t::mrwMulByVec_st_rows) {
				get_self().mrwMulByVec_st_cw(A, pMul, pRCR);
			} else get_self().mrwMulByVec_st_rw(A, pMul, pRCR);
		}
		void mrwMulByVec_mt(realmtx_t& A, const real_t*const pMul)noexcept {
			if (A.rows() < Thresholds_t::mrwMulByVec_mt_rows) {
				get_self().mrwMulByVec_mt_cw(A, pMul);
			} else get_self().mrwMulByVec_mt_rw(A, pMul);
		}
		static void mrwMulByVec_st_cw(realmtx_t& A, const real_t*const pMul, const rowcol_range*const pRCR = nullptr)noexcept {
			_mrwVecOperation_st_cw(A, pMul, 0, pRCR ? *pRCR : rowcol_range(A), _mrw_MUL_mtx_by_vec());
		}
		static void mrwMulByVec_st_rw(realmtx_t& A, const real_t*const pMul, const rowcol_range*const pRCR = nullptr)noexcept {
			_mrwVecOperation_st_rw(A, pMul, pRCR ? *pRCR : rowcol_range(A), _mrw_MUL_mtx_by_vec());
		}
		void mrwMulByVec_mt_cw(realmtx_t& A, const real_t*const pMul)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pMul);
			_processMtx_cw(A, [&A, pMul, this](const rowcol_range& RCR)noexcept {//, const thread_id_t _tid) {
				get_self().mrwMulByVec_st(A, pMul, &RCR);
			});
		}
		void mrwMulByVec_mt_rw(realmtx_t& A, const real_t*const pMul)noexcept {
			_processMtx_rw(A, [&A, pMul, this](const rowcol_range& RCR)noexcept {
				get_self().mrwMulByVec_st(A, pMul, &RCR);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// computes and stores to pVec the count of zeros in A rows 
		//////////////////////////////////////////////////////////////////////////
		// NOT TESTED
		//////////////////////////////////////////////////////////////////////////
		/*template<typename MtxValueT, typename VecValueT>
		void mrwCountZeros(const smatrix<MtxValueT>& A, VecValueT*const pVec)noexcept {
			if (A.numel() < Thresholds_t::mrwCountZeros) {
				get_self().mrwCountZeros_st(A, pVec);
			} else get_self().mrwCountZeros_mt(A, pVec);
		}
		template<typename MtxValueT, typename VecValueT>
		void mrwCountZeros_st(const smatrix<MtxValueT>& A, VecValueT*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			//TODO: should be branched by rows/cols
			if (A.rows() < Thresholds_t::mrwCountZeros_st_rows) {
				get_self().mrwCountZeros_st_cw(A, pVec, pRCR);
			} else get_self().mrwCountZeros_st_rw(A, pVec, pRCR);
		}
		template<typename MtxValueT, typename VecValueT>
		void mrwCountZeros_mt(const smatrix<MtxValueT>& A, VecValueT*const pVec)noexcept {
			if (A.rows() < Thresholds_t::mrwCountZeros_mt_rows) {
				get_self().mrwCountZeros_mt_cw(A, pVec);
			} else get_self().mrwCountZeros_mt_rw(A, pVec);
		}
		template<typename MtxValueT, typename VecValueT>
		static void mrwCountZeros_st_cw(const smatrix<MtxValueT>& A, VecValueT*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			_mrwVecOperation_st_cw(A, pVec, 0, pRCR ? *pRCR : rowcol_range(A), _mrw_COUNT_Zeros());
		}
		template<typename MtxValueT, typename VecValueT>
		static void mrwCountZeros_st_rw(const smatrix<MtxValueT>& A, VecValueT*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			_mrwVecOperation_st_rw(A, pVec, pRCR ? *pRCR : rowcol_range(A), _mrw_COUNT_Zeros());
		}
		template<typename MtxValueT, typename VecValueT>
		void mrwCountZeros_mt_cw(const smatrix<MtxValueT>& A, VecValueT*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			_processMtx_cw(A, [&A, pVec, this](const rowcol_range& RCR)noexcept {//, const thread_id_t _tid) {
				get_self().mrwCountZeros_st(A, pVec, &RCR);
			});
		}
		template<typename MtxValueT, typename VecValueT>
		void mrwCountZeros_mt_rw(const smatrix<MtxValueT>& A, VecValueT*const pVec)noexcept {
			_processMtx_rw(A, [&A, pVec, this](const rowcol_range& RCR)noexcept {
				get_self().mrwCountZeros_st(A, pVec, &RCR);
			});
		}*/

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		template<typename T>
		numel_cnt_t mrwIdxsOfMax_needTempMem(const smatrix<T>& A)const noexcept {
			return get_self().mrwIdxsOfMax_needTempMem<T>(A.rows());
		}
		template<typename T>
		numel_cnt_t mrwIdxsOfMax_needTempMem(const vec_len_t maxRows)const noexcept {
			static_assert(::std::is_pod<T>::value, "");
			// max memory is required by _processMtx_cw() from mrwIdxsOfMax_mt_cw()
			// also see mrwIdxsOfMax_mt_rw
			//note that mrwIdxsOfMax_mt_cw_small() requirements are different from this and are not accounted here!
			return ::std::max(_processMtx_cw_needTempMem<T>(maxRows),numel_cnt_t(maxRows));
		}
		//fills array pDest of size m.rows() with column indexes of greatest element in each row of m
		// #TODO: must meet func.requirements
		// #TODO: must be parametrized by typename T
		void mrwIdxsOfMax(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			if (A.numel() < Thresholds_t::mrwIdxsOfMax) {
				get_self().mrwIdxsOfMax_st(A, pDest);
			} else get_self().mrwIdxsOfMax_mt(A, pDest);
		}
		//pMax must be specified if called from _mt to get rid of threads race condition
		void mrwIdxsOfMax_st(const realmtx_t& A, vec_len_t*const pDest, real_t*const pMax = nullptr, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!(!pMax ^ !pRCR));
// 			if (A.rows() < Thresholds_t::mrwIdxsOfMax_st_rows) {
			get_self().mrwIdxsOfMax_st_rw(A, pDest, pMax, pRCR);
// 			} else get_self().mrwIdxsOfMax_st_cw(A, pDest,pRCR,pMax);
			//get_self().mrwIdxsOfMax_st_rw_small(A, pDest, pRCR, pMax);
		}
		void mrwIdxsOfMax_mt(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			if (A.rows() < Thresholds_t::mrwIdxsOfMax_mt_rows && A.cols() > Thresholds_t::mrwIdxsOfMax_ColsPerThread) {
				get_self().mrwIdxsOfMax_mt_cw(A, pDest);
			} else get_self().mrwIdxsOfMax_mt_rw(A, pDest);
		}
		// not used (was slower than alternatives, but worked. May be good for some other architectures)
		void mrwIdxsOfMax_st_cw(const realmtx_t& A, vec_len_t*const pDest, real_t* pMax = nullptr, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!(!pMax ^ !pRCR));
			NNTL_ASSERT(!A.empty() && pDest && A.numel() > 0);
			const auto rm = A.rows();
			//treat the first column like the max. Then compare other columns with this column and update max'es
			_memset_rowrange<vec_len_t>(pDest, pRCR ? pRCR->colBegin : 0, rm, pRCR);
			if (A.cols() > 1) {
				const bool bSaveMax = !!pMax;
				if (!bSaveMax) pMax = get_self()._istor_alloc(rm);
				//memcpy(pMax, A.data(), sizeof(*pMax)*rm);
				_memcpy_rowcol_range(pMax, A, pRCR);
				if (bSaveMax) {
					_mrwVecOperation_st_cw(A, pMax, 1, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<true>(pDest));
				} else {
					_mrwVecOperation_st_cw(A, pMax, 1, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<false>(pDest));
					get_self()._istor_free(pMax, rm);
				}
			}
		}
		static void mrwIdxsOfMax_st_rw_small(const realmtx_t& A, vec_len_t*const pDest, real_t*const pMax = nullptr, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!(!pMax ^ !pRCR));
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			const bool bSaveMax = !!pMax;
			const auto rm = A.rows();
			NNTL_ASSERT(!pRCR || (rm == pRCR->totalRows() || A.cols() == pRCR->totalCols()));
			const auto ne = pRCR ? realmtx_t::sNumel(rm,pRCR->totalCols()) : A.numel();
			const vec_len_t colBegin = pRCR ? pRCR->colBegin : 0;
			auto pD = A.colDataAsVec(colBegin);//A.data();
			const auto rowEnd = pRCR ? pRCR->rowEnd : rm;
			const vec_len_t colBeginCmp = colBegin + 1;
			if (bSaveMax) {
				for (vec_len_t ri = pRCR ? pRCR->rowBegin : 0; ri < rowEnd; ++ri) {
					auto pV = pD + ri;
					const auto pVEnd = pV + ne;
					auto m = *pV;
					pV += rm;
					vec_len_t c = colBeginCmp, mIdx = colBegin;
					while (pV != pVEnd) {
						const auto v = *pV;
						pV += rm;
						if (v > m) {
							m = v;
							mIdx = c;
						}
						c++;
					}
					pDest[ri] = mIdx;
					pMax[ri] = m;
				}
			} else {
				for (vec_len_t ri = pRCR ? pRCR->rowBegin : 0; ri < rowEnd; ++ri) {
					auto pV = pD + ri;
					const auto pVEnd = pV + ne;
					auto m = *pV;
					pV += rm;
					vec_len_t c = colBeginCmp, mIdx = colBegin;
					while (pV != pVEnd) {
						const auto v = *pV;
						pV += rm;
						if (v > m) {
							m = v;
							mIdx = c;
						}
						c++;
					}
					pDest[ri] = mIdx;
				}
			}			
		}
		void mrwIdxsOfMax_st_rw(const realmtx_t& A, vec_len_t*const pDest, real_t* pMax = nullptr, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!(!pMax ^ !pRCR));
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			const auto bSaveMax = !!pMax;
			if (!bSaveMax) pMax = get_self()._istor_alloc(A.rows());
			if (bSaveMax) {
				_mrwVecOperation_st_rw(A, pMax, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<true>(pDest));
			} else {
				_mrwVecOperation_st_rw(A, pMax, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<false>(pDest));
				get_self()._istor_free(pMax, A.rows());
			}
		}

		void mrwIdxsOfMax_mt_cw(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			_processMtx_cw<vec_len_t>(A, Thresholds_t::mrwIdxsOfMax_ColsPerThread
				, [&A, this](const rowcol_range& RCR, real_t*const pMax, vec_len_t*const pIdxsVec)noexcept
			{
				get_self().mrwIdxsOfMax_st(A, pIdxsVec, pMax, &RCR);
			}, [pDest, this](realmtx_t& fin, vec_len_t*const pIdxsStor)noexcept {
				//now we should gather temporary max'es into the final max'es&indexes
				const auto pMaxStor = fin.data();
				const auto _elmsCnt = fin.numel();
				const auto rm = fin.rows();
				const auto pE = pMaxStor + _elmsCnt;
				auto p = pMaxStor + rm;
				auto pIdxs = pIdxsStor + rm;
				while (p != pE) {
					for (vec_len_t r = 0; r < rm; ++r) {
						const auto pM = pMaxStor + r;
						const auto v = p[r];
						if (v > *pM) {
							*pM = v;
							pIdxsStor[r] = pIdxs[r];
						}
					}
					p += rm;
					pIdxs += rm;
				}
				memcpy(pDest, pIdxsStor, sizeof(vec_len_t)*rm);
			});
		}
		// not used (was slower than alternatives, but worked. May be good for some other architectures)
		//Note that is has differenet mem requirements than of _mt_cw and they are not taken into account in _needTempMem()
		/*
		void mrwIdxsOfMax_mt_cw_small(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			NNTL_ASSERT(Thresholds_t::mrwIdxsOfMax_ColsPerThread >= 3);//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			const auto cm = A.cols(), rm = A.rows();
			NNTL_ASSERT(cm > Thresholds_t::mrwIdxsOfMax_ColsPerThread);
			//if (cm <= Thresholds_t::mrwIdxsOfMax_ColsPerThread) return get_self().mrwIdxsOfMax_st_cw(m, pDest);
			//will calculate max and indexes of each column-set independently and then merge them together
			const auto minThreadsReq = static_cast<thread_id_t>(ceil(real_t(cm) / Thresholds_t::mrwIdxsOfMax_ColsPerThread));
			NNTL_ASSERT(minThreadsReq > 1);
			auto workersCnt = m_threads.cur_workers_count();
			if (minThreadsReq < workersCnt) workersCnt = minThreadsReq;

			// now we'll need rm*workersCnt of real_t's to store temp maxs for each column-set
			// and rm*workersCnt of vec_len_t's to store indexes.
			const auto _elmsCnt = realmtx_t::sNumel(rm, workersCnt);
			const auto tmemSize = _elmsCnt + static_cast<numel_cnt_t>(ceil((real_t(sizeof(vec_len_t)) / sizeof(real_t))*_elmsCnt));
			const auto pTmp = get_self()._istor_alloc(tmemSize);
			const auto pMaxStor = pTmp;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			vec_len_t*const pIdxsStor = reinterpret_cast<vec_len_t*>(pTmp + _elmsCnt);

			//now we may run max calculation in parallel for each column-set into pMaxStor and pIdxsStor
			auto pMD = A.data();
			m_threads.run([rm, pMaxStor, pIdxsStor, pMD](const par_range_t& pr) noexcept{
				const auto _tid = pr.tid();
				const auto _tmpElmOffset = realmtx_t::sNumel(rm, _tid);
				auto pTMax = pMaxStor + _tmpElmOffset;
				auto pTDest = pIdxsStor + _tmpElmOffset;
				//now we have to decide what column range we should process here
				const auto firstColIdx = pr.offset();
				auto p = pMD + firstColIdx*rm;
				const auto pE = p + pr.cnt()*rm;

				//::std::fill(pTDest, pTDest + rm, static_cast<vec_len_t>(firstColIdx)); //doesn't get vectorized!
				{
					const auto v = static_cast<vec_len_t>(firstColIdx);
					for (vec_len_t i = 0; i < rm; i++) pTDest[i] = v;
				}

				memcpy(pTMax, p, sizeof(real_t)*rm);
				p += rm;
				vec_len_t c = static_cast<vec_len_t>(firstColIdx + 1);
				while (p != pE) {
					for (vec_len_t r = 0; r < rm; ++r) {
						const auto pM = pTMax + r;
						const auto v = p[r];
						if (v > *pM) {
							*pM = v;
							pTDest[r] = c;
						}
					}
					++c;
					p += rm;
				}
			}, cm, workersCnt);

			//now we should gather temporary max'es into the final max'es&indexes
			const auto pE = pMaxStor + _elmsCnt;
			auto p = pMaxStor + rm;
			auto pIdxs = pIdxsStor + rm;
			while (p != pE) {
				for (vec_len_t r = 0; r < rm; ++r) {
					const auto pM = pMaxStor + r;
					const auto v = p[r];
					if (v > *pM) {
						*pM = v;
						pIdxsStor[r] = pIdxs[r];
					}
				}
				p += rm;
				pIdxs += rm;
			}
			memcpy(pDest, pIdxsStor, sizeof(vec_len_t)*rm);
			get_self()._istor_free(pTmp, tmemSize);
		}
		}*/
		void mrwIdxsOfMax_mt_rw(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			const auto pMax = get_self()._istor_alloc(A.rows());
			_processMtx_rw(A, [&A, pDest, pMax, this](const rowcol_range& RCR)noexcept {
				NNTL_ASSERT(RCR.colBegin == 0 && RCR.colEnd == A.cols());
				get_self().mrwIdxsOfMax_st_rw(A, pDest, pMax, &RCR);
			});
			get_self()._istor_free(pMax, A.rows());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//fills array pMax of size m.rows() with maximum element in each row of m
		// TODO: must meet func.requirements
		void mrwMax(const realmtx_t& A, real_t*const pMax)noexcept {
			if (A.numel() < Thresholds_t::mrwMax) {
				get_self().mrwMax_st(A, pMax);
			} else get_self().mrwMax_mt(A, pMax);
		}
		void mrwMax_st(const realmtx_t& A, real_t*const pMax, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self().mrwMax_st_cw(A, pMax, pRCR);
		}
		void mrwMax_mt(const realmtx_t& A, real_t*const pMax)noexcept {
			get_self().mrwMax_mt_rw(A, pMax);
		}

		static void mrwMax_st_cw(const realmtx_t& A, real_t*const pMax, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && pMax && A.numel() > 0);
			//treat the first column like the max. Then compare other columns with this column and update max'es
			_memcpy_rowcol_range(pMax, A, pRCR);
			if (A.cols() > 1) _mrwVecOperation_st_cw(A, pMax, 1, pRCR ? *pRCR : rowcol_range(A), _mrwFind_MAX());
		}
		//may be good for some certain datasizes, so leave it for now
		static void mrwMax_st_rw_small(const realmtx_t& A, real_t*const pMax, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pMax);
			const auto rm = A.rows();
			const auto ne = A.numel();
			auto pD = A.data();
			const auto rowEnd = pRCR ? pRCR->rowEnd : rm;
			for (vec_len_t ri = pRCR ? pRCR->rowBegin : 0; ri < rowEnd; ++ri) {
				auto pV = pD + ri;
				const auto pVEnd = pV + ne;
				auto m = *pV;
				pV += rm;
				while (pV != pVEnd) {
					const auto v = *pV;
					pV += rm;
					if (v > m) m = v;
				}
				pMax[ri] = m;
			}
		}
		static void mrwMax_st_rw(const realmtx_t& A, real_t*const pMax, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pMax);
			_mrwVecOperation_st_rw(A, pMax, pRCR ? *pRCR : rowcol_range(A), _mrwFind_MAX());
		}
		void mrwMax_mt_rw(const realmtx_t& A, real_t*const pMax)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pMax);
			_processMtx_rw(A, [&A, pMax, this](const rowcol_range& RCR)noexcept {
				get_self().mrwMax_st(A, pMax, &RCR);
			});
		}

		//code is ok, just is not used. Note, that it requires _processMtx_cw_needTempMem() preallocated!
		/*void mrwMax_mt_cw(const realmtx_t& A, real_t*const pMax)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pMax);
			_processMtx_cw(A, Thresholds_t::mrwMax_mt_cw_ColsPerThread, [&A, this](const rowcol_range& RCR, real_t*const pVec)noexcept {
				get_self().mrwMax_st(A, pVec, &RCR);
			}, [pMax, this](const realmtx_t& fin) noexcept{
				get_self().mrwMax_st(fin, pMax);
			});
		}*/

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		template<typename T>
		nntl_probably_force_inline numel_cnt_t mrwSum_ip_needTempMem(const smatrix<T>& A)const noexcept {
			return get_self().mrwSum_needTempMem(A);
		}		
		// Calculates into the first row of A a sum of columns for each row.
		void mrwSum_ip(realmtx_t& A)noexcept {
			const auto cm = A.cols();
			if (cm <= 1) return;
			get_self().mrwSum_ip_st(A);
		}
		void mrwSum_ip_st(realmtx_t& A, const rowcol_range*const pRCR = nullptr)noexcept {
			const auto cm = A.cols();
			if (cm <= 1) return;
			if (cm < Thresholds_t::mrwSum_ip_st_cols && A.rows() < Thresholds_t::mrwSum_ip_st_rows) {
				get_self().mrwSum_ip_st_cw(A, pRCR);
			} else get_self().mrwSum_ip_st_rw_small(A, pRCR);
		}
		void mrwSum_ip_mt(realmtx_t& A)noexcept {
			const auto cm = A.cols();
			if (cm <= 1) return;
			if (cm <= ::std::max(Thresholds_t::mrwSum_mt_cw_colsPerThread, m_threads.cur_workers_count())) {
				get_self().mrwSum_ip_mt_rw(A);
			} else get_self().mrwSum_ip_mt_cw(A);
		}
		static void mrwSum_ip_st_cw(realmtx_t& A, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			_mrwVecOperation_st_cw(A, A.data(), 1, pRCR ? *pRCR : rowcol_range(A), _mrw_SUM());
		}
		//This version provide some gains over "standartized" version with small data sizes, so I'll leave it here
		static void mrwSum_ip_st_rw_small(realmtx_t& A, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			const auto rm = A.rows();
			const auto neml = A.numel() - rm;
			auto pSum = A.data() + (pRCR ? pRCR->rowBegin : 0);
			auto pRow = pSum + rm;
			const auto pRowE = pRow + (pRCR ? pRCR->totalRows() : rm);
			while (pRow != pRowE) {
				auto pElm = pRow++;
				const auto pElmE = pElm + neml;
				real_t s = real_t(0.0);
				while (pElm != pElmE) {
					s += *pElm;
					pElm += rm;
				}
				*pSum++ += s;
			}
		}
		static void mrwSum_ip_st_rw(realmtx_t& A, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			_mrwVecOperation_st_rw(A, A.data(), pRCR ? *pRCR : rowcol_range(A), _mrw_SUM());
		}
		void mrwSum_ip_mt_cw(realmtx_t& A)noexcept {
			mrwSum_mt_cw(A, A.data());
		}
		void mrwSum_ip_mt_rw(realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			_processMtx_rw(A, [&A, this](const rowcol_range& RCR)noexcept {
				get_self().mrwSum_ip_st(A, &RCR);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		template<typename T>
		nntl_probably_force_inline numel_cnt_t mrwSum_needTempMem(const smatrix<T>& A)const noexcept {
			return _processMtx_cw_needTempMem<T>(A);
		}
		// Calculate rowwise sum into vec
		void mrwSum(const realmtx_t& A, real_t*const pVec)noexcept {
			if (A.numel() < Thresholds_t::mrwSum) {
				get_self().mrwSum_st(A, pVec);
			}else get_self().mrwSum_mt(A, pVec);
		}
		//#TODO: how are we going to branch the code when pRCR is present? Desperately NEED run-time profiler!!!
		void mrwSum_st(const realmtx_t& A, real_t*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			if (A.cols() == 1 || (pRCR && pRCR->colBegin == pRCR->colEnd)) {
				_memcpy_rowcol_range(pVec, A, pRCR);
			} else {
				if (A.numel() < Thresholds_t::mrwSum_st) {
					get_self().mrwSum_st_rw(A, pVec, pRCR);
				}else get_self().mrwSum_st_cw(A, pVec, pRCR);
			}
		}
		void mrwSum_mt(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			const auto cm = A.cols();
			if (cm == 1) {
				memcpy(pVec, A.data(), A.byte_size());
			} else {
				if (cm <= ::std::max(Thresholds_t::mrwSum_mt_cw_colsPerThread, m_threads.cur_workers_count())) {
					get_self().mrwSum_mt_rw(A, pVec);
				}else get_self().mrwSum_mt_cw(A, pVec);				
			}
		}
		static void mrwSum_st_rw(const realmtx_t& A, real_t*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > 1 && pVec);
			_mrwVecOperation_st_rw(A, pVec, pRCR ? *pRCR : rowcol_range(A), _mrw_SUM());
		}
		static void mrwSum_st_cw(const realmtx_t& A, real_t*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > 1 && pVec);
			//memset(pVec, 0, sizeof(*pVec)*A.rows());
			_memset_rowrange(pVec, real_t(0.), A.rows(), pRCR);
			_mrwVecOperation_st_cw(A, pVec, 0, pRCR ? *pRCR : rowcol_range(A), _mrw_SUM());
		}
		void mrwSum_mt_rw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > 1 && pVec);
			_processMtx_rw(A, [&A, pVec, this](const rowcol_range& RCR)noexcept {
				get_self().mrwSum_st(A, pVec, &RCR);
			});
		}
		void mrwSum_mt_cw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > Thresholds_t::mrwSum_mt_cw_colsPerThread && pVec);
			//will sum partial matrices into temp memory, and then sum it to pVec
			_processMtx_cw(A, Thresholds_t::mrwSum_mt_cw_colsPerThread, [&A, this](const rowcol_range& RCR, real_t*const pVec)noexcept {
				get_self().mrwSum_st(A, pVec, &RCR);
			}, [pVec, this](const realmtx_t& fin)noexcept {
				get_self().mrwSum_st(fin, pVec);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// performs rowwise binary OR operation over the matrix A into a pVec.
		// This function is intended to be used on biases only, therefore A must be a (strong) binary (i.e. contain only 1.0 or +0.0)
		// If A contains only zeros and ones, the resulting column vector pVec contains 1 if corresponding row has at least single 1
		// and zero otherwise.
		// Heavily based on mrwBinaryOR
		template<typename T>
		nntl_probably_force_inline numel_cnt_t mrwOr_needTempMem(const smatrix<T>& A)const noexcept {
			return get_self().mrwBinaryOR_needTempMem(A);
		}

		void mrwOr(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}
		void mrwOr_st(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR_st(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}
		void mrwOr_st_cw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR_st_cw(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}
		void mrwOr_st_rw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR_st_rw(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}
		void mrwOr_mt(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR_mt(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}
		void mrwOr_mt_cw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR_mt_cw(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}
		void mrwOr_mt_rw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(A._isBinary());
			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			mrwBinaryOR_mt_rw(reinterpret_cast<const smatrix<similar_FWI_t>&>(A), reinterpret_cast<similar_FWI_t*const>(pVec));
		}


		template<typename T>
		nntl_probably_force_inline numel_cnt_t mrwBinaryOR_needTempMem(const smatrix<T>& A)const noexcept {
			return _processMtx_cw_needTempMem<T>(A);
		}
		template<typename BaseT>
		void mrwBinaryOR(const smatrix<BaseT>& A, BaseT*const pVec)noexcept {
			NNTL_ASSERT(!A.emulatesBiases());
			/*if (A.numel() < Thresholds_t::mrwBinaryOR) {
				get_self().mrwBinaryOR_st(A, pVec);
			} else get_self().mrwBinaryOR_mt(A, pVec);*/
			//stick to probably most universal version until a profiler is ready
			get_self().mrwBinaryOR_st_cw(A, pVec);
		}
		//#TODO: how are we going to branch code when pRCR is present? Desperately NEED run-time profiler!!!
		template<typename BaseT>
		void mrwBinaryOR_st(const smatrix<BaseT>& A, BaseT*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			if (A.cols() == 1 || (pRCR && pRCR->colBegin == pRCR->colEnd)) {
				_memcpy_rowcol_range(pVec, A, pRCR);
			} else {
				/*if (A.numel() < Thresholds_t::mrwBinaryOR_st) {
					get_self().mrwBinaryOR_st_rw(A, pVec, pRCR);
				} else get_self().mrwBinaryOR_st_cw(A, pVec, pRCR);*/
				//stick to probably most universal version until a profiler is ready
				get_self().mrwBinaryOR_st_cw(A, pVec, pRCR);
			}
		}
		template<typename BaseT>
		void mrwBinaryOR_mt(const smatrix<BaseT>& A, BaseT*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			const auto cm = A.cols();
			if (cm == 1) {
				memcpy(pVec, A.data(), A.byte_size());
			} else {
				if (cm <= ::std::max(Thresholds_t::mrwBinaryOR_mt_cw_colsPerThread, m_threads.cur_workers_count())) {
					get_self().mrwBinaryOR_mt_rw(A, pVec);
				} else get_self().mrwBinaryOR_mt_cw(A, pVec);
			}
		}
		template<typename BaseT>
		static void mrwBinaryOR_st_rw(const smatrix<BaseT>& A, BaseT*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			if (A.cols() == 1 || (pRCR && pRCR->colBegin == pRCR->colEnd)) {
				_memcpy_rowcol_range(pVec, A, pRCR);
			} else {
				_mrwVecOperation_st_rw(A, pVec, pRCR ? *pRCR : rowcol_range(A), _mrw_BinaryOR());
			}
		}
		template<typename BaseT>
		static void mrwBinaryOR_st_cw(const smatrix<BaseT>& A, BaseT*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			if (A.cols() == 1 || (pRCR && pRCR->colBegin == pRCR->colEnd)) {
				_memcpy_rowcol_range(pVec, A, pRCR);
			} else {
				//memset(pVec, 0, sizeof(*pVec)*A.rows());
				_memset_rowrange(pVec, BaseT(0.), A.rows(), pRCR);
				_mrwVecOperation_st_cw(A, pVec, 0, pRCR ? *pRCR : rowcol_range(A), _mrw_BinaryOR());
			}
		}
		template<typename BaseT>
		void mrwBinaryOR_mt_rw(const smatrix<BaseT>& A, BaseT*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			if (A.cols() == 1) {
				memcpy(pVec, A.data(), A.byte_size());
			} else {
				_processMtx_rw(A, [&A, pVec, this](const rowcol_range& RCR)noexcept {
					get_self().mrwBinaryOR_st(A, pVec, &RCR);
				});
			}
		}		
		template<typename BaseT>
		void mrwBinaryOR_mt_cw(const smatrix<BaseT>& A, BaseT*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > Thresholds_t::mrwBinaryOR_mt_cw_colsPerThread && pVec);
			//will sum partial matrices into temp memory, and then sum it to pVec
			_processMtx_cw(A, Thresholds_t::mrwBinaryOR_mt_cw_colsPerThread, [&A, this](const rowcol_range& RCR, BaseT*const pVec)noexcept {
				get_self().mrwBinaryOR_st(A, pVec, &RCR);
			}, [pVec, this](const smatrix<BaseT>& fin)noexcept {
				get_self().mrwBinaryOR_st(fin, pVec);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Matrix ColWise operations
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////
		// compute colwise mean
		// pVec must address at least A.cols() elements
		template<bool bNumStab, typename _T>
		void mcwMean(const smatrix<_T>& A, _T*const pVec)noexcept{
			if (A.numel() < Thresholds_t::mcwMean<bNumStab>::v) {
				get_self().mcwMean_st<bNumStab>(A, pVec);
			} else get_self().mcwMean_mt<bNumStab>(A, pVec);
		}

		template<bool bNumStab, typename _T>
		void mcwMean_st(const smatrix<_T>& A, _T*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self()._imcwMean_st<bNumStab>(A, pVec, pRCR ? *pRCR : rowcol_range(A));
		}

		template<bool bNumStab, typename _T>
		void mcwMean_mt(const smatrix<_T>& A, _T*const pVec)noexcept {
			_processMtx_cw(A, [&A, &pVec, this](const rowcol_range& rcr) noexcept {
				get_self()._imcwMean_st<bNumStab>(A, pVec, rcr);
			});
		}

		template<bool bNumStab, typename _T>
		static void _imcwMean_st(const smatrix<_T>& A, _T* pVec, const rowcol_range& rcr)noexcept {
			NNTL_ASSERT(pVec && !A.empty());
			NNTL_ASSERT(rcr.rowBegin == 0 && rcr.rowEnd == A.rows());
			NNTL_ASSERT(!A.emulatesBiases());//we're not expecting matrices with biases here
			const auto* pA = A.colDataAsVec(rcr.colBegin);
			const auto*const pAE = A.colDataAsVec(rcr.colEnd);
			const ptrdiff_t rc = A.rows();
			const ext_real_t N = static_cast<ext_real_t>(rc);
			pVec += rcr.colBegin;

			while (pA != pAE) {
				const auto pCur = pA;
				pA += rc;
				//*pVec++ = ::std::accumulate(pCur, pA, _T(0.)) / rc;
				*pVec++ = static_cast<_T>(static_cast<ext_real_t>(_vec_sum<bNumStab>(pCur, rc)) / N);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// subtract column mean, i.e. m = m-mean(m) colwise.
		template<bool bNumStab, typename _T>
		void mcwDeMean(smatrix<_T>& A)noexcept {
			if (A.numel() < Thresholds_t::mcwDeMean<bNumStab>::v) {
				get_self().mcwDeMean_st<bNumStab>(A);
			} else get_self().mcwDeMean_mt<bNumStab>(A);
		}

		template<bool bNumStab, typename _T>
		void mcwDeMean_st(smatrix<_T>& A, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self()._imcwDeMean_st<bNumStab>(A, pRCR ? *pRCR : rowcol_range(A));
		}

		template<bool bNumStab, typename _T>
		void mcwDeMean_mt(smatrix<_T>& A)noexcept {
			_processMtx_cw(A, [&A, this](const rowcol_range& rcr) noexcept {
				get_self()._imcwDeMean_st<bNumStab>(A, rcr);
			});
		}

		template<bool bNumStab, typename _T>
		static void _imcwDeMean_st(smatrix<_T>& A, const rowcol_range& rcr)noexcept {
			NNTL_ASSERT(!A.empty());
			NNTL_ASSERT(rcr.rowBegin == 0 && rcr.rowEnd == A.rows());
			NNTL_ASSERT(!A.emulatesBiases());//we're not expecting matrices with biases here
			auto* pA = A.colDataAsVec(rcr.colBegin);
			auto*const pAE = A.colDataAsVec(rcr.colEnd);
			const ptrdiff_t rc = A.rows();
			const ext_real_t N = static_cast<ext_real_t>(rc);
			
			while (pA != pAE) {
				const auto pCur = pA;
				pA += rc;
				const auto _mean = static_cast<_T>(static_cast<ext_real_t>(_vec_sum<bNumStab>(pCur, rc)) / N);
				for (ptrdiff_t i = 0; i < rc; ++i) {
					pCur[i] -= _mean;
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// calc and subtract column mean only for nonzero elements, i.e. m(m~=0) = m(m~=0) - mean(m(m~=0)) colwise.
		template<bool bNumStab, typename _T>
		void mcwDeMeanNZ(smatrix<_T>& A, const _T centralPoint = _T(0.))noexcept {
			if (A.numel() < Thresholds_t::mcwDeMeanNZ<bNumStab>::v) {
				get_self().mcwDeMeanNZ_st<bNumStab>(A, centralPoint);
			} else get_self().mcwDeMeanNZ_mt<bNumStab>(A, centralPoint);
		}

		template<bool bNumStab, typename _T>
		void mcwDeMeanNZ_st(smatrix<_T>& A, const _T centralPoint = _T(0.), const rowcol_range*const pRCR = nullptr)noexcept {
			get_self()._imcwDeMeanNZ_st<bNumStab>(A, centralPoint, pRCR ? *pRCR : rowcol_range(A));
		}

		template<bool bNumStab, typename _T>
		void mcwDeMeanNZ_mt(smatrix<_T>& A, const _T centralPoint = _T(0.))noexcept {
			_processMtx_cw(A, [&A, centralPoint, this](const rowcol_range& rcr) noexcept {
				get_self()._imcwDeMeanNZ_st<bNumStab>(A, centralPoint, rcr);
			});
		}

		//#TODO implement tests for that subroutine
		template<bool bNumStab, typename _T>
		static void _imcwDeMeanNZ_st(smatrix<_T>& A, const _T centralPoint, const rowcol_range& rcr)noexcept {
			NNTL_ASSERT(!A.empty());
			NNTL_ASSERT(rcr.rowBegin == 0 && rcr.rowEnd == A.rows());
			NNTL_ASSERT(!A.emulatesBiases());//we're not expecting matrices with biases here
			auto* pA = A.colDataAsVec(rcr.colBegin);
			auto*const pAE = A.colDataAsVec(rcr.colEnd);
			const ptrdiff_t rc = A.rows();
			const ext_real_t N = static_cast<ext_real_t>(rc);

			while (pA != pAE) {
				_T* __restrict pCur = pA;
				pA += rc;

				func_SUMNZ<_T, bNumStab> fsum(centralPoint);
				_vec_apply_func(pCur, rc, fsum);
				const auto _mean = static_cast<_T>(static_cast<ext_real_t>(fsum.result()) / fsum.count() - centralPoint);

				/*numel_cnt_t _cnt = 0;
				_T ret = _T(0.);
				for (ptrdiff_t i = 0; i < rc; ++i) {//nothing helps to vectorize it...
					const auto v = pCur[i];
					const auto b = (v != centralPoint);
					const numel_cnt_t inc = b ? 1 : 0;
					const _T upd = b ? v : _T(0.);
					_cnt += inc;
					ret += upd;
				}
				const auto _mean = static_cast<_T>(static_cast<ext_real_t>(ret) / _cnt - centralPoint);*/


				for (ptrdiff_t i = 0; i < rc; ++i) {//vectorized
					pCur[i] = (pCur[i] == centralPoint) ? centralPoint : pCur[i] - _mean;
				}
			}
		}


		//////////////////////////////////////////////////////////////////////////
		// subtract from each matrix column a corresponding vector element: A(:,j) = A(:,j) - pVec(j)
		template<typename BaseT>
		void mcwSub_ip(smatrix<BaseT>& A, const BaseT* pVec)noexcept {
			if (A.numel() < Thresholds_t::mcwSub_ip) {
				get_self().mcwSub_ip_st(A, pVec);
			} else get_self().mcwSub_ip_mt(A, pVec);
		}
		template<typename BaseT>
		void mcwSub_ip_st(smatrix<BaseT>& A, const BaseT* pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self()._mcwVecOperation_st(A, pVec, pRCR ? *pRCR : rowcol_range(A), _mcwSUB_ip());
		}
		template<typename BaseT>
		void mcwSub_ip_mt(smatrix<BaseT>& A, const BaseT* pVec)noexcept {
			_processMtx_cw(A, [&A, &pVec, this](const rowcol_range& rcr)noexcept {
				get_self().mcwSub_ip_st(A, pVec, &rcr);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//perform elementwise multiplication of a vector of diagonal elements of the matrix B to corresponding columns of matrix A
		// i.e. A(:,j) = A(:,j) .* B(j,j);
		template<typename BaseT>
		void mcwMulDiag_ip(smatrix<BaseT>& A, const smatrix<BaseT>& B)noexcept {
			if (A.numel() < Thresholds_t::mcwMulDiag_ip) {
				get_self().mcwMulDiag_ip_st(A, B);
			} else get_self().mcwMulDiag_ip_mt(A, B);
		}
		template<typename BaseT>
		void mcwMulDiag_ip_st(smatrix<BaseT>& A, const smatrix<BaseT>& B, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self()._imcwMulDiag_ip_st(A, B, pRCR ? *pRCR : rowcol_range(A));
		}
		template<typename BaseT>
		void mcwMulDiag_ip_mt(smatrix<BaseT>& A, const smatrix<BaseT>& B)noexcept {
			_processMtx_cw(A, [&A, &B, this](const rowcol_range& rcr)noexcept {
				get_self()._imcwMulDiag_ip_st(A, B, rcr);
			});
		}
		template<typename _T>
		static void _imcwMulDiag_ip_st(smatrix<_T>& A, const smatrix<_T>& B, const rowcol_range& rcr)noexcept {
			NNTL_ASSERT(!A.empty() && !B.empty());
			NNTL_ASSERT(B.rows() == B.cols() || !"B must be a square matrix!");
			NNTL_ASSERT(A.cols() == B.cols());
			NNTL_ASSERT(rcr.rowBegin == 0 && rcr.rowEnd == A.rows());
			NNTL_ASSERT(rcr.colBegin < rcr.colEnd && rcr.colEnd <= A.cols());
			NNTL_ASSERT(!A.emulatesBiases() && !B.emulatesBiases());//we're not expecting matrices with biases here

			auto* pA = A.colDataAsVec(rcr.colBegin);
			const auto*const pAE = A.colDataAsVec(rcr.colEnd);
			const ptrdiff_t rc = A.rows();
			auto pB = B.colDataAsVec(rcr.colBegin) + rcr.colBegin;
			const ptrdiff_t nextB = B.rows() + 1;

			while (pA != pAE) {
				const auto pCur = pA;
				pA += rc;
				const auto v = *pB;
				pB += nextB;
				for (ptrdiff_t j = 0; j < rc; ++j) {
					pCur[j] *= v;
				}
			}
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// clone matrix columns to another more wide matrix
		// srcCols - matrix with source columns to be copied into dest
		// dest - destination matrix, must have the same rows count as srcCols
		// colSpec - array of vec_len_t of size colSpecCnt==srcCols.cols(). Each element specifies how many
		//		copies of a corresponding srcCols column must be made. For example, if colSpec is {2,3,4}, then
		//		the srcCols must have 3 columns. The first column is copied 2 times into first 2 columns of dest,
		//		the second - 3, the third - 4. Therefore, dest must contain 2+3+4=9 columns.		
		void mCloneCols(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec)noexcept {
			if (srcCols.cols() < Thresholds_t::mCloneCols) {
				get_self().mCloneCols_st(srcCols, dest, pColSpec);
			} else get_self().mCloneCols_mt(srcCols, dest, pColSpec);
		}
		void mCloneCols_st(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec, const vec_len_t firstCol=0, const vec_len_t _lastCol=0)noexcept {
			NNTL_ASSERT(!srcCols.empty() && !dest.empty());
			NNTL_ASSERT(pColSpec && srcCols.cols());
			NNTL_ASSERT(srcCols.rows() == dest.rows());
			NNTL_ASSERT(dest.cols() == ::std::accumulate(pColSpec, pColSpec+srcCols.cols(), vec_len_t(0)));
			NNTL_ASSERT(firstCol < _lastCol || _lastCol == 0);
			NNTL_ASSERT(firstCol < dest.cols() && _lastCol <= dest.cols());

			//now we'll find which index within colSpec corresponds to column number firstCol
			const vec_len_t csIdxMax = srcCols.cols() - 1;
			vec_len_t csIdx = 0//also indexes cols within srcCols
				, colsLeft = 0;
			for (; csIdx <= csIdxMax; ++csIdx) {
				const vec_len_t lc = colsLeft + pColSpec[csIdx];
				if (lc>=firstCol) {
					colsLeft = lc - firstCol;
					//we've found necessary index, it's in csIdx
					break;
				}else colsLeft = lc;
			}

			const vec_len_t destCols = dest.cols();
			NNTL_ASSERT(csIdx <= csIdxMax && colsLeft <= destCols);
			if (csIdx <= csIdxMax && colsLeft <= destCols) {
				auto pDest = dest.colDataAsVec(firstCol);
				const auto pDE = dest.colDataAsVec(_lastCol ? _lastCol : destCols);
				const auto _rows = static_cast<numel_cnt_t>(dest.rows());
				auto pSrc = srcCols.colDataAsVec(csIdx);
				const vec_len_t* pnCS = pColSpec + csIdx+1;
				while (pDest != pDE) {
					//lets find a source column to copy into pDest
					NNTL_ASSERT(pnCS <= &pColSpec[csIdxMax] || colsLeft);
					if (!colsLeft) {//should switch source column
						colsLeft = *pnCS++;
						pSrc += _rows;
					}
					--colsLeft;
					memcpy(pDest, pSrc, sizeof(*pSrc)*_rows);
					pDest += _rows;
				}
			}else{
				//#todo exception here?
				abort();
			}
		}
		void mCloneCols_mt(const realmtx_t& srcCols, realmtx_t& dest, const vec_len_t*const pColSpec)noexcept {
			NNTL_ASSERT(!srcCols.empty() && !dest.empty());
			NNTL_ASSERT(pColSpec && srcCols.cols());
			NNTL_ASSERT(srcCols.rows() == dest.rows());
			NNTL_ASSERT(dest.cols() == ::std::accumulate(pColSpec, pColSpec + srcCols.cols(), vec_len_t(0)));

			m_threads.run([&srcCols, &dest, pColSpec, this](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				get_self().mCloneCols_st(srcCols, dest, pColSpec, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt()));
			},dest.cols());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// clone a matrix column to another more wide matrix dest.cols() number of times
		// (optimized version of mCloneCols where srcCols.cols()==1 )
		void mCloneCol(const realmtx_t& srcCol, realmtx_t& dest)noexcept {
			if (dest.cols() < Thresholds_t::mCloneCol) {
				get_self().mCloneCol_st(srcCol, dest);
			} else get_self().mCloneCol_mt(srcCol, dest);
		}
		void mCloneCol_st(const realmtx_t& srcCol, realmtx_t&dest, const vec_len_t firstCol = 0, const vec_len_t _lastCol = 0)const noexcept {
			NNTL_ASSERT(!srcCol.empty() && !dest.empty());
			NNTL_ASSERT(1 == srcCol.cols());
			NNTL_ASSERT(srcCol.rows() == dest.rows());
			NNTL_ASSERT(firstCol < _lastCol || _lastCol == 0);
			NNTL_ASSERT(firstCol < dest.cols() && _lastCol <= dest.cols());

			const auto _rows = static_cast<numel_cnt_t>(srcCol.rows());
			const auto pSrc = srcCol.data();
			auto pD = dest.colDataAsVec(firstCol);
			const auto pDE = dest.colDataAsVec(_lastCol ? _lastCol : dest.cols());			
			const size_t rowByteSize = sizeof(*pSrc)*static_cast<size_t>(_rows);
			while (pD != pDE) {
				memcpy(pD, pSrc, rowByteSize);
				pD += _rows;
			}
		}
		void mCloneCol_mt(const realmtx_t& srcCol, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!srcCol.empty() && !dest.empty());
			NNTL_ASSERT(1 == srcCol.cols());
			NNTL_ASSERT(srcCol.rows() == dest.rows());

			m_threads.run([&srcCol, &dest, this](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				get_self().mCloneCol_st(srcCol, dest, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt()));
			}, dest.cols());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Transforms a data matrix to be used by tiled layer. For a data with biases it looks like this:
		//																	|x1_1...x1_n 1|		:transformed data_x
		//																	|........... 1|		:to be fed to the layer
		//	data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	===>	|xi_1...xi_n 1|
		//																	|........... 1|
		//																	|xk_1...xk_n 1|
		// For a data without biases the same, just drop all the ones in the picture.
		// If src is biased matrix, then src must be a matrix of size [m, k*n+1], dest - [k*m, n+1], also biased.
		//		Last column of dest is reserved to contain biases and must be preinitialized to 1s
		// If src doesn't have biases, then it's size must be equal to [m, k*n], dest.size() == [k*m, n]
		void mTilingRoll(const realmtx_t& src, realmtx_t& dest)noexcept {
			if (src.numel_no_bias()<Thresholds_t::mTilingRoll) {
				get_self().mTilingRoll_st(src, dest);
			} else get_self().mTilingRoll_mt(src, dest);
		}
		//#TODO Thresholds in this group are nuts.
		void mTilingRoll_st(const realmtx_t& src, realmtx_t& dest)noexcept {
			get_self().mTilingRoll_seqwrite_st(src, dest);
		}
		void mTilingRoll_mt(const realmtx_t& src, realmtx_t& dest)noexcept {
			if (dest.cols_no_bias() < Thresholds_t::mTilingRoll_mt_cols) {
				get_self().mTilingRoll_seqread_mt(src, dest);
			}else get_self().mTilingRoll_seqwrite_mt(src, dest);
			
		}
		//sequential reading version, firstCol and _lastCol applies to src matrix
		void mTilingRoll_seqread_st(const realmtx_t& src, realmtx_t& dest, const vec_len_t firstCol = 0, const vec_len_t _lastCol = 0)const noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() < dest.rows());
			NNTL_ASSERT(dest.cols_no_bias() < src.cols_no_bias());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());

			const vec_len_t lastCol = _lastCol ? _lastCol : src.cols_no_bias();
			NNTL_ASSERT(firstCol < src.cols_no_bias());
			NNTL_ASSERT(lastCol <= src.cols_no_bias());
			NNTL_ASSERT(firstCol <= lastCol);

			const auto m = static_cast<numel_cnt_t>(src.rows());
			const auto km = static_cast<numel_cnt_t>(dest.rows());
			const numel_cnt_t k = km / m;
			NNTL_ASSERT(km == k*m);//to make sure no rounding happened
			const vec_len_t n = dest.cols_no_bias();
			NNTL_ASSERT(src.cols_no_bias() == k*n);

			const vec_len_t kIdx = firstCol / n;
			auto pDFirst = dest.data() + m*kIdx;
			auto pD = pDFirst + km*(firstCol - kIdx*n);
			const auto pDLast = dest.colDataAsVec(n - 1);
			//NNTL_ASSERT(pD + m*(lastCol - firstCol) <= dest.colDataAsVec(n));
			auto pS = src.colDataAsVec(firstCol);
			const auto pSE = pS + m*(lastCol - firstCol);
			NNTL_ASSERT(pSE <= src.colDataAsVec(src.cols_no_bias()));
			while (pS != pSE) {
				NNTL_ASSERT(pD + m <= dest.colDataAsVec(n));
				NNTL_ASSERT(pS + m <= src.colDataAsVec(src.cols_no_bias()));
				memcpy(pD, pS, sizeof(*pD)*m);
				pS += m;
				if (pD < pDLast) {
					pD += km;//nIdx increments
				} else {//should increment kIdx and wrap nIdx to zero
					pDFirst += m;
					pD = pDFirst;
				}
			}

			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());
		}
		void mTilingRoll_seqread_mt(const realmtx_t& src, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() < dest.rows());
			NNTL_ASSERT(dest.cols_no_bias() < src.cols_no_bias());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());

			m_threads.run([&src, &dest, this](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				get_self().mTilingRoll_seqread_st(src, dest, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt()));
			}, src.cols_no_bias());

			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}
		//sequential writing version, firstCol and _lastCol applies to dest matrix
		void mTilingRoll_seqwrite_st(const realmtx_t& src, realmtx_t& dest, const vec_len_t firstCol = 0, const vec_len_t _lastCol = 0)const noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() < dest.rows());
			NNTL_ASSERT(dest.cols_no_bias() < src.cols_no_bias());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());

			const vec_len_t n = dest.cols_no_bias();
			const vec_len_t lastCol = _lastCol ? _lastCol : n;
			NNTL_ASSERT(firstCol < n);
			NNTL_ASSERT(lastCol <= n);
			NNTL_ASSERT(firstCol <= lastCol);

			const auto m = static_cast<numel_cnt_t>(src.rows());
			const auto k = static_cast<numel_cnt_t>(dest.rows()) / m;
			NNTL_ASSERT(dest.rows() == static_cast<vec_len_t>(k*m));//to make sure no rounding happened
			NNTL_ASSERT(src.cols_no_bias() == k*n);

			const numel_cnt_t nm = m*n;

			auto pD = dest.colDataAsVec(firstCol);
			//const auto pDE = pD + static_cast<numel_cnt_t>(dest.rows())*(lastCol - firstCol);
			const auto pDE = pD + __emulu(dest.rows(), lastCol - firstCol);

			auto pSFirst = src.colDataAsVec(firstCol);
			auto pS = pSFirst;
			const auto pSLast = src.colDataAsVec(src.cols_no_bias() - n);
			
			while (pD != pDE) {
				NNTL_ASSERT(pD + m <= dest.colDataAsVec(n));
				NNTL_ASSERT(pS + m <= src.colDataAsVec(src.cols_no_bias()));
				memcpy(pD, pS, sizeof(*pD)*m);
				pD += m;
				if (pS < pSLast) {
					pS += nm;
				} else {
					pSFirst += m;
					pS = pSFirst;
				}
			}
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());
		}
		void mTilingRoll_seqwrite_mt(const realmtx_t& src, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() < dest.rows());
			NNTL_ASSERT(dest.cols_no_bias() < src.cols_no_bias());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());

			m_threads.run([&src, &dest, this](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				get_self().mTilingRoll_seqwrite_st(src, dest, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt()));
			}, dest.cols_no_bias());

			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Transforms a data matrix from a tiled layer format back to normal. For a data with biases it looks like this:
		//																	|x1_1...x1_n 1|		:transformed data_x
		//																	|........... 1|		:to be fed to the layer
		//	data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	<===	|xi_1...xi_n 1|
		//																	|........... 1|
		//																	|xk_1...xk_n 1|
		// For a data without biases the same, just drop all the ones in the picture.
		// If src is biased matrix, then src must be a matrix of size [k*m, n+1], dest - [m, k*n+1], also biased.
		//		Last column of dest is reserved to contain biases and must be preinitialized to 1s
		// If src doesn't have biases, then it's size must be equal to [k*m, n], dest.size() == [k*m, n]
		void mTilingUnroll(const realmtx_t& src, realmtx_t& dest)noexcept {
			if (src.numel_no_bias() < Thresholds_t::mTilingUnroll) {
				get_self().mTilingUnroll_st(src, dest);
			} else get_self().mTilingUnroll_mt(src, dest);
		}
		//#TODO Thresholds in this group are nuts.
		void mTilingUnroll_st(const realmtx_t& src, realmtx_t& dest)noexcept {
			get_self().mTilingUnroll_seqread_st(src, dest);
		}
		void mTilingUnroll_mt(const realmtx_t& src, realmtx_t& dest)noexcept {
			if (dest.cols_no_bias() < Thresholds_t::mTilingUnroll_mt_cols) {
				get_self().mTilingUnroll_seqwrite_mt(src, dest);
			} else get_self().mTilingUnroll_seqread_mt(src, dest);

		}
		//sequential writing version, firstCol and _lastCol applies to dest matrix
		void mTilingUnroll_seqwrite_st(const realmtx_t& src, realmtx_t& dest, const vec_len_t firstCol = 0, const vec_len_t _lastCol = 0)const noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			NNTL_ASSERT(!(dest.emulatesBiases() ^ src.emulatesBiases()));
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() < src.rows());
			NNTL_ASSERT(src.cols_no_bias() < dest.cols_no_bias());
			//NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());

			const vec_len_t lastCol = _lastCol ? _lastCol : dest.cols_no_bias();
			NNTL_ASSERT(firstCol < dest.cols_no_bias());
			NNTL_ASSERT(lastCol <= dest.cols_no_bias());
			NNTL_ASSERT(firstCol <= lastCol);

			const auto m = static_cast<numel_cnt_t>(dest.rows());
			const auto km = static_cast<numel_cnt_t>(src.rows());
			const vec_len_t k = static_cast<vec_len_t>(km / m);
			NNTL_ASSERT(km == k*m);//to make sure no rounding happened
			const vec_len_t n = src.cols_no_bias();
			NNTL_ASSERT(dest.cols_no_bias() == k*n);

			//const vec_len_t srcFirstCol = firstCol / k;
			//auto pSFirst = src.colDataAsVec(srcFirstCol) + m*(firstCol - srcFirstCol*k);
			const vec_len_t kIdx = firstCol / n;
			auto pSFirst = src.data() + m*kIdx;
			auto pS = pSFirst + km*(firstCol - kIdx*n);
			const auto pSLast = src.colDataAsVec(n - 1);
			//NNTL_ASSERT(pS + m*(lastCol - firstCol) <= src.colDataAsVec(n));
			auto pD = dest.colDataAsVec(firstCol);
			const auto pDE = pD + m*(lastCol - firstCol);
			NNTL_ASSERT(pDE <= dest.colDataAsVec(dest.cols_no_bias()));
			while (pD != pDE) {
				NNTL_ASSERT(pS + m <= src.colDataAsVec(n));
				NNTL_ASSERT(pD + m <= dest.colDataAsVec(dest.cols_no_bias()));
				memcpy(pD, pS, sizeof(*pS)*m);
				pD += m;
				if (pS < pSLast) {
					pS += km;
				} else {
					pSFirst += m;
					pS = pSFirst;
				}
			}
			//NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());
		}
		void mTilingUnroll_seqwrite_mt(const realmtx_t& src, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			NNTL_ASSERT(!(dest.emulatesBiases() ^ src.emulatesBiases()));
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() < src.rows());
			NNTL_ASSERT(src.cols_no_bias() < dest.cols_no_bias());
			//NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());

			m_threads.run([&dest, &src, this](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				get_self().mTilingUnroll_seqwrite_st(src, dest, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt()));
			}, dest.cols_no_bias());

			//NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
		}
		//sequential reading version, firstCol and _lastCol applies to src matrix
		void mTilingUnroll_seqread_st(const realmtx_t& src, realmtx_t& dest, const vec_len_t firstCol = 0, const vec_len_t _lastCol = 0)const noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			NNTL_ASSERT(!(dest.emulatesBiases() ^ src.emulatesBiases()));
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() < src.rows());
			NNTL_ASSERT(src.cols_no_bias() < dest.cols_no_bias());
			//NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());

			const vec_len_t n = src.cols_no_bias();
			const vec_len_t lastCol = _lastCol ? _lastCol : n;
			NNTL_ASSERT(firstCol < n);
			NNTL_ASSERT(lastCol <= n);
			NNTL_ASSERT(firstCol <= lastCol);

			const auto m = static_cast<numel_cnt_t>(dest.rows());
			const vec_len_t k = src.rows() / static_cast<vec_len_t>(m);
			NNTL_ASSERT(src.rows() == static_cast<vec_len_t>(m*k));//to make sure no rounding happened
			NNTL_ASSERT(dest.cols_no_bias() == k*n);

			const numel_cnt_t nm = m*n;

			auto pS = src.colDataAsVec(firstCol);
			const auto pSE = pS + src.rows()*(lastCol - firstCol);

			auto pDFirst = dest.colDataAsVec(firstCol);
			auto pD = pDFirst;
			const auto pDLast = dest.colDataAsVec(dest.cols_no_bias() - n);

			while (pS != pSE) {
				NNTL_ASSERT(pS + m <= src.colDataAsVec(n));
				NNTL_ASSERT(pD + m <= dest.colDataAsVec(dest.cols_no_bias()));
				memcpy(pD, pS, sizeof(*pS)*m);
				pS += m;
				if (pD < pDLast) {
					pD += nm;
				} else {
					pDFirst += m;
					pD = pDFirst;
				}
			}
			//NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !dest.emulatesBiases() || dest.test_biases_strict());
			NNTL_ASSERT(firstCol != 0 || _lastCol != 0 || !src.emulatesBiases() || src.test_biases_strict());
		}
		void mTilingUnroll_seqread_mt(const realmtx_t& src, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.rows() && src.cols_no_bias());
			NNTL_ASSERT(dest.rows() && dest.cols_no_bias());
			NNTL_ASSERT(dest.rows() < src.rows());
			NNTL_ASSERT(src.cols_no_bias() < dest.cols_no_bias());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
			//NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());

			m_threads.run([&src, &dest, this](const par_range_t& pr) noexcept{
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				get_self().mTilingUnroll_seqread_st(src, dest, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt()));
			}, src.cols_no_bias());

			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());
			//NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}

	};


	template<typename RealT, typename iThreadsT, typename ThresholdsT = _impl::SMATH_THR<RealT>>
	class SMath final : public _SMath<RealT, iThreadsT, ThresholdsT, SMath<RealT, iThreadsT, ThresholdsT>> {
	public:
		~SMath()noexcept {}
		SMath()noexcept : _SMath<RealT, iThreadsT, ThresholdsT, SMath<RealT, iThreadsT, ThresholdsT>>() {}

	};
}
}
