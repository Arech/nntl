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

//This file contains implementation of some simplest generic-purpose(!) math algorithms over the data
// inside simple_matrix class object

#include "../_i_threads.h"
#include "simple_matrix.h"
#include "simple_math_thr.h"
#include <algorithm>

namespace nntl {
namespace math {

	//////////////////////////////////////////////////////////////////////////
	// _st versions of functions MUST NOT call generic and/or multithreaded function implementations (ONLY _st).
	//		(They may be used in future in some parallel algorithms)
	// _mt and generic function versions may use any suitable implementations.
	// generic, _st and _mt versions MUST accept any datasizes. However, their specializations,
	//		such as _mt_cw MAY put restrictions on acceptable data sizes.

	template<typename RealT, typename iThreadsT, typename ThresholdsT, typename FinalPolymorphChild>
	class _simple_math {
		static_assert(std::is_base_of<threads::_i_threads<typename iThreadsT::range_t>, iThreadsT>::value, "iThreads must implement threads::_i_threads");

	public:
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;

		typedef RealT real_t;
		typedef simple_matrix<real_t> realmtx_t;
		typedef typename realmtx_t::vec_len_t vec_len_t;
		typedef typename realmtx_t::numel_cnt_t numel_cnt_t;

		typedef simple_matrix_deformable<real_t> realmtxdef_t;

		typedef simple_rowcol_range<real_t> rowcol_range;
		typedef simple_elements_range<real_t> elms_range;

		typedef iThreadsT ithreads_t;
		typedef typename ithreads_t::range_t range_t;
		typedef typename ithreads_t::par_range_t par_range_t;
		typedef typename ithreads_t::thread_id_t thread_id_t;

		static_assert(std::is_same<typename realmtx_t::numel_cnt_t, typename iThreadsT::range_t>::value, "iThreads::range_t should be the same as realmtx_t::numel_cnt_t");

		//ALL branching functions require refactoring
		typedef ThresholdsT Thresholds_t;

		//TODO: probably don't need this assert
		static_assert(std::is_base_of<_impl::SIMPLE_MATH_THR<real_t>, Thresholds_t>::value, "Thresholds_t must be derived from _impl::SIMPLE_MATH_THR<real_t>");

	protected:
		typedef std::vector<real_t> thread_temp_storage_t;

		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		ithreads_t m_threads;

		numel_cnt_t m_minTempStorageSize;
		thread_temp_storage_t m_threadTempRawStorage;
		
		//////////////////////////////////////////////////////////////////////////
		// Technical Methods
	protected:
		void _assert_thread_storage_allocated(const numel_cnt_t maxDataSize)const noexcept {
			NNTL_ASSERT(m_minTempStorageSize >= maxDataSize);
			NNTL_ASSERT(m_threadTempRawStorage.size() >= m_minTempStorageSize);
		}

		static real_t _reduce_final_sum(real_t* _ptr, const range_t _cnt)noexcept {
			NNTL_ASSERT(_ptr && _cnt > 0);
			const auto pE = _ptr + _cnt;
			auto ret = *_ptr++;
			while (_ptr != pE) {
				ret += *_ptr++;
			}
			return ret;
		}

	public:
		~_simple_math()noexcept {}
		_simple_math()noexcept : m_minTempStorageSize(0) {}

		self_ref_t get_self() noexcept {
			static_assert(std::is_base_of<_simple_math<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild>, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _simple_math<RealT, iThreadsT, FinalPolymorphChild>");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<_simple_math<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild>, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _simple_math<RealT, iThreadsT, FinalPolymorphChild>");
			return static_cast<self_cref_t>(*this);
		}

		// use with care, it's kind of "internal memory" of the class object. Don't know, if really 
		// should expose it into public (for some testing purposes only at this moment)
		real_t* _get_thread_temp_raw_storage(const numel_cnt_t maxDataSize)noexcept {
			get_self()._assert_thread_storage_allocated(maxDataSize);
			return &m_threadTempRawStorage[0];
		}

		ithreads_t& ithreads()noexcept { return m_threads; }

		//math preinitialization, should be called from each NN layer. n - maximum data length (in real_t), that this layer will use in calls
		//to math interface. Used to calculate max necessary temporary storage length.
		void preinit(const numel_cnt_t n)noexcept {
			if (n > m_minTempStorageSize) m_minTempStorageSize = n;
		}

		//real math initialization, used to allocate necessary temporary storage of size max(preinit::n)
		bool init()noexcept {
			if (m_threadTempRawStorage.size() < m_minTempStorageSize) {
				//TODO: memory allocation exception handling here!
				m_threadTempRawStorage.resize(m_minTempStorageSize);
			}
			//return m_tmpMtx.resize(maxMem);
			return true;
		}
		void deinit()noexcept {
			m_threadTempRawStorage.clear();
		}

		//////////////////////////////////////////////////////////////////////////
		// Math Methods
	protected:
		template<typename T_>
		nntl_force_inline static void _memcpy_rowcol_range(T_* dest, const simple_matrix<T_>& A, const rowcol_range*const pRCR)noexcept {
			const T_* src;
			size_t rm;
			if (pRCR) {
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
		nntl_force_inline static void _memset_rowrange(T_* dest, const T_& v, size_t elems, const rowcol_range*const pRCR)noexcept {
			if (pRCR) {
				dest += pRCR->rowBegin;
				elems = pRCR->totalRows();
			}
			//memset(dest, src, sizeof(T_)*elems);
			std::fill(dest, dest + elems, v);
		}

		enum _OperationType {
			mrw_cw,//processing rows of matrix columnwise
			mrw_rw //processing rows of matrix rowwise
		};

		//////////////////////////////////////////////////////////////////////////
		// operation helpers
		struct _mrwHlpr_rw_InitVecElmByVec {
			static constexpr vec_len_t rw_FirstColumnIdx = 0;

			//size_t mtxRows - size_t by intention!
			template<typename VecBaseT, typename MtxBaseT>
			static VecBaseT rw_beforeInnerLoop(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const size_t mtxRows
				, const vec_len_t colBegin, const vec_len_t r)noexcept 
			{ return vecElm; }
		};
		struct _mrwHlpr_rw_InitVecElmByMtxElm {
			static constexpr vec_len_t rw_FirstColumnIdx = 1;

			//size_t mtxRows - size_t by intention!
			template<typename VecBaseT, typename MtxBaseT>
			static VecBaseT rw_beforeInnerLoop(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const size_t mtxRows
				, const vec_len_t colBegin, const vec_len_t r)noexcept
			{
				const auto v = *pFirstMtxElm;
				pFirstMtxElm += mtxRows;
				return v;
			}
		};
		struct _mrwHlpr_rw_Dont_UpdVecElm {
			template<typename BaseT>
			static void rw_afterInnerLoop(BaseT& vecElm, BaseT& v, const vec_len_t r)noexcept {}
		};
		struct _mrwHlpr_rw_UpdVecElm {
			template<typename BaseT>
			static void rw_afterInnerLoop(BaseT& vecElm, BaseT& v, const vec_len_t r)noexcept {
				vecElm = v;
			}
		};
		struct _mrwHlpr_simpleLoops {
			void beforeMainLoop(const vec_len_t colBegin, const vec_len_t mtxRows)noexcept {};

			void cw_afterInnerLoop(const size_t mtxRows)noexcept {};
		};

		//////////////////////////////////////////////////////////////////////////
		//operations
		struct _mrw_MUL_mtx_by_vec : public _mrwHlpr_rw_InitVecElmByVec, public _mrwHlpr_rw_Dont_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(BaseT& mtxElm, const BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				mtxElm *= vecElm;
			}
		};
		struct _mrw_DIV_mtx_by_vec : public _mrwHlpr_rw_InitVecElmByVec, public _mrwHlpr_rw_Dont_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(BaseT& mtxElm, const BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				mtxElm /= vecElm;
			}
		};
		struct _mrwFind_MAX : public _mrwHlpr_rw_InitVecElmByMtxElm, public _mrwHlpr_rw_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				const auto cv = mtxElm;
				if (cv > vecElm) vecElm = cv;
			}
		};
		template<bool bSaveMaxOnUpdate>
		struct _mrwFindIdxsOf_MAX : public _mrwHlpr_simpleLoops {
			vec_len_t*const pDest;
			vec_len_t _maxColumnIdx;
			_mrwFindIdxsOf_MAX(vec_len_t* pd)noexcept : pDest(pd) {}

			template<_OperationType OpType, typename BaseT>
			std::enable_if_t<OpType == mrw_cw> op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				const auto cv = mtxElm;
				if (cv > vecElm) {
					vecElm = cv;
					pDest[r] = c;
				}
			}
			template<_OperationType OpType, typename BaseT>
			std::enable_if_t<OpType == mrw_rw> op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				const auto cv = mtxElm;
				if (cv > vecElm) {
					vecElm = cv;
					_maxColumnIdx = c;
				}
			}

			static constexpr vec_len_t rw_FirstColumnIdx = 1;

			template<typename VecBaseT, typename MtxBaseT>
			VecBaseT rw_beforeInnerLoop(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const size_t mtxRows
				, const vec_len_t colBegin, const vec_len_t r)noexcept 
			{
				_maxColumnIdx = colBegin;
				const auto v = *pFirstMtxElm;
				pFirstMtxElm += mtxRows;
				return v;
			}

			template<typename BaseT, bool B = bSaveMaxOnUpdate>
			std::enable_if_t<!B> rw_afterInnerLoop(BaseT& vecElm, BaseT& v, const vec_len_t r)noexcept {
				pDest[r] = _maxColumnIdx;
			}
			template<typename BaseT, bool B = bSaveMaxOnUpdate>
			std::enable_if_t<B> rw_afterInnerLoop(BaseT& vecElm, BaseT& v, const vec_len_t r)noexcept {
				pDest[r] = _maxColumnIdx;
				vecElm = v;
			}
		};
		struct _mrw_SUM : public _mrwHlpr_rw_InitVecElmByMtxElm, public _mrwHlpr_rw_UpdVecElm, public _mrwHlpr_simpleLoops {
			template<_OperationType OpType, typename BaseT>
			static void op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				vecElm += mtxElm;
			}
		};

		//////////////////////////////////////////////////////////////////////////
		// Matrix/Vector elementwise operations
		//////////////////////////////////////////////////////////////////////////
		// Apply operation F.op to every element of matrix/vector A
		// #todo: make a wrapper to mate vector api (size(), begin() and so on) and matrix api (numel(), data() ...)
		template<typename ContainerT, typename ewOperationT>
		nntl_force_inline static void _ewOperation_st(ContainerT& A, const elms_range& er, ewOperationT&& F)noexcept {
			const auto pA = A.data();
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) {
				const auto p = pA + i;
				//F.op(pA[i]);
				F.op(*p);
			}
		}
		template<typename ContainerT, typename ewOperationT>
		nntl_force_inline static void _ewOperation_st2(ContainerT& A, const elms_range& er, ewOperationT&& F)noexcept {
			auto pA = A.data() + er.elmBegin;
			const auto pAE = A.data() + er.elmEnd;
			while(pA!=pAE){
				F.op(*pA);//dont do F.op(*pA++) or you'll get serious performance penalty
				pA++;
			}
		}


		//////////////////////////////////////////////////////////////////////////
		// Matrix rowwise operations
		//////////////////////////////////////////////////////////////////////////
		//apply operation F.op to each element of matrix A rows and corresponding element of row-vector pVec (must 
		// have at least A.rows() elements)
		// Columnwise
		template<typename MtxT, typename VecValueT, typename mrwOperationT>
		nntl_force_inline static void _mrwVecOperation_st_cw(MtxT& A, VecValueT*const pVec, vec_len_t colBegin
			, const rowcol_range& RCR, mrwOperationT&& F)noexcept
		{
			static_assert(std::is_same< simple_matrix<std::remove_const_t<VecValueT>>, std::remove_const_t<MtxT> >::value, "Types mismatch");
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			NNTL_ASSERT(colBegin == 0 || colBegin == 1);
			const size_t rm = A.rows(); // , cm = A.cols();
			colBegin += RCR.colBegin;
			NNTL_ASSERT(colBegin <= RCR.colEnd);
			auto pA = A.colDataAsVec(colBegin);
			F.beforeMainLoop(colBegin, A.rows());
			for (auto c = colBegin; c < RCR.colEnd; ++c) {
				for (vec_len_t r = RCR.rowBegin; r < RCR.rowEnd; ++r) {//FOR cycle with offset calculation is generally faster than WHILE,
					const auto pV = pVec + r;//because usually compiler can unfold a cycle into many instructions
					const auto pElm = pA + r;//In WHILE setup with mrwOperationT::op(*pA++, *pV++) it can't
					//and calculating offsets is faster than mrwOperationT::op(*pA++, *pV++);
					//static call mrwOperationT::op vs. object call F.op doesn't make any difference in asm 
					// code (at the moment of testing), but the object call allows to create far more generic algorithms
					F.op<mrw_cw>(*pElm, *pV, r, c, rm);
				}
				pA += rm;
				F.cw_afterInnerLoop(rm);
			}
		}

		/*template<bool _Cond,typename VecValueT>
		nntl_force_inline static std::enable_if_t<!_Cond> _condSetVal(VecValueT* pV, const std::remove_const_t<VecValueT> v)noexcept {}
		template<bool _Cond, typename VecValueT>
		nntl_force_inline static std::enable_if_t<_Cond> _condSetVal(VecValueT* pV, const std::remove_const_t<VecValueT> v)noexcept {
			*pV = v;
		}*/
		//Rowwise
		template<typename MtxT, typename VecValueT, typename mrwOperationT>
		nntl_force_inline static void _mrwVecOperation_st_rw(MtxT& A, VecValueT*const pVec, const rowcol_range& RCR, mrwOperationT&& F)noexcept {
			static_assert(std::is_same< simple_matrix<std::remove_const_t<VecValueT>>, std::remove_const_t<MtxT> >::value, "Types mismatch");
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			const auto pA = A.colDataAsVec(RCR.colBegin); //A.data();
			const size_t rm = A.rows();
			F.beforeMainLoop(RCR.colBegin, A.rows());
			for (vec_len_t r = RCR.rowBegin; r < RCR.rowEnd; ++r) {
				const auto pV = pVec + r;
				auto pElm = pA + r;
				auto v = F.rw_beforeInnerLoop(*pV, pElm, rm, RCR.colBegin, r);
				for (vec_len_t c = RCR.colBegin + mrwOperationT::rw_FirstColumnIdx; c < RCR.colEnd; ++c) {
					F.op<mrw_rw>(*pElm, v, r, c, rm);
					pElm += rm;
				}
				F.rw_afterInnerLoop<VecValueT>(*pV, v, r);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// mt stuff
		thread_id_t _howMuchThreadsNeededForCols(const vec_len_t minColumnsPerThread, const vec_len_t columns)const noexcept {
			NNTL_ASSERT(columns > minColumnsPerThread);
			const auto minThreadsReq = static_cast<thread_id_t>(ceil(real_t(columns) / minColumnsPerThread));
			NNTL_ASSERT(minThreadsReq > 1);
			auto workersCnt = m_threads.workers_count();
			if (minThreadsReq < workersCnt) workersCnt = minThreadsReq;
			return workersCnt;
		}
		//////////////////////////////////////////////////////////////////////////
		// colwise processing
		// Variation to update A without additional tmp vector
		//LambdaF is void(*F)(const rowcol_range& RCR, const thread_id_t _tid)
		template<typename LambdaF>
		nntl_force_inline void _processMtx_cw(const realmtx_t& A, LambdaF&& Func)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			//TODO: for some algorithms and datasizes it may be highly beneficial to make smart partitioning, that takes into account
			//CPU cache size (will probably require more than workers_count() call to worker function, but each call will run significanly
			// faster, due to correct cache use)
			m_threads.run([&A, F{ std::move(Func) }](const par_range_t& pr) {
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				F(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pr.tid());
			}, A.cols());
		}
		//Variation to make a vector out of const A
		//LambdaF is void(*Func)(const rowcol_range& RCR, real_t*const pVec), where pVec is temporary vector of length A.rows() to serve 
		// as F destination
		// LambdaFinal is void(*FinFunc)(realmtx_t& fin), there fin is a matrix of size A.rows()*threadsUsed that store results (columnwise) of
		// each F pVec computation
		template<typename LambdaF, typename LambdaFinal>
		nntl_force_inline void _processMtx_cw(const realmtx_t& A, const vec_len_t mt_cw_ColsPerThread, LambdaF&& Func, LambdaFinal&& FinFunc, real_t*const pTVec=nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			const auto rm = A.rows(), cm = A.cols();
			NNTL_ASSERT(cm > mt_cw_ColsPerThread && mt_cw_ColsPerThread >= 3);//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			const auto threadsToUse = _howMuchThreadsNeededForCols(mt_cw_ColsPerThread, cm);
			const auto pTmpMem = pTVec ? pTVec : get_self()._get_thread_temp_raw_storage(realmtx_t::sNumel(rm, threadsToUse));
			thread_id_t threadsUsed = 0;
			//TODO: for some algorithms and datasizes it may be highly beneficial to make smart partitioning, that takes into account
			//CPU cache size (will probably require more than workers_count() call to worker function, but each call will run significanly
			// faster, due to correct cache use)
			m_threads.run([&A, pTmpMem, rm, F{ std::move(Func) }](const par_range_t& pr) {
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				auto pVec = pTmpMem + realmtx_t::sNumel(rm, pr.tid());
				F(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pVec);
			}, cm, threadsToUse, &threadsUsed);
			NNTL_ASSERT(threadsToUse == threadsUsed);
			realmtx_t fin;
			fin.useExternalStorage(pTmpMem, rm, threadsUsed);
			//FinFunc(static_cast<const MtxT&>(fin));
			FinFunc(fin);
		}

		//Variation to make a vector out of const A, using additional (second) temporary data vector
		//LambdaF is void(*Func)(const realmtx_t& Apart, real_t*const pVec, ScndVecType*const pScndVec), where pVec and pScndVec are temporary 
		// vectors of length A.rows() to serve as F destination
		// LambdaFinal is void(*FinFunc)(const realmtx_t& fin, ScndVecType*const pFullScndMtx), there fin and pFullScndMtx are matrices
		// of size A.rows()*threadsUsed that store results (columnwise) of each F pVec computation
		template<typename ScndVecType, typename LambdaF, typename LambdaFinal>
		nntl_force_inline void _processMtx_cw(const realmtx_t& A, const vec_len_t mt_cw_ColsPerThread, LambdaF&& Func, LambdaFinal&& FinFunc)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			const auto rm = A.rows(), cm = A.cols();
			NNTL_ASSERT(cm > mt_cw_ColsPerThread && mt_cw_ColsPerThread >= 3);//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			const auto threadsToUse = _howMuchThreadsNeededForCols(mt_cw_ColsPerThread, cm);

			const auto elemsCnt = realmtx_t::sNumel(rm, threadsToUse);
			const auto pTmpMem = get_self()._get_thread_temp_raw_storage(elemsCnt 
				+ static_cast<numel_cnt_t>(ceil((real_t(sizeof(ScndVecType)) / sizeof(real_t))*elemsCnt)));

			const auto pMainVec = pTmpMem;
			ScndVecType*const pScndVec = reinterpret_cast<ScndVecType*>(pTmpMem + elemsCnt);

			thread_id_t threadsUsed = 0;
			//TODO: for some algorithms and datasizes it may be highly beneficial to make smart partitioning, that takes into account
			//CPU cache size (will probably require more than workers_count() call to worker function, but each call will run significanly
			// faster, due to correct cache use)
			m_threads.run([&A, pMainVec, pScndVec, rm, F{ std::move(Func) }](const par_range_t& pr) {
				const auto _tmpElmOffset = realmtx_t::sNumel(rm, pr.tid());
				auto pVec = pMainVec + _tmpElmOffset;
				auto pSVec = pScndVec + _tmpElmOffset;
				const auto colBeg = static_cast<vec_len_t>(pr.offset());
				F(rowcol_range(A, colBeg, colBeg + static_cast<vec_len_t>(pr.cnt())), pVec, pSVec);
			}, cm, threadsToUse, &threadsUsed);
			NNTL_ASSERT(threadsToUse == threadsUsed);

			realmtx_t fin;
			fin.useExternalStorage(pMainVec, rm, threadsUsed);
			//FinFunc(static_cast<const MtxT&>(fin), pScndVec);
			FinFunc(fin, pScndVec);
		}

		//////////////////////////////////////////////////////////////////////////
		// Rowwise processing
		//LambdaF is void (*Func) (const rowcol_range& RCR)
		template<typename MtxT, typename LambdaF>
		nntl_force_inline void _processMtx_rw(MtxT& A, LambdaF&& Func)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			m_threads.run([&A, F{ std::move(Func) }](const par_range_t& pr) {
				const auto ofs = static_cast<vec_len_t>(pr.offset());
				F(rowcol_range(ofs, ofs + static_cast<vec_len_t>(pr.cnt()), A));
			}, A.rows());
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
		static real_t ewSumProd_st(const realmtx_t& A, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && !B.empty() && B.size() == A.size());
			const auto pA = A.data(), pB = B.data();
			const elms_range& er = pER ? *pER : elms_range(A);
			real_t ret(0.0);
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i)  ret += pA[i]*pB[i];
			return ret;
		}
		real_t ewSumProd_mt(const realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(!A.empty() && !B.empty() && B.size() == A.size());
			return m_threads.reduce([&A, &B, this](const par_range_t& pr)->real_t {
				return get_self().ewSumProd_st(A, B, &elms_range(pr));
			}, _reduce_final_sum, A.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Matrix RowWise operations
		//////////////////////////////////////////////////////////////////////////
		// divide each matrix A row by corresponding vector d element, A(i,:) = A(i,:) / d(i)
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
			_processMtx_cw(A, [&A, pDiv, this](const rowcol_range& RCR, const thread_id_t _tid) {
				get_self().mrwDivideByVec_st(A, pDiv, &RCR);
			});
		}
		void mrwDivideByVec_mt_rw(realmtx_t& A, const real_t*const pDiv)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDiv);
			_processMtx_rw(A, [&A, pDiv, this](const rowcol_range& RCR) {
				get_self().mrwDivideByVec_st(A, pDiv, &RCR);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// multiply each matrix A row by corresponding vector d element, A(i,:) = A(i,:) / d(i)
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
			_processMtx_cw(A, [&A, pMul, this](const rowcol_range& RCR, const thread_id_t _tid) {
				get_self().mrwMulByVec_st(A, pMul, &RCR);
			});
		}
		void mrwMulByVec_mt_rw(realmtx_t& A, const real_t*const pMul)noexcept {
			_processMtx_rw(A, [&A, pMul, this](const rowcol_range& RCR) {
				get_self().mrwMulByVec_st(A, pMul, &RCR);
			});
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//fills array pDest of size m.rows() with column indexes of greatest element in each row of m
		// TODO: must meet func.requirements
		void mrwIdxsOfMax(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			if (A.numel() < Thresholds_t::mrwIdxsOfMax) {
				get_self().mrwIdxsOfMax_st(A, pDest);
			} else get_self().mrwIdxsOfMax_mt(A, pDest);
		}
		void mrwIdxsOfMax_st(const realmtx_t& A, vec_len_t*const pDest, const rowcol_range*const pRCR = nullptr, real_t*const pMax = nullptr)noexcept {
// 			if (A.rows() < Thresholds_t::mrwIdxsOfMax_st_rows) {
			get_self().mrwIdxsOfMax_st_rw(A, pDest, pRCR, pMax);
// 			} else get_self().mrwIdxsOfMax_st_cw(A, pDest,pRCR,pMax);
			//get_self().mrwIdxsOfMax_st_rw_small(A, pDest, pRCR, pMax);
		}
		void mrwIdxsOfMax_mt(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			if (A.rows() < Thresholds_t::mrwIdxsOfMax_mt_rows && A.cols() > Thresholds_t::mrwIdxsOfMax_ColsPerThread) {
				get_self().mrwIdxsOfMax_mt_cw(A, pDest);
			} else get_self().mrwIdxsOfMax_mt_rw(A, pDest);
		}
		void mrwIdxsOfMax_st_cw(const realmtx_t& A, vec_len_t*const pDest, const rowcol_range*const pRCR = nullptr, real_t* pMax = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && pDest && A.numel() > 0);
			const auto rm = A.rows();
			//treat the first column like the max. Then compare other columns with this column and update max'es
			_memset_rowrange<vec_len_t>(pDest, pRCR ? pRCR->colBegin : 0, rm, pRCR);
			if (A.cols() > 1) {
				const bool bSaveMax = !!pMax;
				if(!bSaveMax) pMax = get_self()._get_thread_temp_raw_storage(rm);
				//memcpy(pMax, A.data(), sizeof(*pMax)*rm);
				_memcpy_rowcol_range(pMax, A, pRCR);
				if (bSaveMax) {
					_mrwVecOperation_st_cw(A, pMax, 1, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<true>(pDest));
				}else _mrwVecOperation_st_cw(A, pMax, 1, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<false>(pDest));
			}
		}
		static void mrwIdxsOfMax_st_rw_small(const realmtx_t& A, vec_len_t*const pDest, const rowcol_range*const pRCR = nullptr, real_t*const pMax = nullptr)noexcept {
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
		void mrwIdxsOfMax_st_rw(const realmtx_t& A, vec_len_t*const pDest, const rowcol_range*const pRCR = nullptr, real_t* pMax = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			const auto bSaveMax = !!pMax;
			if (!bSaveMax) pMax = get_self()._get_thread_temp_raw_storage(A.rows());
			if (bSaveMax) {
				_mrwVecOperation_st_rw(A, pMax, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<true>(pDest));
			}else _mrwVecOperation_st_rw(A, pMax, pRCR ? *pRCR : rowcol_range(A), _mrwFindIdxsOf_MAX<false>(pDest));
		}

		void mrwIdxsOfMax_mt_cw(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			_processMtx_cw<vec_len_t>(A, Thresholds_t::mrwIdxsOfMax_ColsPerThread
				, [&A, this](const rowcol_range& RCR, real_t*const pMax, vec_len_t*const pIdxsVec)
			{
				get_self().mrwIdxsOfMax_st(A, pIdxsVec, &RCR, pMax);
			}, [pDest, this](realmtx_t& fin, vec_len_t*const pIdxsStor) {
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
		void mrwIdxsOfMax_mt_cw_small(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			NNTL_ASSERT(Thresholds_t::mrwIdxsOfMax_ColsPerThread >= 3);//DON'T make it less than 3 or you'll run in troubles with size of temp mem!!!
			const auto cm = A.cols(), rm = A.rows();
			NNTL_ASSERT(cm > Thresholds_t::mrwIdxsOfMax_ColsPerThread);
			//if (cm <= Thresholds_t::mrwIdxsOfMax_ColsPerThread) return get_self().mrwIdxsOfMax_st_cw(m, pDest);
			//will calculate max and indexes of each column-set independently and then merge them together
			const auto minThreadsReq = static_cast<thread_id_t>(ceil(real_t(cm) / Thresholds_t::mrwIdxsOfMax_ColsPerThread));
			NNTL_ASSERT(minThreadsReq > 1);
			auto workersCnt = m_threads.workers_count();
			if (minThreadsReq < workersCnt) workersCnt = minThreadsReq;

			// now we'll need rm*workersCnt of real_t's to store temp maxs for each column-set
			// and rm*workersCnt of vec_len_t's to store indexes.
			const auto _elmsCnt = realmtx_t::sNumel(rm, workersCnt);
			const auto pTmp = get_self()._get_thread_temp_raw_storage(_elmsCnt
				+ static_cast<numel_cnt_t>(ceil((real_t(sizeof(vec_len_t)) / sizeof(real_t))*_elmsCnt)));
			const auto pMaxStor = pTmp;
			vec_len_t*const pIdxsStor = reinterpret_cast<vec_len_t*>(pTmp + _elmsCnt);

			//now we may run max calculation in parallel for each column-set into pMaxStor and pIdxsStor
			auto pMD = A.data();
			m_threads.run([rm, pMaxStor, pIdxsStor, pMD](const par_range_t& pr) {
				const auto _tid = pr.tid();
				const auto _tmpElmOffset = realmtx_t::sNumel(rm, _tid);
				auto pTMax = pMaxStor + _tmpElmOffset;
				auto pTDest = pIdxsStor + _tmpElmOffset;
				//now we have to decide what column range we should process here
				const auto firstColIdx = pr.offset();
				auto p = pMD + firstColIdx*rm;
				const auto pE = p + pr.cnt()*rm;

				std::fill(pTDest, pTDest + rm, static_cast<vec_len_t>(firstColIdx));
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
		}
		void mrwIdxsOfMax_mt_rw(const realmtx_t& A, vec_len_t*const pDest)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pDest);
			_processMtx_rw(A, [&A, pDest, this](const rowcol_range& RCR) {
				get_self().mrwIdxsOfMax_st(A, pDest, &RCR);
			});
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
			_processMtx_rw(A, [&A, pMax, this](const rowcol_range& RCR) {
				get_self().mrwMax_st(A, pMax, &RCR);
			});
		}
		void mrwMax_mt_cw(const realmtx_t& A, real_t*const pMax)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pMax);
			_processMtx_cw(A, Thresholds_t::mrwMax_mt_cw_ColsPerThread, [&A, this](const rowcol_range& RCR, real_t*const pVec) {
				get_self().mrwMax_st(A, pVec, &RCR);
			}, [pMax, this](const realmtx_t& fin) {
				get_self().mrwMax(fin, pMax);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Calculate into first row of A rowwise sum of each row
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
			if (cm <= std::max(Thresholds_t::mrwSum_mt_cw_colsPerThread, m_threads.workers_count())) {
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
			_processMtx_rw(A, [&A, this](const rowcol_range& RCR){
				get_self().mrwSum_ip_st(A, &RCR);
			});
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Calculate rowwise sum into vec
		void mrwSum(const realmtx_t& A, real_t*const pVec)noexcept {
			if (A.numel() < Thresholds_t::mrwSum) {
				get_self().mrwSum_st(A, pVec);
			}else get_self().mrwSum_mt(A, pVec);
		}
		//TODO: how are we going to branch code when pRCR is present? Desperately NEED run-time profiler!!!
		void mrwSum_st(const realmtx_t& A, real_t*const pVec, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && pVec);
			if (A.cols() == 1) {
				_memcpy_rowcol_range(pVec, A, pRCR);
				//memcpy(pVec, A.data(), A.byte_size());
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
				if (cm <= std::max(Thresholds_t::mrwSum_mt_cw_colsPerThread, m_threads.workers_count())) {
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
			memset(pVec, 0, sizeof(*pVec)*A.rows());
			_mrwVecOperation_st_cw(A, pVec, 0, pRCR ? *pRCR : rowcol_range(A), _mrw_SUM());
		}
		void mrwSum_mt_rw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > 1 && pVec);
			_processMtx_rw(A, [&A, pVec, this](const rowcol_range& RCR) {
				get_self().mrwSum_st(A, pVec, &RCR);
			});
		}
		void mrwSum_mt_cw(const realmtx_t& A, real_t*const pVec)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0 && A.cols() > Thresholds_t::mrwSum_mt_cw_colsPerThread && pVec);
			//will sum partial matrices into temp memory, and then sum it to pVec
			_processMtx_cw(A, Thresholds_t::mrwSum_mt_cw_colsPerThread, [&A, this](const rowcol_range& RCR, real_t*const pVec) {
				get_self().mrwSum_st(A, pVec, &RCR);
			}, [pVec, this](const realmtx_t& fin) {
				get_self().mrwSum(fin, pVec);
			});
		}
	};


	template<typename RealT, typename iThreadsT, typename ThresholdsT = _impl::SIMPLE_MATH_THR<RealT>>
	class simple_math final : public _simple_math<RealT, iThreadsT, ThresholdsT, simple_math<RealT, iThreadsT, ThresholdsT>> {
	public:
		~simple_math()noexcept {}
		simple_math()noexcept : _simple_math<RealT, iThreadsT, ThresholdsT, simple_math<RealT, iThreadsT, ThresholdsT>>() {}

	};
}
}