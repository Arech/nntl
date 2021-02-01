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

#include "../_i_math.h"
#include "bindings/b_open_blas.h"
//#include "bindings/b_yeppp.h"


//#include "../../utils/clamp.h"
#include <boost/algorithm/clamp.hpp>

#include <limits>
#include "mathn_thr.h"

#include "smath.h"

namespace nntl {
namespace math {

	// ALL functions of _i_math interface must be tested for ST vs. MT performance and be adjusted accordingly
	// see notes in smath.h

	// this class uses some routines from OpenBLAS to implement _i_math
	template <typename RealT, typename iThreadsT, typename ThresholdsT, typename FinalPolymorphChild, typename bindingBlasT = b_OpenBLAS>
	class _MathN : public _SMath<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild>, public _i_math<RealT> {
	public:
		typedef _SMath<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild> base_class_t;
		typedef bindingBlasT b_BLAS_t;

		using base_class_t::real_t;
		using base_class_t::realmtx_t;
		using base_class_t::realmtxdef_t;
		//using base_class_t::numel_cnt_t;
		//using base_class_t::vec_len_t;

		//TODO: probably don't need this assert
		static_assert(::std::is_base_of<_impl::MATHN_THR<real_t>, Thresholds_t>::value, "Thresholds_t must be derived from _impl::MATHN_THR<real_t>");
				
		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
	#pragma warning(push)
	#pragma warning(disable:4100)
		struct _mrw_SOFTMAXPARTS :public _mrwHlpr_rw_UpdVecElm {
			real_t* pNumerator;//(colmajor) matrix data
			const real_t*const pMax;//row vector

			real_t* pNum;
			const real_t* pMx;

			_mrw_SOFTMAXPARTS(const real_t*const _pMax, real_t*const _pNum)noexcept : pMax(_pMax), pNumerator(_pNum) {}

			template<_OperationType OpType, typename BaseT>
			::std::enable_if_t<OpType == mrw_cw> op(const BaseT& mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t mtxRows)noexcept {
				const auto numerator = ::std::exp(mtxElm - *(pMax + r));
				vecElm += numerator;
				*(pNumerator + r) = numerator;
			}

			template<_OperationType OpType, typename BaseT>
			::std::enable_if_t<OpType == mrw_rw> op(const BaseT& mtxElm, BaseT& vecElm, const numel_cnt_t r, const numel_cnt_t c, const numel_cnt_t mtxRows)noexcept {
				const auto numerator = ::std::exp(mtxElm - *pMx);
				vecElm += numerator;
				*pNum = numerator;
				pNum += mtxRows;
			}

			void initOperation(const vec_len_t colBegin, const vec_len_t mtxRows)noexcept {
				//adjusting matrix data pointer to the beginning of colBegin column
				pNumerator += realmtx_t::sNumel(mtxRows, colBegin);
			};

			void cw_toNextCol(const numel_cnt_t ldM)noexcept {
				//proceeding to next column
				pNumerator += ldM;
			};

			static constexpr vec_len_t rw_FirstColumnIdx = 0;
			template<typename VecBaseT, typename MtxBaseT>
			VecBaseT rw_initVecElm(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const numel_cnt_t mtxRows
				, const numel_cnt_t colBegin, const numel_cnt_t r)noexcept
			{
				pNum = pNumerator + r;
				pMx = pMax + r;
				return VecBaseT(0.0);
			}
		};
	#pragma warning(pop)
		//////////////////////////////////////////////////////////////////////////
		//methods
	public:

		~_MathN()noexcept {};
		_MathN() noexcept : base_class_t() {}

		//////////////////////////////////////////////////////////////////////////
		// i_math interface implementation
		//////////////////////////////////////////////////////////////////////////
		// _st versions of functions MUST NOT call generic and/or multithreaded function implementations (ONLY _st).
		//		(They may be used in future in some parallel algorithms)
		// _mt and generic function versions may use any suitable implementations.
		// generic, _st and _mt versions MUST accept any datasizes. However, their specializations,
		//		such as _mt_cw MAY put restrictions on acceptable data sizes.

		using base_class_t::preinit;
		using base_class_t::init;
		using base_class_t::deinit;
		using base_class_t::ithreads;

		using base_class_t::ewSumProd;
		using base_class_t::ewSumSquares;
		using base_class_t::ewSumSquares_ns;
		//using base_class_t::ewBinarize;
		using base_class_t::mrwIdxsOfMax;
		using base_class_t::mrwMax;

		//////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////
		//SoftMax
		//////////////////////////////////////////////////////////////////////////
		//helper function to calculate softmax (not a part of _i_math, because it's not expected to be called from elsewhere)
		//act - is an activation matrix (WITHOUT biases!)
		// pMax is a vector of size act.rows() with rowwise act maximum values
		// pDenominator is a vector of size act.rows()*m_threads.cur_workers_count() elements. First act.rows() elements will be filled with rowwise_sum_exp()
		// pNumerator is columnmajor matrix(vector) of size act.size() to be filled with exp(Aij - maxj)
		void softmax_parts(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator)noexcept {
			if (act.numel()<Thresholds_t::softmax_parts) {
				get_self().softmax_parts_st(act, pMax, pDenominator, pNumerator);
			} else get_self().softmax_parts_mt(act, pMax, pDenominator, pNumerator);
		}
		void softmax_parts_st(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self().softmax_parts_st_cw(act, pMax, pDenominator, pNumerator, pRCR);
		}
		static void softmax_parts_st_rw(const realmtx_t& act, const real_t* pMax, real_t* pDenominator, real_t* pNumerator, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(act.numel() > 0 && !act.empty() && pMax && pDenominator && pNumerator);
			_mrwVecOperation_st_rw(act, pDenominator, pRCR ? *pRCR : rowcol_range(act), _mrw_SOFTMAXPARTS(pMax, pNumerator));
		}
		static void softmax_parts_st_cw(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(act.numel() > 0 && !act.empty() && pMax && pDenominator && pNumerator);
			_memset_rowrange(pDenominator, real_t(0.0), act.rows(), pRCR);
			_mrwVecOperation_st_cw(act, pDenominator, 0, pRCR ? *pRCR : rowcol_range(act), _mrw_SOFTMAXPARTS(pMax, pNumerator));
		}
		void softmax_parts_mt(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator)noexcept {
			if (act.cols() <= Thresholds_t::softmax_parts_mt_cw_ColsPerThread || act.rows()> Thresholds_t::softmax_parts_mt_rows) {
				get_self().softmax_parts_mt_rw(act, pMax, pDenominator, pNumerator);
			} else get_self().softmax_parts_mt_cw(act, pMax, pDenominator, pNumerator);
		}
		void softmax_parts_mt_rw(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator)noexcept {
			_processMtx_rw(act, [&act, pMax, pDenominator, pNumerator, this](const rowcol_range& RCR) noexcept {
				get_self().softmax_parts_st(act, pMax, pDenominator, pNumerator, &RCR);
			});
		}
		//pDenominator must be able to contain at least sNumel(act.rows(), m_threads.cur_workers_count()) elements!
		//On return first column of pDenominator will contain calculated softmax denominator values
		void softmax_parts_mt_cw(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator)noexcept {
			_processMtx_cw(act, Thresholds_t::softmax_parts_mt_cw_ColsPerThread
				, [&act, pMax, pNumerator, this](const rowcol_range& RCR, real_t*const pVec)noexcept
			{
				get_self().softmax_parts_st(act, pMax, pVec, pNumerator, &RCR);
			}, [this](realmtx_t& fin)noexcept {
				get_self().mrwSum_ip(fin);
			}, pDenominator);
		}

		//////////////////////////////////////////////////////////////////////////
		// helper function that return the amount of temporary memory (in real_t) needed to process by softmax()
		// a matrix of size act.size()
		template<typename T>
		numel_cnt_t ___softmax_needTempMem(const smatrix<T>& act)const noexcept {
			// to compute softmax we'll need a row to store rowwise_max(), at max m_threads.cur_workers_count() rows for
			// rowwise_sum_exp()-denominator of softmax expression, and a
			// 			// whole matrix of exp(Aij - maxj) (numerator of softmax expression).
			// 	and also mrwSum_ip() requirements may apply
			return smatrix_td::sNumel(act.rows(), act.cols_no_bias() + 1 + m_threads.cur_workers_count());
		}
		template<typename T>
		numel_cnt_t softmax_needTempMem(const smatrix<T>& act)const noexcept {
			return get_self().___softmax_needTempMem(act) + get_self().mrwSum_ip_needTempMem(act);
		}
		//////////////////////////////////////////////////////////////////////////
		// MUST ignore biases!
		void softmax(realmtxdef_t& srcdest) noexcept {
			if (srcdest.numel() < Thresholds_t::softmax) {
				get_self().softmax_st(srcdest);
			} else get_self().softmax_mt(srcdest);
		}
		void softmax_st(realmtxdef_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty() && srcdest.numel() > 0);
			const auto bRestoreBiases = srcdest.hide_biases();

			const auto rm = srcdest.rows();
			const auto tmemSize = get_self().___softmax_needTempMem(srcdest);
			const auto pTmp = get_self()._istor_alloc(tmemSize);
			const auto pNumer = pTmp;
			const auto pMax = pTmp + srcdest.numel();
			const auto pDenom = pMax + rm;

			//calc max() rowwise
			get_self().mrwMax_st(srcdest, pMax);
			//calculating denominators and numerators of softmax 
			get_self().softmax_parts_st(srcdest, pMax, pDenom, pNumer);
			//performing division
			memcpy(srcdest.data(), pNumer, srcdest.byte_size());
			get_self().mrwDivideByVec(srcdest, pDenom);

			srcdest.restore_biases(bRestoreBiases);
			get_self()._istor_free(pTmp, tmemSize);
		}
		void softmax_mt(realmtxdef_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty() && srcdest.numel() > 0);
			const auto bRestoreBiases = srcdest.hide_biases();

			const auto rm = srcdest.rows();
			const auto tmemSize = get_self().___softmax_needTempMem(srcdest);
			const auto pTmp = get_self()._istor_alloc(tmemSize);
			const auto pNumer = pTmp;
			const auto pMax = pTmp + srcdest.numel();
			const auto pDenom = pMax + rm;

			//calc max() rowwise
			get_self().mrwMax(srcdest, pMax);
			//calculating denominators and numerators of softmax 
			get_self().softmax_parts(srcdest, pMax, pDenom, pNumer);
			//performing division
			memcpy(srcdest.data(), pNumer, srcdest.byte_size());
			get_self().mrwDivideByVec(srcdest, pDenom);

			srcdest.restore_biases(bRestoreBiases);
			get_self()._istor_free(pTmp, tmemSize);
		}



		//////////////////////////////////////////////////////////////////////////
		// ElementWise operations
		//////////////////////////////////////////////////////////////////////////
		//binarize elements of real-valued matrix according to their relation to frac
		template<typename T>
		void ewBinarize_ip(smatrix<T>& A, const T frac, const T lBnd = T(0.), const T uBnd = T(1.))noexcept {
			if (A.numel_no_bias() < Thresholds_t::ewBinarize_ip) {
				get_self().ewBinarize_ip_st(A, frac, lBnd, uBnd);
			} else get_self().ewBinarize_ip_mt(A, frac, lBnd, uBnd);
		}
		template<typename T>
		void ewBinarize_ip_st(smatrix<T>& A, const T frac, const T lBnd = T(0.), const T uBnd = T(1.), const elms_range*const pER = nullptr)noexcept
		{
			NNTL_ASSERT(!A.empty());
			get_self()._iewBinarize_ip_st(A.data(), frac, lBnd, uBnd, pER ? *pER : elms_range(0, A.numel_no_bias()));
		}
		template<typename T>
		static void _iewBinarize_ip_st(T* pA, const T frac, const T lBnd, const T uBnd, const elms_range& er)noexcept {
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			while (pA != pAE) {//vectorizes
				*pA++ = *pA > frac ? uBnd : lBnd;
			}
			/*for (auto i = er.elmBegin; i < er.elmEnd; ++i) {//doesn't vectorize
				pA[i] = pA[i] > frac ? uBnd : lBnd;
			}*/
		}
		template<typename T>
		void ewBinarize_ip_mt(smatrix<T>& A, const T frac, const T lBnd = T(0.), const T uBnd = T(1.))noexcept {
			NNTL_ASSERT(!A.empty());
			m_threads.run([pA = A.data(), frac, lBnd, uBnd, this](const par_range_t& r) noexcept{
				get_self()._iewBinarize_ip_st(pA, frac, lBnd, uBnd, elms_range(r));
			}, A.numel_no_bias());
		}

		//#TODO finish refactoring; find out which algo is better; move to base SMath class
		/*struct _ew_BINARIZE_IP {
			const real_t frac;
			_ew_BINARIZE_IP(const real_t f)noexcept:frac(f) {}

			template<typename BaseT>
			void op(BaseT& elm)noexcept {
				static_assert(!::std::is_const<BaseT>::value, "BaseT mustn't have a const specifier");
				//NNTL_ASSERT(elm >= BaseT(0.0) && elm <= BaseT(1.0));
				elm = elm > frac ? BaseT(1.0) : BaseT(0.0);
			}
		};
		static void ex_ewBinarize_ip_st(realmtx_t& A, const real_t frac)noexcept {
			_ewOperation_st(A, elms_range(A), _ew_BINARIZE_IP(frac));
		}
		static void ex2_ewBinarize_ip_st(realmtx_t& A, const real_t frac)noexcept {
			_ewOperation_st2(A, elms_range(A), _ew_BINARIZE_IP(frac));
		}*/

		//////////////////////////////////////////////////////////////////////////
		//binarize elements of real-valued matrix according to their relaion to frac into other matrix
		template<typename DestContainerT>
		void ewBinarize(DestContainerT& Dest, const realmtx_t& A, const real_t frac)noexcept {
			NNTL_ASSERT(Numel(Dest) == A.numel_no_bias());
			if (A.numel_no_bias() < Thresholds_t::ewBinarize) {
				get_self().ewBinarize_st(Dest, A, frac);
			} else get_self().ewBinarize_mt(Dest, A, frac);
		}
		template<typename BaseDestT>
		static void _iewBinarize_st(BaseDestT*const pD, const realmtx_t& A, const real_t frac, const elms_range& er)noexcept {
			const auto pA = A.data();
			const auto ee = er.elmEnd;
			//#strictAliasingViolation here if BaseDestT is char?
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i)  pD[i] = pA[i] > frac ? BaseDestT(1.0) : BaseDestT(0.0);
			/*auto pA = A.data();
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			pD += er.elmBegin;
			while (pA != pAE) {
				const auto a = *pA++;
				*pD++ = a > frac ? BaseDestT(1.0) : BaseDestT(0.0);
			}*/
		}
		template<typename DestContainerT>
		static void ewBinarize_st(DestContainerT& Dest, const realmtx_t& A, const real_t frac, const elms_range*const pER = nullptr)noexcept {
			//NNTL_ASSERT(IsSameSizeNumel(A, Dest));
			NNTL_ASSERT(Numel(Dest) == A.numel_no_bias());
			_iewBinarize_st(Dest.data(), A, frac, pER ? *pER : elms_range(0, A.numel_no_bias()));
		}
		template<typename DestContainerT>
		void ewBinarize_mt(DestContainerT& Dest, const realmtx_t& A, const real_t frac)noexcept {
			//NNTL_ASSERT(IsSameSizeNumel(A, Dest));
			NNTL_ASSERT(Numel(Dest) == A.numel_no_bias());
			m_threads.run([pD = Dest.data(), &A, frac, this](const par_range_t& pr) noexcept{
				get_self()._iewBinarize_st(pD, A, frac, elms_range(pr));
			}, A.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		// same as ewBinarize() but designed to fill big destination matrix using small source batch matrices.
		// i.e. it does ewBinarize() for src into a part of dest matrix, starting from row startingDestRow to startingDestRow + src.rows()
		// Biases, if any, are ignored
		template<typename destVT, typename srcVT>
		void ewBinarizeBatch(smatrix<destVT>& dest, const vec_len_t startingDestRow, const smatrix<srcVT>& src, const real_t frac)noexcept {
			if (src.numel_no_bias() < Thresholds_t::ewBinarizeBatch) {
				get_self().ewBinarizeBatch_st(dest, startingDestRow, src, frac);
			} else get_self().ewBinarizeBatch_mt(dest, startingDestRow, src, frac);
		}
		template<typename destVT, typename srcVT>
		static void ewBinarizeBatch_st(smatrix<destVT>& dest, const vec_len_t startingDestRow, const smatrix<srcVT>& src, const real_t frac
			, const elms_range*const pER = nullptr)noexcept
		{
			NNTL_ASSERT(dest.cols_no_bias() == src.cols_no_bias());
			NNTL_ASSERT(startingDestRow >= 0);
			NNTL_ASSERT(dest.rows() >= startingDestRow + src.rows());
			_iewBinarizeBatch_st(dest, startingDestRow, src, frac, pER ? *pER : elms_range(0, src.numel_no_bias()));
		}
		template<typename destVT, typename srcVT>
		void ewBinarizeBatch_mt(smatrix<destVT>& dest, const vec_len_t startingDestRow, const smatrix<srcVT>& src, const real_t frac)noexcept {
			NNTL_ASSERT(dest.cols_no_bias() == src.cols_no_bias());
			NNTL_ASSERT(startingDestRow >= 0);
			NNTL_ASSERT(dest.rows() >= startingDestRow + src.rows());
			m_threads.run([&dest, &src, startingDestRow, frac, this](const par_range_t& pr) noexcept{
				get_self()._iewBinarizeBatch_st(dest, startingDestRow, src, frac, elms_range(pr));
			}, src.numel_no_bias());
		}
		template<typename destVT, typename srcVT>
		static void _iewBinarizeBatch_st(smatrix<destVT>& dest, const vec_len_t startingDestRow, const smatrix<srcVT>& src
			, const real_t frac, const elms_range& srcEr)noexcept
		{
			NNTL_ASSERT(dest.cols_no_bias() == src.cols_no_bias());
			NNTL_ASSERT(startingDestRow >= 0);
			NNTL_ASSERT(dest.rows() >= startingDestRow + src.rows());

			const numel_cnt_t destNextColOfs = dest.rows() - src.rows();
			const numel_cnt_t srcRows = src.rows();

			//calculating <row,col> coordinates of srcEr
			auto _dr = ::std::div(srcEr.elmBegin, srcRows);
			const vec_len_t srcBegCol = static_cast<vec_len_t>(_dr.quot), srcBegRow = static_cast<vec_len_t>(_dr.rem);
			NNTL_ASSERT(srcBegCol < src.cols_no_bias() && srcBegRow < srcRows);
			_dr = ::std::div(srcEr.elmEnd, srcRows);
			const vec_len_t srcEndCol = static_cast<vec_len_t>(_dr.quot);
			const numel_cnt_t srcEndRow = _dr.rem;
			NNTL_ASSERT(srcEndCol <= src.cols_no_bias() && srcEndRow < srcRows);

			NNTL_ASSERT(startingDestRow + srcBegRow <= dest.rows());
			auto* pD = dest.colDataAsVec(srcBegCol) + startingDestRow + srcBegRow;
			const auto* pS = src.colDataAsVec(srcBegCol) + srcBegRow;

			{
				//first col srcBegCol from srcBegRow till end
				const numel_cnt_t _totR1 = (srcBegCol == srcEndCol ? srcEndRow : srcRows) - srcBegRow;
				NNTL_ASSERT(_totR1 >= 0);
				//#strictAliasingViolation here if destVT is char?
				const auto _pD = pD;
				const auto _pS = pS;
				NNTL_ASSERT(_pD + _totR1 <= dest.end_no_bias());
				NNTL_ASSERT(_pS + _totR1 <= src.end_no_bias());
				for (numel_cnt_t r = 0; r < _totR1; ++r) {
					_pD[r] = _pS[r] > frac ? destVT(1.0) : destVT(0.0);
				}
				pD += _totR1 + destNextColOfs; pS += _totR1;
			}

			//intermediate columns
			for (numel_cnt_t c = srcBegCol + 1; c < srcEndCol; ++c) {
				const auto _pD = pD;
				const auto _pS = pS;
				NNTL_ASSERT(_pD + srcRows <= dest.end_no_bias());
				NNTL_ASSERT(_pS + srcRows <= src.end_no_bias());
				for (numel_cnt_t r = 0; r < srcRows; ++r) {
					_pD[r] = _pS[r] > frac ? destVT(1.0) : destVT(0.0);
				}
				pD += srcRows + destNextColOfs; pS += srcRows;
			}

			//last col
			if (srcBegCol != srcEndCol) {
				const auto _pD = pD;
				const auto _pS = pS;
				NNTL_ASSERT(0 == srcEndRow || _pD + srcEndRow <= dest.end_no_bias());
				NNTL_ASSERT(0 == srcEndRow || _pS + srcEndRow <= src.end_no_bias());
				for (numel_cnt_t r = 0; r < srcEndRow; ++r) {
					_pD[r] = _pS[r] > frac ? destVT(1.0) : destVT(0.0);
				}
			}			
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// extracts batches of samples from src matrix into dest.
		// Note that bSampleInColumn() layout offers much more efficient extraction than bSampleInRow().
		// #supportsBatchInRow
		template<typename VT, typename SeqIt>
		void mExtractBatches(const smatrix<VT>& src, const SeqIt& batchIdxsItBegin, smatrix<VT>& dest)noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(src.bBatchInRow() == dest.bBatchInRow());
			NNTL_ASSERT(src.sample_size() == dest.sample_size());
			src.bSampleInColumn()
				? get_self().mExtractCols(src, batchIdxsItBegin, dest)
				: get_self().mExtractRows(src, batchIdxsItBegin, dest);
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Extracts columns from src, addressed by their indexes set in cIdxsItBegin, into columns of dest.
		// cIdxsItBegin must contain dest.cols_no_bias() indexes of columns from src to copy into dest.
		// Bias row (if any) are skipped.
		// Don't mistake the func with mCloneCols(). This one is mainly for batch extraction from matrix with bBatchInRow()
		// prop (though, it's not a requirement)
		// #supportsBatchInRow
		template<typename VT, typename SeqIt>
		void mExtractCols(const smatrix<VT>& src, const SeqIt& cIdxsItBegin, smatrix<VT>& dest)noexcept {
			const auto cnb = dest.cols_no_bias();
			if (cnb < 2 || smatrix_td::sNumel(dest.rows_no_bias(), cnb) < Thresholds_t::mExtractCols) {
				//not using numel_no_bias() to prevent triggering assertions wrong in this context, but helpful in others
			//if (cnb < 2 || cnb <= Thresholds_t::mExtractCols_cols) {
				get_self().mExtractCols_st(src, cIdxsItBegin, dest);
			} else get_self().mExtractCols_mt(src, cIdxsItBegin, dest);
		}
		template<typename VT, typename SeqIt>
		static void _imExtractCols_st(const smatrix<VT>& src, const SeqIt& cIdxsItBegin, smatrix<VT>& dest, const vec_range& destColR)noexcept {
			NNTL_ASSERT(!src.empty() && !dest.empty());
			NNTL_ASSERT(src.bBatchInRow() == dest.bBatchInRow());
			NNTL_ASSERT(src.rows_no_bias() == dest.rows_no_bias());
			NNTL_ASSERT(dest.cols_no_bias() >= destColR.elmEnd);

			const size_t totRowBytes = static_cast<size_t>(src.rows_no_bias())*sizeof(VT);
			const ptrdiff_t colEnd = destColR.elmEnd;
			const ptrdiff_t ldS = src.ldim();
			const ptrdiff_t ldD = dest.ldim();
			
			const VT * __restrict const pSrc = src.data();
			VT * __restrict const pDest = dest.data();

			for (ptrdiff_t ci = destColR.elmBegin; ci < colEnd; ++ci) {
				::std::memcpy(pDest + ldD*ci, pSrc + ldS*cIdxsItBegin[ci], totRowBytes);
			}
		}
		template<typename VT, typename SeqIt>
		void mExtractCols_st(const smatrix<VT>& src, const SeqIt& cIdxsItBegin, smatrix<VT>& dest, const vec_range*const pDestColR=nullptr)noexcept {
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			get_self()._imExtractCols_st(src, cIdxsItBegin, dest, pDestColR ? *pDestColR : vec_range(0, dest.cols_no_bias()));
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}
		template<typename VT, typename SeqIt>
		void mExtractCols_mt(const smatrix<VT>& src, const SeqIt& cIdxsItBegin, smatrix<VT>& dest)noexcept {
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			get_self().ithreads().run([&src, &cIdxsItBegin, &dest, this](const par_range_t& pr)noexcept {
				get_self()._imExtractCols_st(src, cIdxsItBegin, dest, vec_range(pr));
			}, dest.cols_no_bias());
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//extract rows with indexes specified by Contnr ridxs into dest. Biases (if matrices features them) are not copied
		//DOES NOT support non-default emulatesBiases() & bBatchInRow() matrix setting
		// #todo this is extremely slow function
		template<typename VT, typename SeqIt>
		void mExtractRows(const smatrix<VT>& src, const SeqIt& ridxsItBegin, smatrix<VT>& dest)noexcept {
			if (dest.cols_no_bias() < 2 || dest.numel_no_bias() < Thresholds_t::mExtractRows) {
				get_self().mExtractRows_seqWrite_st(src, ridxsItBegin, dest);
			} else get_self().mExtractRows_seqWrite_mt(src, ridxsItBegin, dest);
		}
		//DOES NOT support non-default emulatesBiases() & bBatchInRow() matrix setting
		//#todo mt() should be done over columns for better thread cache locality
		template<typename VT, typename SeqIt>
		static void _imExtractRows_seqWrite_st(const smatrix<VT>& src, const SeqIt& ridxsItBegin, smatrix<VT>& dest, const elms_range& er)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			src.assert_storage_does_not_intersect(dest);
			NNTL_ASSERT(!src.emulatesBiases() || src.bBatchInColumn());
			NNTL_ASSERT(!dest.emulatesBiases() || dest.bBatchInColumn());

			const numel_cnt_t destRows = dest.rows(), srcRows = src.rows();
			NNTL_ASSERT(dest.cols_no_bias() == src.cols_no_bias() && destRows <= srcRows);
			NNTL_ASSERT(er.elmBegin <= destRows && er.elmEnd <= destRows && er.elmBegin <= er.elmEnd);

			const auto rCnt = er.totalElements();

			//TODO: accessing data in sequential order could provide some performance gains. However
			//it requires the content of [ridxsItBegin,ridxsItBegin+ridxsCnt) to be sorted. Therefore, testing is required
			// to decide whether it's all worth it
			// It don't. I've just tried. No significant change on big datasets.
			// Probably a better idea is to transpose the source data to read rows sequentially. Need to implement and test.
			
			auto pSrc = src.data();
			auto pDest = dest.data() + er.elmBegin;
			const auto pDestEnd = pDest + dest.numel_no_bias();//we're leaving bias column intact
			SeqIt pThreadRI = ridxsItBegin + er.elmBegin;

			while (pDest != pDestEnd) {
				SeqIt pRI = pThreadRI;
				auto destCur = pDest;
				pDest += destRows;
				const auto destEnd = destCur + rCnt;
				while (destCur != destEnd) {
					const auto idx = *pRI++;
					NNTL_ASSERT(idx < srcRows);
					*destCur++ = *(pSrc + idx);
				}
				pSrc += srcRows;
			}
		}
		template<typename VT, typename SeqIt>
		static void mExtractRows_seqWrite_st(const smatrix<VT>& src, const SeqIt& ridxsItBegin, smatrix<VT>& dest, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			_imExtractRows_seqWrite_st(src, ridxsItBegin, dest, pER ? *pER : elms_range(0, dest.rows()));
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}
		template<typename VT, typename SeqIt>
		void mExtractRows_seqWrite_mt(const smatrix<VT>& src, const SeqIt& ridxsItBegin, smatrix<VT>& dest)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			src.assert_storage_does_not_intersect(dest);
			//static_assert(::std::is_same<vec_len_t, SeqIt::value_type>::value, "Contnr type should contain vec_len_t data");
			NNTL_ASSERT(dest.cols() == src.cols() && dest.rows() <= src.rows());

			m_threads.run([&src, &dest, &ridxsItBegin](const par_range_t& r) noexcept{
				_imExtractRows_seqWrite_st(src, ridxsItBegin, dest, elms_range(r));
			}, dest.rows());

			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//dest.rows() must be equal to vCountNonZeros(pMask, src.rows())!
		//biases are ignored!
		void mExtractRowsByMask(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest)noexcept {
			if (src.numel_no_bias() < Thresholds_t::mExtractRowsByMask) {
				get_self().mExtractRowsByMask_st(src, pMask, dest);
			} else get_self().mExtractRowsByMask_mt(src, pMask, dest);
		}
		//looks like we'd never need _st variation alone. Moreover, to make this function fully _st compatible,
		// the expression executed if(dest.rows() == src.rows()) must also take into account pER. So just don't care about it now
		void mExtractRowsByMask_st(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest/*, const elms_range*const pER = nullptr*/)noexcept
		{
			NNTL_ASSERT(!src.empty() && pMask && !dest.empty() && src.cols_no_bias() == dest.cols_no_bias());
			NNTL_ASSERT(dest.rows() <= src.rows() && dest.rows());
			NNTL_ASSERT(dest.rows() == static_cast<vec_len_t>(get_self().vCountNonZeros(pMask, src.rows())));
			
			if (dest.rows() == src.rows()) {
				//just copying src to dest
				const auto b = src.copy_data_skip_bias(dest);
				NNTL_ASSERT(b);
			} else {
				get_self()._imExtractRowsByMask_st(src, pMask, dest, /*pER ? *pER :*/ elms_range(0, src.cols_no_bias()));
			}
		}
		static void _imExtractRowsByMask_st(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest, const elms_range& er)noexcept {
			NNTL_ASSERT(dest.rows() < src.rows() && dest.rows());
			NNTL_ASSERT(src.cols_no_bias() >= er.elmEnd);

			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			const auto _z = similar_FWI_pos_zero<real_t>();
			const auto _o = similar_FWI_one<real_t>();
			
			const numel_cnt_t tr = src.rows(), tc = er.totalElements();
			NNTL_ASSERT(tc);
			auto _pS = src.colDataAsVec(static_cast<vec_len_t>(er.elmBegin));
			auto pD = dest.colDataAsVec(static_cast<vec_len_t>(er.elmBegin));
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			const auto pM = reinterpret_cast<const similar_FWI_t*const>(pMask);
			for (numel_cnt_t ci = 0; ci < tc; ++ci) {
				const auto pS = _pS;
				_pS += tr;
				for (numel_cnt_t i = 0; i < tr; ++i) {
					const auto m = pM[i];
					if (m != _z) {
						NNTL_ASSERT(m == _o);
						*pD++ = pS[i];
					}
				}
			}
		}
		void mExtractRowsByMask_mt(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!src.empty() && pMask && !dest.empty() && src.cols_no_bias() == dest.cols_no_bias());
			NNTL_ASSERT(dest.rows() <= src.rows() && dest.rows());
			NNTL_ASSERT(dest.rows() == static_cast<vec_len_t>(get_self().vCountNonZeros(pMask, src.rows())));

			if (dest.rows() == src.rows()) {
				//just copying src to dest
				const auto b = src.copy_data_skip_bias(dest);
				NNTL_ASSERT(b);
			} else {
				m_threads.run([&src, pMask, &dest, this](const par_range_t& r) noexcept{
					get_self()._imExtractRowsByMask_st(src, pMask, dest, elms_range(r));
				}, src.cols_no_bias());
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//biases are ignored!
		void mFillRowsByMask(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest)noexcept {
			if (dest.numel_no_bias() < Thresholds_t::mFillRowsByMask) {
				get_self().mFillRowsByMask_st(src, pMask, dest);
			} else get_self().mFillRowsByMask_mt(src, pMask, dest);
		}
		void mFillRowsByMask_st(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest/*, const elms_range*const pER = nullptr*/)noexcept
		{
			NNTL_ASSERT(!src.empty() && pMask && !dest.empty() && src.cols_no_bias() == dest.cols_no_bias());
			NNTL_ASSERT(src.rows() <= dest.rows() && src.rows());
			NNTL_ASSERT(src.rows() == static_cast<vec_len_t>(get_self().vCountNonZeros(pMask, dest.rows())));

			if (dest.rows() == src.rows()) {
				//just copying src to dest
				const auto b = src.copy_data_skip_bias(dest);
				NNTL_ASSERT(b);
			} else {
				get_self()._imFillRowsByMask_st(src, pMask, dest, /*pER ? *pER :*/ elms_range(0, dest.cols_no_bias()));
			}
		}
		static void _imFillRowsByMask_st(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest, const elms_range& er)noexcept {
			NNTL_ASSERT(dest.cols_no_bias() >= er.elmEnd);
			NNTL_ASSERT(src.rows() < dest.rows() && src.rows());

			typedef real_t_limits<real_t>::similar_FWI_t similar_FWI_t;
			const auto _z = similar_FWI_pos_zero<real_t>();
			const auto _o = similar_FWI_one<real_t>();

			const numel_cnt_t tr = dest.rows(), tc = er.totalElements();
			NNTL_ASSERT(tc);
			auto pS = src.colDataAsVec(static_cast<vec_len_t>(er.elmBegin));
			auto _pD = dest.colDataAsVec(static_cast<vec_len_t>(er.elmBegin));
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			const auto pM = reinterpret_cast<const similar_FWI_t*const>(pMask);
			for (numel_cnt_t ci = 0; ci < tc; ++ci) {
				NNTL_ASSERT(pS == src.colDataAsVec(static_cast<vec_len_t>(er.elmBegin + ci)));
				const auto pD = _pD;
				_pD += tr;
				for (numel_cnt_t i = 0; i < tr; ++i) {
					const auto m = pM[i];
					NNTL_ASSERT(m == _o || m == _z);
					pD[i] = m != _z ? *pS++ : real_t(0);
				}
			}
		}
		void mFillRowsByMask_mt(const realmtx_t& src, const real_t*const pMask, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!src.empty() && pMask && !dest.empty() && src.cols_no_bias() == dest.cols_no_bias());
			NNTL_ASSERT(src.rows() <= dest.rows() && src.rows());
			NNTL_ASSERT(src.rows() == static_cast<vec_len_t>(get_self().vCountNonZeros(pMask, dest.rows())));

			if (dest.rows() == src.rows()) {
				//just copying src to dest
				const auto b = src.copy_data_skip_bias(dest);
				NNTL_ASSERT(b);
			} else {
				m_threads.run([&src, pMask, &dest, this](const par_range_t& r) noexcept{
					get_self()._imFillRowsByMask_st(src, pMask, dest, elms_range(r));
				}, dest.cols_no_bias());
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Extract whole submatrix dest from source matrix src starting at rowOfs.
		template<typename T>
		void mExtractRowsSeq(const smatrix<T>& src, const vec_len_t rowOfs, smatrix<T>& dest)noexcept {
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			if (dest.cols_no_bias() < 2 || dest.numel_no_bias() < Thresholds_t::mExtractRowsSeq) {
				get_self().mExtractRowsSeq_st(src, rowOfs, dest);
			} else get_self().mExtractRowsSeq_mt(src, rowOfs, dest);
		}
		
		template<typename T>
		void mExtractRowsSeq_st(const smatrix<T>& src, const vec_len_t rowOfs, smatrix<T>& dest, const elms_range*const pCR = nullptr)noexcept {
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
			/*if (src.emulatesBiases()) {
				dest.holey_biases(src.isHoleyBiases());
			}*/
			get_self()._imExtractRowsSeq_st(src, rowOfs, dest, pCR ? *pCR : elms_range(0, dest.cols_no_bias()));

			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}

		template<typename T>
		static void _imExtractRowsSeq_st(const smatrix<T>& src, const vec_len_t rowOfs, smatrix<T>& dest, const elms_range& CR)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			src.assert_storage_does_not_intersect(dest);

			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.cols() == dest.cols());
			NNTL_ASSERT(dest.rows() + rowOfs <= src.rows());
			NNTL_ASSERT(CR.elmBegin >= 0 && CR.elmEnd <= dest.cols_no_bias());

			const vec_len_t colBeg = static_cast<vec_len_t>(CR.elmBegin);
			auto* pD = dest.colDataAsVec(colBeg);
			auto* pS = src.colDataAsVec(colBeg) + rowOfs;
			const numel_cnt_t destRows = dest.rows(), srcRows = src.rows(), totCols = CR.totalElements();
			const size_t bytesToCopy = static_cast<size_t>(destRows) * sizeof(T);

			for (numel_cnt_t c = 0; c < totCols; ++c) {
				NNTL_ASSERT(pD <= dest.end_no_bias());
				NNTL_ASSERT(pS <= src.end_no_bias() + rowOfs);

				::std::memcpy(pD, pS, bytesToCopy);
				pD += destRows;
				pS += srcRows;
			}
		}
		template<typename T>
		void mExtractRowsSeq_mt(const smatrix<T>& src, const vec_len_t rowOfs, smatrix<T>& dest)noexcept {
			NNTL_ASSERT(!(src.emulatesBiases() ^ dest.emulatesBiases()));
			NNTL_ASSERT(src.cols() == dest.cols());
			NNTL_ASSERT(dest.rows() + rowOfs <= src.rows());
			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());

			/*if (src.emulatesBiases()) {
				dest.holey_biases(src.isHoleyBiases());
			}*/

			m_threads.run([&src, rowOfs, &dest, this](const par_range_t& r) noexcept{
				get_self()._imExtractRowsSeq_st(src, rowOfs, dest, elms_range(r));
			}, dest.cols_no_bias());

			NNTL_ASSERT(!src.emulatesBiases() || src.test_biases_strict());//if there're biases, they can't be holey
			NNTL_ASSERT(!dest.emulatesBiases() || dest.test_biases_strict());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// compute squared L2norm of each matrix A row into a vector pNormsVec: pNormsVec(i) = norm(A(i,:)) (rowwise sum of squares)
		// ATTENTION! pNormsVec MUST address at least ( A.rows()*m_threads.cur_workers_count() ) elements!
		// 
		template<typename T>
		nntl_probably_force_inline numel_cnt_t mrwL2NormSquared_needTempMem(const smatrix<T>& A)const noexcept {
			static_assert(::std::is_same<T, real_t>::value, "");
			return get_self().mrwSum_needTempMem(A);
		}
		void mrwL2NormSquared(const realmtx_t& A, real_t*const pNormsVec)noexcept {
			NNTL_ASSERT(pNormsVec);
			return (A.cols() <= Thresholds_t::mrwL2NormSquared_mt_cw_ColsPerThread || A.numel() < Thresholds_t::mrwL2NormSquared)
				? get_self().mrwL2NormSquared_st(A, pNormsVec)
				: get_self().mrwL2NormSquared_mt(A, pNormsVec);
		}

		//pNormsVec MUST address at least A.rows() elements
		void mrwL2NormSquared_st(const realmtx_t& A, real_t*const pNormsVec, const rowcol_range*const pRCR = nullptr)noexcept {
			get_self()._imrwL2NormSquared_st(A, pNormsVec, pRCR ? *pRCR : rowcol_range(A));
		}
		// #todo implement using _processMtx_rw/_processMtx_cw and move a whole family to SMath::
		void _imrwL2NormSquared_st(const realmtx_t& A, real_t*const pNormsVec, const rowcol_range& RCR)noexcept {
			NNTL_ASSERT(pNormsVec);
			NNTL_ASSERT(RCR.colBegin <= A.cols() && RCR.colEnd <= A.cols() && RCR.colBegin < RCR.colEnd);
			
			const auto mRows = A.rows();
			NNTL_ASSERT(0 == RCR.rowBegin && mRows == RCR.rowEnd);

			memset(pNormsVec, 0, sizeof(*pNormsVec)*mRows);

			const auto dataCnt = realmtx_t::sNumel(mRows, RCR.totalCols());
			const real_t* pCol = A.colDataAsVec(RCR.colBegin);
			const auto pColE = pCol + dataCnt;
			while (pCol != pColE) {
				auto pElm = pCol;
				pCol += mRows;
				const auto pElmE = pCol;
				auto pN = pNormsVec;
				while (pElm != pElmE) {
					const auto v = *pElm++;
					*pN++ += v*v;
				}
			}
		}
		void mrwL2NormSquared_mt(const realmtx_t& A, real_t*const pNormsVec)noexcept {
			NNTL_ASSERT(!A.empty());
			if (A.cols() <= Thresholds_t::mrwL2NormSquared_mt_cw_ColsPerThread) {
				get_self().mrwL2NormSquared_st(A, pNormsVec);
			} else {
				_processMtx_cw(A, Thresholds_t::mrwL2NormSquared_mt_cw_ColsPerThread
					, [&A, this](const rowcol_range& RCR, real_t*const pVec)noexcept
				{
					get_self()._imrwL2NormSquared_st(A, pVec, RCR);
				},
					[this](realmtx_t& fin)noexcept
				{
					get_self().mrwSum_ip(fin);
				}, pNormsVec);
			}
		}

		
		//////////////////////////////////////////////////////////////////////////
		//#todo implement
		/*void apply_max_norm(realmtxdef_t& W, const real_t maxL2NormSquared, const bool bNormIncludesBias)noexcept {

			NNTL_ASSERT(!W.empty() && maxL2NormSquared > real_t(0.));

			//готовим временное хранилище для вычисленных норм
			const auto mRows = W.rows();
			auto pRowsNorm = get_self()._get_thread_temp_raw_storage(mRows);
			//#todo _memset_rowrange()
			memset(pRowsNorm, 0, sizeof(*pRowsNorm)*mRows);

			//вычисляем нормы
			if (!bNormIncludesBias) W.//hide_last_col();
			//#todo - _st версию
			mrwL2NormSquared(W, pRowsNorm);
			if (!bNormIncludesBias) W.restore_last_col();

			//определяем масштабирующий коэффициент
		}

		//#todo reimplement using more generic and fast smath:: functions
		void apply_max_norm_st(realmtxdef_t& W, const real_t maxL2NormSquared, const bool bNormIncludesBias)noexcept {
		}*/

		

		//////////////////////////////////////////////////////////////////////////
		// treat matrix as a set of row-vectors (matrices in col-major mode!). For each row-vector check, whether
		// its length/norm is not longer, than predefined value. If it's longer, than rescale vector to this max length
		// (for use in max-norm weights regularization)
		// #TODO reimplement as apply_max_norm()
		template<typename T>
		nntl_probably_force_inline numel_cnt_t mCheck_normalize_rows_needTempMem(const smatrix<T>& A)const noexcept {
			static_assert(::std::is_same<T, real_t>::value, "");
			return smatrix_td::sNumel(A.rows(), m_threads.cur_workers_count()) //mCheck_normalize_rows_mt
				+ get_self().mrwL2NormSquared_needTempMem(A);
		}

		void mCheck_normalize_rows(realmtxdef_t& A, const real_t maxL2NormSquared, const bool bNormIncludesBias)noexcept {
			if (A.numel() < Thresholds_t::mCheck_normalize_rows) {
				get_self().mCheck_normalize_rows_st(A, maxL2NormSquared, bNormIncludesBias);
			} else get_self().mCheck_normalize_rows_mt(A, maxL2NormSquared, bNormIncludesBias);
		}
		//static constexpr real_t sCheck_normalize_rows_MULT = real_t(32.0);
		void mCheck_normalize_rows_st(realmtxdef_t& A, const real_t maxNormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty() && maxNormSquared > real_t(0.0));

			const auto mRows = A.rows();
			auto pTmp = get_self()._istor_alloc(mRows);
			
			//A could be (and almost always is) a weight matrix that doesn't have correct emulatesBias() property, therefore
			//have to use unconditional .hide_last_col() instead of .hide_biases()
			if (!bNormIncludesBias) A.hide_last_col();
			get_self().mrwL2NormSquared_st(A, pTmp);
			if (!bNormIncludesBias) A.restore_last_col();

			//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, 
			// that doesn't need.
			// Making newNorm slightly less, than maxNormSquared to make sure the result will be less than max norm.
			//const real_t newNorm = maxNormSquared - math::real_t_limits<real_t>::eps_lower_n(maxNormSquared, sCheck_normalize_rows_MULT);
			// removed. it's not a big deal if resulting norm will be slightly bigger
			const real_t newNorm = maxNormSquared;// -2 * ::std::sqrt(math::real_t_limits<real_t>::eps_lower(maxNormSquared));
			auto pCurNorm = pTmp;
			const auto pTmpE = pTmp + mRows;
			while (pCurNorm != pTmpE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = rowNorm > maxNormSquared ? ::std::sqrt(newNorm / rowNorm) : real_t(1.0);
			}

			//renormalize (multiply each rowvector to corresponding coefficient from pTmp)
			get_self().mrwMulByVec_st(A, pTmp);
			get_self()._istor_free(pTmp, mRows);
		}
		//TODO: might be good to make separate _cw and _rw versions of this algo
		void mCheck_normalize_rows_mt(realmtxdef_t& A, const real_t maxNormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty() && maxNormSquared > real_t(0.0));
						
			const auto mRows = A.rows();
			const auto tmemSize = smatrix_td::sNumel(mRows, m_threads.cur_workers_count());
			const auto pTmpStor = get_self()._istor_alloc(tmemSize);

			//A could be (and almost always is) a weight matrix that doesn't have correct emulatesBias() property, therefore
			//have to use unconditional .hide_last_col() instead of .hide_biases()
			if (!bNormIncludesBias) A.hide_last_col();
			get_self().mrwL2NormSquared(A, pTmpStor);
			if (!bNormIncludesBias) A.restore_last_col();

			// calc scaling coefficients
			const auto pRowNormE = pTmpStor + mRows;
			const real_t newNorm = maxNormSquared;// -2 * ::std::sqrt(math::real_t_limits<real_t>::eps_lower(maxNormSquared));
			auto pCurNorm = pTmpStor;
			while (pCurNorm != pRowNormE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = rowNorm > maxNormSquared ? ::std::sqrt(newNorm / rowNorm) : real_t(1.0);
			}

			// 3. multiplying
			get_self().mrwMulByVec(A, pTmpStor);
			get_self()._istor_free(pTmpStor, tmemSize);
		}

		//////////////////////////////////////////////////////////////////////////
		// scales each row-vector of matrix A such that it would have l2norm^2 == L2NormSquared.
		// bNormIncludesBias flag controls whether to include last column (bias or bias weight usually) in norm calculation or not.
		
		template<typename T>
		void mrwSetL2Norm(smatrix<T>& A, const T L2NormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty());
			smatrix_deform<T> Def(A.data(), A);
			mrwSetL2Norm(Def, L2NormSquared, bNormIncludesBias);
		}
		template<typename T>
		void mrwSetL2Norm(smatrix_deform<T>& A, const T L2NormSquared, const bool bNormIncludesBias)noexcept {
			if (A.numel() < Thresholds_t::mCheck_normalize_rows) { //algo almost the same as mCheck_normalize_rows, so leaving its threshold
				get_self().mrwSetL2Norm_st(A, L2NormSquared, bNormIncludesBias);
			} else get_self().mrwSetL2Norm_mt(A, L2NormSquared, bNormIncludesBias);
		}
		template<typename T>
		nntl_probably_force_inline numel_cnt_t mrwSetL2Norm_needTempMem(const smatrix<T>& A)const noexcept {
			static_assert(::std::is_same<T, real_t>::value, "");
			return smatrix_td::sNumel(A.rows(), m_threads.cur_workers_count()) //mrwSetL2Norm_mt
				+ get_self().mrwL2NormSquared_needTempMem(A);
		}
		template<typename T>
		void mrwSetL2Norm_st(smatrix_deform<T>& A, const T L2NormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty() && L2NormSquared > real_t(0.0));

			const auto mRows = A.rows();
			auto pTmp = get_self()._istor_alloc(mRows);

			//A could be (and almost always is) a weight matrix that doesn't have correct emulatesBias() property, therefore
			//have to use unconditional .hide_last_col() instead of .hide_biases()
			if (!bNormIncludesBias) A.hide_last_col();
			get_self().mrwL2NormSquared_st(A, pTmp);
			if (!bNormIncludesBias) A.restore_last_col();

			//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, 
			// that doesn't need.
			// Making newNorm slightly less, than L2NormSquared to make sure the result will be less than max norm.
			//const real_t newNorm = L2NormSquared - math::real_t_limits<real_t>::eps_lower_n(L2NormSquared, sCheck_normalize_rows_MULT);
			// removed. it's not a big deal if resulting norm will be slightly bigger
			const real_t newNorm = L2NormSquared;// -2 * ::std::sqrt(math::real_t_limits<real_t>::eps_lower(L2NormSquared));
			auto pCurNorm = pTmp;
			const auto pTmpE = pTmp + mRows;
			while (pCurNorm != pTmpE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = ::std::sqrt(newNorm / rowNorm);
			}

			//renormalize (multiply each rowvector to corresponding coefficient from pTmp)
			get_self().mrwMulByVec_st(A, pTmp);
			get_self()._istor_free(pTmp, mRows);
		}
		//TODO: might be good to make separate _cw and _rw versions of this algo
		template<typename T>
		void mrwSetL2Norm_mt(smatrix_deform<T>& A, const T L2NormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty() && L2NormSquared > real_t(0.0));

			const auto mRows = A.rows();
			const auto tmemSize = smatrix_td::sNumel(mRows, m_threads.cur_workers_count());
			const auto pTmpStor = get_self()._istor_alloc(tmemSize);

			//A could be (and almost always is) a weight matrix that doesn't have correct emulatesBias() property, therefore
			//have to use unconditional .hide_last_col() instead of .hide_biases()
			if (!bNormIncludesBias) A.hide_last_col();
			get_self().mrwL2NormSquared(A, pTmpStor);
			if (!bNormIncludesBias) A.restore_last_col();

			// calc scaling coefficients
			const auto pRowNormE = pTmpStor + mRows;
			const real_t newNorm = L2NormSquared;// -2 * ::std::sqrt(math::real_t_limits<real_t>::eps_lower(L2NormSquared));
			auto pCurNorm = pTmpStor;
			while (pCurNorm != pRowNormE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = ::std::sqrt(newNorm / rowNorm);
			}

			// 3. multiplying
			get_self().mrwMulByVec(A, pTmpStor);
			get_self()._istor_free(pTmpStor, tmemSize);
		}

		//////////////////////////////////////////////////////////////////////////
		// Returns the number of elements equal to zero
		// Don't expect big n here, so no _mt version
		template<typename T>
		static ::std::enable_if_t<::std::is_floating_point<T>::value, numel_cnt_t> vCountNonZeros(const T* pVec, const numel_cnt_t ne)noexcept {
			NNTL_ASSERT(pVec && ne > 0);
			numel_cnt_t nz = 0;
			const auto pE = pVec + ne;
			while (pVec != pE) {//#DIDNT_VECTORIZE
				const auto v = *pVec++;
				nz += (v > T(+0.)) + (v < T(-0.));
			}
			return nz;
		}

		//Strict version counts only a positive zero as a zero. Works about a twice as fast as vCountNonZeros
		template<typename T>
		static ::std::enable_if_t<::std::is_floating_point<T>::value, numel_cnt_t> vCountNonZerosStrict(const T* pVec, const numel_cnt_t ne)noexcept {
			NNTL_ASSERT(pVec && ne > 0);
			typedef typename real_t_limits<T>::similar_FWI_t similar_FWI_t;
			NNTL_ASSERT(0 == similar_FWI_pos_zero<T>());
			static_assert(::std::is_unsigned<similar_FWI_t>::value, "");

			similar_FWI_t nz = 0;
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			const similar_FWI_t* ptr = reinterpret_cast<const similar_FWI_t*>(pVec);
			const auto pE = ptr + ne;
			while (ptr != pE) {
				const auto v = *ptr++;
				nz += v > 0;
			}
			return nz;
		}


		//////////////////////////////////////////////////////////////////////////
		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		// Note that ADataOfs should only be used if Contnr is a 1D container (::std::vector, for example, or a matrix with only 1 column)
		template<typename Contnr>
		numel_cnt_t vCountSame(const Contnr& A, const Contnr& B, const vec_len_t ADataOfs = 0)noexcept {
			return get_self().vCountSame_st_naive(A, B, ADataOfs);
			// 			if (A.size()<=50000) {
			// 				return vCountSame_st_naive(A, B);
			// 			}else return vCountSame_mt_naive(A, B);
		}
		template<typename Contnr>
		static numel_cnt_t vCountSame_st_naive(const Contnr& A, const Contnr& B, const vec_len_t ADataOfs = 0)noexcept {
			NNTL_ASSERT(ADataOfs >= 0);
			NNTL_ASSERT(Numel(A) - ADataOfs >= Numel(B));

			const auto pA = A.data() + ADataOfs;
			const auto pB = B.data();
			numel_cnt_t ret = 0;
			const auto dataCnt = Numel(B);
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				//if (A[i] == B[i]) ret++;
				//ret += A[i] == B[i] ? 1 : 0;
				ret += numel_cnt_t(pA[i] == pB[i]);
			}
			return ret;
		}
		template<typename Contnr>
		numel_cnt_t vCountSame_mt_naive(const Contnr& A, const Contnr& B, const vec_len_t ADataOfs = 0)noexcept {
			NNTL_ASSERT(ADataOfs >= 0);
			NNTL_ASSERT(Numel(A) - ADataOfs >= Numel(B));

			auto pAc = A.data() + ADataOfs;
			auto pBc = B.data();
			numel_cnt_t ret = m_threads.reduce([pAc, pBc](const par_range_t& r)noexcept->reduce_data_t {
				const auto ofs = r.offset();
				numel_cnt_t ret = 0;
				const auto pA = &pAc[ofs];
				const auto pB = &pBc[ofs];
				const auto cnt = r.cnt();
				for (range_t i = 0; i < cnt; ++i) {
					//if (pA[i] == pB[i]) ret++;
					ret += numel_cnt_t(pA[i] == pB[i]);
				}
				return converter_reduce_data_tpl<numel_cnt_t>::to(ret);
			}, _reduce_vec_sum<numel_cnt_t>, Numel(B));
			return ret;
		}
		//////////////////////////////////////////////////////////////////////////
		//vCountSame() that compares only a part of bigger matrix with a smaller matrix
		template<typename TA, typename TB>
		numel_cnt_t vCountSameBatch(const smatrix<TA>& A, const vec_len_t rOfsA, const smatrix<TB>& B)noexcept {
			return get_self().vCountSameBatch_st_naive(A, rOfsA, B);
			// 			if (A.size()<=50000) {
			// 				return vCountSame_st_naive(A, B);
			// 			}else return vCountSame_mt_naive(A, B);
		}
		template<typename TA, typename TB>
		static numel_cnt_t vCountSameBatch_st_naive(const smatrix<TA>& A, const vec_len_t rOfsA, const smatrix<TB>& B)noexcept {
			NNTL_ASSERT(A.cols() == B.cols() && B.rows() + rOfsA <= A.rows());

			const TA* pA = A.data() + rOfsA;
			const TB* pB = B.data();
			const numel_cnt_t aRows = A.rows();
			const numel_cnt_t bRows = B.rows();
			numel_cnt_t ret = 0;

			const auto cls = B.cols();
			for (vec_len_t c = 0; c < cls; ++c) {
				const auto _pA = pA;
				NNTL_ASSERT(_pA + aRows <= A.end() + rOfsA);
				const auto _pB = pB;
				NNTL_ASSERT(_pB + bRows <= B.end());
				for (numel_cnt_t r = 0; r < bRows; ++r) {
					ret += numel_cnt_t(_pA[r] == _pB[r]);
				}
				pA += aRows;
				pB += bRows;
			}
			return ret;
		}
		//#TODO: probably should implement multithreaded version (same as ewBinarizeBatch()) 


		//////////////////////////////////////////////////////////////////////////
		//clamps vector values into range
		void evClamp(realmtx_t& m, real_t lo, real_t hi)noexcept {
			if (m.numel() < Thresholds_t::evClamp) {
				get_self().evClamp_st(m, lo, hi);
			} else get_self().evClamp_mt(m, lo, hi);
		}
		static void evClamp_st(realmtx_t& m, real_t lo, real_t hi)noexcept {
			NNTL_ASSERT(m.numel() > 0 && !m.empty());
			NNTL_ASSERT(lo < hi);

			auto p = m.data();
			//utils::boost::algorithm::clamp_range(p, p + m.numel(), p, lo, hi);
			::boost::algorithm::clamp_range(p, p + m.numel(), p, lo, hi);
		}
		void evClamp_mt(realmtx_t& m, real_t lo, real_t hi)noexcept {
			NNTL_ASSERT(m.numel() > 0 && !m.empty());
			NNTL_ASSERT(lo < hi);

			auto ptr = m.data();
			m_threads.run([ptr, lo, hi](const par_range_t& r) noexcept{
				auto p = ptr + r.offset();
				//utils::boost::algorithm::clamp_range(p, p + r.cnt(), p, lo, hi);
				::boost::algorithm::clamp_range(p, p + r.cnt(), p, lo, hi);
			}, m.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//on entry the dropoutMask must be filled with random values in range [0,1]
		//Function binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// dropPercAct - probability of keeping unit active
		// act must be used in "no_bias" mode.
		// Actually, the function implements so called "inverted Dropout", see http://cs231n.github.io/neural-networks-2/
		// And by the way, it seems to work faster, than using bernuolli_vector(), see TEST(TestPerfDecisions, makeDropoutPerf)
		void make_dropout(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask)noexcept {
			if (dropoutMask.numel() < Thresholds_t::make_dropout) {
				get_self().make_dropout_st(act, dropPercAct, dropoutMask);
			} else get_self().make_dropout_mt(act, dropPercAct, dropoutMask);
		}
		void make_dropout_st(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask, const elms_range*const pER = nullptr) const noexcept {
			get_self()._imake_dropout_st(act, dropPercAct, dropoutMask, pER ? *pER : elms_range(dropoutMask));
		}
		static void _imake_dropout_st(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask, const elms_range& er) noexcept {
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);

			/*const real_t dropPercActInv = real_t(1.) / dropPercAct;
			auto pDM = dropoutMask.data()+er.elmBegin;
			const auto pDME = pDM + er.totalElements();
			while (pDM != pDME) {
				const auto v = *pDM;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				*pDM++ = v < dropPercAct ? dropPercActInv : real_t(0.);
			}
			get_self()._ievMul_ip_st(act.data(), dropoutMask.data(), er);*/

			const real_t dropPercActInv = real_t(1.) / dropPercAct;
			auto pDM = dropoutMask.data() + er.elmBegin;
			auto pA = act.data() + er.elmBegin;
			const auto pDME = pDM + er.totalElements();
			while (pDM != pDME) { //vectorized!
				const auto v = *pDM;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				const real_t dv = v < dropPercAct ? dropPercActInv : real_t(0.);
				*pA++ *= dv;
				*pDM++ = dv;
			}
		}
		void make_dropout_mt(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask)noexcept {
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);
			m_threads.run([&act, &dropoutMask, dropPercAct,this](const par_range_t& r) noexcept{
				get_self()._imake_dropout_st(act, dropPercAct, dropoutMask, elms_range(r));
			}, dropoutMask.numel());
		}
		
		//////////////////////////////////////////////////////////////////////////
		// Similar to make_dropout(), but it's not an inverted but direct dropout
		// and it doesn't change the dropoutMask parameter and therefore faster.
		// on entry the dropoutMask must be filled with random values in range [0,1]
		// Function binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to passed matrix
		// dropPercAct - probability of keeping unit active
		// mtx must be used in "no_bias" mode.
		// Borrowed from make_dropout note: And by the way, it seems to work faster, than using bernuolli_vector(), see TEST(TestPerfDecisions, makeDropoutPerf)
		void apply_dropout_mask(realmtx_t& mtx, const real_t dropPercAct, const realmtx_t& dropoutMask)noexcept {
			if (dropoutMask.numel() < Thresholds_t::apply_dropout_mask) {
				get_self().apply_dropout_mask_st(mtx, dropPercAct, dropoutMask);
			} else get_self().apply_dropout_mask_mt(mtx, dropPercAct, dropoutMask);
		}
		void apply_dropout_mask_st(realmtx_t& mtx, const real_t dropPercAct, const realmtx_t& dropoutMask, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iapply_dropout_mask_st(mtx, dropPercAct, dropoutMask, pER ? *pER : elms_range(dropoutMask));
		}
		static void _iapply_dropout_mask_st(realmtx_t& mtx, const real_t dropPercAct, const realmtx_t& dropoutMask, const elms_range& er) noexcept {
			NNTL_ASSERT(mtx.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);

			auto pDM = dropoutMask.data() + er.elmBegin;
			auto pA = mtx.data() + er.elmBegin;
			const auto pDME = pDM + er.totalElements();
			while (pDM != pDME) { //#DIDNT_VECTORIZE
				const auto v = *pDM++;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				//const real_t dv = v < dropPercAct ? real_t(1.) : real_t(0.);
				*pA++ *= v < dropPercAct ? real_t(1.) : real_t(0.);
			}
		}
		void apply_dropout_mask_mt(realmtx_t& mtx, const real_t dropPercAct, const realmtx_t& dropoutMask)noexcept {
			NNTL_ASSERT(mtx.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);
			m_threads.run([&mtx, &dropoutMask, dropPercAct, this](const par_range_t& r) noexcept{
				get_self()._iapply_dropout_mask_st(mtx, dropPercAct, dropoutMask, elms_range(r));
			}, dropoutMask.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		// on entry the dropoutMask must be filled with random values in range [0,1]
		// For Alpha-dropout description see arxiv:1706.02515 "Self-Normalizing Neural Networks", by Günter Klambauer et al.
		// Here we compute
		// {dropoutMask <- 0, mtxB <- (a*(-Alpha*Lambda) + b) } with probability (1-p) and
		// {dropoutMask <- a, mtxB <- b } with probability p
		// Then we compute the post-dropout activations A3 <- A.*dropoutMask + mtxB
		void make_alphaDropout(realmtx_t& act, const real_t dropPercAct
			, const real_t a_dmKeepVal, const real_t b_mbKeepVal, const real_t mbDropVal, realmtx_t& dropoutMask)noexcept
		{
			if (dropoutMask.numel() < Thresholds_t::make_alphaDropout) {
				get_self().make_alphaDropout_st(act, dropPercAct, a_dmKeepVal, b_mbKeepVal, mbDropVal, dropoutMask);
			} else get_self().make_alphaDropout_mt(act, dropPercAct, a_dmKeepVal, b_mbKeepVal, mbDropVal, dropoutMask);
		}
		void make_alphaDropout_st(realmtx_t& act, const real_t dropPercAct
			, const real_t a_dmKeepVal, const real_t b_mbKeepVal, const real_t mbDropVal
			, realmtx_t& dropoutMask, const elms_range*const pER = nullptr) const noexcept
		{
			get_self()._imake_alphaDropout_st(act, dropPercAct, a_dmKeepVal, b_mbKeepVal, mbDropVal, dropoutMask
				, pER ? *pER : elms_range(dropoutMask));
		}
		static void _imake_alphaDropout_st(realmtx_t& act, const real_t dropPercAct, const real_t a_dmKeepVal
			, const real_t b_mbKeepVal, const real_t mbDropVal, realmtx_t& dropoutMask, const elms_range& er) noexcept
		{
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);
			NNTL_ASSERT(er.elmEnd <= dropoutMask.numel());
			NNTL_ASSERT(a_dmKeepVal > real_t(0));//need it to break into two loops

			/*real_t* pA = act.data() + er.elmBegin;
			real_t* pDM = dropoutMask.data() + er.elmBegin;
			const auto pDME = pDM + er.totalElements();
			while (pDM != pDME) {
				const real_t v = *pDM;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
 				const auto bKeep = v < dropPercAct;
 				*pDM++ = bKeep ? a_dmKeepVal : real_t(0.);
// 				*pA++ = bKeep ? (*pA*a_dmKeepVal + b_mbKeepVal) : mbDropVal;	//prevents loop vectorization
			}

			pDM = dropoutMask.data() + er.elmBegin;
			while (pDM != pDME) {
				*pA++ = *pDM++ > real_t(0.) ? (*pA * a_dmKeepVal + b_mbKeepVal) : mbDropVal;
			}*/
			
			real_t* pA = act.data() + er.elmBegin;
			real_t* pDM = dropoutMask.data() + er.elmBegin;
			const auto pDME = pDM + er.totalElements();
			const real_t b_div_a = b_mbKeepVal / a_dmKeepVal;
			const real_t mbDV_div_a = mbDropVal / a_dmKeepVal;
			while (pDM != pDME) { //vectorizes!
				const real_t v = *pDM;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				const real_t dv = v < dropPercAct ? a_dmKeepVal : real_t(0.);
				*pA++ = dv*(*pA + b_div_a) + (a_dmKeepVal - dv)*mbDV_div_a;
				*pDM++ = dv;
			}
		}
		void make_alphaDropout_mt(realmtx_t& act, const real_t dropPercAct
			, const real_t a_dmKeepVal, const real_t b_mbKeepVal, const real_t mbDropVal, realmtx_t& dropoutMask)noexcept
		{
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);

			m_threads.run([&act, &dropoutMask, a_dmKeepVal, b_mbKeepVal, mbDropVal, dropPercAct, this](const par_range_t& r) noexcept{
				get_self()._imake_alphaDropout_st(act, dropPercAct, a_dmKeepVal, b_mbKeepVal, mbDropVal, dropoutMask, elms_range(r));
			}, dropoutMask.numel());
		}

		////////////////////////////////////////////////////////////////////////// 
		//////////////////////////////////////////////////////////////////////////
		//apply individual learning rate to dLdW
		template<typename T>
		nntl_probably_force_inline numel_cnt_t apply_ILR_needTempMem(const smatrix_td::mtx_size_t& dLdWMaxSize)const noexcept {
			static_assert(::std::is_same<T, real_t>::value, "");
			return smatrix_td::sNumel(dLdWMaxSize);
		}
		void apply_ILR(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			const auto dataCnt = dLdW.numel();
			if (dataCnt < Thresholds_t::apply_ILR_mt) {
				if (dataCnt <= Thresholds_t::apply_ILR_st_vec) {
					get_self().apply_ILR_st_vec(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
				} else get_self().apply_ILR_st_naive(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
			} else {
				if (dataCnt <= Thresholds_t::apply_ILR_mt_vec || dataCnt >= Thresholds_t::apply_ILR_mt_vec2) {
					get_self().apply_ILR_mt_vec(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
				} else get_self().apply_ILR_mt_naive(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
			}
		}
		static void apply_ILR_st_naive(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			NNTL_ASSERT(dLdW.size() == prevdLdW.size() && dLdW.size() == ILRGain.size());
			NNTL_ASSERT(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

			//TODO: probably not the most efficient implementation

			const auto dataCnt = dLdW.numel();

			auto pdW = dLdW.data();
			const auto prevdW = prevdLdW.data();
			auto pGain = ILRGain.data();

			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				auto pW = pdW + i;
				auto pG = pGain + i;
				const real_t cond = prevdW[i] * (*pW);
				auto g = *pG;
				
				/*if (cond > real_t(+0.0)) {
					if (g < capHigh) g *= incr;
				} else if (cond < real_t(-0.0)) {
					if (g > capLow) g *= decr;
				}
				*/

				const auto bUp = (cond > real_t(+0.0))&(g < capHigh)
					, bDown = (cond < real_t(-0.0)) & (g > capLow);

				g *= (!(bUp | bDown))*real_t(1.) + bUp*incr + bDown*decr;
				*pG = g;
				*pW *= g;
			}
		}
		void apply_ILR_mt_naive(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			NNTL_ASSERT(dLdW.size() == prevdLdW.size() && dLdW.size() == ILRGain.size());
			NNTL_ASSERT(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

			//TODO: probably not the most efficient implementation

			auto pdW = dLdW.data();
			const auto prevdW = prevdLdW.data();
			auto pGain = ILRGain.data();
			m_threads.run([pdW, prevdW, pGain, decr, incr, capLow, capHigh](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					auto pW = pdW + i;
					auto pG = pGain + i;
					const auto cond = prevdW[i] * (*pW);
					auto g = *pG;
					
					/*if (cond > real_t(+0.0)) {
						if (g < capHigh) g *= incr;
					} else if (cond < real_t(-0.0)) {
						if (g > capLow) g *= decr;
					}*/
					const auto bUp = (cond > real_t(+0.0))&(g < capHigh)
						, bDown = (cond < real_t(-0.0)) & (g > capLow);

					g *= (!(bUp | bDown))*real_t(1.) + bUp*incr + bDown*decr;
					*pG = g;
					*pW *= g;
				}
			}, dLdW.numel());
		}
		void apply_ILR_st_vec(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			NNTL_ASSERT(dLdW.size() == prevdLdW.size() && dLdW.size() == ILRGain.size());
			NNTL_ASSERT(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

			//TODO: probably not the most efficient implementation

			const auto dataCnt = dLdW.numel();
			auto pCond = get_self()._istor_alloc(dataCnt);

			auto pdW = dLdW.data();
			const auto prevdW = prevdLdW.data();
			auto pGain = ILRGain.data();

			for (numel_cnt_t i = 0; i < dataCnt; ++i) pCond[i] = pdW[i] * prevdW[i];

			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				auto pG = pGain + i;
				const auto cond = pCond[i];
				const auto g = *pG;

				/*if (cond > real_t(+0.0)) {
					if (g < capHigh) g *= incr;
				} else if (cond < real_t(-0.0)) {
					if (g > capLow) g *= decr;
				}
				*pG = g;*/

				const auto bUp = (cond > real_t(+0.0))&(g < capHigh)
					, bDown = (cond < real_t(-0.0)) & (g > capLow);

				*pG *= (!(bUp | bDown))*real_t(1.) + bUp*incr + bDown*decr;
			}

			get_self()._istor_free(pCond, dataCnt);

			for (numel_cnt_t i = 0; i < dataCnt; ++i) pdW[i] *= pGain[i];
		}

		void apply_ILR_mt_vec(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			NNTL_ASSERT(dLdW.size() == prevdLdW.size() && dLdW.size() == ILRGain.size());
			NNTL_ASSERT(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

			//TODO: probably not the most efficient implementation

			const auto dataCnt = dLdW.numel();
			const auto pTmpMem = get_self()._istor_alloc(dataCnt);
			const auto pdW = dLdW.data(), pGain = ILRGain.data();
			const auto prevdW = prevdLdW.data();

			m_threads.run([pdW, prevdW, pGain, decr, incr, capLow, capHigh, pTmpMem](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				const auto cnt = r.cnt();
				//auto pCond = pTmpMem[r.tid()];
				auto pCond = pTmpMem + ofs;
				const auto pW = pdW + ofs;
				const auto prW = prevdW + ofs;
				const auto pGn = pGain + ofs;

				for (numel_cnt_t i = 0; i < cnt; ++i) pCond[i] = pW[i] * prW[i];

				for (numel_cnt_t i = 0; i < cnt; ++i) {
					auto pG = pGn + i;
					const auto cond = pCond[i];
					const auto g = *pG;
					
					/*if (cond > real_t(+0.0)) {
						if (g < capHigh) g *= incr;
					} else if (cond < real_t(-0.0)) {
						if (g > capLow) g *= decr;
					}
					*pG = g;*/
					const auto bUp = (cond > real_t(+0.0))&(g < capHigh)
						, bDown = (cond < real_t(-0.0)) & (g > capLow);

					*pG *= (!(bUp | bDown))*real_t(1.) + bUp*incr + bDown*decr;
				}

				for (numel_cnt_t i = 0; i < cnt; ++i) pW[i] *= pGn[i];
			}, dataCnt);
			get_self()._istor_free(pTmpMem, dataCnt);
		}

		//////////////////////////////////////////////////////////////////////////
		//apply momentum vW = momentum.*vW + dW
		void apply_momentum(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			if (vW.numel() < Thresholds_t::apply_momentum) {
				get_self().apply_momentum_st(vW, momentum, dW);
			} else get_self().apply_momentum_mt(vW, momentum, dW);
		}
		static void apply_momentum_st(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			NNTL_ASSERT(vW.size() == dW.size());
			NNTL_ASSERT(!vW.empty() && !dW.empty());

			const auto dataCnt = vW.numel();
			const auto pV = vW.data();
			const auto pdW = dW.data();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				pV[i] = momentum*pV[i] + pdW[i];
			}
		}
		void apply_momentum_mt(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			NNTL_ASSERT(vW.size() == dW.size());
			NNTL_ASSERT(!vW.empty() && !dW.empty());

			const auto pV = vW.data();
			const auto pdW = dW.data();
			m_threads.run([pV, pdW, momentum](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					pV[i] = momentum*pV[i] + pdW[i];
				}
			}, vW.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		// scale and offset vector, i.e. A = sc*A - ofs
		template<typename T>
		static void vMulCAddC_ip_st(T* __restrict pA, T* __restrict const pAE, const T sc, const T ofs)noexcept {
			NNTL_ASSERT(pA && pAE && pA <= pAE);
			NNTL_ASSERT(sc != T(0));
			NNTL_ASSERT(!::std::is_floating_point<T>::value ||
				(!::std::isnan(sc) && ::std::isfinite(sc) && !::std::isnan(ofs) && ::std::isfinite(ofs)));
			//#TODO should help a compiler with FMA (probably, should code by hand - check it)?
			while (pA != pAE) { // #vectorized
				const auto v = *pA;
				*pA++ = sc*v + ofs;
			};
		}

		//////////////////////////////////////////////////////////////////////////
		// Add constant and rescale every matrix element, A = sc*A - ofs
		// #supportsBatchInRow
		template<typename T>
		void evMulCAddC_ip(smatrix<T>& A, const T sc, const T ofs, const bool bIgnoreBias = false) noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(sc) && ::std::isfinite(sc) && !::std::isnan(ofs) && ::std::isfinite(ofs));
			NNTL_ASSERT(sc != T(0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			//#TODO T must be used when branching here
			if (A.numel() < Thresholds_t::evMulCAddC_ip) {
				get_self().evMulCAddC_ip_st(A, sc, ofs, bIgnoreBias);
			} else get_self().evMulCAddC_ip_mt(A, sc, ofs, bIgnoreBias);
		}
		template<typename T>
		void evMulCAddC_ip_nb(smatrix<T>& A, const T sc, const T ofs) noexcept {
			get_self().evMulCAddC_ip(A, sc, ofs, true);
		}
		template<typename T>
		static void evMulCAddC_ip_st(smatrix<T>& A, const T sc, const T ofs, const bool bIgnoreBias)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(sc) && ::std::isfinite(sc) && !::std::isnan(ofs) && ::std::isfinite(ofs));
			NNTL_ASSERT(sc != T(0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			self_t::_ievMulCAddC_ip_st(A, sc, ofs, bIgnoreBias, par_range_t(A._universal_range_BiR_aware(bIgnoreBias)));
		}
		template<typename T>
		void evMulCAddC_ip_mt(smatrix<T>& A, const T sc, const T ofs, const bool bIgnoreBias) noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(sc) && ::std::isfinite(sc) && !::std::isnan(ofs) && ::std::isfinite(ofs));
			NNTL_ASSERT(sc != T(0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			get_self().ithreads().run([&A, sc, ofs, bIgnoreBias](const par_range_t& pr)noexcept {
				self_t::_ievMulCAddC_ip_st(A, sc, ofs, bIgnoreBias, pr);
			}, A._universal_range_BiR_aware(bIgnoreBias));
		}
		template<typename T>
		static void _ievMulCAddC_ip_st(smatrix<T>& A, const T sc, const T ofs, const bool bIgnoreBias, const par_range_t& pr)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(sc) && ::std::isfinite(sc) && !::std::isnan(ofs) && ::std::isfinite(ofs));
			NNTL_ASSERT(sc != T(0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			if (bIgnoreBias & A.bBatchInRow() & A.emulatesBiases()) { //we had to exclude bias row from adjustments here
				NNTL_ASSERT(pr.end() <= A.cols());
				const ptrdiff_t rowsNb = A.rows() - 1;
				auto pA = A.colDataAsVec(static_cast<vec_len_t>(pr.offset()));
				const auto pAE = A.colDataAsVec(static_cast<vec_len_t>(pr.end()));
				while (pA < pAE) {
					const auto tpA = pA;
					pA += rowsNb;
					self_t::vMulCAddC_ip_st(tpA, pA, sc, ofs);
					++pA;
				}
			} else {
				const auto pA = A.data();
				self_t::vMulCAddC_ip_st(pA + pr.offset(), pA + pr.end(), sc, ofs);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = b.*A
		/*static void _ievMulC_ip_st(real_t* pA, const real_t b, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && er.totalElements() > 0);
			const real_t*const pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			while (pA != pAE) *pA++ *= b;
		}*/
		template<typename T>
		static nntl_force_inline void vMulC_ip_st(T* __restrict pA, T* __restrict const pAE, const T C)noexcept {
			NNTL_ASSERT(pA && pAE && pA <= pAE);
			NNTL_ASSERT(C != T(0));
			while (pA != pAE) { // #vectorized
				*pA++ *= C;
			};
		}
		//////////////////////////////////////////////////////////////////////////
		void evMulC_ip(real_t* pA, const numel_cnt_t n, const real_t b)noexcept {
			if (n < Thresholds_t::evMulC_ip) {
				get_self().evMulC_ip_st(pA, n, b);
			} else get_self().evMulC_ip_mt(pA, n, b);
		}
		void evMulC_ip_st(real_t* pA, const numel_cnt_t n, const real_t b)const noexcept { //, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(pA && n > 0);
			//NNTL_ASSERT(!pER || pER->elmEnd <= n);
			//get_self()._ievMulC_ip_st(pA, b, pER ? *pER : elms_range(0, n));
			get_self().vMulC_ip_st(pA, pA + n, b);
		}
		void evMulC_ip_mt(real_t* pA, const numel_cnt_t n, const real_t b)noexcept {
			NNTL_ASSERT(pA && n > 0);
			m_threads.run([pA, b, this](const par_range_t& pr) noexcept{
				//get_self()._ievMulC_ip_st(pA, b, elms_range(pr));
				get_self().vMulC_ip_st(pA + pr.offset(), pA + pr.end(), b);
			}, n);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = c.*A
		// #supportsBatchInRow
		template<typename T>
		void evMulC_ip(smatrix<T>& A, const T c, const bool bIgnoreBias = false)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c) && c != T(0.0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			if (A.numel() < Thresholds_t::evMulC_ip) {
				get_self().evMulC_ip_st(A, c, bIgnoreBias);
			} else get_self().evMulC_ip_mt(A, c, bIgnoreBias);
		}
		template<typename T>
		void evMulC_ip_nb(smatrix<T>& A, const T c)noexcept {
			get_self().evMulC_ip(A, c, true);
		}
		template<typename T>
		static void evMulC_ip_st(smatrix<T>& A, const T c, const bool bIgnoreBias) noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c) && c != T(0.0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			self_t::_ievMulC_ip_st(A, c, bIgnoreBias, par_range_t(A._universal_range_BiR_aware(bIgnoreBias)));
		}
		template<typename T>
		void evMulC_ip_mt(smatrix<T>& A, const T c, const bool bIgnoreBias)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c) && c != T(0.0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			get_self().ithreads().run([&A, c, bIgnoreBias](const par_range_t& pr)noexcept {
				self_t::_ievMulC_ip_st(A, c, bIgnoreBias, pr);
			}, A._universal_range_BiR_aware(bIgnoreBias));
		}
		template<typename T>
		static void _ievMulC_ip_st(smatrix<T>& A, const T c, const bool bIgnoreBias, const par_range_t& pr)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c) && c != T(0.0));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			if (bIgnoreBias & A.bBatchInRow() & A.emulatesBiases()) { //we had to exclude bias row from adjustments here
				NNTL_ASSERT(pr.end() <= A.cols());
				const ptrdiff_t rowsNb = A.rows() - 1;
				auto pA = A.colDataAsVec(static_cast<vec_len_t>(pr.offset()));
				const auto pAE = A.colDataAsVec(static_cast<vec_len_t>(pr.end()));
				while (pA < pAE) {
					const auto tpA = pA;
					pA += rowsNb;
					self_t::vMulC_ip_st(tpA, pA, c);
					++pA;
				}
			} else {
				const auto pA = A.data();
				self_t::vMulC_ip_st(pA + pr.offset(), pA + pr.end(), c);
			}
		}
		
		//////////////////////////////////////////////////////////////////////////
		// inplace element-wise _no_bias operation A <- (A-M) .* c
		void evSubMtxMulC_ip_nb(realmtx_t& A, const realmtx_t& M, const real_t c)noexcept {
			if (M.numel() < Thresholds_t::evSubMtxMulC_ip_nb) {
				get_self().evSubMtxMulC_ip_nb_st(A, M, c);
			} else get_self().evSubMtxMulC_ip_nb_mt(A, M, c);
		}
		void evSubMtxMulC_ip_nb_st(realmtx_t& A, const realmtx_t& M, const real_t c, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
			NNTL_ASSERT(!M.empty() && A.size_no_bias() == M.size());
			NNTL_ASSERT(c);
			NNTL_ASSERT(!pER || pER->elmEnd <= A.numel_no_bias());
			get_self()._ievSubMtxMulC_ip_nb_st(A.data(), M.data(), c, pER ? *pER : elms_range(M));
		}

		static void _ievSubMtxMulC_ip_nb_st(real_t*const /*__restrict*/ pA, const real_t*const /*__restrict*/ pM, const real_t c, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pM && c && er.totalElements() > 0);
			/*const auto elmend = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < elmend; ++i) {
				const auto v = pA[i];
				pA[i] = (v - pM[i])*c;//c1200 not vectorized
			}*/
			real_t* /*__restrict*/ pA1 = pA + er.elmBegin;
			real_t*const /*__restrict*/ pAE = pA + er.elmEnd;
			const real_t* /*__restrict*/ pM1 = pM + er.elmBegin;
			while (pA1 != pAE) {
				const auto v = *pA1;
				*pA1++ = (v - *pM1++)*c; //vectorized! and it has nothing to do with __restrict
			}
		}

		void evSubMtxMulC_ip_nb_mt(realmtx_t& A, const realmtx_t& M, const real_t c)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
			NNTL_ASSERT(!M.empty() && A.size_no_bias() == M.size());
			NNTL_ASSERT(c);
			m_threads.run([pA = A.data(), pM = M.data(), c, this](const par_range_t& pr) noexcept{
				get_self()._ievSubMtxMulC_ip_nb_st(pA, pM, c, elms_range(pr));
			}, M.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = A.*B
		void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evMul_ip) {
				get_self().evMul_ip_st(A, B);
			} else get_self().evMul_ip_mt(A, B);
		}
		void evMul_ip_st(realmtx_t& A, const realmtx_t& B, const elms_range*const pER=nullptr)const noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size() == B.size());
			get_self()._ievMul_ip_st(A.data(), B.data(), pER ? *pER : elms_range(A));
		}
		/*static void _ievMul_ip_st(real_t*const ptrA, const real_t*const ptrB, const elms_range& er) noexcept {
			const bool bOdd = er.totalElements() & 1;
			if (bOdd) {
				ptrA[er.elmBegin] *= ptrB[er.elmBegin];
			}
			for (auto i = er.elmBegin + bOdd; i < er.elmEnd; i += 2) {
				ptrA[i] *= ptrB[i];
				const auto j = i + 1;
				ptrA[j] *= ptrB[j];
			}
		}*/
		//__declspec(noalias) static void _ievMul_ip_st(real_t* __restrict ptrA, const real_t* __restrict ptrB, const elms_range& __restrict er) noexcept {
		static void _ievMul_ip_st(real_t* ptrA, const real_t* ptrB, const elms_range& er) noexcept {
			const auto pE = ptrA + er.elmEnd;
			ptrA += er.elmBegin;
			ptrB += er.elmBegin;
			while (ptrA != pE) {
				*ptrA++ *= *ptrB++;
			}
		}
		void evMul_ip_mt(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size() == B.size());
			get_self()._ievMul_ip_mt(A.data(), B.data(), A.numel());
		}
		void _ievMul_ip_mt(real_t*const ptrA, const real_t*const ptrB, const numel_cnt_t dataCnt) noexcept {
			m_threads.run([ptrA, ptrB, this](const par_range_t& r) noexcept{
				get_self()._ievMul_ip_st(ptrA, ptrB, elms_range(r));
			}, dataCnt);
		}

		//inplace elementwise multiplication A(no_bias) = A(no_bias).*B, - A is taken in no_bias mode
		void evMul_ip_Anb(realmtx_t& A, const realmtx_t& B)noexcept {
			if (B.numel() < Thresholds_t::evMul_ip) {
				get_self().evMul_ip_Anb_st(A, B);
			} else get_self().evMul_ip_Anb_mt(A, B);
		}
		void evMul_ip_Anb_st(realmtx_t& A, const realmtx_t& B, const elms_range*const pER = nullptr)const noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size_no_bias() == B.size());
			get_self()._ievMul_ip_st(A.data(), B.data(), pER ? *pER : elms_range(B));
		}
		void evMul_ip_Anb_mt(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size_no_bias() == B.size());
			get_self()._ievMul_ip_mt(A.data(), B.data(), B.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		// vector += constant single threaded only
		template<typename T>
		static void vAddC_ip_st(T*__restrict pA, T*__restrict const pAE, const T c)noexcept {
			NNTL_ASSERT(pA && pAE && pA <= pAE);
			NNTL_ASSERT(!::std::is_floating_point<T>::value || (!::std::isnan(c) && ::std::isfinite(c)));
			while (pA != pAE) { // #vectorized
				*pA++ += c;
			}
		}
		//////////////////////////////////////////////////////////////////////////
		// Add constant to every matrix element, A += c
		// #supportsBatchInRow
		template<typename T>
		void evAddC_ip(smatrix<T>& A, const T c, const bool bIgnoreBias = false) noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			//#TODO T must be used when branching here
			if (A.numel() < Thresholds_t::evAddC_ip) {
				get_self().evAddC_ip_st(A, c, bIgnoreBias);
			} else get_self().evAddC_ip_mt(A, c, bIgnoreBias);
		}
		template<typename T>
		void evAddC_ip_nb(smatrix<T>& A, const T c) noexcept {
			get_self().evAddC_ip(A, c, true);
		}
		template<typename T>
		static void evAddC_ip_st(smatrix<T>& A, const T c, const bool bIgnoreBias)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			self_t::_ievAddC_ip_st(A, c, bIgnoreBias, par_range_t(A._universal_range_BiR_aware(bIgnoreBias)));
		}
		template<typename T>
		void evAddC_ip_mt(smatrix<T>& A, const T c, const bool bIgnoreBias) noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			get_self().ithreads().run([&A, c, bIgnoreBias](const par_range_t& pr)noexcept {
				self_t::_ievAddC_ip_st(A, c, bIgnoreBias, pr);
			}, A._universal_range_BiR_aware(bIgnoreBias));
		}
		template<typename T>
		static void _ievAddC_ip_st(smatrix<T>& A, const T c, const bool bIgnoreBias, const par_range_t& pr)noexcept {
			NNTL_ASSERT(!A.empty() && !::std::isnan(c) && ::std::isfinite(c));
			NNTL_ASSERT(bIgnoreBias || !A.emulatesBiases());//we generally never want to touch bias column
			if (bIgnoreBias & A.bBatchInRow() & A.emulatesBiases()) { //we had to exclude bias row from adjustments here
				NNTL_ASSERT(pr.end() <= A.cols());
				const ptrdiff_t rowsNb = A.rows() - 1;
				auto pA = A.colDataAsVec(static_cast<vec_len_t>(pr.offset()));
				const auto pAE = A.colDataAsVec(static_cast<vec_len_t>(pr.end()));
				while (pA < pAE) {
					const auto tpA = pA;
					pA += rowsNb;
					self_t::vAddC_ip_st(tpA, pA, c);
					++pA;
				}
			} else {
				const auto pA = A.data();
				self_t::vAddC_ip_st(pA + pr.offset(), pA + pr.end(), c);
			}
		}

		/*template<typename T>
		static void _ievAddC_ip_st_nb_BiR(smatrix<T>& A, const T c, const s_vec_range& colR)noexcept {
			NNTL_ASSERT(A.bBatchInRow() && A.emulatesBiases());
			const ptrdiff_t rowsNb = A.rows() - 1;
			auto pA = A.colDataAsVec(colR.elmBegin);
			const auto pAE = A.colDataAsVec(colR.elmEnd);
			while (pA < pAE) {
				const auto tpA = pA;
				pA += rowsNb;
				self_t::vAddC_ip_st(tpA, pA, c);
				++pA;
			}
		}*/

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition *pA = *pA + *pB
		void vAdd_ip(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt)noexcept {
			NNTL_ASSERT(pA && pB && dataCnt);
			if (dataCnt < Thresholds_t::evAdd_ip) {
				get_self().vAdd_ip_st(pA, pB, dataCnt);
			} else get_self().vAdd_ip_mt(pA, pB, dataCnt);
		}
		static void _ivAdd_ip_st(real_t* pA, const real_t* pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB);
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			pB += er.elmBegin;
			while (pA != pAE) {
				*pA++ += *pB++;
			}
			//for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) pA[i] += pB[i];
		}
		void vAdd_ip_st(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt, const elms_range*const pER=nullptr)noexcept {
			NNTL_ASSERT(pA && pB && (pER || dataCnt));
			_ivAdd_ip_st(pA, pB, pER ? *pER : elms_range(0, dataCnt));
		}
		void vAdd_ip_mt(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt)noexcept {
			NNTL_ASSERT(pA && pB && dataCnt);
			m_threads.run([pA, pB](const par_range_t& pr) noexcept{
				_ivAdd_ip_st(pA, pB, elms_range(pr));
			}, dataCnt);
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition A = A+B
		void evAdd_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evAdd_ip) {
				get_self().evAdd_ip_st(A, B);
			} else get_self().evAdd_ip_mt(A, B);
		}
		static void evAdd_ip_st(realmtx_t& A, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty());
			_ivAdd_ip_st(A.data(), B.data(), pER ? *pER : elms_range(A));
		}
		void evAdd_ip_mt(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty());
			m_threads.run([pA=A.data(), pB=B.data(), this](const par_range_t& pr) noexcept{
				get_self()._ivAdd_ip_st(pA, pB, elms_range(pr));
			}, A.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition A(nobias) = A(nobias)+B
		void evAdd_Anb_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (B.numel() < Thresholds_t::evAdd_ip) {
				get_self().evAdd_Anb_ip_st(A, B);
			} else get_self().evAdd_Anb_ip_mt(A, B);
		}
		static void evAdd_Anb_ip_st(realmtx_t& A, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(A.size_no_bias() == B.size() && !A.empty() && !B.empty());
			_ivAdd_ip_st(A.data(), B.data(), pER ? *pER : elms_range(B));
		}
		void evAdd_Anb_ip_mt(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size_no_bias() == B.size() && !A.empty() && !B.empty());
			m_threads.run([pA = A.data(), pB = B.data(), this](const par_range_t& pr) noexcept{
				get_self()._ivAdd_ip_st(pA, pB, elms_range(pr));
			}, B.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition of scaled vector: A = A + c*B;
		void evAddScaled_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evAddScaled_ip) {
				get_self().evAddScaled_ip_st(A, c, B);
			} else get_self().evAddScaled_ip_mt(A, c, B);
		}
		void evAddScaled_ip_st(realmtx_t& A, const real_t c, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			get_self()._ievAddScaled_ip_st(A.data(), c, B.data(), pER ? *pER : elms_range(A));
		}
		static void _ievAddScaled_ip_st(real_t* pA, const real_t c, const real_t* pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB && c);
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			pB += er.elmBegin;
			while (pA != pAE) {//vectorizes
				*pA++ += c*(*pB++);
			}
			//for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) pA[i] += c*pB[i]; //doesn't vectorize, code 500/1200
		}
		void evAddScaled_ip_mt(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			m_threads.run([pA=A.data(), pB=B.data(), c, this](const par_range_t& r) noexcept{
				get_self()._ievAddScaled_ip_st(pA, c, pB, elms_range(r));
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition of scaled vector: A = { A + c*B | A!=0, 0|A==0}
		void evNZAddScaled_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evNZAddScaled_ip) {
				get_self().evNZAddScaled_ip_st(A, c, B);
			} else get_self().evNZAddScaled_ip_mt(A, c, B);
		}
		void evNZAddScaled_ip_st(realmtx_t& A, const real_t c, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			get_self()._ievNZAddScaled_ip_st(A.data(), c, B.data(), pER ? *pER : elms_range(A));
		}
		static void _ievNZAddScaled_ip_st(real_t* pA, const real_t c, const real_t* pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB && c);
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			pB += er.elmBegin;
			while (pA != pAE) {//vectorizes
				const real_t a = *pA;
				const real_t b = *pB++;
				*pA++ = !a ? real_t(0) : a + c*b;
			}
			/*const numel_cnt_t ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) {
				pA[i] = (pA[i] == real_t(0.) ? real_t(0) : pA[i] + c*pB[i]);
			}*/
		}
		void evNZAddScaled_ip_mt(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			m_threads.run([pA = A.data(), pB = B.data(), c, this](const par_range_t& r) noexcept{
				get_self()._ievNZAddScaled_ip_st(pA, c, pB, elms_range(r));
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition of scaled signum: A = A + c*sign(B);
		//(L1 regularization, dLdW update step)
		void evAddScaledSign_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evAddScaledSign_ip) {
				get_self().evAddScaledSign_ip_st(A, c, B);
			} else get_self().evAddScaledSign_ip_mt(A, c, B);
		}
		void evAddScaledSign_ip_st(realmtx_t& A, const real_t c, const realmtx_t& B, const elms_range*const pER=nullptr)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			get_self()._ievAddScaledSign_ip_st(A.data(),c,B.data(), pER ? *pER : elms_range(A));
		}
		static void _ievAddScaledSign_ip_st(real_t* pA, const real_t c, const real_t* pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB && c != real_t(0.0));
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			pB += er.elmBegin;
			while (pA != pAE) {
				*pA++ += c*math::sign(*pB++);
			}

			//for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) pA[i] += c*math::sign(pB[i]);
		}
		void evAddScaledSign_ip_mt(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			m_threads.run([pA=A.data(), pB=B.data(), c, this](const par_range_t& r) noexcept{
				get_self()._ievAddScaledSign_ip_st(pA, c, pB, elms_range(r));
			}, A.numel());
		}

		void evNZAddScaledSign_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evNZAddScaledSign_ip) {
				get_self().evNZAddScaledSign_ip_st(A, c, B);
			} else get_self().evNZAddScaledSign_ip_mt(A, c, B);
		}
		void evNZAddScaledSign_ip_st(realmtx_t& A, const real_t c, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			get_self()._ievNZAddScaledSign_ip_st(A.data(), c, B.data(), pER ? *pER : elms_range(A));
		}
		static void _ievNZAddScaledSign_ip_st(real_t*const pA, const real_t c, const real_t*const pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB && c != real_t(0.0));
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) {
				pA[i] = (pA[i] == real_t(0.) ? real_t(0) : pA[i] + c*math::sign(pB[i]));
				//pA[i] += c*math::sign(pB[i]);
			}
		}
		void evNZAddScaledSign_ip_mt(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			m_threads.run([pA = A.data(), pB = B.data(), c, this](const par_range_t& r) noexcept{
				get_self()._ievNZAddScaledSign_ip_st(pA, c, pB, elms_range(r));
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//signum: A = sign(B);
		void evSign(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evSign) {
				get_self().evSign_st(A, B);
			} else get_self().evSign_mt(A, B);
		}
		void evSign_st(realmtx_t& A, const realmtx_t& B, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty());
			get_self()._ievSign_st(A.data(), B.data(), pER ? *pER : elms_range(A));
		}
		static void _ievSign_st(real_t*const pA, const real_t*const pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB);
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) pA[i] = math::sign(pB[i]);
		}
		void evSign_mt(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty());
			m_threads.run([pA = A.data(), pB = B.data(), this](const par_range_t& r) noexcept{
				get_self()._ievSign_st(pA, pB, elms_range(r));
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// For a (strict) binary matrix gate (i.e. when g \in gate is real_t \in {0,1}) makes a complement matrix gcompl = (1-gate)
		void evOneCompl(const realmtx_t& gate, realmtx_t& gcompl)noexcept {
			if (gate.numel_no_bias() < Thresholds_t::evOneCompl) {
				get_self().evOneCompl_st(gate, gcompl);
			} else get_self().evOneCompl_mt(gate, gcompl);
		}
		void evOneCompl_st(const realmtx_t& gate, realmtx_t& gcompl, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(gate.size_no_bias() == gcompl.size_no_bias() && !gate.empty() && !gcompl.empty());
			get_self()._ievOneCompl_st(gate.data(), gcompl.data(), pER ? *pER : elms_range(gate, tag_noBias()));
		}
		/*template<typename T>
		static void _ievOneCompl_st(const T*const _pG, T*const _pGc, const elms_range& er)noexcept {
			NNTL_ASSERT(_pG && _pGc);
			typedef real_t_limits<T>::similar_FWI_t similar_FWI_t;

			const auto _zero = similar_FWI_pos_zero<T>();
			const auto _one = similar_FWI_one<T>();

			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			const similar_FWI_t* pG = reinterpret_cast<const similar_FWI_t*>(_pG) + er.elmBegin;
			similar_FWI_t* pGc = reinterpret_cast<similar_FWI_t*>(_pGc) + er.elmBegin;
			const auto pGE = pG + er.totalElements();

			while (pG != pGE) {
				const auto g = *pG++;
				NNTL_ASSERT(g == _one || g == _zero);
				const auto gc = _one - g;
				NNTL_ASSERT(gc == _one || gc == _zero);
				*pGc++ = gc;
			}
		}*/
		//fuck it
		template<typename T>
		static void _ievOneCompl_st(const T* pG, T* pGc, const elms_range& er)noexcept {
			NNTL_ASSERT(pG && pGc);

			pG += er.elmBegin;
			pGc += er.elmBegin;
			const auto pGE = pG + er.totalElements();

			while (pG != pGE) {
				const auto g = *pG++;
				NNTL_ASSERT(g == T(1) || g == T(0));
				const auto gc = T(1) - g;
				NNTL_ASSERT(gc == T(1) || gc == T(0));
				*pGc++ = gc;
			}
		}

		void evOneCompl_mt(const realmtx_t& gate, realmtx_t& gcompl)noexcept {
			NNTL_ASSERT(gate.size_no_bias() == gcompl.size_no_bias() && !gate.empty() && !gcompl.empty());
			m_threads.run([pG = gate.data(), pGc = gcompl.data(), this](const par_range_t& r) noexcept{
				get_self()._ievOneCompl_st(pG, pGc, elms_range(r));
			}, gate.numel_no_bias());
		}

		////////////////////////////////////////////////////////////////////////// 
		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise subtraction A = A-B
		void evSub_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evSub_ip) {
				get_self().evSub_ip_st_naive(A, B);
			} else get_self().evSub_ip_mt_naive(A, B);
		}
		void evSub_ip_st_naive(realmtx_t& A, const realmtx_t& B, const elms_range* pER = nullptr)const noexcept {
			NNTL_ASSERT(A.size() == B.size());
			get_self()._ievSub_ip_st(A.data(), B.data(), pER ? *pER : elms_range(A));
		}
// 		static void _ievSub_ip_st(real_t*const pA, const real_t*const pB, const elms_range& er)noexcept {
// 			NNTL_ASSERT(pA && pB);
// 			const numel_cnt_t _mi = er.elmEnd;
// 			for (numel_cnt_t i = er.elmBegin; i < _mi/*er.elmEnd*/; ++i) pA[i] -= pB[i];
// 		} //doesn't vectorize, err 1200
		static void _ievSub_ip_st(real_t* pA, const real_t* pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB);
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			pB += er.elmBegin;
			while (pA != pAE) {//vectorizeable
				*pA++ -= *pB++;
			}
		}
		void evSub_ip_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());
			m_threads.run([pA= A.data(), pB=B.data(), this](const par_range_t& r) noexcept{
				get_self()._ievSub_ip_st(pA, pB, elms_range(r));
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//elementwise subtraction C = A-B
		void evSub(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			if (A.numel() < Thresholds_t::evSub) {
				get_self().evSub_st_naive(A, B, C);
			} else get_self().evSub_mt_naive(A, B, C);
		}
		static void evSub_st_naive(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			NNTL_ASSERT(A.size() == B.size() && A.size() == C.size());
			const auto dataCnt = A.numel();
			const auto pA = A.data(), pB = B.data();
			const auto pC = C.data();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pC[i] = pA[i] - pB[i];
		}
		void evSub_mt_naive(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			NNTL_ASSERT(A.size() == B.size() && A.size() == C.size());
			const auto pA = A.data(), pB = B.data();
			const auto pC = C.data();
			m_threads.run([pA, pB, pC](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) pC[i] = pA[i] - pB[i];
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise scaling and subtracting: vW = momentum.*vW, W = W-vW;
		//(it's pre-fprop step of Nesterov Momentum method)
		void evMulC_ip_Sub_ip(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
			if (vW.numel() < Thresholds_t::evMulC_ip_Sub_ip) {
				get_self().evMulC_ip_Sub_ip_st(vW, momentum, W);
			} else get_self().evMulC_ip_Sub_ip_mt(vW, momentum, W);
		}
		static void evMulC_ip_Sub_ip_st(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
			NNTL_ASSERT(vW.size() == W.size() && !vW.empty() && !W.empty()); //NNTL_ASSERT(momentum > real_t(0.0) && momentum < real_t(1.0));
			auto pV = vW.data();
			const auto pVE = pV + vW.numel();
			auto pW = W.data();
			while (pV != pVE) {
				const auto v = *pV * momentum;
				*pV++ = v;
				*pW++ -= v;
			}
		}
		void evMulC_ip_Sub_ip_mt(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
			NNTL_ASSERT(vW.size() == W.size() && !vW.empty() && !W.empty()); //NNTL_ASSERT(momentum > real_t(0.0) && momentum < real_t(1.0));

			auto pVf = vW.data();
			auto pWf = W.data();
			m_threads.run([pVf, pWf, momentum](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				auto pV = pVf + ofs;
				const auto pVE = pV + r.cnt();
				auto pW = pWf + ofs;
				while (pV != pVE) {
					const auto v = *pV * momentum;
					*pV++ = v;
					*pW++ -= v;
				}
			}, vW.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//elementwise squaring dest = src.^2;
		void evSquare(realmtx_t& dest, const realmtx_t& src)noexcept {
			if (src.numel() < Thresholds_t::evSquare) {
				get_self().evSquare_st(dest, src);
			} else get_self().evSquare_mt(dest, src);
		}
		static void evSquare_st(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.data();
			auto pD = dest.data();
			const auto dataCnt = src.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto s = pS[i];
				pD[i] = s*s;
			}
		}
		void evSquare_mt(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.data();
			auto pD = dest.data();
			m_threads.run([pS, pD](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto s = pS[i];
					pD[i] = s*s;
				}
			}, src.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//finds sum of squares of elements (squared L2 norm): return sum( A.^2 )
		// see SMath::ewSumSquares()
		/*real_t vSumSquares(const realmtx_t& A)noexcept {
			if (A.numel() < Thresholds_t::vSumSquares) {
				return get_self().vSumSquares_st(A);
			} else return get_self().vSumSquares_mt(A);
		}
		static real_t vSumSquares_st(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			real_t ret(0.0), C(0.), Y, T;
			auto p = A.data();
			const auto pE = p + A.numel();
			while (p != pE) {
				const auto v = *p++;
				Y = v*v - C;
				T = ret + Y;
				C = T - ret - Y;
				ret = T;
			}
			return ret;
		}
		real_t vSumSquares_mt(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			const auto pA = A.data();
			return m_threads.reduce([pA](const par_range_t& r)noexcept->real_t {
				real_t ret(0.0), C(0.), Y, T;
				auto p = pA + r.offset();
				const auto pE = p + r.cnt();
				while (p != pE) {
					const auto v = *p++;
					Y = v*v - C;
					T = ret + Y;
					C = T - ret - Y;
					ret = T;
				}
				return ret;
			}, _vec_sum<true, real_t>, A.numel());
		}*/

		//////////////////////////////////////////////////////////////////////////
		//finding elementwise absolute values dest = abs(src);
		void evAbs(realmtx_t& dest, const realmtx_t& src)noexcept {
			if (src.numel() < Thresholds_t::evAbs) {
				get_self().evAbs_st(dest, src);
			} else get_self().evAbs_mt(dest, src);
		}
		static void evAbs_st(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.data();
			auto pD = dest.data();
			const auto dataCnt = src.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i)  pD[i] = ::std::abs(pS[i]);
		}
		void evAbs_mt(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.data();
			auto pD = dest.data();
			m_threads.run([pS, pD](const par_range_t& r) noexcept{
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i)  pD[i] = ::std::abs(pS[i]);
			}, src.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//finds sum of abs values (L1 norm): return sum( abs(A) );
		real_t vSumAbs(const realmtx_t& A)noexcept {
			if (A.numel() < Thresholds_t::vSumAbs) {
				return get_self().vSumAbs_st(A);
			} else return get_self().vSumAbs_mt(A);
		}
		static real_t vSumAbs_st(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			real_t ret(0.0);
			auto p = A.data();
			const auto pE = p + A.numel();
			while (p != pE) ret += ::std::abs(*p++);
			return ret;
		}
		real_t vSumAbs_mt(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			const auto pA = A.data();
			return m_threads.reduce([pA](const par_range_t& r)noexcept->reduce_data_t {
				real_t ret(0.0);
				auto p = pA + r.offset();
				const auto pE = p + r.cnt();
				while (p != pE) ret += ::std::abs(*p++);
				return converter_reduce_data_tpl<real_t>::to(ret);
			}, _reduce_vec_sum<real_t>, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//C = A * B, - matrix multiplication
		static void mMulAB_C(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			NNTL_ASSERT(acols == B.rows() && A.rows() == C.rows() && B.cols() == C.cols());

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			A._breakWhenDenormal();
			B._breakWhenDenormal();
#endif

			b_BLAS_t::gemm(false, false, A.rows(), C.cols(), acols, real_t(1.0), A.data(), A.rows(), B.data(), B.rows(),
				real_t(0.0), C.data(), C.rows());
#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			C._breakWhenDenormal();
#endif
		}
		//////////////////////////////////////////////////////////////////////////
		//matrix multiplication C(no bias) = A * B` (B transposed). C could have emulated biases (they will be left untouched)
		static void mMulABt_Cnb(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto ccols = C.cols_no_bias();
			NNTL_ASSERT(A.cols() == B.cols() && A.rows() == C.rows() && B.rows() == ccols);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			A._breakWhenDenormal();
			B._breakWhenDenormal();
#endif

			b_BLAS_t::gemm(false, true, A.rows(), ccols, A.cols(), real_t(1.0), A.data(), A.rows(), B.data(), ccols,
				real_t(0.0), C.data(), C.rows());

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			C._breakWhenDenormal();
#endif
		}
		//////////////////////////////////////////////////////////////////////////
		//C = a*(A` * B) - matrix multiplication of transposed A times B with result normalization
		static void mScaledMulAtB_C(const real_t alpha, const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			//A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			const auto arows = A.rows();
			NNTL_ASSERT(arows == B.rows() && acols == C.rows() && B.cols() == C.cols());

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			enable_denormals();
			if (::std::fpclassify(alpha) == FP_SUBNORMAL) {
				__debugbreak();
			}
			A._breakWhenDenormal();
			B._breakWhenDenormal();
			global_denormalized_floats_mode();
#endif

			b_BLAS_t::gemm(true, false, acols, B.cols(), arows, alpha, A.data(), arows, B.data(), arows,
				real_t(0.0), C.data(), acols);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			C._breakWhenDenormal();
#endif
		}

		//////////////////////////////////////////////////////////////////////////
		// single entry for calculating preactivation values for fullyconnected layer based on activations of prev.layer and weight
		// matrix. prevAct must contain biases, act doesn't have to contain biases (just don't touched if any).
		// weight matrix must be sized [n,p] where n==act.sample_size() and p==prevAct.sample_size()+1 (+1 for bias weight)
		// #supportsBatchInRow for prevAct&act only. weights MUST have the standard bBatchInColumn() layout.
		template<typename T>
		static void mMul_prevAct_weights_2_act(const smatrix<T>& prevAct, const smatrix<T>& weights, smatrix<T>& act)noexcept {
			//layout checks
			NNTL_ASSERT(prevAct.emulatesBiases() && !weights.emulatesBiases());
			NNTL_ASSERT(weights.bBatchInColumn());
			NNTL_ASSERT(prevAct.sample_size() + 1 == weights.cols() && act.sample_size() == weights.rows());
			NNTL_ASSERT(prevAct.batch_size() == act.batch_size());
			//normal checks
			prevAct.assert_storage_does_not_intersect(weights);
			prevAct.assert_storage_does_not_intersect(act);
			weights.assert_storage_does_not_intersect(act);

		#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			enable_denormals();
			prevAct._breakWhenDenormal();
			weights._breakWhenDenormal();
			global_denormalized_floats_mode();
		#endif

			const T* p1, *p2;
			vec_len_t d1, d2;
			bool bT1, bT2;
			const auto bABiR = act.bBatchInRow();

			if (bABiR) {
// 				if (prevAct.bBatchInRow()) {
// 					// AT[n,m] = Wc[n,p] * PT[p,m]
// 
// 					b_BLAS_t::gemm(false, false, act.rows_no_bias(), act.cols(), weights.cols(), real_t(1.), weights.data(), weights.ldim()
// 						, prevAct.data(), prevAct.ldim(), real_t(0), act.data(), act.ldim());
// 				} else {
// 					// AT[n,m] = Wc[n,p] * (Pc[m,p])`
// 
// 					b_BLAS_t::gemm(false, true, act.rows_no_bias(), act.cols(), weights.cols(), real_t(1.), weights.data(), weights.ldim()
// 						, prevAct.data(), prevAct.ldim(), real_t(0), act.data(), act.ldim());
// 				}
				bT1 = false;
				bT2 = !prevAct.bBatchInRow();
				d1 = weights.ldimAsVecLen();
				d2 = prevAct.ldimAsVecLen();
				p1 = weights.data();
				p2 = prevAct.data();
			} else {
				//will call BLAS directly here to reserve for future tricks
				// note that .sample_size() ignores biases if any, however, here we should deal with prevAct matrix here that ALWAYS have biases
				/* the code is fine, just do it in one call without branching
				if (prevAct.bBatchInRow()) {
				// PT[p,m] & Wc[n,p],   4.   Ac[m,n] = PT'*Wc'
				b_BLAS_t::gemm(true, true, act.rows(), act.sample_size(), prevAct.sample_size()+1, real_t(1.), prevAct.data(), prevAct.ldim()
				, weights.data(), weights.ldim(), real_t(0), act.data(), act.ldim());
				} else {
				// Pc[m,p] & Wc[n,p],   2.   Ac[m,n] = Pc*Wc'
				b_BLAS_t::gemm(false, true, act.rows(), act.sample_size(), prevAct.sample_size()+1, real_t(1.), prevAct.data(), prevAct.ldim()
				, weights.data(), weights.ldim(), real_t(0), act.data(), act.ldim());
				}*/

				// prevAct.bBatchInRow():  PT[p,m] & Wc[n,p],   4.   Ac[m,n] = PT'*Wc'
				//!prevAct.bBatchInRow():  Pc[m,p] & Wc[n,p],   2.   Ac[m,n] = Pc*Wc'
// 				b_BLAS_t::gemm(prevAct.bBatchInRow(), true, act.rows(), act.cols_no_bias(), weights.cols() //prevAct.sample_size() + 1
// 					, real_t(1.), prevAct.data(), prevAct.ldim()
// 					, weights.data(), weights.ldim(), real_t(0), act.data(), act.ldim());

				bT1 = prevAct.bBatchInRow();
				bT2 = true;
				d1 = prevAct.ldimAsVecLen();
				d2 = weights.ldimAsVecLen();
				p1 = prevAct.data();
				p2 = weights.data();
			}

			b_BLAS_t::gemm(bT1, bT2, act.rows(bABiR), act.cols(!bABiR), weights.cols(), real_t(1.), p1, d1, p2, d2
				, real_t(0), act.data(), act.ldimAsVecLen());

		#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			act._breakWhenDenormal();
		#endif
		}

		//////////////////////////////////////////////////////////////////////////
		// single entry for calculating dL/dAPrev values for fullyconnected layer based on current dL/dZ and weight
		// matrix. There must be no biases in matrices.
		// weight matrix must be sized [n,p+1] where n==dLdZ.sample_size() and p==dLdAPrev.sample_size(),
		//		the last column of weight matrix must contain weights of bias units, they will be hidden during calculation
		// #supportsBatchInRow for dLdAPrev only. dLdZ & weights MUST have the standard bBatchInColumn() layout.
		// There's no problem to support bBatchInRow() for dLdZ matrix, however it won't be used anyway, b/c of issue with
		// making dLdA from activations in bprop()
		template<typename T>
		static void mMul_dLdZ_weights_2_dLdAPrev(const smatrix<T>& dLdZ, smatrix_deform<T>& weights, smatrix<T>& dLdAPrev)noexcept {
			//layout checks
			NNTL_ASSERT(weights.bBatchInColumn() && dLdZ.bBatchInColumn());
			NNTL_ASSERT(dLdAPrev.sample_size() + 1 == weights.cols() && dLdZ.sample_size() == weights.rows());
			NNTL_ASSERT(dLdAPrev.batch_size() == dLdZ.batch_size());
			NNTL_ASSERT(!dLdZ.emulatesBiases() && !weights.emulatesBiases() && !dLdAPrev.emulatesBiases());
			//normal checks
			dLdAPrev.assert_storage_does_not_intersect(weights);
			dLdAPrev.assert_storage_does_not_intersect(dLdZ);
			weights.assert_storage_does_not_intersect(dLdZ);

		#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			enable_denormals();
			dLdZ._breakWhenDenormal();
			weights._breakWhenDenormal();
			global_denormalized_floats_mode();
		#endif

			//hiding bias weights
			weights.hide_last_col();

			//will call BLAS directly here to reserve for future tricks
			const T* p1, *p2;
			vec_len_t d1, d2;
			const bool dLdAPrev_bir = dLdAPrev.bBatchInRow();
			//const bool bTr1 = dLdAPrev_bir, bTr2 = dLdAPrev_bir;
			if (dLdAPrev_bir) {
				// PT[p,m] = (W[n,p])' * (Dc[m,n])'
				//bTr1 = bTr2 = true;
				p1 = weights.data(); d1 = weights.ldimAsVecLen();
				p2 = dLdZ.data(); d2 = dLdZ.ldimAsVecLen();
				// b_BLAS_t::gemm(true, true, dLdAPrev.rows(), dLdAPrev.cols(), weights.rows(), real_t(1.), weights.data(), weights.ldim()
// 					, dLdZ.data(), dLdZ.ldim(), real_t(0), dLdAPrev.data(), dLdAPrev.ldim());
			} else {
				// Pc[m,p] = Dc[m,n] * W[n,p]
				p2 = weights.data(); d2 = weights.ldimAsVecLen();
				p1 = dLdZ.data(); d1 = dLdZ.ldimAsVecLen();
				//bTr1 = bTr2 = false;
// 				b_BLAS_t::gemm(false, false, dLdAPrev.rows(), dLdAPrev.cols(), dLdZ.sample_size(), real_t(1.), dLdZ.data(), dLdZ.ldim()
// 					, weights.data(), weights.ldim(), real_t(0), dLdAPrev.data(), dLdAPrev.ldim());
			}

			b_BLAS_t::gemm(dLdAPrev_bir, dLdAPrev_bir, dLdAPrev.rows(), dLdAPrev.cols(), dLdZ.sample_size(), real_t(1.), p1, d1
				, p2, d2, real_t(0), dLdAPrev.data(), dLdAPrev.ldimAsVecLen());

			weights.restore_last_col();//restore bias weights back

		#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			dLdAPrev._breakWhenDenormal();
		#endif
		}

		//////////////////////////////////////////////////////////////////////////
		// single entry for calculating scaled dL/dW values for fullyconnected layer based on activations of prev.layer and dL/dZ
		// matrix. prevAct must contain biases (but they are ignored), dLdZ and weights must not have biases.
		// weight matrix must be sized [n,p] where n==dLdZ.sample_size() and p==prevAct.sample_size()+1 (+1 for bias weight)
		// #supportsBatchInRow for prevAct only. act & weights MUST have the standard bBatchInColumn() layout.
		// Probably there's no problem to support bBatchInRow() for act matrix with a decreased rows count and proper ldim() trick, however
		// not 100% sure and testing is required. Anyway now it won't be used anyway, b/c of issue with
		// making dLdA from activations in bprop()
		template<typename T>
		static void mMulScaled_dLdZ_prevAct_2_dLdW(const T Sc, const smatrix<T>& dLdZ, const smatrix<T>& prevAct, smatrix<T>& dLdW)noexcept {
			//layout checks
			NNTL_ASSERT(dLdW.bBatchInColumn() && dLdZ.bBatchInColumn());
			NNTL_ASSERT(prevAct.sample_size() + 1 == dLdW.cols() && dLdZ.sample_size() == dLdW.rows());
			NNTL_ASSERT(prevAct.batch_size() == dLdZ.batch_size());
			//normal checks
			prevAct.assert_storage_does_not_intersect(dLdW);
			prevAct.assert_storage_does_not_intersect(dLdZ);
			dLdW.assert_storage_does_not_intersect(dLdZ);

		#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			enable_denormals();
			prevAct._breakWhenDenormal();
			dLdZ._breakWhenDenormal();
			if (::std::fpclassify(Sc) == FP_SUBNORMAL) 
				__debugbreak();
			global_denormalized_floats_mode();
		#endif

			//will call BLAS directly here to reserve for future tricks
			//the code is fine, just do it in one call without branching
			/*if (prevAct.bBatchInRow()) {
				// dW [n,p] = (dZ[m,n])' * (PT[p,m])'
				b_BLAS_t::gemm(true, true, dLdW.rows(), dLdW.cols(), dLdZ.batch_size(), Sc, dLdZ.data(), dLdZ.ldim()
					, prevAct.data(), prevAct.ldim(), real_t(0), dLdW.data(), dLdW.ldim());
			} else {
				// dW [n,p] = (dZ[m,n])' * Pc[m,p]
				b_BLAS_t::gemm(true, false, dLdW.rows(), dLdW.cols(), dLdZ.batch_size(), Sc, dLdZ.data(), dLdZ.ldim()
					, prevAct.data(), prevAct.ldim(), real_t(0), dLdW.data(), dLdW.ldim());
			}*/

			b_BLAS_t::gemm(true, prevAct.bBatchInRow(), dLdW.rows(), dLdW.cols(), dLdZ.batch_size(), Sc, dLdZ.data(), dLdZ.ldimAsVecLen()
				, prevAct.data(), prevAct.ldimAsVecLen(), real_t(0), dLdW.data(), dLdW.ldimAsVecLen());

		#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			dLdW._breakWhenDenormal();
		#endif
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// Computes a symmetrical matrix C = 1/ARowsCnt  A' * A.
		// If the columns of A are zero meaned, the resulting matrix is the actual covariance matrix for columns of A
		// Actual C content stored only in the upper or lower triangular part of C
		// Generally, it seems that for tall matrices (rows>cols) upper triangular matrix work a bit faster
		static void mColumnsCov(const realmtx_t& A, realmtx_t& C, const bool bCLowerTriangl)noexcept {
			NNTL_ASSERT(A.cols() > 1);
			A.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			const auto arows = A.rows();
			NNTL_ASSERT(C.rows() == acols && C.cols() == acols);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			A._breakWhenDenormal();
#endif

			b_BLAS_t::syrk(bCLowerTriangl, true, acols, arows, real_t(1.) / real_t(arows), A.data(), arows, real_t(0.), C.data(), acols);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			C._breakWhenDenormal();
#endif
		}
		// it might be helpful to have a templated version, because we don't expect we'll have a need to switch triangles in a run-time
		template<bool bCLowerTriangl>
		static void mColumnsCov(const realmtx_t& A, realmtx_t& C)noexcept {
			NNTL_ASSERT(A.cols() > 1);
			A.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			const auto arows = A.rows();
			NNTL_ASSERT(C.rows() == acols && C.cols() == acols);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			A._breakWhenDenormal();
#endif

			b_BLAS_t::syrk(bCLowerTriangl, true, acols, arows, real_t(1.) / real_t(arows), A.data(), arows, real_t(0.), C.data(), acols);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			C._breakWhenDenormal();
#endif
		}

		//////////////////////////////////////////////////////////////////////////
		//Elements of SVD (singular value decomposition)
		// mSVD_Orthogonalize_ss(A) performs SVD of m*n matrix A and returns in A same sized corresponding orthogonal matrix of singular vectors
		//		returns true if SVD was successful
		//		Restrictions: MUST NOT use the math object's local storage (the function is intended to be used during
		//			the weight initialization phase when the local storage is generally not initialized yet)
		//	https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
		bool mSVD_Orthogonalize_ss(realmtx_t& A)noexcept {
			const auto m = A.rows(), n = A.cols();
			const bool bGetU = m >= n;
			const auto minmn = bGetU ? n : m;

			::std::vector<real_t> S(2 * minmn);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			A._breakWhenDenormal();
#endif

			const auto r = b_BLAS_t::gesvd(bGetU ? 'O' : 'N', bGetU ? 'N' : 'O', m, n
				, A.data(), m, &S[0], static_cast<real_t*>(nullptr), m, static_cast<real_t*>(nullptr), n, &S[minmn]);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			enable_denormals();
			A._breakWhenDenormal();
			for (auto& e : S) {
				if (::std::fpclassify(e) == FP_SUBNORMAL) {
					__debugbreak();
				}
			}
			global_denormalized_floats_mode();
#endif

			NNTL_ASSERT(0 == r || !"b_BLAS_t::gesvd failed!");
			NNTL_ASSERT(get_self()._mIsOrthogonal(A, bGetU) || !"SVD returned non orthogonal matrix!");
			return 0 == r;
		}



		//////////////////////////////////////////////////////////////////////////
		template<typename base_t> struct _mIsOrthogonal_defEps {};
		template<> struct _mIsOrthogonal_defEps<double> { static constexpr double eps = 1e-11; };
		template<> struct _mIsOrthogonal_defEps<float> { static constexpr float eps = 1e-4f; };
		// This function checks whether A is orthogonal, i.e. A'*A is identity matrix.
		// Not optimized, FOR DEBUG PURPOSES ONLY!
		static bool _mIsOrthogonal(const realmtx_t& A,  bool bFirstTransposed = true, const real_t epsV = _mIsOrthogonal_defEps<real_t>::eps)noexcept {
			NNTL_ASSERT(!A.empty());
			const vec_len_t opArows = bFirstTransposed ? A.cols() : A.rows()
				, opAcols = bFirstTransposed ? A.rows() : A.cols()
				, ldab = bFirstTransposed ? opAcols : opArows;
			realmtx_t ICand(opArows, opArows);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			A._breakWhenDenormal();
#endif

			b_BLAS_t::gemm(bFirstTransposed, !bFirstTransposed, opArows, opArows, opAcols
				, real_t(1.), A.data(), ldab, A.data(), ldab,
				real_t(0.0), ICand.data(), opArows);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			ICand._breakWhenDenormal();
#endif

			bool r = true;
			for (vec_len_t ri = 0; ri < opArows; ++ri) {
				for (vec_len_t ci = 0; ci < opArows; ++ci) {
					if ( ::std::abs(real_t(ri==ci) - ICand.get(ri,ci)) > epsV  ) {
						r = false;
						break;
					}
				}
				if (!r) break;
			}
			return r;
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//not really using it now
// 		template<typename T>
// 		void mTranspose_ip_ignore_bias(smatrix_deform<T>& src_dest)noexcept {
// 			//#TODO
// 		}
		
		// #performance #WARNING current OpenBLAS implementation is slower, than it can be. See TEST(TestPerfDecisions, mTranspose) in test_perf_decisions.cpp
		// and https://github.com/xianyi/OpenBLAS/issues/1243
		// https://github.com/xianyi/OpenBLAS/issues/2532
		//just leaving it for the case... Note that LIBXSMM libxsmm_otrans() may be superior in some cases
		// (see commented out section in test_imath_basic.cpp aroung TEST(TestMathN, mTranspose)
		// or search for string "LIBXSMM vs. OpenBLAS vs. naive results" there
		// #supportsBatchInRow
		/*
		template<typename T, bool ce = ::std::is_floating_point<T>::value>
		static ::std::enable_if_t<ce> mTranspose_ignore_bias_BLAS(const smatrix<T>& src, smatrix<T>& dest)noexcept {
			NNTL_ASSERT(src.bBatchInRow() == !dest.bBatchInRow());
			NNTL_ASSERT(src.cols_no_bias() == dest.rows_no_bias() && src.rows_no_bias() == dest.cols_no_bias());
			NNTL_ASSERT(src.if_biases_test_strict() && dest.if_biases_test_strict());
			b_BLAS_t::omatcopy(true, src.rows_no_bias(), src.cols_no_bias(), T(1.0), src.data(), src.ldim(), dest.data(), dest.ldim());
			NNTL_ASSERT(dest.if_biases_test_strict());
		}
		// DOES NOT SUPPORT non-default m_bBatchesInRows when emulatesBiases()
		template<typename T, bool ce = ::std::is_floating_point<T>::value>
		static ::std::enable_if_t<ce> mTranspose_ip_ignore_bias_BLAS(smatrix_deform<T>& src_dest)noexcept {
			NNTL_ASSERT(!src_dest.emulatesBiases() || src_dest.bBatchInColumn());
			const auto sRows = src_dest.rows_no_bias(), sCols = src_dest.cols_no_bias();
			NNTL_ASSERT(src_dest._isOkToDeform(sCols, sRows, src_dest.emulatesBiases(), src_dest.bBatchInRow()));
			b_BLAS_t::imatcopy(true, sRows, sCols, T(1.0), src_dest.data(), src_dest.ldim(), sCols);

			const bool bBias = src_dest.emulatesBiases();
			src_dest.deform(sCols, sRows + bBias);
			if (bBias) src_dest.set_biases();
		}
		*/

		//////////////////////////////////////////////////////////////////////////
		// matrix transposition. Bias row/column (if any in src or dest) is treated just like any other row/column (also transposed).
		// Destination matrix as always must be properly sized
		// #supportsBatchInRow
		// Note that is doesn't change bBatchInRow property of dest matrix
		template<typename T>
		void mTranspose(const smatrix<T>& src, smatrix<T>& dest, const bool bIgnoreBias = false)noexcept {
			NNTL_ASSERT(src.bBatchInRow() == !dest.bBatchInRow());
			const bool bIsWide = (src.rows() < src.cols());
			//#TODO: that threshold below depends on hw architecture and current use-case (esp. cache cleanliness; and libxsmm could be better)
			//but it's insanity to try to hardcode them all.
			//so just leaving here mine threshold until run-time profiler ready. It should not degrade performance very much, thought
			// testing is required
			const bool bIsBig = (src.numel() >= Thresholds_t::mTransposeTrsh);
			if (bIsWide ^ bIsBig) {
				get_self().mTranspose_seq_write(src, dest, bIgnoreBias);
			} else get_self().mTranspose_seq_read(src, dest, bIgnoreBias);
		}

		// matrix transposition. Biases (if any in src or dest) is ignored. Note that is doesn't change bBatchInRow property of dest matrix
		// dest matrix MUST be properly sized and src.bBatchInRow() == !dest.bBatchInRow() must eval to TRUE on entry
		// #supportsBatchInRow
		template<typename T>
		void mTranspose_ignore_bias(const smatrix<T>& src, smatrix<T>& dest)noexcept {
			NNTL_ASSERT(src.bBatchInRow() == !dest.bBatchInRow());
			const bool bIsWide = (src.rows_no_bias() < src.cols_no_bias());
			//#TODO: that threshold below depends on hw architecture and current use-case (esp. cache cleanliness; and libxsmm could be better)
			//but it's insanity to try to hardcode them all.
			//so just leaving here mine threshold until run-time profiler ready. It should not degrade performance very much, thought
			// testing is required
			const bool bIsBig = (src.numel() >= Thresholds_t::mTransposeTrsh);
			if (bIsWide ^ bIsBig) {
				get_self().mTranspose_seq_write(src, dest, true);
			} else get_self().mTranspose_seq_read(src, dest, true);
		}
		// #supportsBatchInRow
		// dest matrix MUST be properly sized and src.bBatchInRow() == !dest.bBatchInRow() must eval to TRUE on entry
		template<typename T>
		static void mTranspose_seq_read(const smatrix<T>& src, smatrix<T>& dest, const bool bIgnoreBias) noexcept {
			NNTL_ASSERT(src.bBatchInRow() == !dest.bBatchInRow());
			NNTL_ASSERT(src.rows(bIgnoreBias) == dest.cols(bIgnoreBias) && src.cols(bIgnoreBias) == dest.rows(bIgnoreBias));
			NNTL_ASSERT(src.if_biases_test_strict() && dest.if_biases_test_strict());

			const ptrdiff_t sRows = src.rows(bIgnoreBias), ldDest = dest.ldim();
			const ptrdiff_t ibSrcBiasRow = (bIgnoreBias & src.has_bias_row());

			const T* __restrict pSrc = src.data();
			T* __restrict pDest = dest.data();
			const auto pDE = pDest + dest.rows(bIgnoreBias);

			while (pDest != pDE) {
				T* __restrict pD = pDest++;
				const T* __restrict pS = pSrc;
				pSrc += sRows;
				const auto pSE = pSrc;
				pSrc += ibSrcBiasRow;
				while (pS != pSE) {
					*pD = *pS++;
					pD += ldDest;
				}
			}

			NNTL_ASSERT(dest.if_biases_test_strict());
		}
		// #supportsBatchInRow
		// dest matrix MUST be properly sized and src.bBatchInRow() == !dest.bBatchInRow() must eval to TRUE on entry
		template<typename T>
		static void mTranspose_seq_write(const smatrix<T>& src, smatrix<T>& dest, const bool bIgnoreBias) noexcept {
			NNTL_ASSERT(src.bBatchInRow() == !dest.bBatchInRow());
			NNTL_ASSERT(src.rows(bIgnoreBias) == dest.cols(bIgnoreBias) && src.cols(bIgnoreBias) == dest.rows(bIgnoreBias));
			NNTL_ASSERT(src.if_biases_test_strict() && dest.if_biases_test_strict());

			const ptrdiff_t ldSrc = src.ldim(), dRows = dest.rows(bIgnoreBias);
			const ptrdiff_t ibDestBiasRow = (bIgnoreBias & dest.has_bias_row());

			const T* __restrict pSrc = src.data();
			const auto pSE = pSrc + src.rows(bIgnoreBias);
			T*__restrict pDest = dest.data();

			while (pSrc != pSE) {
				const T* __restrict pS = pSrc++;
				T*__restrict pD = pDest;
				pDest += dRows;
				const auto pDE = pDest;
				pDest += ibDestBiasRow;
				while (pD != pDE) {
					*pD++ = *pS;
					pS += ldSrc;
				}
			}

			NNTL_ASSERT(dest.if_biases_test_strict());
		}

		//////////////////////////////////////////////////////////////////////////
		//full matrix transposition in-place
		// MUST #supportsBatchInRow if changed
		// MUST be static (hence prefix s_)
		template<typename T, bool ce = ::std::is_floating_point<T>::value>
		static inline ::std::enable_if_t<ce> s_mTranspose_ip(smatrix_deform<T>& src_dest)noexcept {
			s_mTranspose_ip_BLAS(src_dest);
		}

		//the following 2 functions is a dirty hack, use at own risk only!
		template<typename T, bool ce = ::std::is_floating_point<T>::value, size_t ts = sizeof(T)>
		static inline ::std::enable_if_t<!ce && (sizeof(float) == ts)> s_mTranspose_ip(smatrix_deform<T>& src_dest)noexcept {
			smatrix_deform<float> prxy(reinterpret_cast<float*>(src_dest.data()), src_dest);
			s_mTranspose_ip_BLAS(prxy);
			src_dest.on_transposition();
		}
		template<typename T, bool ce = ::std::is_floating_point<T>::value, size_t ts = sizeof(T)>
		static inline ::std::enable_if_t<!ce && (sizeof(double) == ts)> s_mTranspose_ip(smatrix_deform<T>& src_dest)noexcept {
			smatrix_deform<double> prxy(reinterpret_cast<double*>(src_dest.data()), src_dest);
			s_mTranspose_ip_BLAS(prxy);
			src_dest.on_transposition();
		}

		// #supportsBatchInRow
		template<typename T, bool ce = ::std::is_floating_point<T>::value>
		static inline ::std::enable_if_t<ce> s_mTranspose_ip_BLAS(smatrix_deform<T>& src_dest)noexcept {
			NNTL_ASSERT(src_dest.if_biases_test_strict());
			const auto sr = src_dest.rows(), sc = src_dest.cols();
			if (sr > 1 && sc > 1) {
				b_BLAS_t::imatcopy(true, sr, sc, T(1.0), src_dest.data(), src_dest.ldimAsVecLen(), sc);
			}//else no need to do anything
			src_dest.on_transposition();
			NNTL_ASSERT(src_dest.if_biases_test_strict());
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//#TODO refactor other suitable dL/dZ computation types to use this function
		template<typename WlT>
		void dLoss_dZ(const smatrix<typename WlT::real_t>& data_y, smatrix<typename WlT::real_t>& act_dLdZ)noexcept {
			if (act_dLdZ.numel() < Thresholds_t::dLoss_dZ<typename WlT::tag_dLdZ>::thr) {
				get_self().dLoss_dZ_st<WlT>(data_y, act_dLdZ);
			} else get_self().dLoss_dZ_mt<WlT>(data_y, act_dLdZ);
		}
		template<typename WlT>
		void dLoss_dZ_st(const smatrix<typename WlT::real_t>& data_y, smatrix<typename WlT::real_t>& act_dLdZ, const elms_range*const pER = nullptr) noexcept {
			get_self()._idLoss_dZ_st<WlT>(data_y, act_dLdZ, pER ? *pER : elms_range(act_dLdZ));
		}
		template<typename WlT>
		static void _idLoss_dZ_st(const smatrix<typename WlT::real_t>& data_y, smatrix<typename WlT::real_t>& act_dLdZ, const elms_range& er)noexcept {
			typedef typename WlT::real_t real_t;
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());

			auto pY = data_y.data() + er.elmBegin;
			auto pSD = act_dLdZ.data() + er.elmBegin;
			const auto pSDE = pSD + er.totalElements();
			while (pSD != pSDE) {
				const real_t a = *pSD;
				const real_t y = *pY++;
				*pSD++ = WlT::dLdZ(y, a);
			}
		}
		template<typename WlT>
		void dLoss_dZ_mt(const smatrix<typename WlT::real_t>& data_y, smatrix<typename WlT::real_t>& act_dLdZ)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			m_threads.run([&data_y, &act_dLdZ, this](const par_range_t& r) noexcept{
				get_self()._idLoss_dZ_st<WlT>(data_y, act_dLdZ, elms_range(r));
			}, act_dLdZ.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//#TODO refactor other suitable loss computation types to use this function
		template<typename WlT>
		real_t compute_loss(const smatrix<typename WlT::real_t>& activations, const smatrix<typename WlT::real_t>& data_y)noexcept {
			if (activations.numel() < Thresholds_t::compute_loss<typename WlT::tag_loss>::thr) {
				return get_self().compute_loss_st<WlT>(activations, data_y);
			} else return get_self().compute_loss_mt<WlT>(activations, data_y);
		}
		template<typename WlT>
		static real_t _icompute_loss_st(const smatrix<typename WlT::real_t>& activations, const smatrix<typename WlT::real_t>& data_y, const elms_range& er)noexcept {
			typedef typename WlT::real_t real_t;
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());

			auto pA = activations.data();
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			auto pY = data_y.data() + er.elmBegin;
			real_t ret(0.0);
			while (pA != pAE) {
				const real_t a = *pA++, y = *pY++;
				ret += WlT::loss(a, y);
			}
			return ret;
		}
		template<typename WlT>
		static real_t compute_loss_st(const smatrix<typename WlT::real_t>& activations, const smatrix<typename WlT::real_t>& data_y, const elms_range*const pER = nullptr)noexcept {
			return WlT::normalize(_icompute_loss_st<WlT>(activations, data_y, pER ? *pER : elms_range(activations)), activations.rows());
		}
		template<typename WlT>
		real_t compute_loss_mt(const smatrix<typename WlT::real_t>& activations, const smatrix<typename WlT::real_t>& data_y)noexcept {
			typedef typename WlT::real_t real_t;
			const real_t ql = m_threads.reduce([&activations, &data_y](const par_range_t& r)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					_icompute_loss_st<WlT>(activations, data_y, elms_range(r))
				);
			}, _reduce_vec_sum<real_t>, activations.numel());
			return WlT::normalize(ql, activations.rows());
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// L = -y*log(a)-(1-y)log(1-a) (dL/dZ = dL/dA * dA/dZ = (a-y)/(a*(1-a)) * dA/dZ )
		// dA/dZ = 1 /////{0|a==0, 1|a!=0}
		// because activations comes from the output layer, expecting no biases there
		void dIdentityXEntropyLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ)noexcept {
			if (act_dLdZ.numel() < Thresholds_t::dIdentityXEntropyLoss_dZ) {
				get_self().dIdentityXEntropyLoss_dZ_st(data_y, act_dLdZ);
			} else get_self().dIdentityXEntropyLoss_dZ_mt(data_y, act_dLdZ);
		}
		void dIdentityXEntropyLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const elms_range*const pER = nullptr)noexcept {
			get_self()._idIdentityXEntropyLoss_dZ_st(data_y, act_dLdZ, pER ? *pER : elms_range(act_dLdZ));
		}
		static void _idIdentityXEntropyLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const elms_range& er)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());

			auto pY = data_y.data() + er.elmBegin;
			auto pSD = act_dLdZ.data() + er.elmBegin;
			const auto pSDE = pSD + er.totalElements();
			while (pSD != pSDE) {
				const auto av = *pSD;
				NNTL_ASSERT(real_t(0.) <= av && av <= real_t(1.));
				const auto y = *pY++;
				NNTL_ASSERT(real_t(0.) <= y && y <= real_t(1.));

				//#numstab ?
				//*pSD++ = (av - y)*(av != real_t(0.)) / (av*(real_t(1.) - av));
				*pSD++ = (av - y) / (av*(real_t(1.) - av));
			}
		}
		void dIdentityXEntropyLoss_dZ_mt(const realmtx_t& data_y, realmtx_t& act_dLdZ)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			m_threads.run([&data_y, &act_dLdZ, this](const par_range_t& r) noexcept{
				get_self()._idIdentityXEntropyLoss_dZ_st(data_y, act_dLdZ, elms_range(r));
			}, act_dLdZ.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		// sigmoid function
		//////////////////////////////////////////////////////////////////////////
		void sigm(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::sigm) {
				get_self().sigm_st(srcdest);
			}else get_self().sigm_mt(srcdest);
		}
		// MUST ignore biases!
		void sigm_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) const noexcept {
			get_self()._isigm_st(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _isigm_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
// 			const auto ptr = srcdest.data();
// 			for (range_t i = er.elmBegin; i < er.elmEnd; ++i) ptr[i] = real_t(1.0) / (real_t(1.0) + ::std::exp(-ptr[i]));

			auto pA = srcdest.data() + er.elmBegin;
			const auto pAE = pA + er.totalElements();

			/*while (pA != pAE) {
				NNTL_ASSERT(!::std::isnan(*pA++));
			}
			pA = srcdest.data() + er.elmBegin;*/

			while (pA != pAE) {
				const auto x = *pA;
				const auto a = real_t(1.0) / (real_t(1.0) + ::std::exp(-x));
				//FUUUUUUU!!!!!
				// when the code is vectorized, ::std::exp(-x) may produce -nan(ind) instead of standard 0.
				*pA++ = a;
			}

			/*pA = srcdest.data() + er.elmBegin;
			while (pA != pAE) {
				NNTL_ASSERT(!::std::isnan(*pA++));
			}*/
		}
		void sigm_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& pr) noexcept{
				get_self()._isigm_st(srcdest, elms_range(pr));
			}, srcdest.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		// d(sigm)/d(arg) - sigmoid derivative df = f.*(1-f), where fValue is activation value (used in no_bias version)
		void dsigm(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::dsigm) {
				get_self().dsigm_st(f_df);
			} else get_self().dsigm_mt(f_df);
		}
		void dsigm_st(realmtx_t& f_df, const elms_range*const pER = nullptr) noexcept {
			get_self()._idsigm_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idsigm_st(realmtx_t& f_df, const elms_range& er) noexcept {
			NNTL_ASSERT(!f_df.empty());
			auto pF = f_df.data() + er.elmBegin;
			const auto pFE = pF + er.totalElements();
			while (pF != pFE) {
				const auto f = *pF;
				NNTL_ASSERT(f >= real_t(0.) && f <= real_t(1.));
				*pF++ = f*(real_t(1.0) - f);
			}
		}
		void dsigm_mt(realmtx_t& f_df) noexcept {
			m_threads.run([&f_df, this](const par_range_t& pr) noexcept{
				get_self()._idsigm_st(f_df, elms_range(pr));
			}, f_df.numel());
		}
		
		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of quadratic loss function for sigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		//dL/dZ = (err===a-y)*a*(1-a)
		// because activations comes from the output layer, expecting no biases there
		void dSigmQuadLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ) {
			if (act_dLdZ.numel() < Thresholds_t::dSigmQuadLoss_dZ) {
				get_self().dSigmQuadLoss_dZ_st(data_y, act_dLdZ);
			}else get_self().dSigmQuadLoss_dZ_mt(data_y, act_dLdZ);
		}
		//usually error is defined as diffrence between data_y and last layer activation, i.e. nn.e=y-nn.a{n}, but
		//that will lead to necessity of negation of error in back propagation algorithm. To get rid of that negation,
		// we'll define error as nn.a{n}-y. This won't bother loss calculation, because it is either squares error
		// (conventional quadratic loss function) or doesn't use that error definition at all (crossentropy error)
		void dSigmQuadLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const elms_range*const pER = nullptr) {
			get_self()._idSigmQuadLoss_dZ_st(data_y, act_dLdZ, pER ? *pER : elms_range(act_dLdZ));
		}
		static void _idSigmQuadLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const elms_range& er) {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());

			auto pY = data_y.data() + er.elmBegin;
			auto pSD = act_dLdZ.data() + er.elmBegin;
			const auto pSDE = pSD + er.totalElements();
			while (pSD != pSDE) {
				const auto a = *pSD;
				NNTL_ASSERT(real_t(0.) <= a && a <= real_t(1.));
				const auto y = *pY++;
				NNTL_ASSERT(real_t(0.) <= y && y <= real_t(1.));
				*pSD++ = (a - y)*a*(real_t(1.0) - a);
			}
		}
		void dSigmQuadLoss_dZ_mt(const realmtx_t& data_y, realmtx_t& act_dLdZ) {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			m_threads.run([&data_y, &act_dLdZ, this](const par_range_t& r) noexcept{
				get_self()._idSigmQuadLoss_dZ_st(data_y, act_dLdZ, elms_range(r));
			}, act_dLdZ.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ReLU
		// MUST ignore biases!
		void relu(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::relu) {
				get_self().relu_st(srcdest);
			} else get_self().relu_mt(srcdest);
		}
		void relu_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) noexcept {
			get_self()._irelu_st(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _irelu_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				/*if (*pV < real_t(+0.0))  *pV = real_t(0.0);
				++pV;*/
				const auto v = *pV;
				*pV++ = v <= real_t(0.) ? real_t(0.) : v;
			}
		}
		void relu_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest,this](const par_range_t& r) noexcept{
				get_self()._irelu_st(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		// d(ReLU)/dZ
		void drelu(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::drelu) {
				get_self().drelu_st(f_df);
			} else get_self().drelu_mt(f_df);
		}
		void drelu_st(realmtx_t& f_df, const elms_range*const pER = nullptr) noexcept {
			get_self()._idrelu_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idrelu_st(realmtx_t& f_df, const elms_range& er) noexcept {			
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v <= real_t(0.) ? real_t(0.) : real_t(1.0);
			}
		}
		void drelu_mt(realmtx_t& f_df) noexcept {			
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df,this](const par_range_t& r) noexcept{
				get_self()._idrelu_st(f_df, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//Leaky ReLU
		void leakyrelu(realmtx_t& srcdest, const real_t leak) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::leakyrelu) {
				get_self().leakyrelu_st(srcdest, leak);
			} else get_self().leakyrelu_mt(srcdest, leak);
		}
		void leakyrelu_st(realmtx_t& srcdest, const real_t leak, const elms_range*const pER = nullptr) noexcept {
			get_self()._ileakyrelu_st(srcdest, leak, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ileakyrelu_st(realmtx_t& srcdest, const real_t leak, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(leak > real_t(0.0));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*if (v < real_t(+0.0))  *pV = v*leak;
				++pV;*/
				*pV++ = v < real_t(0.0) ? v*leak : v;
			}
		}
		void leakyrelu_mt(realmtx_t& srcdest, const real_t leak) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, leak, this](const par_range_t& r) noexcept{
				get_self()._ileakyrelu_st(srcdest, leak, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(LeakyReLU)/dZ
		void dleakyrelu(realmtx_t& f_df, const real_t leak) noexcept {
			if (f_df.numel() < Thresholds_t::dleakyrelu) {
				get_self().dleakyrelu_st(f_df, leak);
			} else get_self().dleakyrelu_mt(f_df, leak);
		}
		void dleakyrelu_st(realmtx_t& f_df, const real_t leak, const elms_range*const pER = nullptr) noexcept {
			get_self()._idleakyrelu_st(f_df, leak, pER ? *pER : elms_range(f_df));
		}
		static void _idleakyrelu_st(realmtx_t& f_df, const real_t leak, const elms_range& er) noexcept {
			NNTL_ASSERT(leak > real_t(0.0));
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v <= real_t(0.) ? leak : real_t(1.0);
			}
		}
		void dleakyrelu_mt(realmtx_t& f_df, const real_t leak) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, leak,this](const par_range_t& r) noexcept{
				get_self()._idleakyrelu_st(f_df, leak, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ELU
		void elu(realmtx_t& srcdest, const real_t alpha) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elu) {
				get_self().elu_st(srcdest, alpha);
			} else get_self().elu_mt(srcdest, alpha);
		}
		void elu_st(realmtx_t& srcdest, const real_t alpha, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielu_st(srcdest, alpha, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielu_st(realmtx_t& srcdest, const real_t alpha, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*if (v < real_t(+0.0)) *pV = (::std::exp(v) - real_t(1.0))*alpha;
				++pV;*/
				//*pV++ = v < real_t(0.) ? (::std::exp(v) - real_t(1.0))*alpha : v;
				*pV++ = v < real_t(0.) ? math::expm1(v)*alpha : v;
			}
		}
		void elu_mt(realmtx_t& srcdest, const real_t alpha) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, alpha, this](const par_range_t& r) noexcept{
				get_self()._ielu_st(srcdest, alpha, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ
		void delu(realmtx_t& f_df, const real_t alpha) noexcept {
			if (f_df.numel() < Thresholds_t::delu) {
				get_self().delu_st(f_df, alpha);
			} else get_self().delu_mt(f_df, alpha);
		}
		void delu_st(realmtx_t& f_df, const real_t alpha, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelu_st(f_df, alpha, pER ? *pER : elms_range(f_df));
		}
		static void _idelu_st(realmtx_t& f_df, const real_t alpha, const elms_range& er) noexcept {
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + alpha) : real_t(1.0);
			}
		}
		void delu_mt(realmtx_t& f_df, const real_t alpha) noexcept {			
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, alpha, this](const par_range_t& r) noexcept{
				get_self()._idelu_st(f_df, alpha, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ELU with alpha==1.0
		void elu_unitalpha(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elu_unitalpha) {
				get_self().elu_unitalpha_st(srcdest);
			} else get_self().elu_unitalpha_mt(srcdest);
		}
		void elu_unitalpha_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielu_unitalpha_st(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielu_unitalpha_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*if (v < real_t(+0.0)) *pV = (::std::exp(v) - real_t(1.0));
				++pV;*/
				//*pV++ = v < real_t(0.0) ? (::std::exp(v) - real_t(1.0)) : v;
				*pV++ = v < real_t(0.0) ? math::expm1(v) : v;
			}
		}
		void elu_unitalpha_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& r) noexcept{
				get_self()._ielu_unitalpha_st(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ
		void delu_unitalpha(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::delu_unitalpha) {
				get_self().delu_unitalpha_st(f_df);
			} else get_self().delu_unitalpha_mt(f_df);
		}
		void delu_unitalpha_st(realmtx_t& f_df, const elms_range*const pER = nullptr) noexcept {
			get_self()._idelu_unitalpha_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idelu_unitalpha_st(realmtx_t& f_df, const elms_range& er) noexcept {
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + real_t(1.0)) : real_t(1.0);
			}
		}
		void delu_unitalpha_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) noexcept{
				get_self()._idelu_unitalpha_st(f_df, elms_range(r));
			}, f_df.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ELogU : alpha*(exp(x)-1) | x<0,    log(x+1)/log(b) | x>0
		void elogu(realmtx_t& srcdest, const real_t alpha, const real_t b) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu) {
				get_self().elogu_st(srcdest, alpha, b);
			} else get_self().elogu_mt(srcdest, alpha, b);
		}
		void elogu_st(realmtx_t& srcdest, const real_t alpha, const real_t b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_st(srcdest, alpha, b, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_st(realmtx_t& srcdest, const real_t alpha, const real_t b, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			const real_t lbi = real_t(1.) / ::std::log(b);
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (::std::exp(v) - real_t(1.))*alpha : log(v + real_t(1.))*lbi;
				*pV++ = v < real_t(0.0) ? math::expm1(v)*alpha : math::log1p(v)*lbi;
			}
		}
		void elogu_mt(realmtx_t& srcdest, const real_t alpha, const real_t b) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&srcdest, alpha, b, this](const par_range_t& r) noexcept{
				get_self()._ielogu_st(srcdest, alpha, b, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu(realmtx_t& f_df, const real_t alpha, const real_t b) noexcept {
			if (f_df.numel() < Thresholds_t::delogu) {
				get_self().delogu_st(f_df, alpha, b);
			} else get_self().delogu_mt(f_df, alpha, b);
		}
		void delogu_st(realmtx_t& f_df, const real_t alpha, const real_t b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_st(f_df, alpha, b, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_st(realmtx_t& f_df, const real_t alpha, const real_t b, const elms_range& er) noexcept {
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());

			const ext_real_t _lb = ::std::log(ext_real_t(b));
			const real_t nllb = -static_cast<real_t>(::std::log(_lb)), nlb = -static_cast<real_t>(_lb);

			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + alpha) : ::std::exp(v*nlb + nllb);
			}
		}
		void delogu_mt(realmtx_t& f_df, const real_t alpha, const real_t b) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&f_df, alpha, b, this](const par_range_t& r) noexcept{
				get_self()._idelogu_st(f_df, alpha, b, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// ELogU with alpha==1.
		void elogu_ua(realmtx_t& srcdest, const real_t b) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu_ua) {
				get_self().elogu_ua_st(srcdest, b);
			} else get_self().elogu_ua_mt(srcdest, b);
		}
		void elogu_ua_st(realmtx_t& srcdest, const real_t b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_ua_st(srcdest, b, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_ua_st(realmtx_t& srcdest, const real_t b, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b > real_t(1.0));
			const real_t lbi = real_t(1.) / ::std::log(b);
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (::std::exp(v) - real_t(1.)) : log(v + real_t(1.))*lbi;
				*pV++ = v < real_t(0.0) ? math::expm1(v) : math::log1p(v)*lbi;
			}
		}
		void elogu_ua_mt(realmtx_t& srcdest, const real_t b) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&srcdest, b, this](const par_range_t& r) noexcept{
				get_self()._ielogu_ua_st(srcdest, b, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu_ua(realmtx_t& f_df, const real_t b) noexcept {
			if (f_df.numel() < Thresholds_t::delogu_ua) {
				get_self().delogu_ua_st(f_df, b);
			} else get_self().delogu_ua_mt(f_df, b);
		}
		void delogu_ua_st(realmtx_t& f_df, const real_t b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_ua_st(f_df, b, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_ua_st(realmtx_t& f_df, const real_t b, const elms_range& er) noexcept {
			NNTL_ASSERT(b > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());

			const ext_real_t _lb = ::std::log(ext_real_t(b));
			const real_t nllb = -static_cast<real_t>(::std::log(_lb)), nlb = -static_cast<real_t>(_lb);

			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + real_t(1.)) : ::std::exp(v*nlb + nllb);
			}
		}
		void delogu_ua_mt(realmtx_t& f_df, const real_t b) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&f_df, b, this](const par_range_t& r) noexcept{
				get_self()._idelogu_ua_st(f_df, b, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ELogU with natural base, b==exp(1)
		void elogu_nb(realmtx_t& srcdest, const real_t alpha) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu_nb) {
				get_self().elogu_nb_st(srcdest, alpha);
			} else get_self().elogu_nb_mt(srcdest, alpha);
		}
		void elogu_nb_st(realmtx_t& srcdest, const real_t alpha, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_nb_st(srcdest, alpha, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_nb_st(realmtx_t& srcdest, const real_t alpha, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (::std::exp(v) - real_t(1.))*alpha : log(v + real_t(1.));
				*pV++ = v < real_t(0.0) ? math::expm1(v)*alpha : math::log1p(v);
			}
		}
		void elogu_nb_mt(realmtx_t& srcdest, const real_t alpha) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			m_threads.run([&srcdest, alpha, this](const par_range_t& r) noexcept{
				get_self()._ielogu_nb_st(srcdest, alpha, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu_nb(realmtx_t& f_df, const real_t alpha) noexcept {
			if (f_df.numel() < Thresholds_t::delogu_nb) {
				get_self().delogu_nb_st(f_df, alpha);
			} else get_self().delogu_nb_mt(f_df, alpha);
		}
		void delogu_nb_st(realmtx_t& f_df, const real_t alpha, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_nb_st(f_df, alpha, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_nb_st(realmtx_t& f_df, const real_t alpha, const elms_range& er) noexcept {
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());

			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + alpha) : ::std::exp(-v);
			}
		}
		void delogu_nb_mt(realmtx_t& f_df, const real_t alpha) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			m_threads.run([&f_df, alpha, this](const par_range_t& r) noexcept{
				get_self()._idelogu_nb_st(f_df, alpha, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//ELogU with unit alpha and natural base, b==exp(1)
		void elogu_ua_nb(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu_ua_nb) {
				get_self().elogu_ua_nb_st(srcdest);
			} else get_self().elogu_ua_nb_mt(srcdest);
		}
		void elogu_ua_nb_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_ua_nb_st(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_ua_nb_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (::std::exp(v) - real_t(1.)) : log(v + real_t(1.));
				*pV++ = v < real_t(0.0) ? math::expm1(v) : math::log1p(v);
			}
		}
		void elogu_ua_nb_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& r) noexcept{
				get_self()._ielogu_ua_nb_st(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu_ua_nb(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::delogu_ua_nb) {
				get_self().delogu_ua_nb_st(f_df);
			} else get_self().delogu_ua_nb_mt(f_df);
		}
		void delogu_ua_nb_st(realmtx_t& f_df, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_ua_nb_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_ua_nb_st(realmtx_t& f_df, const elms_range& er) noexcept {
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + real_t(1.)) : ::std::exp(-v);
			}
		}
		void delogu_ua_nb_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) noexcept{
				get_self()._idelogu_ua_nb_st(f_df, elms_range(r));
			}, f_df.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//LogLogU : -log(1-x)/log(b_neg) | x<0,   log(x+1)/log(b_pos) | x>0
		void loglogu(realmtx_t& srcdest, const real_t b_neg, const real_t b_pos) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu) {
				get_self().loglogu_st(srcdest, b_neg, b_pos);
			} else get_self().loglogu_mt(srcdest, b_neg, b_pos);
		}
		void loglogu_st(realmtx_t& srcdest, const real_t b_neg, const real_t b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_st(srcdest, b_neg, b_pos, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_st(realmtx_t& srcdest, const real_t b_neg, const real_t b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			const real_t lbposi = real_t(ext_real_t(1.) / ::std::log(ext_real_t(b_pos)))
				, nlbnegi = real_t(ext_real_t (-1.) / ::std::log(ext_real_t(b_neg)));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//const auto isNeg = v < real_t(0.0);
// 				const auto lv = isNeg ? (real_t(1.) - v) : (v + real_t(1.));
// 				const auto bv = isNeg ? nlbnegi : lbposi;
// 				*pV++ = bv*::std::log(lv);

				//*pV++ = (isNeg ? nlbnegi : lbposi)*math::log1p(isNeg ? -v : v);
				*pV++ = (v < real_t(0.0) ? nlbnegi : lbposi)*math::log1p(::std::fabs(v));//a bit faster
			}
		}
		void loglogu_mt(realmtx_t& srcdest, const real_t b_neg, const real_t b_pos) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&srcdest, b_neg, b_pos, this](const par_range_t& r) noexcept{
				get_self()._iloglogu_st(srcdest, b_neg, b_pos, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu(realmtx_t& f_df, const real_t b_neg, const real_t b_pos) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu) {
				get_self().dloglogu_st(f_df, b_neg, b_pos);
			} else get_self().dloglogu_mt(f_df, b_neg, b_pos);
		}
		void dloglogu_st(realmtx_t& f_df, const real_t b_neg, const real_t b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_st(f_df, b_neg, b_pos, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_st(realmtx_t& f_df, const real_t b_neg, const real_t b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());
			const ext_real_t _lbpos = ::std::log(ext_real_t(b_pos)), _lbneg = ::std::log(ext_real_t(b_neg));
			const real_t nllbpos = -static_cast<real_t>(::std::log(_lbpos)), nlbpos = -static_cast<real_t>(_lbpos);
			const real_t nllbneg = -static_cast<real_t>(::std::log(_lbneg)), lbneg = static_cast<real_t>(_lbneg);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = ::std::exp(v < real_t(0.) ? (v*lbneg + nllbneg) : (v*nlbpos + nllbpos));
			}
		}
		void dloglogu_mt(realmtx_t& f_df, const real_t b_neg, const real_t b_pos) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&f_df, b_neg, b_pos, this](const par_range_t& r) noexcept{
				get_self()._idloglogu_st(f_df, b_neg, b_pos, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//#TODO code should be improved. And it's slower than a generic version.
		void loglogu_nbn(realmtx_t& srcdest, const real_t b_pos) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu_nbn) {
				get_self().loglogu_nbn_st(srcdest, b_pos);
			} else get_self().loglogu_nbn_mt(srcdest, b_pos);
		}
		void loglogu_nbn_st(realmtx_t& srcdest, const real_t b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_nbn_st(srcdest, b_pos, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_nbn_st(realmtx_t& srcdest, const real_t b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_pos > real_t(1.0));
			const real_t lbposi = real_t(ext_real_t(1.) / ::std::log(ext_real_t(b_pos)));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*const auto isNeg = v < real_t(0.0);
				const auto lv = isNeg ? (real_t(1.) - v) : (v + real_t(1.));
				const auto bv = isNeg ? real_t(-1.) : lbposi;
				*pV++ = bv*log(lv);*/
				//*pV++ = v < real_t(0.0) ? -log(real_t(1.) - v) : lbposi*log(v + real_t(1.));
				*pV++ = v < real_t(0.0) ? -math::log1p(-v) : lbposi*math::log1p(v);
			}
		}
		void loglogu_nbn_mt(realmtx_t& srcdest, const real_t b_pos) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&srcdest, b_pos, this](const par_range_t& r) noexcept{
				get_self()._iloglogu_nbn_st(srcdest, b_pos, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu_nbn(realmtx_t& f_df, const real_t b_pos) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu_nbn) {
				get_self().dloglogu_nbn_st(f_df, b_pos);
			} else get_self().dloglogu_nbn_mt(f_df, b_pos);
		}
		void dloglogu_nbn_st(realmtx_t& f_df, const real_t b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_nbn_st(f_df, b_pos, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_nbn_st(realmtx_t& f_df, const real_t b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(b_pos > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());
			const ext_real_t _lbpos = ::std::log(ext_real_t(b_pos));
			const real_t nllbpos = -static_cast<real_t>(::std::log(_lbpos)), nlbpos = -static_cast<real_t>(_lbpos);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = ::std::exp(v < real_t(0.) ? v : (v*nlbpos + nllbpos));
			}
		}
		void dloglogu_nbn_mt(realmtx_t& f_df, const real_t b_pos) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&f_df, b_pos, this](const par_range_t& r) noexcept{
				get_self()._idloglogu_nbn_st(f_df, b_pos, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		void loglogu_nbp(realmtx_t& srcdest, const real_t b_neg) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu_nbp) {
				get_self().loglogu_nbp_st(srcdest, b_neg);
			} else get_self().loglogu_nbp_mt(srcdest, b_neg);
		}
		void loglogu_nbp_st(realmtx_t& srcdest, const real_t b_neg, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_nbp_st(srcdest, b_neg, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_nbp_st(realmtx_t& srcdest, const real_t b_neg, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));			
			const real_t nlbnegi = real_t(ext_real_t (-1.) / ::std::log(ext_real_t(b_neg)));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*const auto isNeg = v < real_t(0.0);
				const auto lv = isNeg ? (real_t(1.) - v) : (v + real_t(1.));
				const auto bv = isNeg ? nlbnegi : lbposi;
				*pV++ = bv*log(lv);*/
				//*pV++ = v < real_t(0.0) ? nlbnegi*log(real_t(1.) - v) : log(v + real_t(1.));
				*pV++ = v < real_t(0.0) ? nlbnegi*math::log1p(-v) : math::log1p(v);
			}
		}
		void loglogu_nbp_mt(realmtx_t& srcdest, const real_t b_neg) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));			
			m_threads.run([&srcdest, b_neg, this](const par_range_t& r) noexcept{
				get_self()._iloglogu_nbp_st(srcdest, b_neg, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu_nbp(realmtx_t& f_df, const real_t b_neg) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu_nbp) {
				get_self().dloglogu_nbp_st(f_df, b_neg);
			} else get_self().dloglogu_nbp_mt(f_df, b_neg);
		}
		void dloglogu_nbp_st(realmtx_t& f_df, const real_t b_neg, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_nbp_st(f_df, b_neg, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_nbp_st(realmtx_t& f_df, const real_t b_neg, const elms_range& er) noexcept {
			NNTL_ASSERT(b_neg > real_t(1.0));			
			NNTL_ASSERT(!f_df.empty());
			const ext_real_t _lbneg = ::std::log(ext_real_t(b_neg));
			const real_t nllbneg = -static_cast<real_t>(::std::log(_lbneg)), lbneg = static_cast<real_t>(_lbneg);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = ::std::exp(v < real_t(0.) ? (v*lbneg + nllbneg) : -v);
			}
		}
		void dloglogu_nbp_mt(realmtx_t& f_df, const real_t b_neg) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));			
			m_threads.run([&f_df, b_neg, this](const par_range_t& r) noexcept{
				get_self()._idloglogu_nbp_st(f_df, b_neg, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		void loglogu_nbn_nbp(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu_nbn_nbp) {
				get_self().loglogu_nbn_nbp_st(srcdest);
			} else get_self().loglogu_nbn_nbp_mt(srcdest);
		}
		void loglogu_nbn_nbp_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_nbn_nbp_st(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_nbn_nbp_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*const auto isNeg = v < real_t(0.0);
				const auto lv = isNeg ? (real_t(1.) - v) : (v + real_t(1.));
				const auto bv = isNeg ? nlbnegi : lbposi;
				*pV++ = bv*log(lv);*/
				
				//*pV++ = v < real_t(0.0) ? -log(real_t(1.) - v) : log(v + real_t(1.));
				
				*pV++ = v < real_t(0.0) ? -math::log1p(-v) : math::log1p(v);
			}
		}
		void loglogu_nbn_nbp_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());			
			m_threads.run([&srcdest, this](const par_range_t& r) noexcept{
				get_self()._iloglogu_nbn_nbp_st(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu_nbn_nbp(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu_nbn_nbp) {
				get_self().dloglogu_nbn_nbp_st(f_df);
			} else get_self().dloglogu_nbn_nbp_mt(f_df);
		}
		void dloglogu_nbn_nbp_st(realmtx_t& f_df, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_nbn_nbp_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_nbn_nbp_st(realmtx_t& f_df, const elms_range& er) noexcept {
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = ::std::exp(v < real_t(0.) ? v : -v);
			}
		}
		void dloglogu_nbn_nbp_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) noexcept{
				get_self()._idloglogu_nbn_nbp_st(f_df, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// y = (x/(a+|x|)), dy/dx = (1-|y|)^2 /a, parameter 'a' controls the slope of the curve
		void softsign_uc(realmtx_t& srcdest, const real_t a) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::softsign) {
				get_self().softsign_uc_st(srcdest, a);
			} else get_self().softsign_uc_mt(srcdest, a);
		}
		void softsign_uc_st(realmtx_t& srcdest, const real_t a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._isoftsign_uc_st(srcdest, a, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _isoftsign_uc_st(realmtx_t& srcdest, const real_t a, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));

			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = v / (a + ::std::abs(v));
			}
		}
		void softsign_uc_mt(realmtx_t& srcdest, const real_t a) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&srcdest, a, this](const par_range_t& r) noexcept{
				get_self()._isoftsign_uc_st(srcdest, a, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// y = c*(x/(a+|x|)), dy/dx = (c-|y|)^2 /(c*a), parameter 'a' controls the slope of the curve, c- amplitude
		void softsign(realmtx_t& srcdest, const real_t a, const real_t c) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::softsign) {
				get_self().softsign_st(srcdest, a, c);
			} else get_self().softsign_mt(srcdest, a, c);
		}
		void softsign_st(realmtx_t& srcdest, const real_t a, const real_t c, const elms_range*const pER = nullptr) const noexcept {
			get_self()._isoftsign_st(srcdest, a, c, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _isoftsign_st(realmtx_t& srcdest, const real_t a, const real_t c, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(c > real_t(0.0));

			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = (c*v) / (a + ::std::abs(v));
			}
		}
		void softsign_mt(realmtx_t& srcdest, const real_t a, const real_t c) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(c > real_t(0.0));
			m_threads.run([&srcdest, a, c, this](const par_range_t& r) noexcept{
				get_self()._isoftsign_st(srcdest, a, c, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		//dy / dx = (1 - |y|)^2
		void dsoftsign_ua_uc(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::dsoftsign_ua_uc) {
				get_self().dsoftsign_ua_uc_st(f_df);
			} else get_self().dsoftsign_ua_uc_mt(f_df);
		}
		void dsoftsign_ua_uc_st(realmtx_t& f_df, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idsoftsign_ua_uc_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idsoftsign_ua_uc_st(realmtx_t& f_df, const elms_range& er) noexcept {
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				NNTL_ASSERT(real_t(-1.) <= v && v <= real_t(1.));
				const auto s = real_t(1.) - ::std::abs(v);
				*ptrDF++ = s*s;
			}
		}
		void dsoftsign_ua_uc_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) noexcept{
				get_self()._idsoftsign_ua_uc_st(f_df, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//dy/dx = (c-|y|)^2 /(c*a)
		void dsoftsign(realmtx_t& f_df, const real_t a, const real_t c) noexcept {
			if (f_df.numel() < Thresholds_t::dsoftsign) {
				get_self().dsoftsign_st(f_df, a, c);
			} else get_self().dsoftsign_mt(f_df, a, c);
		}
		void dsoftsign_st(realmtx_t& f_df, const real_t a, const real_t c, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idsoftsign_st(f_df, a, c, pER ? *pER : elms_range(f_df));
		}
		static void _idsoftsign_st(realmtx_t& f_df, const real_t a, const real_t c, const elms_range& er) noexcept {
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(c > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());
			const auto mult = real_t(1.) / (c*a);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				NNTL_ASSERT(real_t(-c) <= v && v <= c);
				const auto s = c - ::std::abs(v);
				*ptrDF++ = mult*(s*s);
			}
		}
		void dsoftsign_mt(realmtx_t& f_df, const real_t a, const real_t c) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(c > real_t(0.0));
			m_threads.run([&f_df, a, c, this](const par_range_t& r) noexcept{
				get_self()._idsoftsign_st(f_df, a, c, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// y = (x/(2*(a+|x|)) +.5 ), dy/dx = (.5-|y-.5|)^2 * 2/a, parameter 'a' controls the slope of the curve
		void softsigm(realmtx_t& srcdest, const real_t a) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::softsigm) {
				get_self().softsigm_st(srcdest, a);
			} else get_self().softsigm_mt(srcdest, a);
		}
		void softsigm_st(realmtx_t& srcdest, const real_t a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._isoftsigm_st(srcdest, a, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _isoftsigm_st(realmtx_t& srcdest, const real_t a, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));

			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = real_t(.5) + real_t(.5)* v / (a + ::std::abs(v));
			}
		}
		void softsigm_mt(realmtx_t& srcdest, const real_t a) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&srcdest, a, this](const par_range_t& r) noexcept{
				get_self()._isoftsigm_st(srcdest, a, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		//dy/dx = (.5-|y-.5|)^2 * 2/a
		void dsoftsigm(realmtx_t& f_df, const real_t a) noexcept {
			if (f_df.numel() < Thresholds_t::dsoftsigm) {
				get_self().dsoftsigm_st(f_df, a);
			} else get_self().dsoftsigm_mt(f_df, a);
		}
		void dsoftsigm_st(realmtx_t& f_df, const real_t a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idsoftsigm_st(f_df, a, pER ? *pER : elms_range(f_df));
		}
		static void _idsoftsigm_st(realmtx_t& f_df, const real_t a, const elms_range& er) noexcept {
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());
			const auto dainv = real_t(2.) / a;
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				NNTL_ASSERT(real_t(0.) <= v && v <= real_t(1.));
				const auto s = real_t(.5) - ::std::abs(v - real_t(.5));
				*ptrDF++ = dainv*s*s;
			}
		}
		void dsoftsigm_mt(realmtx_t& f_df, const real_t a) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&f_df, a, this](const par_range_t& r) noexcept{
				get_self()._idsoftsigm_st(f_df, a, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of quadratic loss function for softsigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		//dL/dZ = (err===a-y)*dSoftSigm/dZ
		// because activations comes from the output layer, expecting no biases there
		void dSoftSigmQuadLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a)noexcept {
			if (act_dLdZ.numel() < Thresholds_t::dSoftSigmQuadLoss_dZ) {
				get_self().dSoftSigmQuadLoss_dZ_st(data_y, act_dLdZ, a);
			} else get_self().dSoftSigmQuadLoss_dZ_mt(data_y, act_dLdZ, a);
		}
		//usually error is defined as diffrence between data_y and last layer activation, i.e. nn.e=y-nn.a{n}, but
		//that will lead to necessity of negation of error in back propagation algorithm. To get rid of that negation,
		// we'll define error as nn.a{n}-y. This won't bother loss calculation, because it is either squares error
		// (conventional quadratic loss function) or doesn't use that error definition at all (crossentropy error)
		void dSoftSigmQuadLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a, const elms_range*const pER = nullptr)noexcept {
			get_self()._idSoftSigmQuadLoss_dZ_st(data_y, act_dLdZ, a, pER ? *pER : elms_range(act_dLdZ));
		}
		static void _idSoftSigmQuadLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a, const elms_range& er)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			NNTL_ASSERT(a > real_t(0.0));

			const auto dainv = real_t(2.) / a;
			auto pY = data_y.data() + er.elmBegin;
			auto pSD = act_dLdZ.data() + er.elmBegin;
			const auto pSDE = pSD + er.totalElements();
			while (pSD != pSDE) {
				const auto av = *pSD;
				NNTL_ASSERT(real_t(0.) <= av && av <= real_t(1.));
				const auto y = *pY++;
				NNTL_ASSERT(real_t(0.) <= y && y <= real_t(1.));
				const auto s = real_t(.5) - ::std::abs(av - real_t(.5));
				*pSD++ = (av - y)*dainv*s*s;
			}
		}
		void dSoftSigmQuadLoss_dZ_mt(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&data_y, &act_dLdZ, a, this](const par_range_t& r) noexcept{
				get_self()._idSoftSigmQuadLoss_dZ_st(data_y, act_dLdZ, a, elms_range(r));
			}, act_dLdZ.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of cross-entropy loss function for softsigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		// L = -y*log(a)-(1-y)log(1-a) (dL/dZ = dL/dA * dA/dZ = (a-y)/(a*(1-a)) * dA/dZ )
		// dL/dZ = (a-y)/(a*(1-a)) * dSoftSigm/dZ
		// because activations comes from the output layer, expecting no biases there
		void dSoftSigmXEntropyLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a)noexcept {
			if (act_dLdZ.numel() < Thresholds_t::dSoftSigmXEntropyLoss_dZ) {
				get_self().dSoftSigmXEntropyLoss_dZ_st(data_y, act_dLdZ, a);
			} else get_self().dSoftSigmXEntropyLoss_dZ_mt(data_y, act_dLdZ, a);
		}
		void dSoftSigmXEntropyLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a, const elms_range*const pER = nullptr)noexcept {
			get_self()._idSoftSigmXEntropyLoss_dZ_st(data_y, act_dLdZ, a, pER ? *pER : elms_range(act_dLdZ));
		}
		static void _idSoftSigmXEntropyLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a, const elms_range& er)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			NNTL_ASSERT(a > real_t(0.0));

			const auto dainv = real_t(2.) / a;
			auto pY = data_y.data() + er.elmBegin;
			auto pSD = act_dLdZ.data() + er.elmBegin;
			const auto pSDE = pSD + er.totalElements();
			while (pSD != pSDE) {
				const auto av = *pSD;
				NNTL_ASSERT(real_t(0.) <= av && av <= real_t(1.));
				const auto y = *pY++;
				NNTL_ASSERT(real_t(0.) <= y && y <= real_t(1.));

				const auto s = real_t(.5) - ::std::abs(av - real_t(.5));
				//#numstab
				*pSD++ = (av - y)*((s*s*dainv) / (av*(real_t(1.) - av)));
			}
		}
		void dSoftSigmXEntropyLoss_dZ_mt(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t a)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&data_y, &act_dLdZ, a, this](const par_range_t& r) noexcept{
				get_self()._idSoftSigmXEntropyLoss_dZ_st(data_y, act_dLdZ, a, elms_range(r));
			}, act_dLdZ.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//SELU, see arxiv:1706.02515 "Self-Normalizing Neural Networks", by Günter Klambauer et al.
		void selu(realmtx_t& srcdest, const real_t alpha_t_lambda, const real_t lambda) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::selu) {
				get_self().selu_st(srcdest, alpha_t_lambda, lambda);
			} else get_self().selu_mt(srcdest, alpha_t_lambda, lambda);
		}
		void selu_st(realmtx_t& srcdest, const real_t alpha_t_lambda, const real_t lambda, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iselu_st(srcdest, alpha_t_lambda, lambda, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iselu_st(realmtx_t& srcdest, const real_t alpha_t_lambda, const real_t lambda, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha_t_lambda > real_t(0.0));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				/*if (v < real_t(+0.0)) *pV = (::std::exp(v) - real_t(1.0))*alpha;
				++pV;*/
				//*pV++ = v < real_t(0.) ? (::std::exp(v) - real_t(1.0))*alpha : v;
				*pV++ = v < real_t(0.) ? math::expm1(v)*alpha_t_lambda : v*lambda;
			}
		}
		void selu_mt(realmtx_t& srcdest, const real_t alpha_t_lambda, const real_t lambda) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, alpha_t_lambda, lambda, this](const par_range_t& r) noexcept{
				get_self()._iselu_st(srcdest, alpha_t_lambda, lambda, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(selu)/dZ
		void dselu(realmtx_t& f_df, const real_t alpha_t_lambda, const real_t lambda) noexcept {
			if (f_df.numel() < Thresholds_t::dselu) {
				get_self().dselu_st(f_df, alpha_t_lambda, lambda);
			} else get_self().dselu_mt(f_df, alpha_t_lambda, lambda);
		}
		void dselu_st(realmtx_t& f_df, const real_t alpha_t_lambda, const real_t lambda, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idselu_st(f_df, alpha_t_lambda, lambda, pER ? *pER : elms_range(f_df));
		}
		// FFFUUUUUUUU....!!!!
		// Declare alpha_t_lambda and lambda parameters as references and get x10 slowdown!
		// I guess it happens because compiler can't assume that they are allocated outside of mutable *ptrDF.
		// C++ is indeed a nice rope to shoot own leg... BTW, __restrict, as well as __declspec(noalias) doesn't seem to work!
		static void _idselu_st(realmtx_t& f_df, const real_t alpha_t_lambda, const real_t lambda, const elms_range& er) noexcept {
			NNTL_ASSERT(alpha_t_lambda > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + alpha_t_lambda) : lambda;
			}
		}
		void dselu_mt(realmtx_t& f_df, const real_t alpha_t_lambda, const real_t lambda) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, alpha_t_lambda, lambda, this](const par_range_t& r) noexcept{
				get_self()._idselu_st(f_df, alpha_t_lambda, lambda, elms_range(r));
			}, f_df.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//step activation unit
		void step(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::step) {
				get_self().step_st(srcdest);
			} else get_self().step_mt(srcdest);
		}
		void step_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) const noexcept {
			get_self()._istep_st(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _istep_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = v < real_t(0.0) ? real_t(0.) : real_t(1.);
			}
		}
		void step_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& r) noexcept{
				get_self()._istep_st(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////



		//////////////////////////////////////////////////////////////////////////
		// #TODO: probably, it is better to rewrite asymmetric activation functions processing using two templated
		// functions, one for f(x) and the other for dF/dX. However, possible performance penalty should be considered
		// -- well, it's a bit slower than Indian-style copy&pasted code...
		/* see test_perf_decisions for better solution. will update code later...
		template<numel_cnt_t MtThreshold, typename FunctorT>
		void act_asymm(realmtx_t& srcdest) noexcept {
			if (srcdest.numel_no_bias() < MtThreshold) {
				get_self().act_asymm_st<FunctorT>(srcdest);
			} else get_self().act_asymm_mt<FunctorT>(srcdest);
		}
		template<typename FunctorT>
		void act_asymm_st(realmtx_t& srcdest, const elms_range*const pER = nullptr) noexcept {
			get_self()._iact_asymm_st<FunctorT>(srcdest, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		template<typename FunctorT>
		void act_asymm_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& r) noexcept{
				get_self()._iact_asymm_st<FunctorT>(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		template<typename FunctorT>
		void _iact_asymm_st(realmtx_t& srcdest, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = v < real_t(+0.0) ? FunctorT::f_neg(v) : FunctorT::f_pos(v);
			}
		}
		//////////////////////////////////////////////////////////////////////////
		template<numel_cnt_t MtThreshold, typename FunctorT>
		void dact_asymm(realmtx_t& f_df) noexcept {
			if (f_df.numel() < MtThreshold) {
				get_self().dact_asymm_st<FunctorT>(f_df);
			} else get_self().dact_asymm_mt<FunctorT>(f_df);
		}
		template<typename FunctorT>
		void dact_asymm_st(realmtx_t& f_df, const elms_range*const pER = nullptr) noexcept {
			get_self()._idact_asymm_st(f_df, pER ? *pER : elms_range(f_df));
		}
		template<typename FunctorT>
		void dact_asymm_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) noexcept{
				get_self()._idact_asymm_st(f_df, elms_range(r));
			}, f_df.numel());
		}
		template<typename FunctorT>
		void _idact_asymm_st(realmtx_t& f_df, const elms_range& er) noexcept {
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(+0.) ? FunctorT::df_neg(v) : FunctorT::df_pos(v);
			}
		}*/


		//////////////////////////////////////////////////////////////////////////
		//loss functions
		//////////////////////////////////////////////////////////////////////////
		real_t loss_quadratic(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_quadratic) {
				return get_self().loss_quadratic_st_naive(activations, data_y);
			} else return get_self().loss_quadratic_mt_naive(activations, data_y);
		}
		static real_t _iloss_quadratic_st_naive(const realmtx_t& activations, const realmtx_t& data_y, const elms_range& er)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());

			auto pA = activations.data();
			const auto pAE = pA + er.elmEnd;
			pA += er.elmBegin;
			auto pY = data_y.data() + er.elmBegin;
			real_t ret(0.0);
			while (pA != pAE) {//gets vectorized
				const real_t e = *pA++ - *pY++;
				ret += e*e;
			}
			return ret;
		}
		static real_t loss_quadratic_st_naive(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			return _iloss_quadratic_st_naive(activations, data_y, pER ? *pER : elms_range(activations)) / (2 /** activations.rows()*/);
		}
		real_t loss_quadratic_mt_naive(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			const real_t ql = m_threads.reduce([&activations, &data_y](const par_range_t& r)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					_iloss_quadratic_st_naive(activations, data_y, elms_range(r))
				);
			}, _reduce_vec_sum<real_t>, activations.numel());
			return ql / (2 /** activations.rows()*/);
		}

		//////////////////////////////////////////////////////////////////////////
		//numerically stabilized
		real_t loss_quadratic_ns(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_quadratic_ns) {
				return get_self().loss_quadratic_st_naive_ns(activations, data_y);
			} else return get_self().loss_quadratic_mt_naive_ns(activations, data_y);
		}

		static real_t _iloss_quadratic_st_naive_ns(const realmtx_t& activations, const realmtx_t& data_y, const elms_range& er)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());

			const auto pA = activations.data();
			const auto pY = data_y.data();
			real_t sum(0.0), C(0.), Y, T;
			//for KahanSum() see https://msdn.microsoft.com/en-us/library/aa289157(v=vs.71).aspx
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) {
				const real_t e = pA[i] - pY[i];
				Y = e*e - C;
				T = sum + Y;
				C = T - sum - Y;
				sum = T;
			}
			return sum;
		}

		static real_t loss_quadratic_st_naive_ns(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			return _iloss_quadratic_st_naive_ns(activations, data_y, pER ? *pER : elms_range(activations)) / (2 /** activations.rows()*/);
		}
		real_t loss_quadratic_mt_naive_ns(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			const real_t ql = m_threads.reduce([&activations, &data_y](const par_range_t& r)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					_iloss_quadratic_st_naive_ns(activations, data_y, elms_range(r))
				);
			}, _reduce_vec_sum<real_t, true>, activations.numel());
			return ql / (2 /** activations.rows()*/);
		}

		//////////////////////////////////////////////////////////////////////////
		// cross entropy function (applicable ONLY for binary data_y and sigmoid-style activation function)
		// L = -y*log(a)-(1-y)log(1-a) (dL/dZ = dL/dA * dA/dZ = (a-y)/(a*(1-a)) * dA/dZ )
		real_t loss_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_xentropy) {
				return get_self().loss_xentropy_st(activations, data_y);
			} else return get_self().loss_xentropy_mt(activations, data_y);
		}
		real_t loss_xentropy_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			return -get_self()._iloss_xentropy_st(activations, data_y, pER ? *pER : elms_range(activations)) /*/ activations.rows()*/;
		}
		static real_t _iloss_xentropy_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range& er)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
			const auto ptrA = activations.data(), ptrY = data_y.data();
			real_t ql = 0;
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) {
				const auto y = ptrY[i];
				const auto a = ptrA[i];
				NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
				NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

				if (y > real_t(0.0)) {
					ql += math::log_eps(a);
				} else {
					//const auto oma = real_t(1.0) - a;
					//ql += (oma == real_t(0.0) ? log_zero : log(oma));
#if NNTL_CFG_CAREFULL_LOG_EXP
					ql += (a == real_t(1.0) ? math::real_t_limits<real_t>::log_almost_zero : math::log1p(-a));
#else
					ql += math::log1p_eps(-a);
#endif
				}
				NNTL_ASSERT(!isnan(ql));
			}
			return ql;
		}
		real_t loss_xentropy_mt(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			return -m_threads.reduce([&activations, &data_y, this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iloss_xentropy_st(activations, data_y, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t>, activations.numel()) /*/ activations.rows()*/;
		}
		//////////////////////////////////////////////////////////////////////////
		real_t loss_xentropy_ns(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_xentropy_ns) {
				return get_self().loss_xentropy_ns_st(activations, data_y);
			} else return get_self().loss_xentropy_ns_mt(activations, data_y);
		}
		real_t loss_xentropy_ns_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			return -get_self()._iloss_xentropy_ns_st(activations, data_y, pER ? *pER : elms_range(activations)) /*/ activations.rows()*/;
		}
		static real_t _iloss_xentropy_ns_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range& er)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
			const auto ptrA = activations.data(), ptrY = data_y.data();
			real_t sum(0.), C(0.), Y, T, ev;
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) {
				const auto y = ptrY[i];
				const auto a = ptrA[i];
				NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
				NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

				if (y > real_t(0.0)) {
					ev= math::log_eps(a);
				} else {
					//const auto oma = real_t(1.0) - a;
					//sum += (oma == real_t(0.0) ? log_zero : log(oma));
#if NNTL_CFG_CAREFULL_LOG_EXP
					ev = (a == real_t(1.0) ? math::real_t_limits<real_t>::log_almost_zero : math::log1p(-a));
#else
					ev = math::log1p_eps(-a);
#endif
				}
				Y = ev - C;
				T = sum + Y;
				C = T - sum - Y;
				sum = T;
				NNTL_ASSERT(!isnan(sum));
			}
			return sum;
		}
		real_t loss_xentropy_ns_mt(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			return -m_threads.reduce([&activations, &data_y, this](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					get_self()._iloss_xentropy_ns_st(activations, data_y, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t, true>, activations.numel()) /*/ activations.rows()*/;
		}



		//////////////////////////////////////////////////////////////////////////
		// cross entropy function for softmax (applicable for data_y in range [0,1])
		// L = sum( -y*log(a) )/activations.rows(), dL/dz=a-y
		real_t loss_softmax_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_softmax_xentropy) {
				return get_self().loss_softmax_xentropy_st(activations, data_y);
			}else return get_self().loss_softmax_xentropy_mt(activations, data_y);
		}
		static real_t _iloss_softmax_xentropy_sum_st(const real_t*const pA, const real_t*const pY, const elms_range& er)noexcept {
			real_t ret(0.0);
			const auto ee = er.elmEnd;
			for (numel_cnt_t i = er.elmBegin; i < ee; ++i) {
				auto a = pA[i];
				const auto y = -pY[i];
				NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));
				NNTL_ASSERT(y <= real_t(0.0) && y >= real_t(-1.0));
				a = a > real_t(0.0) ? ::std::log(a) : math::real_t_limits<real_t>::log_almost_zero;
				ret += y*a;
				NNTL_ASSERT(!isnan(ret));
			}
			return ret;
		}
		static real_t loss_softmax_xentropy_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!activations.empty() && !data_y.empty() && data_y.size() == activations.size());
			return _iloss_softmax_xentropy_sum_st(activations.data(), data_y.data(), pER ? *pER : elms_range(activations)) /*/ activations.rows()*/;
		}
		real_t loss_softmax_xentropy_mt(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(!activations.empty() && !data_y.empty() && data_y.size() == activations.size());
			const auto pA = activations.data(), pY = data_y.data();
			return m_threads.reduce([pA, pY](const par_range_t& pr)noexcept->reduce_data_t {
				return converter_reduce_data_tpl<real_t>::to(
					_iloss_softmax_xentropy_sum_st(pA, pY, elms_range(pr))
				);
			}, _reduce_vec_sum<real_t>, activations.numel()) /*/ activations.rows()*/;
		}


		//////////////////////////////////////////////////////////////////////////
		//gradient application procedures
		void RMSProp_Hinton(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			if (dW.numel() < Thresholds_t::RMSProp_Hinton) {
				get_self().RMSProp_Hinton_st(dW, rmsF, learningRate, emaDecay, numericStabilizer);
			}else get_self().RMSProp_Hinton_mt(dW, rmsF, learningRate, emaDecay, numericStabilizer);
		}
		static void RMSProp_Hinton_st(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//#TODO: this implementation probably isn't vectorized well
			//#consider switching for() to a while(). See the Adam()'s comments
			const auto pdW = dW.data(), prmsF = rmsF.data();
			const auto _1_emaDecay = 1 - emaDecay;
			const auto dataCnt = dW.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto pW = pdW + i, pF = prmsF + i;
				const auto w = *pW;
				const auto rms = emaDecay*(*pF) + w*w*_1_emaDecay;
				*pF = rms;
				*pW = learningRate*(w / (::std::sqrt(rms) + numericStabilizer));
			}
		}
		void RMSProp_Hinton_mt(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//#TODO: this implementation probably isn't vectorized well
			//#consider refactoring to a single call to _st() version
			const auto pdW = dW.data(), prmsF = rmsF.data();
			m_threads.run([pdW,prmsF,learningRate,emaDecay,numericStabilizer](const par_range_t& r) noexcept{
				const auto _1_emaDecay = 1 - emaDecay;
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto pW = pdW + i, pF = prmsF + i;
					const auto w = *pW;
					const auto rms = emaDecay*(*pF) + w*w*_1_emaDecay;
					*pF = rms;
					*pW = learningRate*(w / (::std::sqrt(rms) + numericStabilizer));
				}
			}, dW.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		void RMSProp_Graves(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept 
		{
			if (dW.numel() < Thresholds_t::RMSProp_Graves) {
				get_self().RMSProp_Graves_st(dW, rmsF, rmsG, learningRate, emaDecay, numericStabilizer);
			} else get_self().RMSProp_Graves_mt(dW, rmsF, rmsG, learningRate, emaDecay, numericStabilizer);
		}
		static void RMSProp_Graves_st(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size() && rmsF.size()==rmsG.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//#TODO: this implementation probably isn't vectorized well
			//#consider switching for() to a while(). See the Adam()'s comments
			const auto pdW = dW.data();
			const auto prmsF = rmsF.data();
			const auto prmsG = rmsG.data();
			const auto _1_emaDecay = 1 - emaDecay;
			const auto dataCnt = dW.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto pW = pdW + i, pF = prmsF + i, pG = prmsG + i;
				const auto w = *pW;
				const auto wdec = w*_1_emaDecay;
				const auto rF = emaDecay*(*pF) + w*wdec;
				*pF = rF;
				const auto rG = emaDecay*(*pG) + wdec;
				*pG = rG;
				*pW = learningRate*(w / (::std::sqrt(rF - rG*rG + numericStabilizer)));
			}
		}
		void RMSProp_Graves_mt(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size() && rmsF.size() == rmsG.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//#TODO: this implementation probably isn't vectorized well
			//#consider refactoring to a single call to _st() version
			const auto pdW = dW.data();
			const auto prmsF = rmsF.data();
			const auto prmsG = rmsG.data();
			m_threads.run([pdW, prmsF, prmsG, learningRate, emaDecay, numericStabilizer](const par_range_t& r) noexcept{
				const auto _1_emaDecay = 1 - emaDecay;
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto pW = pdW + i, pF = prmsF + i, pG = prmsG + i;
					const auto w = *pW;
					const auto wdec = w*_1_emaDecay;
					const auto rF = emaDecay*(*pF) + w*wdec;
					*pF = rF;
					const auto rG = emaDecay*(*pG) + wdec;
					*pG = rG;
					*pW = learningRate*(w / (::std::sqrt(rF - rG*rG + numericStabilizer)));
				}
			}, dW.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//RProp or inplace L1 regularization
		void RProp(realmtx_t& dW, const real_t learningRate)noexcept {
			if (dW.numel() < Thresholds_t::RProp) {
				get_self().RProp_st(dW, learningRate);
			} else get_self().RProp_mt(dW, learningRate);
		}
		void RProp_st(realmtx_t& dW, const real_t learningRate)noexcept {
			get_self()._iRProp_st(dW, learningRate, elms_range(dW));
		}
		static void _iRProp_st(realmtx_t& dW, const real_t learningRate, const elms_range& er)noexcept {
			auto p = dW.data() + er.elmBegin;
			const auto pE = p + er.totalElements();
			//TODO: verify vectorization
			while (p != pE) {
				*p++ = learningRate*math::sign(*p);
			}
		}
		void RProp_mt(realmtx_t& dW, const real_t learningRate)noexcept {
			get_self().ithreads().run([&dW, learningRate,this](const par_range_t& r) noexcept{
				get_self()._iRProp_st(dW, learningRate, elms_range(r));
			}, dW.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		// ModProp - like RMSProp, but devide dW by abs( ema(dW) ), instead of
		//		sqrt(ema(dW ^ 2)).Seems no significant changes, but faster. And sometimes works when other doesn't
		void ModProp(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			if (dW.numel() < Thresholds_t::ModProp) {
				get_self().ModProp_st(dW, rmsF, learningRate, emaDecay, numericStabilizer);
			} else get_self().ModProp_mt(dW, rmsF, learningRate, emaDecay, numericStabilizer);
		}
		static void ModProp_st(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//#TODO: this implementation probably isn't vectorized well
			//#consider switching for() to a while(). See the Adam()'s comments
			auto pdW = dW.data();
			auto prmsF = rmsF.data();
			const auto _1_emaDecay = 1 - emaDecay;
			const auto dataCnt = dW.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto pW = pdW + i;
				const auto pF = prmsF + i;
				const auto w = *pW;
				const auto ema = (*pF)*emaDecay + ::std::abs(w)*_1_emaDecay;
				*pF = ema;
				*pW = learningRate*(w / (ema + numericStabilizer));
			}
		}
		void ModProp_mt(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//#TODO: this implementation probably isn't vectorized well
			//#consider refactoring to a single call to _st() version
			auto pdW = dW.data();
			auto prmsF = rmsF.data();
			m_threads.run([pdW, prmsF, learningRate, emaDecay, numericStabilizer](const par_range_t& r) noexcept{
				const auto _1_emaDecay = 1 - emaDecay;
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto pW = pdW + i;
					const auto pF = prmsF + i;
					const auto w = *pW;
					const auto ema = (*pF)*emaDecay + ::std::abs(w)*_1_emaDecay;
					*pF = ema;
					*pW = learningRate*(w / (ema + numericStabilizer));
				}
			}, dW.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		// Adam - A Method for Stochastic Optimization.1412.6980v8 implementation
		//on a first call beta1t and beta2t must be initialized with 1
		void Adam(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept
		{
			if (dW.numel() < Thresholds_t::Adam) {
				get_self().Adam_st(dW, Mt, Vt, beta1t, beta2t, learningRate, beta1, beta2, numericStabilizer);
			} else get_self().Adam_mt(dW, Mt, Vt, beta1t, beta2t, learningRate, beta1, beta2, numericStabilizer);
		}
		void Adam_st(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer, const elms_range*const pER = nullptr) noexcept
		{
			get_self()._iAdam_st(dW, Mt, Vt, beta1t, beta2t, learningRate, beta1, beta2, numericStabilizer, pER ? *pER : elms_range(0, dW.numel()), !!pER);
		}
		static void _iAdam_st(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer, const elms_range& er, const bool bInsideMT = true) noexcept
		{
			NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Vt.size());
			NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta1 && beta1 < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta2 && beta2 < real_t(1.));
			NNTL_ASSERT(real_t(0.) < numericStabilizer && numericStabilizer < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= beta1t && beta1t <= real_t(1.));
			NNTL_ASSERT(real_t(0.) <= beta2t && beta2t <= real_t(1.));

			if (!bInsideMT) {//This means, we're running outside of _mt version
				beta1t *= beta1;
				beta2t *= beta2;
				NNTL_ASSERT(beta1t < real_t(1.));
				NNTL_ASSERT(beta2t < real_t(1.));
			}
			const auto alphat = learningRate*::std::sqrt(real_t(1.) - beta2t) / (real_t(1.) - beta1t);
			const auto ombeta1 = real_t(1.) - beta1, ombeta2 = real_t(1.) - beta2;

// 			if (::std::isnan(alphat)) {
// 				__debugbreak();
// 			}

			auto pdW = dW.data()+ er.elmBegin, pMt = Mt.data()+ er.elmBegin, pVt = Vt.data()+ er.elmBegin;
			const auto pDWE = pdW + er.totalElements();
			while (pdW != pDWE) {
				const auto g = *pdW;
				const auto m = (*pMt)*beta1 + g*ombeta1;
				*pMt++ = m;
				const auto v = (*pVt)*beta2 + (g*g)*ombeta2;
				*pVt++ = v;

				const auto ndw = alphat*m / (::std::sqrt(v) + numericStabilizer);
// 				if (::std::isnan(g) || ::std::isnan(m) || ::std::isnan(v) || ::std::isnan(ndw)) {
// 					__debugbreak();
// 				}
				*pdW++ = ndw;
			}
			//FFFUUUUUUUUCK! for() cycle (commented out below) works about 4-5 times slower, than while().
			// Don't know why did I choose to use for() for an RMSProp_*() implementation.
			// #TODO refactor & check all related functions, esp RMSProp_*() family.
			// 
			/*const auto pdW = dW.data(), pMt = Mt.data(), pVt = Vt.data();
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) {
				const auto pG = pdW + i, pM = pMt + i, pV = pVt + i;
				const auto g = *pG;
				const auto m = (*pM)*beta1 + g*ombeta1;
				*pM = m;
				const auto v = (*pV)*beta2 + g*g*ombeta2;
				*pV = v;
				*pG = alphat*m / (::std::sqrt(v) + numericStabilizer);
			}*/
		}
		void Adam_mt(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Vt, real_t& beta1t, real_t& beta2t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer) noexcept
		{
			NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Vt.size());
			NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta1 && beta1 < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta2 && beta2 < real_t(1.));
			NNTL_ASSERT(real_t(0.) < numericStabilizer && numericStabilizer < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= beta1t && beta1t <= real_t(1.));
			NNTL_ASSERT(real_t(0.) <= beta2t && beta2t <= real_t(1.));
			beta1t *= beta1;
			beta2t *= beta2;
			NNTL_ASSERT(beta1t < real_t(1.));
			NNTL_ASSERT(beta2t < real_t(1.));
			m_threads.run([&dW, &Mt, &Vt, &beta1t, &beta2t, learningRate, beta1, beta2, numericStabilizer,this](const par_range_t& r) noexcept{
				get_self()._iAdam_st(dW, Mt, Vt, beta1t, beta2t, learningRate, beta1, beta2, numericStabilizer, elms_range(r), true);
			}, dW.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		// AdaMax - 1412.6980v8 implementation
		//on a first call beta1t must be initialized with 1
		void AdaMax(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer)noexcept
		{
			if (dW.numel() < Thresholds_t::AdaMax) {
				get_self().AdaMax_st(dW, Mt, Ut, beta1t, learningRate, beta1, beta2, numericStabilizer);
			} else get_self().AdaMax_mt(dW, Mt, Ut, beta1t, learningRate, beta1, beta2, numericStabilizer);
		}
		void AdaMax_st(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer, const elms_range*const pER = nullptr) noexcept
		{
			get_self()._iAdaMax_st(dW, Mt, Ut, beta1t, learningRate, beta1, beta2, numericStabilizer, pER ? *pER : elms_range(0, dW.numel()), !!pER);
		}
		static void _iAdaMax_st(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer, const elms_range& er, const bool bInsideMT = true) noexcept
		{
			NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Ut.size());
			NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta1 && beta1 < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta2 && beta2 < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= beta1t && beta1t <= real_t(1.));

			if (!bInsideMT) {//This means, we're running outside of _mt version
				beta1t *= beta1;
				NNTL_ASSERT(beta1t < real_t(1.));
			}
			const auto alphat = learningRate / (real_t(1.) - beta1t);
			const auto ombeta1 = real_t(1.) - beta1;

			auto pdW = dW.data() + er.elmBegin, pMt = Mt.data() + er.elmBegin, pUt = Ut.data() + er.elmBegin;
			const auto pDWE = pdW + er.totalElements();
			while (pdW != pDWE) {
				const auto g = *pdW;
				const auto m = (*pMt)*beta1 + g*ombeta1;
				*pMt++ = m;
				//const auto u = ::std::max({::std::abs(g),beta2*(*pUt)});
				const auto u = ::std::max(::std::abs(g), beta2*(*pUt));
				*pUt++ = u;
				*pdW++ = alphat*m / (u + numericStabilizer);
			}
		}
		void AdaMax_mt(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Ut, real_t& beta1t, const real_t learningRate,
			const real_t beta1, const real_t beta2, const real_t numericStabilizer) noexcept
		{
			NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Ut.size());
			NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta1 && beta1 < real_t(1.));
			NNTL_ASSERT(real_t(0.) < beta2 && beta2 < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= beta1t && beta1t <= real_t(1.));
			beta1t *= beta1;
			NNTL_ASSERT(beta1t < real_t(1.));
			m_threads.run([&dW, &Mt, &Ut, &beta1t, learningRate, beta1, beta2, numericStabilizer, this](const par_range_t& r) noexcept{
				get_self()._iAdaMax_st(dW, Mt, Ut, beta1t, learningRate, beta1, beta2, numericStabilizer, elms_range(r), true);
			}, dW.numel());
		}



		//////////////////////////////////////////////////////////////////////////
		// Radam - Reweighted Adaptive Moment Estimation (see https://github.com/tdozat/Optimization)
		// It is a further generalization of Nadam (Timothy Dozat, ICLR 2016, "Incorporating Nesterov Momentum into Adam")
		// that includes the Nadam as a special case (just pass (0) as gamma parameter).
		// 
		//on a first call mu_pow_t and eta_pow_t must be initialized with 1.
		void RNadam(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Nt, real_t& mu_pow_t, real_t& eta_pow_t
			, const real_t learningRate, const real_t mu, const real_t eta, const real_t gamma, const real_t numericStabilizer)noexcept
		{
			if (dW.numel() < Thresholds_t::RNadam) {
				get_self().RNadam_st(dW, Mt, Nt, mu_pow_t, eta_pow_t, learningRate, mu, eta, gamma, numericStabilizer);
			} else get_self().RNadam_mt(dW, Mt, Nt, mu_pow_t, eta_pow_t, learningRate, mu, eta, gamma, numericStabilizer);
		}
		void RNadam_st(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Nt, real_t& mu_pow_t, real_t& eta_pow_t
			, const real_t learningRate, const real_t mu, const real_t eta, const real_t gamma, const real_t numericStabilizer
			, const elms_range*const pER = nullptr) noexcept
		{
			get_self()._iRNadam_st(dW, Mt, Nt, mu_pow_t, eta_pow_t, learningRate, mu, eta, gamma, numericStabilizer, pER ? *pER : elms_range(0, dW.numel()), !!pER);
		}
		static void _iRNadam_st(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Nt, real_t& mu_pow_t, real_t& eta_pow_t
			, const real_t learningRate, const real_t mu, const real_t eta, const real_t gamma, const real_t numericStabilizer
			, const elms_range& er, const bool bInsideMT = true) noexcept
		{
			NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Nt.size());
			NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
			NNTL_ASSERT(real_t(0.) < mu && mu < real_t(1.));
			NNTL_ASSERT(real_t(0.) < eta && eta < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= gamma && gamma < real_t(1.));
			NNTL_ASSERT(real_t(0.) < numericStabilizer && numericStabilizer < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= mu_pow_t && mu_pow_t <= real_t(1.));
			NNTL_ASSERT(real_t(0.) <= eta_pow_t && eta_pow_t <= real_t(1.));

			if (!bInsideMT) {//This means, we're running outside of _mt version
				mu_pow_t *= mu;
				eta_pow_t *= eta;
				NNTL_ASSERT(mu_pow_t < real_t(1.));
				NNTL_ASSERT(eta_pow_t < real_t(1.));
			}

			const real_t mu_t = (mu - mu_pow_t) / (real_t(1.) - mu_pow_t), eta_t = (eta - eta_pow_t) / (real_t(1.) - eta_pow_t);

			const auto o_m_mu_t = real_t(1.) - mu_t, o_m_eta_t = real_t(1.) - eta_t;

			const bool bIsNadam = gamma == real_t(0);

			const real_t mHat_c_mt = bIsNadam ? mu*((real_t(1.) - mu_pow_t) / (real_t(1) - mu*mu_pow_t)) : (real_t(1.) - gamma);
			const real_t mHat_c_g = bIsNadam ? o_m_mu_t : gamma;

			auto pdW = dW.data() + er.elmBegin, pMt = Mt.data() + er.elmBegin, pNt = Nt.data() + er.elmBegin;
			const auto pDWE = pdW + er.totalElements();
			while (pdW != pDWE) {
				const auto g = *pdW;

				const auto n = (*pNt)*eta_t + (g*g)*o_m_eta_t;
				*pNt++ = n;
				const auto n_hat = ::std::sqrt(n) + numericStabilizer;

				const auto m = (*pMt)*mu_t + g*o_m_mu_t;
				*pMt++ = m;

				//m_hat = o_m_gamma*m_t + gamma*g
				const auto m_hat = mHat_c_mt*m + mHat_c_g*g;

				const auto ndw = learningRate*(m_hat / n_hat);
				
#if NNTL_DEBUGBREAK_ON_DENORMALS
				enable_denormals();
				if (::std::fpclassify(ndw) == FP_SUBNORMAL) {
					__debugbreak();
				}
				global_denormalized_floats_mode();
#endif

				*pdW++ = ndw;
			}
			//FFFUUUUUUUUCK! for() cycle (commented out below) works about 4-5 times slower, than while().
		}
		void RNadam_mt(realmtx_t& dW, realmtx_t& Mt, realmtx_t& Nt, real_t& mu_pow_t, real_t& eta_pow_t
			, const real_t learningRate, const real_t mu, const real_t eta, const real_t gamma, const real_t numericStabilizer) noexcept
		{
			NNTL_ASSERT(dW.size() == Mt.size() && dW.size() == Nt.size());
			NNTL_ASSERT(real_t(0.) < learningRate && learningRate < real_t(1.));
			NNTL_ASSERT(real_t(0.) < mu && mu < real_t(1.));
			NNTL_ASSERT(real_t(0.) < eta && eta < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= gamma && gamma < real_t(1.));
			NNTL_ASSERT(real_t(0.) < numericStabilizer && numericStabilizer < real_t(1.));
			NNTL_ASSERT(real_t(0.) <= mu_pow_t && mu_pow_t <= real_t(1.));
			NNTL_ASSERT(real_t(0.) <= eta_pow_t && eta_pow_t <= real_t(1.));
			mu_pow_t *= mu;
			eta_pow_t *= eta;
			NNTL_ASSERT(mu_pow_t < real_t(1.));
			NNTL_ASSERT(eta_pow_t < real_t(1.));
			m_threads.run([&dW, &Mt, &Nt, &mu_pow_t, &eta_pow_t, learningRate, mu, eta, gamma, numericStabilizer, this](const par_range_t& r) noexcept{
				get_self()._iRNadam_st(dW, Mt, Nt, mu_pow_t, eta_pow_t, learningRate, mu, eta, gamma, numericStabilizer, elms_range(r), true);
			}, dW.numel());
		}




		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// loss addendum functions
		// 
		//////////////////////////////////////////////////////////////////////////
		// deCov
		// returns how much internal temporarily memory must be available to the to calculate loss_deCov() and dLoss_deCov
		static numel_cnt_t loss_DeCov_needTempMem(const bool bWillDoTraining, const smatrix_td::mtx_size_t biggestMtx)noexcept {
			NNTL_UNREF(bWillDoTraining);
			// we'll need some temporary memory to compute covariation and derivative correctly.
			// In general, we'll need memory for:
			// - vector of colwise means of biggestMtx, size == biggestMtx.cols_no_bias() 
			// - de-mean'ed matrix of the same size as biggestMtx
			// - covariance matrix of size (biggestMtx.cols_no_bias(), biggestMtx.cols_no_bias()). Will be used after means-vector are freed
			return realmtx_t::sNumel(biggestMtx) + realmtx_t::sNumel(biggestMtx.second, biggestMtx.second);
		}
		// Implements DeCov regularizer from the paper "Reducing Overfitting in Deep Neural Networks by Decorrelating Representations", 2015, ArXiv:1511.06068
		// (similar to “Discovering Hidden Factors of Variation in Deep Networks”, ArXiv:1412.6583)
		// BTW, there's wrong derivative presented in "Reducing Overfitting...". Actual derivative must be 2 times greater, than printed in paper.
		// 
		// N=size(A,1);
		// actCnt = size(A, 2);
		// DM = A - mean(A);
		// C = DM'*DM./N;
		// L = (norm(C, 'fro'). ^ 2 - norm(diag(C)). ^ 2). / 2;
		template<bool bLowerTriangl, bool bNumStab>
		real_t loss_deCov(const realmtx_t& Vals)noexcept {
			NNTL_ASSERT(!Vals.emulatesBiases() || !Vals.isHoleyBiases() || !"Current deCov algorithm does not support holey biases!");
			//to support holey biases algorithm must ignore rows with zeroed biases. Looks like the easiest (and probably the
			//fastest to run) way to do it is to make a something like a Vals.clone_to_droping_holes(DeMeaned) and then proceed as normal
			NNTL_ASSERT(Vals.cols_no_bias() > 1);

			const auto valsNumelNoBias = Vals.numel_no_bias();
			real_t*const pDeMeaned = get_self()._istor_alloc(valsNumelNoBias);
			realmtx_t DeMeaned(pDeMeaned, Vals, realmtx_t::tag_noBias());
			NNTL_ASSERT(!DeMeaned.emulatesBiases());
			const auto _clone_result = Vals.clone_to_no_bias(DeMeaned);
			NNTL_ASSERT(_clone_result);

// 			real_t*const pVecMeans = get_self()._istor_alloc(DeMeaned.cols());
// 
// 			get_self().mcwMean<bNumStab>(DeMeaned, pVecMeans);
// 			get_self().mcwSub_ip(DeMeaned, pVecMeans);
// 
// 			get_self()._istor_free(pVecMeans, DeMeaned.cols());
 			get_self().mcwDeMean<bNumStab>(DeMeaned);

			const auto covMtxNumel = realmtx_t::sNumel(DeMeaned.cols(), DeMeaned.cols());
			real_t*const pCovMtx = get_self()._istor_alloc(covMtxNumel);
			realmtx_t CovMtx(pCovMtx, DeMeaned.cols(), DeMeaned.cols());

			get_self().mColumnsCov<bLowerTriangl>(DeMeaned, CovMtx);
			//sum over a single triangle of a symmetric matrix is a half smaller than the sum over the whole matrix (excluding the main diagonal)
			//therefore we shouldn't divide it by 2 to fit the formula.
			//const auto ret = get_self().ewSumSquaresTriang<bLowerTriangl, bNumStab>(CovMtx);

			const auto ret = static_cast<real_t>(static_cast<ext_real_t>(get_self().ewSumSquaresTriang<bLowerTriangl, bNumStab>(CovMtx))
				/ static_cast<ext_real_t>(Vals.cols_no_bias() - 1));
			
			// - and to the amount of active neurons
			//const auto ret = get_self().ewSumSquaresTriang<bLowerTriangl, bNumStab>(CovMtx) / (valsNumelNoBias * (DeMeaned.cols() - 1));

			get_self()._istor_free(pCovMtx, covMtxNumel);

			get_self()._istor_free(pDeMeaned, valsNumelNoBias);
			return ret;
		}

		template<bool bLowerTriangl, bool bNumStab>
		void dLoss_deCov(const realmtx_t& Vals, realmtx_t& dLossdVals, const real_t deCov_scale = real_t(1.0))noexcept {
			NNTL_ASSERT(Vals.cols_no_bias() == dLossdVals.cols() && Vals.rows() == dLossdVals.rows());

		#ifndef NNTL_DECOV_DONT_CARE_ON_HOLEY_BIASES
			NNTL_ASSERT(!Vals.emulatesBiases() || !Vals.isHoleyBiases() || !"Current deCov algorithm does not support holey biases!");
			//to support holey biases algorithm must ignore rows with zeroed biases. Looks like the easiest (and probably the
			//fastest to run) way to do it is to make a something like a Vals.clone_to_droping_holes(DeMeaned) and then proceed as normal
		#endif // !NNTL_DECOV_DONT_CARE_ON_HOLEY_BIASES

			NNTL_ASSERT(Vals.cols_no_bias() > 1);
			NNTL_ASSERT(!dLossdVals.emulatesBiases());

			const auto bCr = Vals.clone_to_no_bias(dLossdVals);
			NNTL_ASSERT(bCr);
			get_self().dLoss_deCov_ip<bLowerTriangl, bNumStab>(dLossdVals, deCov_scale);
		}

		// N=size(A,1);
		// actCnt = size(A, 2);
		// DM = A - mean(A);
		// C = DM'*DM./N;
		// dL = (DM*C - diag(C)'.*DM).*2./N;
		template<bool bLowerTriangl, bool bNumStab>
		void dLoss_deCov_ip(realmtx_t& Vals_dLossdVals, const real_t deCov_scale = real_t(1.0))noexcept {
			NNTL_ASSERT(!Vals_dLossdVals.emulatesBiases());

			const auto vRows = Vals_dLossdVals.rows(), vCols = Vals_dLossdVals.cols();

			const auto valsNumelNoBias = Vals_dLossdVals.numel();
			real_t*const pDeMeaned = get_self()._istor_alloc(valsNumelNoBias);
			realmtx_t DeMeaned(pDeMeaned, Vals_dLossdVals, realmtx_t::tag_noBias());
			NNTL_ASSERT(!DeMeaned.emulatesBiases());
			const auto _clone_result = Vals_dLossdVals.clone_to_no_bias(DeMeaned);
			NNTL_ASSERT(_clone_result);

			get_self().mcwDeMean<bNumStab>(DeMeaned);

			const auto covMtxNumel = realmtx_t::sNumel(vCols, vCols);
			real_t*const pCovMtx = get_self()._istor_alloc(covMtxNumel);
			realmtx_t CovMtx(pCovMtx, vCols, vCols);

			get_self().mColumnsCov<bLowerTriangl>(DeMeaned, CovMtx);
			
			DeMeaned.clone_to(Vals_dLossdVals);
			get_self().mcwMulDiag_ip(Vals_dLossdVals, CovMtx);

			//dL = (DM*C - diag(C)'.*DM).*2./N;
			
			//const real_t cmnScale = (real_t(2.)*deCov_scale) / static_cast<real_t>(vRows);
			
			const real_t cmnScale = static_cast<real_t>( (ext_real_t(2.)* static_cast<ext_real_t>(deCov_scale))
				/ static_cast<ext_real_t>(valsNumelNoBias - vRows));

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			enable_denormals();
			Vals_dLossdVals._breakWhenDenormal();
			CovMtx._breakWhenDenormal();
			DeMeaned._breakWhenDenormal();
			if (::std::fpclassify(cmnScale) == FP_SUBNORMAL) {
				__debugbreak();
			}
			global_denormalized_floats_mode();
#endif

			b_BLAS_t::symm(false, bLowerTriangl, vRows, vCols, cmnScale, CovMtx.data(), vCols
				, DeMeaned.data(), vRows, -cmnScale, Vals_dLossdVals.data(), vRows);

#if NNTL_DEBUGBREAK_ON_OPENBLAS_DENORMALS
			Vals_dLossdVals._breakWhenDenormal();
#endif

			get_self()._istor_free(pCovMtx, covMtxNumel);
			get_self()._istor_free(pDeMeaned, valsNumelNoBias);
		}

	};

	template <typename RealT, typename iThreadsT, typename ThresholdsT = _impl::MATHN_THR<RealT>>
	class MathN final : public _MathN<RealT, iThreadsT, ThresholdsT, MathN<RealT, iThreadsT, ThresholdsT>> {
	public:
		~MathN()noexcept {}
		MathN()noexcept : _MathN<RealT, iThreadsT, ThresholdsT, MathN<RealT, iThreadsT, ThresholdsT>>() {}
	};

}
}
