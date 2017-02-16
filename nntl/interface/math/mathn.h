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

	// this class uses some routines from OpenBLAS to implement _i_math
	template <typename RealT, typename iThreadsT, typename ThresholdsT, typename FinalPolymorphChild, typename bindingBlasT = b_OpenBLAS>
	class _MathN : public _SMath<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild>, public _i_math<RealT> {
	public:
		typedef _SMath<RealT, iThreadsT, ThresholdsT, FinalPolymorphChild> base_class_t;
		typedef bindingBlasT b_BLAS_t;

		using base_class_t::real_t;
		using base_class_t::realmtx_t;
		using base_class_t::realmtxdef_t;
		using base_class_t::numel_cnt_t;
		using base_class_t::vec_len_t;

		//TODO: probably don't need this assert
		static_assert(std::is_base_of<_impl::MATHN_THR<real_t>, Thresholds_t>::value, "Thresholds_t must be derived from _impl::MATHN_THR<real_t>");
				
		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		struct _mrw_SOFTMAXPARTS :public _mrwHlpr_rw_UpdVecElm {
			real_t* pNumerator;//(colmajor) matrix data
			const real_t*const pMax;//row vector

			real_t* pNum;
			const real_t* pMx;

			_mrw_SOFTMAXPARTS(const real_t*const _pMax, real_t*const _pNum)noexcept : pMax(_pMax), pNumerator(_pNum) {}

			template<_OperationType OpType, typename BaseT>
			std::enable_if_t<OpType == mrw_cw> op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				const auto numerator = std::exp(mtxElm - *(pMax + r));
				vecElm += numerator;
				*(pNumerator + r) = numerator;
			}

			template<_OperationType OpType, typename BaseT>
			std::enable_if_t<OpType == mrw_rw> op(const BaseT& mtxElm, BaseT& vecElm, const vec_len_t r, const vec_len_t c, const size_t mtxRows)noexcept {
				const auto numerator = std::exp(mtxElm - *pMx);
				vecElm += numerator;
				*pNum = numerator;
				pNum += mtxRows;
			}

			void beforeMainLoop(const vec_len_t colBegin, const vec_len_t mtxRows)noexcept {
				//adjusting matrix data pointer to the beginning of colBegin column
				pNumerator += realmtx_t::sNumel(mtxRows, colBegin);
			};

			void cw_afterInnerLoop(const size_t mtxRows)noexcept {
				//proceeding to next column
				pNumerator += mtxRows;
			};

			static constexpr vec_len_t rw_FirstColumnIdx = 0;
			template<typename VecBaseT, typename MtxBaseT>
			VecBaseT rw_beforeInnerLoop(VecBaseT& vecElm, MtxBaseT*& pFirstMtxElm, const size_t mtxRows
				, const vec_len_t colBegin, const vec_len_t r)noexcept
			{
				pNum = pNumerator + r;
				pMx = pMax + r;
				return VecBaseT(0.0);
			}
		};

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
		// pDenominator is a vector of size act.rows()*m_threads.workers_count() elements. First act.rows() elements will be filled with rowwise_sum_exp()
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
			_processMtx_rw(act, [&act, pMax, pDenominator, pNumerator, this](const rowcol_range& RCR) {
				get_self().softmax_parts_st(act, pMax, pDenominator, pNumerator, &RCR);
			});
		}
		//pDenominator must be able to contain at least sNumel(act.rows(), m_threads.workers_count()) elements!
		//On return first column of pDenominator will contain calculated softmax denominator values
		void softmax_parts_mt_cw(const realmtx_t& act, const real_t*const pMax, real_t*const pDenominator, real_t*const pNumerator)noexcept {
			_processMtx_cw(act, Thresholds_t::softmax_parts_mt_cw_ColsPerThread
				, [&act, pMax, pNumerator, this](const rowcol_range& RCR, real_t*const pVec)
			{
				get_self().softmax_parts_st(act, pMax, pVec, pNumerator, &RCR);
			}, [this](realmtx_t& fin) {
				get_self().mrwSum_ip(fin);
			}, pDenominator);
		}

		//////////////////////////////////////////////////////////////////////////
		// helper function that return the amount of temporary memory (in real_t) needed to process by softmax()
		// a matrix of size act.size()
		numel_cnt_t softmax_needTempMem(const realmtx_t& act)const noexcept {
			// to compute softmax we'll need a row to store rowwise_max(), at max m_threads.workers_count() rows for
			// rowwise_sum_exp()-denominator of softmax expression, and a
			// 			// whole matrix of exp(Aij - maxj) (numerator of softmax expression).
			return realmtx_t::sNumel(act.rows(), act.cols_no_bias() + 1 + m_threads.workers_count());
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
			const auto pTmp = get_self()._get_thread_temp_raw_storage(get_self().softmax_needTempMem(srcdest));
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

			if (bRestoreBiases) srcdest.restore_biases();
		}
		void softmax_mt(realmtxdef_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty() && srcdest.numel() > 0);
			const auto bRestoreBiases = srcdest.hide_biases();

			const auto rm = srcdest.rows();
			const auto pTmp = get_self()._get_thread_temp_raw_storage(get_self().softmax_needTempMem(srcdest));
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

			if (bRestoreBiases) srcdest.restore_biases();
		}



		//////////////////////////////////////////////////////////////////////////
		// ElementWise operations
		//////////////////////////////////////////////////////////////////////////
		//binarize elements of real-valued matrix according to their relaion to frac
		void ewBinarize_ip(realmtx_t& A, const real_t& frac, const real_t& lBnd = real_t(0.), const real_t& uBnd = real_t(1.))noexcept {
			if (A.numel() < Thresholds_t::ewBinarize_ip) {
				get_self().ewBinarize_ip_st(A, frac, lBnd, uBnd);
			} else get_self().ewBinarize_ip_mt(A, frac, lBnd, uBnd);
		}
		static void ewBinarize_ip_st(realmtx_t& A, const real_t frac, const real_t lBnd = real_t(0.), const real_t uBnd = real_t(1.))noexcept {
			auto pA = A.data();
			const auto pAE = pA + A.numel();
			while (pA != pAE) {
				const auto v = *pA;
				//NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				*pA++ = v > frac ? uBnd : lBnd;
			}
		}
		void ewBinarize_ip_mt(realmtx_t& A, const real_t frac, const real_t lBnd = real_t(0.), const real_t uBnd = real_t(1.))noexcept {
			auto pA = A.data();
			m_threads.run([pA, frac, lBnd, uBnd](const par_range_t& r) {
				auto p = pA + r.offset();
				const auto pAE = p + r.cnt();
				while (p != pAE) {
					const auto v = *p;
					//NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
					*p++ = v > frac ? uBnd : lBnd;
				}
			}, A.numel());
		}

		//#TODO finish refactoring; find out which algo is better; move to base SMath class
		/*struct _ew_BINARIZE_IP {
			const real_t frac;
			_ew_BINARIZE_IP(const real_t f)noexcept:frac(f) {}

			template<typename BaseT>
			void op(BaseT& elm)noexcept {
				static_assert(!std::is_const<BaseT>::value, "BaseT mustn't have a const specifier");
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
			if (A.numel() < Thresholds_t::ewBinarize) {
				get_self().ewBinarize_st(Dest, A, frac);
			} else get_self().ewBinarize_mt(Dest, A, frac);
		}
		template<typename BaseDestT>
		static void _iewBinarize_st(BaseDestT*const pD, const realmtx_t& A, const real_t frac, const elms_range& er)noexcept {
			const auto pA = A.data();
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i)  pD[i] = pA[i] > frac ? BaseDestT(1.0) : BaseDestT(0.0);
			//its significantly faster, than while (pA != pAE)  *pD++ = *pA++ > frac ? BaseDestT(1.0) : BaseDestT(0.0);
		}
		template<typename DestContainerT>
		static void ewBinarize_st(DestContainerT& Dest, const realmtx_t& A, const real_t frac, const elms_range*const pER = nullptr)noexcept {
			//NNTL_ASSERT(A.size() == Dest.size()); //not compatible with std vector
			_iewBinarize_st(Dest.data(), A, frac, pER ? *pER : elms_range(A));
		}
		template<typename DestContainerT>
		void ewBinarize_mt(DestContainerT& Dest, const realmtx_t& A, const real_t frac)noexcept {
			m_threads.run([&Dest, &A, frac, this](const par_range_t& pr) {
				get_self()._iewBinarize_st(Dest.data(), A, frac, elms_range(pr));
			}, A.numel());
		}



		//////////////////////////////////////////////////////////////////////////
		//extract rows with indexes specified by Contnr ridxs into dest.
		template<typename SeqIt>
		void mExtractRows(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept {
			if (dest.numel() < Thresholds_t::mExtractRows) {
				get_self().mExtractRows_st_naive(src, ridxsItBegin, ridxsCnt, dest);
			} else get_self().mExtractRows_mt_naive(src, ridxsItBegin, ridxsCnt, dest);
		}
		template<typename SeqIt>
		static void mExtractRows_st_naive(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			src.assert_storage_does_not_intersect(dest);
			static_assert(std::is_same<vec_len_t, SeqIt::value_type>::value, "Contnr type should contain vec_len_t data");

			const numel_cnt_t destRows = dest.rows(), srcRows = src.rows();
			NNTL_ASSERT(dest.cols() == src.cols() && destRows == ridxsCnt && ridxsCnt <= srcRows);

			//TODO: accessing data in sequential order could provide some performance gains. However
			//it requires the content of [ridxsItBegin,ridxsItBegin+ridxsCnt) to be sorted. Therefore, testing is required
			// to decide whether it's all worth it

			auto pSrc = src.data();
			auto pDest = dest.data();
			const auto pDestEnd = pDest + dest.numel();

			while (pDest != pDestEnd) {
				auto pRI = ridxsItBegin;
				auto destCur = pDest;
				pDest += destRows;
				const auto destEnd = destCur + ridxsCnt;
				while (destCur != destEnd) {
					*destCur++ = *(pSrc + *pRI++);
				}
				pSrc += srcRows;
			}
		}
		template<typename SeqIt>
		void mExtractRows_mt_naive(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			src.assert_storage_does_not_intersect(dest);
			static_assert(std::is_same<vec_len_t, SeqIt::value_type>::value, "Contnr type should contain vec_len_t data");
			NNTL_ASSERT(dest.cols() == src.cols() && dest.rows() == ridxsCnt && ridxsCnt <= src.rows());

			m_threads.run([&src, &dest, ridxsItBegin](const par_range_t& r) {
				const numel_cnt_t destRows = dest.rows(), srcRows = src.rows();
				const auto rOfs = r.offset();
				const auto rCnt = r.cnt();

				//TODO: accessing data in sequential order could provide some performance gains. However
				//it requires the content of [ridxsItBegin,ridxsItBegin+ridxsCnt) to be sorted. Therefore, testing is required
				// to decide whether it's all worth it

				auto pSrc = src.data();
				auto pDest = dest.data() + rOfs;
				const auto pDestEnd = pDest + dest.numel();//BUGBUG!! dest.numel() is a bug probably 
				auto pThreadRI = ridxsItBegin + rOfs;

				while (pDest != pDestEnd) {
					auto pRI = pThreadRI;
					auto destCur = pDest;// [c*destRows];
					pDest += destRows;
					const auto destEnd = destCur + rCnt;
					//const auto thisBeg = &pSrc[c*srcRows];
					while (destCur != destEnd) {
						*destCur++ = *(pSrc + *pRI++);
					}
					pSrc += srcRows;
				}
				/*for (numel_cnt_t c = 0; c < destCols; ++c) {
					auto pRI = pThreadRI;
					auto destCur = &pDest[c*destRows];
					const auto destEnd = destCur + rCnt;
					const auto thisBeg = &pSrc[c*srcRows];
					while (destCur != destEnd) {
						*destCur++ = *(thisBeg + *pRI++);
					}
				}*/
			}, ridxsCnt);
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// compute squared L2norm of each matrix A row into a vector pNormsVec: pNormsVec(i) = norm(A(i,:)) (rowwise sum of squares)
		// ATTENTION! At this moment pNormsVec parameter is used for a description purpose only! It MUST be the result of
		// the call to get_self()._get_thread_temp_raw_storage( A.rows()*m_threads.workers_count() ).
		// 
		void mrwL2NormSquared(const realmtx_t& A, real_t*const pNormsVec=nullptr)noexcept {
			NNTL_ASSERT(pNormsVec == nullptr || pNormsVec == get_self()._get_thread_temp_raw_storage(realmtx_t::sNumel(A.rows(), m_threads.workers_count())));
			return (A.cols() <= Thresholds_t::mrwL2NormSquared_mt_cw_ColsPerThread || A.numel() < Thresholds_t::mrwL2NormSquared)
				? get_self().mrwL2NormSquared_st(A)
				: get_self().mrwL2NormSquared_mt(A);
		}

		void mrwL2NormSquared_st(const realmtx_t& A, real_t*const pNormsVec=nullptr, const rowcol_range*const pRCR = nullptr)noexcept {
			NNTL_ASSERT(pNormsVec == nullptr || pNormsVec == get_self()._get_thread_temp_raw_storage(A.rows()));
			get_self()._imrwL2NormSquared_st(A, get_self()._get_thread_temp_raw_storage(A.rows()), pRCR ? *pRCR : rowcol_range(A));
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
		void mrwL2NormSquared_mt(const realmtx_t& A, real_t*const pNormsVec=nullptr)noexcept {
			NNTL_ASSERT(!A.empty());
			if (A.cols() <= Thresholds_t::mrwL2NormSquared_mt_cw_ColsPerThread) {
				get_self().mrwL2NormSquared_st(A, pNormsVec);
			} else {
				const auto pTmpStor = get_self()._get_thread_temp_raw_storage(realmtx_t::sNumel(A.rows(), m_threads.workers_count()));
				NNTL_ASSERT(pNormsVec == nullptr || pNormsVec == pTmpStor);

				_processMtx_cw(A, Thresholds_t::mrwL2NormSquared_mt_cw_ColsPerThread
					, [&A, this](const rowcol_range& RCR, real_t*const pVec)noexcept
				{
					get_self()._imrwL2NormSquared_st(A, pVec, RCR);
				},
					[this](realmtx_t& fin)noexcept
				{
					get_self().mrwSum_ip(fin);
				}, pTmpStor);
			}
		}

		
		//////////////////////////////////////////////////////////////////////////
		//#todo implement
		/*void apply_max_norm(realmtxdef_t& W, const real_t& maxL2NormSquared, const bool bNormIncludesBias)noexcept {

			NNTL_ASSERT(!W.empty() && maxL2NormSquared > real_t(0.));

			//������� ��������� ��������� ��� ����������� ����
			const auto mRows = W.rows();
			auto pRowsNorm = get_self()._get_thread_temp_raw_storage(mRows);
			//#todo _memset_rowrange()
			memset(pRowsNorm, 0, sizeof(*pRowsNorm)*mRows);

			//��������� �����
			if (!bNormIncludesBias) W.hide_last_col();
			//#todo - _st ������
			mrwL2NormSquared(W, pRowsNorm);
			if (!bNormIncludesBias) W.restore_last_col();

			//���������� �������������� �����������
		}

		//#todo reimplement using more generic and fast smath:: functions
		void apply_max_norm_st(realmtxdef_t& W, const real_t& maxL2NormSquared, const bool bNormIncludesBias)noexcept {
		}*/

		

		//////////////////////////////////////////////////////////////////////////
		// treat matrix as a set of row-vectors (matrices in col-major mode!). For each row-vector check, whether
		// its length/norm is not longer, than predefined value. If it's longer, than rescale vector to this max length
		// (for use in max-norm weights regularization)
		// #TODO reimplement as apply_max_norm()
		void mCheck_normalize_rows(realmtxdef_t& A, const real_t& maxL2NormSquared, const bool bNormIncludesBias)noexcept {
			if (A.numel() < Thresholds_t::mCheck_normalize_rows) {
				get_self().mCheck_normalize_rows_st(A, maxL2NormSquared, bNormIncludesBias);
			} else get_self().mCheck_normalize_rows_mt(A, maxL2NormSquared, bNormIncludesBias);
		}
		//static constexpr real_t sCheck_normalize_rows_MULT = real_t(32.0);
		void mCheck_normalize_rows_st(realmtxdef_t& A, const real_t& maxNormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty() && maxNormSquared > real_t(0.0));

			const auto mRows = A.rows();
			auto pTmp = get_self()._get_thread_temp_raw_storage(mRows);
			
			if (!bNormIncludesBias) A.hide_last_col();
			get_self().mrwL2NormSquared_st(A, pTmp);
			if (!bNormIncludesBias) A.restore_last_col();

			//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, 
			// that doesn't need.
			// Making newNorm slightly less, than maxNormSquared to make sure the result will be less than max norm.
			//const real_t newNorm = maxNormSquared - math::real_t_limits<real_t>::eps_lower_n(maxNormSquared, sCheck_normalize_rows_MULT);
			// removed. it's not a big deal if resulting norm will be slightly bigger
			const real_t newNorm = maxNormSquared;// -2 * sqrt(math::real_t_limits<real_t>::eps_lower(maxNormSquared));
			auto pCurNorm = pTmp;
			const auto pTmpE = pTmp + mRows;
			while (pCurNorm != pTmpE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = rowNorm > maxNormSquared ? sqrt(newNorm / rowNorm) : real_t(1.0);
			}

			//renormalize (multiply each rowvector to corresponding coefficient from pTmp)
			get_self().mrwMulByVec_st(A, pTmp);
		}
		//TODO: might be good to make separate _cw and _rw versions of this algo
		void mCheck_normalize_rows_mt(realmtxdef_t& A, const real_t& maxNormSquared, const bool bNormIncludesBias)noexcept {
			NNTL_ASSERT(!A.empty() && maxNormSquared > real_t(0.0));
						
			const auto mRows = A.rows();
			const auto pTmpStor = get_self()._get_thread_temp_raw_storage(realmtx_t::sNumel(mRows, m_threads.workers_count()));

			if (!bNormIncludesBias) A.hide_last_col();
			get_self().mrwL2NormSquared_mt(A, pTmpStor);
			if (!bNormIncludesBias) A.restore_last_col();

			// calc scaling coefficients
			const auto pRowNormE = pTmpStor + mRows;
			const real_t newNorm = maxNormSquared;// -2 * sqrt(math::real_t_limits<real_t>::eps_lower(maxNormSquared));
			auto pCurNorm = pTmpStor;
			while (pCurNorm != pRowNormE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = rowNorm > maxNormSquared ? sqrt(newNorm / rowNorm) : real_t(1.0);
			}

			// 3. multiplying
			get_self().mrwMulByVec(A, pTmpStor);
		}

		//////////////////////////////////////////////////////////////////////////
		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		template<typename Contnr>
		size_t vCountSame(const Contnr& A, const Contnr& B)noexcept {
			return get_self().vCountSame_st_naive(A, B);
			// 			if (A.size()<=50000) {
			// 				return vCountSame_st_naive(A, B);
			// 			}else return vCountSame_mt_naive(A, B);
		}
		template<typename Contnr>
		static size_t vCountSame_st_naive(const Contnr& A, const Contnr& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());

			size_t ret = 0;
			const auto dataCnt = A.size();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				//if (A[i] == B[i]) ret++;
				//ret += A[i] == B[i] ? 1 : 0;
				ret += size_t(A[i] == B[i]);
			}
			return ret;
		}
		template<typename Contnr>
		size_t vCountSame_mt_naive(const Contnr& A, const Contnr& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());

			auto pAc = &A[0];
			auto pBc = &B[0];
			real_t ret = m_threads.reduce([pAc, pBc](const par_range_t& r)->real_t {
				const auto ofs = r.offset();
				size_t ret = 0;
				const auto pA = &pAc[ofs];
				const auto pB = &pBc[ofs];
				const auto cnt = r.cnt();
				for (range_t i = 0; i < cnt; ++i) {
					//if (pA[i] == pB[i]) ret++;
					ret += size_t(pA[i] == pB[i]);
				}
				return static_cast<real_t>(ret);
			}, _reduce_final_sum, A.size());

			return static_cast<size_t>(ret);
		}

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
			boost::algorithm::clamp_range(p, p + m.numel(), p, lo, hi);
		}
		void evClamp_mt(realmtx_t& m, real_t lo, real_t hi)noexcept {
			NNTL_ASSERT(m.numel() > 0 && !m.empty());
			NNTL_ASSERT(lo < hi);

			auto ptr = m.data();
			m_threads.run([ptr, lo, hi](const par_range_t& r) {
				auto p = ptr + r.offset();
				//utils::boost::algorithm::clamp_range(p, p + r.cnt(), p, lo, hi);
				boost::algorithm::clamp_range(p, p + r.cnt(), p, lo, hi);
			}, m.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//on entry the dropoutMask must be filled with random values in range [0,1]
		//Function binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// dropPercAct - probability of keeping unit active
		// act must be used in "no_bias" mode.
		// Actually, the function must implement so called "inverted Dropout", see http://cs231n.github.io/neural-networks-2/
		void make_dropout(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask)noexcept {
			if (act.numel_no_bias() < Thresholds_t::make_dropout) {
				get_self().make_dropout_st(act, dropPercAct, dropoutMask);
			} else get_self().make_dropout_mt(act, dropPercAct, dropoutMask);
		}
		void make_dropout_st(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask, const elms_range*const pER = nullptr) const noexcept {
			get_self()._imake_dropout_st(act, dropPercAct, dropoutMask, pER ? *pER : elms_range(0, dropoutMask.numel()));
		}
		static void _imake_dropout_st(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask, const elms_range& er) noexcept {
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);

			const real_t dropPercActInv = real_t(1.) / dropPercAct;
			auto pDM = dropoutMask.data()+er.elmBegin;
			const auto pDME = pDM + er.totalElements();
			while (pDM != pDME) {
				const auto v = *pDM;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				*pDM++ = v < dropPercAct ? dropPercActInv : real_t(0.);
			}

			const auto pA = act.data();
			pDM = dropoutMask.data();
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) pA[i] *= pDM[i];
		}
		void make_dropout_mt(realmtx_t& act, const real_t dropPercAct, realmtx_t& dropoutMask)noexcept {
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dropPercAct > 0 && dropPercAct < 1);
			m_threads.run([&act, &dropoutMask, dropPercAct,this](const par_range_t& r) {
				get_self()._imake_dropout_st(act, dropPercAct, dropoutMask, elms_range(r));
			}, dropoutMask.numel());
		}
		
		//////////////////////////////////////////////////////////////////////////
		//apply individual learning rate to dLdW
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
			m_threads.run([pdW, prevdW, pGain, decr, incr, capLow, capHigh](const par_range_t& r) {
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
			auto pCond = get_self()._get_thread_temp_raw_storage(dataCnt);

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

			for (numel_cnt_t i = 0; i < dataCnt; ++i) pdW[i] *= pGain[i];
		}

		void apply_ILR_mt_vec(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			NNTL_ASSERT(dLdW.size() == prevdLdW.size() && dLdW.size() == ILRGain.size());
			NNTL_ASSERT(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

			//TODO: probably not the most efficient implementation

			const auto dataCnt = dLdW.numel();
			const auto pTmpMem = get_self()._get_thread_temp_raw_storage(dataCnt);
			const auto pdW = dLdW.data(), pGain = ILRGain.data();
			const auto prevdW = prevdLdW.data();

			m_threads.run([pdW, prevdW, pGain, decr, incr, capLow, capHigh, pTmpMem](const par_range_t& r) {
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
			m_threads.run([pV, pdW, momentum](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					pV[i] = momentum*pV[i] + pdW[i];
				}
			}, vW.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = b.*A
		void evMulC_ip(realmtx_t& A, const real_t b)noexcept {
			if (A.numel() < Thresholds_t::evMulC_ip) {
				get_self().evMulC_ip_st_naive(A, b);
			} else get_self().evMulC_ip_mt_naive(A, b);
		}
		void evMulC_ip_st_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			get_self().ievMulC_ip_st_naive(A.data(), A.numel(), b);
		}
		static void ievMulC_ip_st_naive(real_t* ptrA, const numel_cnt_t dataCnt, const real_t b)noexcept {
			const auto ptrAE = ptrA + dataCnt;
			while (ptrA != ptrAE)  *ptrA++ *= b;
		}
		void evMulC_ip_mt_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel() > 0);
			get_self().ievMulC_ip_mt_naive(A.data(), A.numel(), b);
		}
		void ievMulC_ip_mt_naive(real_t* ptrA, const numel_cnt_t dataCnt, const real_t b)noexcept {
			m_threads.run([ptrA, b](const par_range_t& r) {
				auto p = ptrA + r.offset();
				const auto pe = p + r.cnt();
				while (p != pe) {
					*p++ *= b;
				}
			}, dataCnt);
		}

		//inplace elementwise multiplication A(no_bias) = b.*A(no_bias)
		void evMulC_ip_Anb(realmtx_t& A, const real_t b)noexcept {
			if (A.numel_no_bias() < Thresholds_t::evMulC_ip_Anb) {
				get_self().evMulC_ip_Anb_st_naive(A, b);
			} else get_self().evMulC_ip_Anb_mt_naive(A, b);
		}
		void evMulC_ip_Anb_st_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
			get_self().ievMulC_ip_st_naive(A.data(), A.numel_no_bias(), b);
		}
		void evMulC_ip_Anb_mt_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
			get_self().ievMulC_ip_mt_naive(A.data(), A.numel_no_bias(), b);
		}


		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = A.*B
		void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evMul_ip) {
				get_self().evMul_ip_st_naive(A, B);
			} else get_self().evMul_ip_mt_naive(A, B);
		}
		void evMul_ip_st_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size() == B.size());
			get_self().ievMul_ip_st_naive(A.data(), B.data(), A.numel());
		}
		static void ievMul_ip_st_naive(real_t* ptrA, const real_t*ptrB, numel_cnt_t dataCnt) noexcept {
			//for (numel_cnt_t i = 0; i < dataCnt; ++i) ptrA[i] *= ptrB[i];
			const bool bOdd = dataCnt & 1;
			if (bOdd) --dataCnt;
			for (numel_cnt_t i = 0; i < dataCnt; i += 2) {
				ptrA[i] *= ptrB[i];
				const auto j = i + 1;
				ptrA[j] *= ptrB[j];
			}
			if (bOdd) ptrA[dataCnt] *= ptrB[dataCnt];
		}
		void evMul_ip_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size() == B.size());
			get_self().ievMul_ip_mt_naive(A.data(), B.data(), A.numel());
		}
		void ievMul_ip_mt_naive(real_t* ptrA, const real_t*ptrB, numel_cnt_t dataCnt) noexcept {
			m_threads.run([ptrA, ptrB](const par_range_t& r) {
				const auto ofs = r.offset();
				auto cnt = r.cnt();
				const bool bOdd = cnt & 1;
				if (bOdd) --cnt;
				const auto im = ofs + cnt;
				//for (numel_cnt_t i = ofs; i < im; ++i) ptrA[i] *= ptrB[i];
				for (numel_cnt_t i = ofs; i < im; i += 2) {
					ptrA[i] *= ptrB[i];
					const auto j = i + 1;
					ptrA[j] *= ptrB[j];
				}
				if (bOdd) ptrA[im] *= ptrB[im];
			}, dataCnt);
		}


		//inplace elementwise multiplication A(no_bias) = A(no_bias).*B, - A is taken in no_bias mode
		void evMul_ip_Anb(realmtx_t& A, const realmtx_t& B)noexcept {
			const auto dataCnt = B.numel();
			if (dataCnt < Thresholds_t::evMul_ip_Anb) {
				get_self().evMul_ip_Anb_st_naive(A, B);
			} else get_self().evMul_ip_Anb_mt_naive(A, B);
		}
		void evMul_ip_Anb_st_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size_no_bias() == B.size());
			get_self().ievMul_ip_st_naive(A.data(), B.data(), B.numel());
		}
		void evMul_ip_Anb_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size_no_bias() == B.size());
			get_self().ievMul_ip_mt_naive(A.data(), B.data(), B.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition *pA = *pA + *pB
		void vAdd_ip(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt)noexcept {
			NNTL_ASSERT(pA && pB && dataCnt);
			if (dataCnt < Thresholds_t::evAdd_ip) {
				get_self().vAdd_ip_st(pA, pB, dataCnt);
			} else get_self().vAdd_ip_mt(pA, pB, dataCnt);
		}
		static void _ivAdd_ip_st(real_t*const pA, const real_t*const pB, const elms_range& er)noexcept {
			NNTL_ASSERT(pA && pB);
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) pA[i] += pB[i];
		}
		void vAdd_ip_st(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt, const elms_range*const pER=nullptr)noexcept {
			NNTL_ASSERT(pA && pB && dataCnt);
			_ivAdd_ip_st(pA, pB, pER ? *pER : elms_range(0, dataCnt));
		}
		void vAdd_ip_mt(real_t*const pA, const real_t*const pB, const numel_cnt_t dataCnt)noexcept {
			NNTL_ASSERT(pA && pB && dataCnt);
			m_threads.run([pA, pB](const par_range_t& pr) {
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
			m_threads.run([&A, &B](const par_range_t& pr) {
				_ivAdd_ip_st(A.data(), B.data(), elms_range(pr));
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise adding of scaled vector: A = A + c*B;
		void evAddScaled_ip(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evAddScaled_ip) {
				get_self().evAddScaled_ip_st(A, c, B);
			} else get_self().evAddScaled_ip_mt(A, c, B);
		}
		static void evAddScaled_ip_st(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			const auto pA = A.data();
			const auto dataCnt = A.numel();
			const auto pB = B.data();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += c*pB[i];
		}
		void evAddScaled_ip_mt(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			const auto pA = A.data();
			const auto pB = B.data();
			m_threads.run([pA, pB, c](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) pA[i] += c*pB[i];
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
		static void evAddScaledSign_ip_st(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			const auto pA = A.data();
			const auto dataCnt = A.numel();
			const auto pB = B.data();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += c*math::sign(pB[i]);
		}
		void evAddScaledSign_ip_mt(realmtx_t& A, const real_t c, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty() && c != real_t(0.0));
			const auto pA = A.data();
			const auto pB = B.data();
			m_threads.run([pA, pB, c](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) pA[i] += c*math::sign(pB[i]);
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise subtraction A = A-B
		void evSub_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evSub_ip) {
				get_self().evSub_ip_st_naive(A, B);
			} else get_self().evSub_ip_mt_naive(A, B);
		}
		static void evSub_ip_st_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());
			const auto dataCnt = A.numel();
			const auto pA = A.data();
			const auto pB = B.data();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] -= pB[i];
		}
		void evSub_ip_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());
			const auto pA = A.data();
			const auto pB = B.data();
			m_threads.run([pA, pB](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) pA[i] -= pB[i];
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
			m_threads.run([pA, pB, pC](const par_range_t& r) {
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
			m_threads.run([pVf, pWf, momentum](const par_range_t& r) {
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
			m_threads.run([pS, pD](const par_range_t& r) {
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
		real_t vSumSquares(const realmtx_t& A)noexcept {
			if (A.numel() < Thresholds_t::vSumSquares) {
				return get_self().vSumSquares_st(A);
			} else return get_self().vSumSquares_mt(A);
		}
		static real_t vSumSquares_st(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			real_t ret(0.0);
			auto p = A.data();
			const auto pE = p + A.numel();
			while (p != pE) {
				const auto v = *p++;
				ret += v*v;
			}
			return ret;
		}
		real_t vSumSquares_mt(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			const auto pA = A.data();
			return m_threads.reduce([pA](const par_range_t& r)->real_t {
				real_t ret(0.0);
				auto p = pA + r.offset();
				const auto pE = p + r.cnt();
				while (p != pE) {
					const auto v = *p++;
					ret += v*v;
				}
				return ret;
			}, _reduce_final_sum, A.numel());
		}

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
			for (numel_cnt_t i = 0; i < dataCnt; ++i)  pD[i] = std::abs(pS[i]);
		}
		void evAbs_mt(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.data();
			auto pD = dest.data();
			m_threads.run([pS, pD](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i)  pD[i] = std::abs(pS[i]);
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
			while (p != pE) ret += std::abs(*p++);
			return ret;
		}
		real_t vSumAbs_mt(const realmtx_t& A)noexcept {
			NNTL_ASSERT(!A.empty());

			const auto pA = A.data();
			return m_threads.reduce([pA](const par_range_t& r)->real_t {
				real_t ret(0.0);
				auto p = pA + r.offset();
				const auto pE = p + r.cnt();
				while (p != pE) ret += std::abs(*p++);
				return ret;
			}, _reduce_final_sum, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//C = A * B, - matrix multiplication
		static void mMulAB_C(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			NNTL_ASSERT(acols == B.rows() && A.rows() == C.rows() && B.cols() == C.cols());

			b_BLAS_t::gemm(false, false, A.rows(), C.cols(), acols, real_t(1.0), A.data(), A.rows(), B.data(), B.rows(),
				real_t(0.0), C.data(), C.rows());
		}
		//////////////////////////////////////////////////////////////////////////
		//matrix multiplication C(no bias) = A * B` (B transposed). C could have emulated biases (they will be left untouched)
		static void mMulABt_Cnb(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto ccols = C.cols_no_bias();
			NNTL_ASSERT(A.cols() == B.cols() && A.rows() == C.rows() && B.rows() == ccols);

			b_BLAS_t::gemm(false, true, A.rows(), ccols, A.cols(), real_t(1.0), A.data(), A.rows(), B.data(), ccols,
				real_t(0.0), C.data(), C.rows());
		}
		//////////////////////////////////////////////////////////////////////////
		//C = a*(A` * B) - matrix multiplication of transposed A times B with result normalization
		static void mScaledMulAtB_C(real_t alpha, const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			const auto arows = A.rows();
			NNTL_ASSERT(arows == B.rows() && acols == C.rows() && B.cols() == C.cols());

			b_BLAS_t::gemm(true, false, acols, B.cols(), arows, alpha, A.data(), arows, B.data(), arows,
				real_t(0.0), C.data(), acols);
		}


		//////////////////////////////////////////////////////////////////////////
		//Elements of SVD (singular value decomposition)
		// mSVD_Orthogonalize_ss(A) performs SVD of m*n matrix A and returns in A same sized corresponding orthogonal matrix of singular vectors
		//		returns true if SVD was successful
		//		Restrictions: MUST NOT use the math object's local storage (the function is intended to be used during
		//			the weight initialization phase when the local storage is generally not initialized yet)
		bool mSVD_Orthogonalize_ss(realmtx_t& A)noexcept {
			const auto m = A.rows(), n = A.cols();
			const bool bGetU = m >= n;
			const auto minmn = bGetU ? n : m;

			std::vector<real_t> S(2 * minmn);

			const auto r = b_BLAS_t::gesvd(bGetU ? 'O' : 'N', bGetU ? 'N' : 'O', m, n
				, A.data(), m, &S[0], static_cast<real_t*>(nullptr), m, static_cast<real_t*>(nullptr), n, &S[minmn]);

			NNTL_ASSERT(0 == r || !"b_BLAS_t::gesvd failed!");
			NNTL_ASSERT(get_self()._mIsOrthogonal(A, bGetU) || !"SVD returned non orthogonal matrix!");
			return 0 == r;
		}

		template<typename base_t> struct _mIsOrthogonal_defEps {};
		template<> struct _mIsOrthogonal_defEps<double> { static constexpr double eps = 1e-11; };
		template<> struct _mIsOrthogonal_defEps<float> { static constexpr float eps = 1e-5f; };
		// This function checks whether A is orthogonal, i.e. A'*A is identity matrix.
		// Not optimized, FOR DEBUG PURPOSES ONLY!
		static bool _mIsOrthogonal(const realmtx_t& A,  bool bFirstTransposed = true, const real_t epsV = _mIsOrthogonal_defEps<real_t>::eps)noexcept {
			NNTL_ASSERT(!A.empty());
			const vec_len_t opArows = bFirstTransposed ? A.cols() : A.rows()
				, opAcols = bFirstTransposed ? A.rows() : A.cols()
				, ldab = bFirstTransposed ? opAcols : opArows;
			realmtx_t ICand(opArows, opArows);

			b_BLAS_t::gemm(bFirstTransposed, !bFirstTransposed, opArows, opArows, opAcols
				, real_t(1.), A.data(), ldab, A.data(), ldab,
				real_t(0.0), ICand.data(), opArows);

			bool r = true;
			for (vec_len_t ri = 0; ri < opArows; ++ri) {
				for (vec_len_t ci = 0; ci < opArows; ++ci) {
					if ( std::abs(real_t(ri==ci) - ICand.get(ri,ci)) > epsV  ) {
						r = false;
						break;
					}
				}
				if (!r) break;
			}
			return r;
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
// 			for (range_t i = er.elmBegin; i < er.elmEnd; ++i) ptr[i] = real_t(1.0) / (real_t(1.0) + std::exp(-ptr[i]));
			auto pA = srcdest.data() + er.elmBegin;
			const auto pAE = pA + er.totalElements();
			while (pA != pAE) {
				const auto x = *pA;
				*pA++ = real_t(1.0) / (real_t(1.0) + std::exp(-x));

			}
		}
		void sigm_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& pr) {
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
			m_threads.run([&f_df, this](const par_range_t& pr) {
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
			m_threads.run([&data_y, &act_dLdZ, this](const par_range_t& r) {
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
			m_threads.run([&srcdest,this](const par_range_t& r) {
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
			m_threads.run([&f_df,this](const par_range_t& r) {
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
			m_threads.run([&srcdest, leak, this](const par_range_t& r) {
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
			m_threads.run([&f_df, leak,this](const par_range_t& r) {
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
				/*if (v < real_t(+0.0)) *pV = (std::exp(v) - real_t(1.0))*alpha;
				++pV;*/
				//*pV++ = v < real_t(0.) ? (std::exp(v) - real_t(1.0))*alpha : v;
				*pV++ = v < real_t(0.) ? math::expm1(v)*alpha : v;
			}
		}
		void elu_mt(realmtx_t& srcdest, const real_t alpha) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, alpha, this](const par_range_t& r) {
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
			m_threads.run([&f_df, alpha, this](const par_range_t& r) {
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
				/*if (v < real_t(+0.0)) *pV = (std::exp(v) - real_t(1.0));
				++pV;*/
				//*pV++ = v < real_t(0.0) ? (std::exp(v) - real_t(1.0)) : v;
				*pV++ = v < real_t(0.0) ? math::expm1(v) : v;
			}
		}
		void elu_unitalpha_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& r) {
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
			m_threads.run([&f_df, this](const par_range_t& r) {
				get_self()._idelu_unitalpha_st(f_df, elms_range(r));
			}, f_df.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ELogU : alpha*(exp(x)-1) | x<0,    log(x+1)/log(b) | x>0
		void elogu(realmtx_t& srcdest, const real_t& alpha, const real_t& b) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu) {
				get_self().elogu_st(srcdest, alpha, b);
			} else get_self().elogu_mt(srcdest, alpha, b);
		}
		void elogu_st(realmtx_t& srcdest, const real_t& alpha, const real_t& b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_st(srcdest, alpha, b, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_st(realmtx_t& srcdest, const real_t& alpha, const real_t& b, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			const real_t lbi = real_t(1.) / std::log(b);
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (std::exp(v) - real_t(1.))*alpha : log(v + real_t(1.))*lbi;
				*pV++ = v < real_t(0.0) ? math::expm1(v)*alpha : math::log1p(v)*lbi;
			}
		}
		void elogu_mt(realmtx_t& srcdest, const real_t& alpha, const real_t& b) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&srcdest, &alpha, &b, this](const par_range_t& r) {
				get_self()._ielogu_st(srcdest, alpha, b, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu(realmtx_t& f_df, const real_t& alpha, const real_t& b) noexcept {
			if (f_df.numel() < Thresholds_t::delogu) {
				get_self().delogu_st(f_df, alpha, b);
			} else get_self().delogu_mt(f_df, alpha, b);
		}
		void delogu_st(realmtx_t& f_df, const real_t& alpha, const real_t& b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_st(f_df, alpha, b, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_st(realmtx_t& f_df, const real_t& alpha, const real_t& b, const elms_range& er) noexcept {
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());

			const ext_real_t _lb = std::log(ext_real_t(b));
			const real_t nllb = -static_cast<real_t>(std::log(_lb)), nlb = -static_cast<real_t>(_lb);

			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + alpha) : std::exp(v*nlb + nllb);
			}
		}
		void delogu_mt(realmtx_t& f_df, const real_t& alpha, const real_t& b) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&f_df, &alpha, &b, this](const par_range_t& r) {
				get_self()._idelogu_st(f_df, alpha, b, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// ELogU with alpha==1.
		void elogu_ua(realmtx_t& srcdest, const real_t& b) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu_ua) {
				get_self().elogu_ua_st(srcdest, b);
			} else get_self().elogu_ua_mt(srcdest, b);
		}
		void elogu_ua_st(realmtx_t& srcdest, const real_t& b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_ua_st(srcdest, b, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_ua_st(realmtx_t& srcdest, const real_t& b, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b > real_t(1.0));
			const real_t lbi = real_t(1.) / std::log(b);
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (std::exp(v) - real_t(1.)) : log(v + real_t(1.))*lbi;
				*pV++ = v < real_t(0.0) ? math::expm1(v) : math::log1p(v)*lbi;
			}
		}
		void elogu_ua_mt(realmtx_t& srcdest, const real_t& b) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&srcdest, &b, this](const par_range_t& r) {
				get_self()._ielogu_ua_st(srcdest, b, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu_ua(realmtx_t& f_df, const real_t& b) noexcept {
			if (f_df.numel() < Thresholds_t::delogu_ua) {
				get_self().delogu_ua_st(f_df, b);
			} else get_self().delogu_ua_mt(f_df, b);
		}
		void delogu_ua_st(realmtx_t& f_df, const real_t& b, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_ua_st(f_df, b, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_ua_st(realmtx_t& f_df, const real_t& b, const elms_range& er) noexcept {
			NNTL_ASSERT(b > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());

			const ext_real_t _lb = std::log(ext_real_t(b));
			const real_t nllb = -static_cast<real_t>(std::log(_lb)), nlb = -static_cast<real_t>(_lb);

			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + real_t(1.)) : std::exp(v*nlb + nllb);
			}
		}
		void delogu_ua_mt(realmtx_t& f_df, const real_t& b) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b > real_t(1.0));
			m_threads.run([&f_df, &b, this](const par_range_t& r) {
				get_self()._idelogu_ua_st(f_df, b, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//ELogU with natural base, b==exp(1)
		void elogu_nb(realmtx_t& srcdest, const real_t& alpha) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::elogu_nb) {
				get_self().elogu_nb_st(srcdest, alpha);
			} else get_self().elogu_nb_mt(srcdest, alpha);
		}
		void elogu_nb_st(realmtx_t& srcdest, const real_t& alpha, const elms_range*const pER = nullptr) const noexcept {
			get_self()._ielogu_nb_st(srcdest, alpha, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _ielogu_nb_st(realmtx_t& srcdest, const real_t& alpha, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//*pV++ = v < real_t(0.0) ? (std::exp(v) - real_t(1.))*alpha : log(v + real_t(1.));
				*pV++ = v < real_t(0.0) ? math::expm1(v)*alpha : math::log1p(v);
			}
		}
		void elogu_nb_mt(realmtx_t& srcdest, const real_t& alpha) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			m_threads.run([&srcdest, &alpha, this](const par_range_t& r) {
				get_self()._ielogu_nb_st(srcdest, alpha, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(-y*log(b)-log(log(b))) | x>0
		void delogu_nb(realmtx_t& f_df, const real_t& alpha) noexcept {
			if (f_df.numel() < Thresholds_t::delogu_nb) {
				get_self().delogu_nb_st(f_df, alpha);
			} else get_self().delogu_nb_mt(f_df, alpha);
		}
		void delogu_nb_st(realmtx_t& f_df, const real_t& alpha, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idelogu_nb_st(f_df, alpha, pER ? *pER : elms_range(f_df));
		}
		static void _idelogu_nb_st(realmtx_t& f_df, const real_t& alpha, const elms_range& er) noexcept {
			NNTL_ASSERT(alpha > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());

			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = v < real_t(0.) ? (v + alpha) : std::exp(-v);
			}
		}
		void delogu_nb_mt(realmtx_t& f_df, const real_t& alpha) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(alpha > real_t(0.0));
			m_threads.run([&f_df, &alpha, this](const par_range_t& r) {
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
				//*pV++ = v < real_t(0.0) ? (std::exp(v) - real_t(1.)) : log(v + real_t(1.));
				*pV++ = v < real_t(0.0) ? math::expm1(v) : math::log1p(v);
			}
		}
		void elogu_ua_nb_mt(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			m_threads.run([&srcdest, this](const par_range_t& r) {
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
				*ptrDF++ = v < real_t(0.) ? (v + real_t(1.)) : std::exp(-v);
			}
		}
		void delogu_ua_nb_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) {
				get_self()._idelogu_ua_nb_st(f_df, elms_range(r));
			}, f_df.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//LogLogU : -log(1-x)/log(b_neg) | x<0,   log(x+1)/log(b_pos) | x>0
		void loglogu(realmtx_t& srcdest, const real_t& b_neg, const real_t& b_pos) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu) {
				get_self().loglogu_st(srcdest, b_neg, b_pos);
			} else get_self().loglogu_mt(srcdest, b_neg, b_pos);
		}
		void loglogu_st(realmtx_t& srcdest, const real_t& b_neg, const real_t& b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_st(srcdest, b_neg, b_pos, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_st(realmtx_t& srcdest, const real_t& b_neg, const real_t& b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			const real_t lbposi = real_t(ext_real_t(1.) / std::log(ext_real_t(b_pos)))
				, nlbnegi = real_t(ext_real_t (-1.) / std::log(ext_real_t(b_neg)));
			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				//const auto isNeg = v < real_t(0.0);
// 				const auto lv = isNeg ? (real_t(1.) - v) : (v + real_t(1.));
// 				const auto bv = isNeg ? nlbnegi : lbposi;
// 				*pV++ = bv*std::log(lv);

				//*pV++ = (isNeg ? nlbnegi : lbposi)*math::log1p(isNeg ? -v : v);
				*pV++ = (v < real_t(0.0) ? nlbnegi : lbposi)*math::log1p(std::fabs(v));//a bit faster
			}
		}
		void loglogu_mt(realmtx_t& srcdest, const real_t& b_neg, const real_t& b_pos) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&srcdest, &b_neg, &b_pos, this](const par_range_t& r) {
				get_self()._iloglogu_st(srcdest, b_neg, b_pos, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu(realmtx_t& f_df, const real_t& b_neg, const real_t& b_pos) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu) {
				get_self().dloglogu_st(f_df, b_neg, b_pos);
			} else get_self().dloglogu_mt(f_df, b_neg, b_pos);
		}
		void dloglogu_st(realmtx_t& f_df, const real_t& b_neg, const real_t& b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_st(f_df, b_neg, b_pos, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_st(realmtx_t& f_df, const real_t& b_neg, const real_t& b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());
			const ext_real_t _lbpos = std::log(ext_real_t(b_pos)), _lbneg = std::log(ext_real_t(b_neg));
			const real_t nllbpos = -static_cast<real_t>(std::log(_lbpos)), nlbpos = -static_cast<real_t>(_lbpos);
			const real_t nllbneg = -static_cast<real_t>(std::log(_lbneg)), lbneg = static_cast<real_t>(_lbneg);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = std::exp(v < real_t(0.) ? (v*lbneg + nllbneg) : (v*nlbpos + nllbpos));
			}
		}
		void dloglogu_mt(realmtx_t& f_df, const real_t& b_neg, const real_t& b_pos) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&f_df, &b_neg, &b_pos, this](const par_range_t& r) {
				get_self()._idloglogu_st(f_df, b_neg, b_pos, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//#TODO code should be improved. And it's slower than a generic version.
		void loglogu_nbn(realmtx_t& srcdest, const real_t& b_pos) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu_nbn) {
				get_self().loglogu_nbn_st(srcdest, b_pos);
			} else get_self().loglogu_nbn_mt(srcdest, b_pos);
		}
		void loglogu_nbn_st(realmtx_t& srcdest, const real_t& b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_nbn_st(srcdest, b_pos, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_nbn_st(realmtx_t& srcdest, const real_t& b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_pos > real_t(1.0));
			const real_t lbposi = real_t(ext_real_t(1.) / std::log(ext_real_t(b_pos)));
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
		void loglogu_nbn_mt(realmtx_t& srcdest, const real_t& b_pos) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&srcdest, &b_pos, this](const par_range_t& r) {
				get_self()._iloglogu_nbn_st(srcdest, b_pos, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu_nbn(realmtx_t& f_df, const real_t& b_pos) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu_nbn) {
				get_self().dloglogu_nbn_st(f_df, b_pos);
			} else get_self().dloglogu_nbn_mt(f_df, b_pos);
		}
		void dloglogu_nbn_st(realmtx_t& f_df, const real_t& b_pos, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_nbn_st(f_df, b_pos, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_nbn_st(realmtx_t& f_df, const real_t& b_pos, const elms_range& er) noexcept {
			NNTL_ASSERT(b_pos > real_t(1.0));
			NNTL_ASSERT(!f_df.empty());
			const ext_real_t _lbpos = std::log(ext_real_t(b_pos));
			const real_t nllbpos = -static_cast<real_t>(std::log(_lbpos)), nlbpos = -static_cast<real_t>(_lbpos);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = std::exp(v < real_t(0.) ? v : (v*nlbpos + nllbpos));
			}
		}
		void dloglogu_nbn_mt(realmtx_t& f_df, const real_t& b_pos) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b_pos > real_t(1.0));
			m_threads.run([&f_df, &b_pos, this](const par_range_t& r) {
				get_self()._idloglogu_nbn_st(f_df, b_pos, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		void loglogu_nbp(realmtx_t& srcdest, const real_t& b_neg) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::loglogu_nbp) {
				get_self().loglogu_nbp_st(srcdest, b_neg);
			} else get_self().loglogu_nbp_mt(srcdest, b_neg);
		}
		void loglogu_nbp_st(realmtx_t& srcdest, const real_t& b_neg, const elms_range*const pER = nullptr) const noexcept {
			get_self()._iloglogu_nbp_st(srcdest, b_neg, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _iloglogu_nbp_st(realmtx_t& srcdest, const real_t& b_neg, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));			
			const real_t nlbnegi = real_t(ext_real_t (-1.) / std::log(ext_real_t(b_neg)));
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
		void loglogu_nbp_mt(realmtx_t& srcdest, const real_t& b_neg) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));			
			m_threads.run([&srcdest, &b_neg, this](const par_range_t& r) {
				get_self()._iloglogu_nbp_st(srcdest, b_neg, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		// d(ELU)/dZ = exp(y*log(b_neg)-log(log(b_neg))) | x<0 ,  exp(-y*log(b_pos)-log(log(b_pos))) | x>0
		void dloglogu_nbp(realmtx_t& f_df, const real_t& b_neg) noexcept {
			if (f_df.numel() < Thresholds_t::dloglogu_nbp) {
				get_self().dloglogu_nbp_st(f_df, b_neg);
			} else get_self().dloglogu_nbp_mt(f_df, b_neg);
		}
		void dloglogu_nbp_st(realmtx_t& f_df, const real_t& b_neg, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idloglogu_nbp_st(f_df, b_neg, pER ? *pER : elms_range(f_df));
		}
		static void _idloglogu_nbp_st(realmtx_t& f_df, const real_t& b_neg, const elms_range& er) noexcept {
			NNTL_ASSERT(b_neg > real_t(1.0));			
			NNTL_ASSERT(!f_df.empty());
			const ext_real_t _lbneg = std::log(ext_real_t(b_neg));
			const real_t nllbneg = -static_cast<real_t>(std::log(_lbneg)), lbneg = static_cast<real_t>(_lbneg);
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				*ptrDF++ = std::exp(v < real_t(0.) ? (v*lbneg + nllbneg) : -v);
			}
		}
		void dloglogu_nbp_mt(realmtx_t& f_df, const real_t& b_neg) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(b_neg > real_t(1.0));			
			m_threads.run([&f_df, &b_neg, this](const par_range_t& r) {
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
			m_threads.run([&srcdest, this](const par_range_t& r) {
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
				*ptrDF++ = std::exp(v < real_t(0.) ? v : -v);
			}
		}
		void dloglogu_nbn_nbp_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) {
				get_self()._idloglogu_nbn_nbp_st(f_df, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// y = (x/(a+|x|)), dy/dx = (1-|y|)^2 /a, parameter 'a' controls the slope of the curve
		void softsign(realmtx_t& srcdest, const real_t& a) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::softsign) {
				get_self().softsign_st(srcdest, a);
			} else get_self().softsign_mt(srcdest, a);
		}
		void softsign_st(realmtx_t& srcdest, const real_t& a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._isoftsign_st(srcdest, a, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _isoftsign_st(realmtx_t& srcdest, const real_t& a, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));

			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = v/(a+std::abs(v));
			}
		}
		void softsign_mt(realmtx_t& srcdest, const real_t& a) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&srcdest, &a, this](const par_range_t& r) {
				get_self()._isoftsign_st(srcdest, a, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		//dy / dx = (1 - |y|)^2
		void dsoftsign_ua(realmtx_t& f_df) noexcept {
			if (f_df.numel() < Thresholds_t::dsoftsign_ua) {
				get_self().dsoftsign_ua_st(f_df);
			} else get_self().dsoftsign_ua_mt(f_df);
		}
		void dsoftsign_ua_st(realmtx_t& f_df, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idsoftsign_ua_st(f_df, pER ? *pER : elms_range(f_df));
		}
		static void _idsoftsign_ua_st(realmtx_t& f_df, const elms_range& er) noexcept {
			NNTL_ASSERT(!f_df.empty());
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				NNTL_ASSERT(real_t(-1.) <= v && v <= real_t(1.));
				const auto s = real_t(1.) - std::abs(v);
				*ptrDF++ = s*s;
			}
		}
		void dsoftsign_ua_mt(realmtx_t& f_df) noexcept {
			NNTL_ASSERT(!f_df.empty());
			m_threads.run([&f_df, this](const par_range_t& r) {
				get_self()._idsoftsign_ua_st(f_df, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//dy/dx = (1-|y|)^2 /a
		void dsoftsign(realmtx_t& f_df, const real_t& a) noexcept {
			if (f_df.numel() < Thresholds_t::dsoftsign) {
				get_self().dsoftsign_st(f_df, a);
			} else get_self().dsoftsign_mt(f_df, a);
		}
		void dsoftsign_st(realmtx_t& f_df, const real_t& a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idsoftsign_st(f_df, a, pER ? *pER : elms_range(f_df));
		}
		static void _idsoftsign_st(realmtx_t& f_df, const real_t& a, const elms_range& er) noexcept {
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());
			const auto ainv = real_t(1.) / a;
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				NNTL_ASSERT(real_t(-1.) <= v && v <= real_t(1.));
				const auto s = real_t(1.) - std::abs(v);
				*ptrDF++ = ainv*s*s;
			}
		}
		void dsoftsign_mt(realmtx_t& f_df, const real_t& a) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&f_df, &a, this](const par_range_t& r) {
				get_self()._idsoftsign_st(f_df, a, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		// y = (x/(2*(a+|x|)) +.5 ), dy/dx = (.5-|y-.5|)^2 * 2/a, parameter 'a' controls the slope of the curve
		void softsigm(realmtx_t& srcdest, const real_t& a) noexcept {
			if (srcdest.numel_no_bias() < Thresholds_t::softsigm) {
				get_self().softsigm_st(srcdest, a);
			} else get_self().softsigm_mt(srcdest, a);
		}
		void softsigm_st(realmtx_t& srcdest, const real_t& a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._isoftsigm_st(srcdest, a, pER ? *pER : elms_range(0, srcdest.numel_no_bias()));
		}
		static void _isoftsigm_st(realmtx_t& srcdest, const real_t& a, const elms_range& er) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));

			auto pV = srcdest.data() + er.elmBegin;
			const auto pVE = pV + er.totalElements();
			while (pV != pVE) {
				const auto v = *pV;
				*pV++ = real_t(.5) + real_t(.5)* v / (a + std::abs(v));
			}
		}
		void softsigm_mt(realmtx_t& srcdest, const real_t& a) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&srcdest, &a, this](const par_range_t& r) {
				get_self()._isoftsigm_st(srcdest, a, elms_range(r));
			}, srcdest.numel_no_bias());
		}
		//////////////////////////////////////////////////////////////////////////
		//dy/dx = (.5-|y-.5|)^2 * 2/a
		void dsoftsigm(realmtx_t& f_df, const real_t& a) noexcept {
			if (f_df.numel() < Thresholds_t::dsoftsigm) {
				get_self().dsoftsigm_st(f_df, a);
			} else get_self().dsoftsigm_mt(f_df, a);
		}
		void dsoftsigm_st(realmtx_t& f_df, const real_t& a, const elms_range*const pER = nullptr) const noexcept {
			get_self()._idsoftsigm_st(f_df, a, pER ? *pER : elms_range(f_df));
		}
		static void _idsoftsigm_st(realmtx_t& f_df, const real_t& a, const elms_range& er) noexcept {
			NNTL_ASSERT(a > real_t(0.0));
			NNTL_ASSERT(!f_df.empty());
			const auto dainv = real_t(2.) / a;
			auto ptrDF = f_df.data() + er.elmBegin;
			const auto ptrDFE = ptrDF + er.totalElements();
			while (ptrDF != ptrDFE) {
				const auto v = *ptrDF;
				NNTL_ASSERT(real_t(0.) <= v && v <= real_t(1.));
				const auto s = real_t(.5) - std::abs(v - real_t(.5));
				*ptrDF++ = dainv*s*s;
			}
		}
		void dsoftsigm_mt(realmtx_t& f_df, const real_t& a) noexcept {
			NNTL_ASSERT(!f_df.empty());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&f_df, &a, this](const par_range_t& r) {
				get_self()._idsoftsigm_st(f_df, a, elms_range(r));
			}, f_df.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of quadratic loss function for softsigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		//dL/dZ = (err===a-y)*dSoftSigm/dZ
		// because activations comes from the output layer, expecting no biases there
		void dSoftSigmQuadLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a)noexcept {
			if (act_dLdZ.numel() < Thresholds_t::dSoftSigmQuadLoss_dZ) {
				get_self().dSoftSigmQuadLoss_dZ_st(data_y, act_dLdZ, a);
			} else get_self().dSoftSigmQuadLoss_dZ_mt(data_y, act_dLdZ, a);
		}
		//usually error is defined as diffrence between data_y and last layer activation, i.e. nn.e=y-nn.a{n}, but
		//that will lead to necessity of negation of error in back propagation algorithm. To get rid of that negation,
		// we'll define error as nn.a{n}-y. This won't bother loss calculation, because it is either squares error
		// (conventional quadratic loss function) or doesn't use that error definition at all (crossentropy error)
		void dSoftSigmQuadLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a, const elms_range*const pER = nullptr)noexcept {
			get_self()._idSoftSigmQuadLoss_dZ_st(data_y, act_dLdZ, a, pER ? *pER : elms_range(act_dLdZ));
		}
		static void _idSoftSigmQuadLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a, const elms_range& er)noexcept {
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
				const auto s = real_t(.5) - std::abs(av - real_t(.5));
				*pSD++ = (av - y)*dainv*s*s;
			}
		}
		void dSoftSigmQuadLoss_dZ_mt(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&data_y, &act_dLdZ, &a, this](const par_range_t& r) {
				get_self()._idSoftSigmQuadLoss_dZ_st(data_y, act_dLdZ, a, elms_range(r));
			}, act_dLdZ.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of cross-entropy loss function for softsigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		// L = -y*log(a)-(1-y)log(1-a) (dL/dz = dL/dA * dA/dZ = (a-y)/(a*(1-a)) * dA/dZ )
		// dL/dZ = (a-y)/(a*(1-a)) * dSoftSigm/dZ
		// because activations comes from the output layer, expecting no biases there
		void dSoftSigmXEntropyLoss_dZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a)noexcept {
			if (act_dLdZ.numel() < Thresholds_t::dSoftSigmXEntropyLoss_dZ) {
				get_self().dSoftSigmXEntropyLoss_dZ_st(data_y, act_dLdZ, a);
			} else get_self().dSoftSigmXEntropyLoss_dZ_mt(data_y, act_dLdZ, a);
		}
		void dSoftSigmXEntropyLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a, const elms_range*const pER = nullptr)noexcept {
			get_self()._idSoftSigmXEntropyLoss_dZ_st(data_y, act_dLdZ, a, pER ? *pER : elms_range(act_dLdZ));
		}
		static void _idSoftSigmXEntropyLoss_dZ_st(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a, const elms_range& er)noexcept {
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

				const auto s = real_t(.5) - std::abs(av - real_t(.5));
				//#numstab
				*pSD++ = (av - y)*((s*s*dainv) / (av*(real_t(1.) - av)));
			}
		}
		void dSoftSigmXEntropyLoss_dZ_mt(const realmtx_t& data_y, realmtx_t& act_dLdZ, const real_t& a)noexcept {
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			NNTL_ASSERT(act_dLdZ.size() == data_y.size());
			NNTL_ASSERT(a > real_t(0.0));
			m_threads.run([&data_y, &act_dLdZ, &a, this](const par_range_t& r) {
				get_self()._idSoftSigmXEntropyLoss_dZ_st(data_y, act_dLdZ, a, elms_range(r));
			}, act_dLdZ.numel());
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
			m_threads.run([&srcdest, this](const par_range_t& r) {
				get_self()._istep_st(srcdest, elms_range(r));
			}, srcdest.numel_no_bias());
		}


		//////////////////////////////////////////////////////////////////////////
		// #TODO: probably, it is better to rewrite asymmetric activation functions processing using two templated
		// functions, one for f(x) and the other for dF/dX. However, possible performance penalty should be considered
		// -- well, it's a bit slower than Indian-style copy&pasted code...
		/* see test_perf_decisions for better solution. will update code later...
		template<size_t MtThreshold, typename FunctorT>
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
			m_threads.run([&srcdest, this](const par_range_t& r) {
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
		template<size_t MtThreshold, typename FunctorT>
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
			m_threads.run([&f_df, this](const par_range_t& r) {
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
		static real_t loss_quadratic_st_naive(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
			const auto dataCnt = activations.numel();
			const auto ptrA = activations.data(), ptrY = data_y.data();
			real_t ql(0.0);
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const real_t e = ptrA[i] - ptrY[i];
				ql += e*e;
			}
			return ql / (2 * activations.rows());
		}
		real_t loss_quadratic_mt_naive(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());

			const auto pA = activations.data();
			const auto pY = data_y.data();

			real_t ql = m_threads.reduce([pA,pY](const par_range_t& r)->real_t {
				const auto ofs = r.offset();
				const numel_cnt_t im = ofs + r.cnt();
				real_t ret(0.0);
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const real_t e = pA[i] - pY[i];
					ret += e*e;
				}
				return ret;
			}, _reduce_final_sum, activations.numel());

			return ql / (2 * activations.rows());
		}

		//////////////////////////////////////////////////////////////////////////
		// cross entropy function (applicable ONLY for binary data_y and sigmoid activation function)
		// L = -y*log(a)-(1-y)log(1-a) (dL/dz = dL/dA * dA/dZ = (a-y)/(a*(1-a)) * dA/dZ )
		real_t loss_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_xentropy) {
				return get_self().loss_xentropy_st(activations, data_y);
			} else return get_self().loss_xentropy_mt(activations, data_y);
		}
		real_t loss_xentropy_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			return -get_self()._iloss_xentropy_st(activations, data_y, pER ? *pER : elms_range(activations)) / activations.rows();
		}
		static real_t _iloss_xentropy_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range& er)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
			const auto ptrA = activations.data(), ptrY = data_y.data();
			constexpr auto log_zero = math::real_t_limits<real_t>::log_almost_zero;
			real_t ql = 0;
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) {
				const auto y = ptrY[i];
				const auto a = ptrA[i];
				NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
				NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

				if (y > real_t(0.0)) {
					ql += (a == real_t(0.0) ? log_zero : std::log(a));
				} else {
					//const auto oma = real_t(1.0) - a;
					//ql += (oma == real_t(0.0) ? log_zero : log(oma));
					ql += (a == real_t(1.0) ? log_zero : math::log1p(-a));
				}
				NNTL_ASSERT(!isnan(ql));
			}
			return ql;
		}
		real_t loss_xentropy_mt(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			return -m_threads.reduce([&activations, &data_y, this](const par_range_t& pr)->real_t {
				return get_self()._iloss_xentropy_st(activations, data_y, elms_range(pr));
			}, _reduce_final_sum, activations.numel()) / activations.rows();
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
			for (numel_cnt_t i = er.elmBegin; i < er.elmEnd; ++i) {
				auto a = pA[i];
				const auto y = -pY[i];
				NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));
				NNTL_ASSERT(y <= real_t(0.0) && y >= real_t(-1.0));
				a = a > real_t(0.0) ? std::log(a) : math::real_t_limits<real_t>::log_almost_zero;
				ret += y*a;
				NNTL_ASSERT(!isnan(ret));
			}
			return ret;
		}
		static real_t loss_softmax_xentropy_st(const realmtx_t& activations, const realmtx_t& data_y, const elms_range*const pER = nullptr)noexcept {
			NNTL_ASSERT(!activations.empty() && !data_y.empty() && data_y.size() == activations.size());
			return _iloss_softmax_xentropy_sum_st(activations.data(), data_y.data(), pER ? *pER : elms_range(activations)) / activations.rows();
		}
		real_t loss_softmax_xentropy_mt(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(!activations.empty() && !data_y.empty() && data_y.size() == activations.size());
			const auto pA = activations.data(), pY = data_y.data();
			return m_threads.reduce([pA, pY](const par_range_t& pr)->real_t {
				return _iloss_softmax_xentropy_sum_st(pA, pY, elms_range(pr));
			}, _reduce_final_sum, activations.numel()) / activations.rows();
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
				*pW = learningRate*(w / (sqrt(rms) + numericStabilizer));
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
			m_threads.run([pdW,prmsF,learningRate,emaDecay,numericStabilizer](const par_range_t& r) {
				const auto _1_emaDecay = 1 - emaDecay;
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto pW = pdW + i, pF = prmsF + i;
					const auto w = *pW;
					const auto rms = emaDecay*(*pF) + w*w*_1_emaDecay;
					*pF = rms;
					*pW = learningRate*(w / (sqrt(rms) + numericStabilizer));
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
				*pW = learningRate*(w / (sqrt(rF - rG*rG + numericStabilizer)));
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
			m_threads.run([pdW, prmsF, prmsG, learningRate, emaDecay, numericStabilizer](const par_range_t& r) {
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
					*pW = learningRate*(w / (sqrt(rF - rG*rG + numericStabilizer)));
				}
			}, dW.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		void RProp(realmtx_t& dW, const real_t learningRate)noexcept {
			if (dW.numel() < Thresholds_t::RProp) {
				get_self().RProp_st(dW, learningRate);
			} else get_self().RProp_mt(dW, learningRate);
		}
		static void RProp_st(realmtx_t& dW, const real_t learningRate)noexcept {
			auto p = dW.data();
			const auto pE = p + dW.numel();
			//TODO: verify vectorization
			while (p != pE) {
				*p++ = learningRate*math::sign(*p);
			}
		}
		void RProp_mt(realmtx_t& dW, const real_t learningRate)noexcept {
			auto pW = dW.data();
			m_threads.run([pW, learningRate](const par_range_t& r) {
				auto p = pW + r.offset();
				const auto pE = p + r.cnt();
				//TODO: verify vectorization
				while (p != pE) {
					*p++ = learningRate*math::sign(*p);
				}
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
				const auto ema = (*pF)*emaDecay + std::abs(w)*_1_emaDecay;
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
			m_threads.run([pdW, prmsF, learningRate, emaDecay, numericStabilizer](const par_range_t& r) {
				const auto _1_emaDecay = 1 - emaDecay;
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto pW = pdW + i;
					const auto pF = prmsF + i;
					const auto w = *pW;
					const auto ema = (*pF)*emaDecay + std::abs(w)*_1_emaDecay;
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
			const auto alphat = learningRate*sqrt(real_t(1.) - beta2t) / (real_t(1.) - beta1t);
			const auto ombeta1 = real_t(1.) - beta1, ombeta2 = real_t(1.) - beta2;

			auto pdW = dW.data()+ er.elmBegin, pMt = Mt.data()+ er.elmBegin, pVt = Vt.data()+ er.elmBegin;
			const auto pDWE = pdW + er.totalElements();
			while (pdW != pDWE) {
				const auto g = *pdW;
				const auto m = (*pMt)*beta1 + g*ombeta1;
				*pMt++ = m;
				const auto v = (*pVt)*beta2 + g*g*ombeta2;
				*pVt++ = v;
				*pdW++ = alphat*m / (sqrt(v) + numericStabilizer);
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
				*pG = alphat*m / (sqrt(v) + numericStabilizer);
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
			m_threads.run([&dW, &Mt, &Vt, &beta1t, &beta2t, learningRate, beta1, beta2, numericStabilizer,this](const par_range_t& r) {
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
				const auto u = std::max({std::abs(g),beta2*(*pUt)});
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
			m_threads.run([&dW, &Mt, &Ut, &beta1t, learningRate, beta1, beta2, numericStabilizer, this](const par_range_t& r) {
				get_self()._iAdaMax_st(dW, Mt, Ut, beta1t, learningRate, beta1, beta2, numericStabilizer, elms_range(r), true);
			}, dW.numel());
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
