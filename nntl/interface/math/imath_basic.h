/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "_i_math.h"
#include "bindings/b_open_blas.h"
//#include "bindings/b_yeppp.h"
#include "../threads.h"

#include "../../utils/clamp.h"

#include <limits>
#include "imath_basic_thresholds.h"

namespace nntl {
namespace math {

	// ALL functions of _i_math interface must be tested for ST vs. MT performance and be adjusted accordingly

	// this class uses some routines from OpenBLAS to implement _i_math
	template <typename iThreads>// = threads::Std>
	class iMath_basic : public _i_math {
		static_assert(std::is_base_of<threads::_i_threads<typename iThreads::range_t>, iThreads>::value, "iThreads must implement threads::_i_threads");
		static_assert(std::is_same<realmtx_t::numel_cnt_t, typename iThreads::range_t>::value, "iThreads::range_t should be the same as realmtx_t::numel_cnt_t");

	public:
		typedef iThreads ithreads_t;
		typedef typename ithreads_t::range_t range_t;
		typedef typename ithreads_t::par_range_t par_range_t;
		typedef typename ithreads_t::thread_id_t thread_id_t;

		typedef math_types::realmtxdef_ty realmtxdef_t;

		typedef _impl::IMATH_BASIC_THRESHOLDS<real_t> Thresholds_t;

	protected:
		typedef std::vector<real_t*> thread_temp_storage_ptrs_t;
		typedef std::vector<real_t> thread_temp_storage_t;

		//////////////////////////////////////////////////////////////////////////
		// members
	protected:
		ithreads_t m_threads;
		//b_Yeppp m_Yeppp;

		numel_cnt_t m_minTempStorageSize, m_minPerThreadTempStorageSize;
		thread_temp_storage_t m_threadTempRawStorage;
		thread_temp_storage_ptrs_t m_threadTempRawStoragePtrs;

		//realmtxdef_t m_tmpMtx;

		//bool m_succeded;

		//////////////////////////////////////////////////////////////////////////
		//methods
		// 
	
	public:

		~iMath_basic()noexcept {};
		iMath_basic() noexcept : m_minTempStorageSize(0), m_minPerThreadTempStorageSize(0) {
			//TODO: memory allocation exception handling here!
			m_threadTempRawStoragePtrs.resize(m_threads.workers_count());
		}

		ithreads_t& ithreads()noexcept { return m_threads; }

		//math preinitialization, should be called from each NN layer. n - maximum data length (in real_t), that this layer will use in calls
		//to math interface. Used to calculate max necessary temporary storage length.
		void preinit(const numel_cnt_t n)noexcept {
			static_assert(std::is_same<numel_cnt_t, range_t>::value, "WTF? realmtx_t::numel_cnt_t and ithreads_t::range_t must be the same!");
			if (n > m_minTempStorageSize) m_minTempStorageSize = n;
		}
		//real math initialization, used to allocate necessary temporary storage of size max(preinit::n)
		bool init()noexcept {
			const auto threadsCnt = m_threadTempRawStoragePtrs.size();
			m_minPerThreadTempStorageSize = static_cast<decltype(m_minPerThreadTempStorageSize)>(ceil(static_cast<double>(m_minTempStorageSize) / threadsCnt));
			const range_t maxMem = m_minPerThreadTempStorageSize*threadsCnt;
			if (m_minTempStorageSize < maxMem) m_minTempStorageSize = maxMem;

			if (m_threadTempRawStorage.size() < maxMem) {
				//TODO: memory allocation exception handling here!
				m_threadTempRawStorage.resize(maxMem);
				for (range_t i = 0, o = 0; i < threadsCnt; ++i, o += m_minPerThreadTempStorageSize) {
					m_threadTempRawStoragePtrs[i] = &m_threadTempRawStorage[o];
				}
			}

			//return m_tmpMtx.resize(maxMem);
			return true;
		}
		void deinit()noexcept {
			//m_threadTempRawStoragePtrs mustn't be freed!
			//m_threadTempRawStoragePtrs.clear();

			m_threadTempRawStorage.clear();
		}

		//bool initialized()const noexcept { return m_succeded; }

		//////////////////////////////////////////////////////////////////////////
		// i_math interface implementation

		//////////////////////////////////////////////////////////////////////////
		// Contnr dest is a std::vector-like container of vec_len_t, sized to m.rows(). Will contain for each row column index
		//of greatest element in a row.
		template<typename Contnr>
		void mFindIdxsOfMaxRowwise(const realmtx_t& m, Contnr& dest)noexcept {
			if (dest.size() < Thresholds_t::mFindIdxsOfMaxRowwise) {
				mFindIdxsOfMaxRowwise_st_naive(m, dest);
			} else mFindIdxsOfMaxRowwise_mt_naive(m, dest);
		}
		template<typename Contnr>
		void mFindIdxsOfMaxRowwise_st_naive(const realmtx_t& m, Contnr& dest)noexcept {
			const auto rows = m.rows();
			const auto ne = m.numel();
			NNTL_ASSERT(rows == dest.size());

			auto pD = m.dataAsVec();
			for (vec_len_t ri = 0; ri < rows; ++ri) {
				auto pV = &pD[ri];
				const auto pVEnd = pV + ne;
				auto m = std::numeric_limits<real_t>::min();
				Contnr::value_type mIdx = 0;
				vec_len_t c = 0;
				while (pV != pVEnd) {
					const auto v = *pV;
					pV += rows;
					if (v > m) {
						m = v;
						mIdx = c;
					}
					c++;
				}
				dest[ri] = mIdx;
			}
		}
		template<typename Contnr>
		void mFindIdxsOfMaxRowwise_mt_naive(const realmtx_t& m, Contnr& dest)noexcept {
			const auto rows = m.rows();
			const auto ne = m.numel();
			NNTL_ASSERT(rows == dest.size());

			auto pD = m.dataAsVec();
			m_threads.run([pD, ne, rows, &dest](const par_range_t& r) {
				const auto ofs = static_cast<vec_len_t>(r.offset());
				const auto riMax = ofs + static_cast<vec_len_t>(r.cnt());
				for (vec_len_t ri = ofs; ri < riMax; ++ri) {
					auto pV = &pD[ri];
					const auto pVEnd = pV + ne;
					auto m = std::numeric_limits<real_t>::min();
					Contnr::value_type mIdx = 0;
					vec_len_t c = 0;
					while (pV != pVEnd) {
						const auto v = *pV;
						pV += rows;
						if (v > m) {
							m = v;
							mIdx = c;
						}
						c++;
					}
					dest[ri] = mIdx;
				}
			}, rows);
		}

		//////////////////////////////////////////////////////////////////////////
		//extract rows with indexes specified by Contnr ridxs into dest.
		template<typename SeqIt>
		void mExtractRows(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept {
			if (dest.numel() < Thresholds_t::mExtractRows) {
				mExtractRows_st_naive(src, ridxsItBegin, ridxsCnt, dest);
			} else mExtractRows_mt_naive(src, ridxsItBegin, ridxsCnt, dest);
		}
		template<typename SeqIt>
		void mExtractRows_st_naive(const realmtx_t& src, SeqIt ridxsItBegin, const numel_cnt_t ridxsCnt, realmtx_t& dest)noexcept {
			NNTL_ASSERT(!dest.empty() && !src.empty());
			src.assert_storage_does_not_intersect(dest);
			static_assert(std::is_same<vec_len_t, SeqIt::value_type>::value, "Contnr type should contain vec_len_t data");

			const numel_cnt_t destRows = dest.rows(), srcRows = src.rows();
			NNTL_ASSERT(dest.cols() == src.cols() && destRows == ridxsCnt && ridxsCnt <= srcRows);

			//TODO: accessing data in sequential order could provide some performance gains. However
			//it requires the content of [ridxsItBegin,ridxsItBegin+ridxsCnt) to be sorted. Therefore, testing is required
			// to decide whether it's all worth it

			auto pSrc = src.dataAsVec();
			auto pDest = dest.dataAsVec();
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

				auto pSrc = src.dataAsVec();
				auto pDest = dest.dataAsVec() + rOfs;
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
		//binarize real-valued matrix with values in [0,1] according to 0<=frac<=1
		void mBinarize(realmtx_t& A, const real_t frac)noexcept {
			if (A.numel() < Thresholds_t::mBinarize) {
				mBinarize_st(A, frac);
			}else mBinarize_mt(A, frac);
		}
		void mBinarize_st(realmtx_t& A, const real_t frac)noexcept {
			auto pA = A.dataAsVec();
			const auto pAE = pA + A.numel();
			while (pA != pAE) {
				const auto v = *pA;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				*pA++ = v > frac ? real_t(1.0) : real_t(0.0);
			}
		}
		void mBinarize_mt(realmtx_t& A, const real_t frac)noexcept {
			auto pA = A.dataAsVec();
			m_threads.run([pA,frac](const par_range_t& r) {
				auto p = pA + r.offset();
				const auto pAE = p + r.cnt();
				while (p != pAE) {
					const auto v = *p;
					NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
					*p++ = v > frac ? real_t(1.0) : real_t(0.0);
				}
			},A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		// treat matrix as a set of row-vectors (matrices in col-major mode!). For each row-vector check, whether
		// its length/norm is not longer, than predefined value. If it's longer, than rescale vector to this max length
		// (for use in max-norm weights regularization)
		void mCheck_normalize_rows(realmtx_t& A, const real_t maxNormSquared)noexcept {
			if (A.numel() < Thresholds_t::mCheck_normalize_rows) {
				mCheck_normalize_rows_st(A, maxNormSquared);
			}else mCheck_normalize_rows_mt(A, maxNormSquared);
		}
		//static constexpr real_t sCheck_normalize_rows_MULT = real_t(32.0);
		void mCheck_normalize_rows_st(realmtx_t& A, const real_t maxNormSquared)noexcept {
			NNTL_ASSERT(!A.empty() && maxNormSquared > real_t(0.0));

			const auto mRows = A.rows();
			auto pTmp = _get_thread_temp_raw_storage(mRows);
			memset(pTmp, 0, sizeof(*pTmp)*mRows);
			
			//calculate current norms of row-vectors into pTmp
			const auto dataCnt = A.numel();
			real_t* pCol = A.dataAsVec();
			const auto pColE = pCol + dataCnt;
			while (pCol != pColE) {
				const real_t* pElm = pCol;
				pCol += mRows;
				const auto pElmE = pCol;
				auto pN = pTmp;
				while (pElm != pElmE) {
					const auto v = *pElm++;
					*pN++ += v*v;
				}
			}

			//Saving normalization coefficient into pTmp for those rows, that needs normalization, or ones for those, 
			// that doesn't need.
			// Making newNorm slightly less, than maxNormSquared to make sure the result will be less than max norm.
			//const real_t newNorm = maxNormSquared - math::real_ty_limits<real_t>::eps_lower_n(maxNormSquared, sCheck_normalize_rows_MULT);
			const real_t newNorm = maxNormSquared - sqrt(math::real_ty_limits<real_t>::eps_lower(maxNormSquared));
			auto pCurNorm = pTmp;
			const auto pTmpE = pTmp + mRows;
			while (pCurNorm != pTmpE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = rowNorm > maxNormSquared ? sqrt(newNorm / rowNorm) : real_t(1.0);
			}

			//renormalize (multiply each rowvector to corresponding coefficient from pTmp)
			pCol = A.dataAsVec();
			while (pCol != pColE) {
				real_t* pElm = pCol;
				pCol += mRows;
				const auto pElmE = pCol;
				auto pN = pTmp;
				while (pElm != pElmE) {
					*pElm++ *= *pN++;
				}
			}
		}
		void mCheck_normalize_rows_mt(realmtx_t& A, const real_t maxNormSquared)noexcept {
			NNTL_ASSERT(!A.empty() && maxNormSquared > real_t(0.0));

			// 1. each thread will get it's own sequential set of columns (which means sequential underlying representation)
			//		and will find for that set norm (sum of squares) of their rowvectors.
			// 2. master thread will sum these partial norms into total norms of whole rowvectors of original matrix and
			//		calculate corresponding normalization coefficients.
			// 3. then again each thread will apply these calculated normalization coefficients to their own sequential set of
			//		columns.

			const auto mRows = A.rows(), mCols=A.cols();
			auto ppTmps = _get_thread_temp_storage_ptrs_head(mRows*m_threads.workers_count());
			thread_id_t threadsCnt;
			// 1.
			m_threads.run([&A, ppTmps](const par_range_t& r)noexcept{
				const vec_len_t startingCol = static_cast<vec_len_t>(r.offset());
				const vec_len_t colsToProcess = static_cast<vec_len_t>(r.cnt());

				const auto mRows = A.rows();
				real_t* pTmp = ppTmps[r.tid()];
				memset(pTmp, 0, sizeof(*pTmp)*mRows);

				//calculate current norms of row-vectors into pTmp
				const auto dataCnt = realmtx_t::sNumel(mRows, colsToProcess);
				real_t* pCol = A.colDataAsVec(startingCol);
				const auto pColE = pCol + dataCnt;
				while (pCol != pColE) {
					const real_t* pElm = pCol;
					pCol += mRows;
					const auto pElmE = pCol;
					auto pN = pTmp;
					while (pElm != pElmE) {
						const auto v = *pElm++;
						*pN++ += v*v;
					}
				}
			}, mCols, &threadsCnt);

			//2. sum partial norms into total pTmp
			auto pHead = ppTmps + 1;
			const auto pTail = ppTmps + threadsCnt;//threadsCnt already includes main thread which we'll use in pTmp
			real_t* pTmp = ppTmps[0];
			const auto pTmpE = pTmp + mRows;
			while (pHead != pTail) {
				auto pPart = *pHead++;
				auto pT = pTmp;
				while (pT != pTmpE) {
					*pT++ += *pPart++;
				}
			}
			// calc scaling coefficients
			const real_t newNorm = maxNormSquared - sqrt(math::real_ty_limits<real_t>::eps_lower(maxNormSquared));
			auto pCurNorm = pTmp;
			while (pCurNorm != pTmpE) {
				const auto rowNorm = *pCurNorm;
				*pCurNorm++ = rowNorm > maxNormSquared ? sqrt(newNorm / rowNorm) : real_t(1.0);
			}
			
			// 3. multiplying
			m_threads.run([&A, pTmp](const par_range_t& r)noexcept {
				const vec_len_t startingCol = static_cast<vec_len_t>(r.offset());
				const vec_len_t colsToProcess = static_cast<vec_len_t>(r.cnt());

				const auto mRows = A.rows();
				const auto dataCnt = realmtx_t::sNumel(mRows, colsToProcess);
				real_t* pCol = A.colDataAsVec(startingCol);
				const auto pColE = pCol + dataCnt;
				while (pCol != pColE) {
					real_t* pElm = pCol;
					pCol += mRows;
					const auto pElmE = pCol;
					auto pN = pTmp;
					while (pElm != pElmE) {
						*pElm++ *= *pN++;
					}
				}
			}, mCols);
		}

		//////////////////////////////////////////////////////////////////////////
		//returns how many elements in two vectors has exactly the same value. Vectors must have the same length
		template<typename Contnr>
		size_t vCountSame(const Contnr& A, const Contnr& B)noexcept {
			return vCountSame_st_naive(A, B);
			// 			if (A.size()<=50000) {
			// 				return vCountSame_st_naive(A, B);
			// 			}else return vCountSame_mt_naive(A, B);
		}
		template<typename Contnr>
		size_t vCountSame_st_naive(const Contnr& A, const Contnr& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());

			size_t ret = 0;
			const auto dataCnt = A.size();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				if (A[i] == B[i]) ret++;
			}
			return ret;
		}
		template<typename Contnr>
		size_t vCountSame_mt_naive(const Contnr& A, const Contnr& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());

			auto pAc = &A[0];
			auto pBc = &B[0];
			real_t ret = m_threads.reduce([pAc,pBc](const par_range_t& r)->real_t {
				const auto ofs = r.offset();
				size_t ret = 0;
				const auto pA = &pAc[ofs];
				const auto pB = &pBc[ofs];
				const auto cnt = r.cnt();
				for (range_t i = 0; i < cnt; ++i) {
					if (pA[i] == pB[i]) ret++;
				}
				return static_cast<real_t>(ret);
			}, [](const real_t* ptr, const range_t cnt)->real_t {
				real_t ret = ptr[0];
				for (numel_cnt_t i = 1; i < cnt; ++i) ret += ptr[i];
				return ret;
			}, A.size());

			return static_cast<size_t>(ret);
		}

		//////////////////////////////////////////////////////////////////////////
		//clamps vector values into range
		void evClamp(realmtx_t& m, real_t lo, real_t hi)noexcept {
			if (m.numel() < Thresholds_t::evClamp) {
				evClamp_st(m, lo, hi);
			} else evClamp_mt(m, lo, hi);
		}
		void evClamp_st(realmtx_t& m, real_t lo, real_t hi)noexcept {
			NNTL_ASSERT(m.numel() > 0 && !m.empty());
			NNTL_ASSERT(lo < hi);

			auto p = m.dataAsVec();
			utils::boost::algorithm::clamp_range(p, p + m.numel(), p, lo, hi);
		}
		void evClamp_mt(realmtx_t& m, real_t lo, real_t hi)noexcept {
			NNTL_ASSERT(m.numel() > 0 && !m.empty());
			NNTL_ASSERT(lo < hi);

			auto ptr = m.dataAsVec();
			m_threads.run([ptr, lo, hi](const par_range_t& r) {
				auto p = ptr + r.offset();
				utils::boost::algorithm::clamp_range(p, p + r.cnt(), p, lo, hi);
			}, m.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//on entry dropoutMask must be filled with random values in [0,1]
		//binarizes dropoutMask according to dropoutFraction value and applies dropoutMask to activations
		// act must be used in "no_bias" mode
		void make_dropout(realmtx_t& act, real_t dfrac, realmtx_t& dropoutMask)noexcept {
			if (act.numel_no_bias() < Thresholds_t::make_dropout) {
				make_dropout_st(act,dfrac, dropoutMask);
			} else make_dropout_mt(act, dfrac, dropoutMask);
		}
		void make_dropout_st(realmtx_t& act, real_t dfrac, realmtx_t& dropoutMask)noexcept {
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dfrac > 0 && dfrac < 1);

			const auto dataCnt = act.numel_no_bias();
			auto pDM = dropoutMask.dataAsVec();
			const auto pDME = pDM + dataCnt;
			while (pDM != pDME) {
				const auto v = *pDM;
				NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
				*pDM++ = v > dfrac ? real_t(1.0) : real_t(0.0);
			}

			const auto pA = act.dataAsVec();
			pDM = dropoutMask.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] *= pDM[i];
		}
		void make_dropout_mt(realmtx_t& act, real_t dfrac, realmtx_t& dropoutMask)noexcept {
			NNTL_ASSERT(act.emulatesBiases() && !dropoutMask.emulatesBiases());
			NNTL_ASSERT(act.size_no_bias() == dropoutMask.size());
			NNTL_ASSERT(dfrac > 0 && dfrac < 1);
			
			const auto pDM = dropoutMask.dataAsVec();
			const auto pA = act.dataAsVec();
			m_threads.run([pA,pDM,dfrac](const par_range_t& r) {
				const auto pD = pDM + r.offset();
				auto p = pD;
				const auto cnt = r.cnt();
				const auto pDE = pD + cnt;
				while (p != pDE) {
					const auto v = *p;
					NNTL_ASSERT(v >= real_t(0.0) && v <= real_t(1.0));
					*p++ = v > dfrac ? real_t(1.0) : real_t(0.0);
				}

				const auto pAct = pA + r.offset();
				for (numel_cnt_t i = 0; i < cnt; ++i) pAct[i] *= pD[i];
			}, act.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		//apply individual learning rate to dLdW
		void apply_ILR(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			const auto dataCnt = dLdW.numel();
			if (dataCnt < Thresholds_t::apply_ILR_st) {
				if (std::is_same<float, real_t>::value) {
					apply_ILR_st_vec(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
				}else apply_ILR_st_naive(dLdW,prevdLdW,ILRGain,decr,incr,capLow,capHigh);
			} else if (dataCnt < Thresholds_t::apply_ILR_mt_lo || dataCnt > Thresholds_t::apply_ILR_mt_hi) {
 				apply_ILR_mt_naive(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
			}else apply_ILR_mt_vec(dLdW, prevdLdW, ILRGain, decr, incr, capLow, capHigh);
		}
		void apply_ILR_st_naive(realmtx_t& dLdW, const realmtx_t& prevdLdW, realmtx_t& ILRGain,
			const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh)noexcept
		{
			NNTL_ASSERT(dLdW.size() == prevdLdW.size() && dLdW.size() == ILRGain.size());
			NNTL_ASSERT(decr > 0 && decr < 1 && incr>1 && capLow < capHigh && capLow>0);

			//TODO: probably not the most efficient implementation

			const auto dataCnt = dLdW.numel();

			auto pdW = dLdW.dataAsVec();
			const auto prevdW = prevdLdW.dataAsVec();
			auto pGain = ILRGain.dataAsVec();

			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				auto pW = pdW + i;
				auto pG = pGain + i;
				const auto cond = prevdW[i] * (*pW);
				auto g = *pG;
				if (cond > 0) {
					g *= incr;
					if (g > capHigh) g = capHigh;
				} else if (cond < 0) {
					g *= decr;
					if (g < capLow) g = capLow;
				}
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

			auto pdW = dLdW.dataAsVec();
			const auto prevdW = prevdLdW.dataAsVec();
			auto pGain = ILRGain.dataAsVec();
			m_threads.run([pdW, prevdW, pGain, decr, incr, capLow, capHigh](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					auto pW = pdW + i;
					auto pG = pGain + i;
					const auto cond = prevdW[i] * (*pW);
					auto g = *pG;
					if (cond > 0) {
						g *= incr;
						if (g > capHigh) g = capHigh;
					} else if (cond < 0) {
						g *= decr;
						if (g < capLow) g = capLow;
					}
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
			auto pCond = _get_thread_temp_raw_storage(dataCnt);

			auto pdW = dLdW.dataAsVec();
			const auto prevdW= prevdLdW.dataAsVec();
			auto pGain = ILRGain.dataAsVec();

			for (numel_cnt_t i = 0; i < dataCnt; ++i) pCond[i] = pdW[i] * prevdW[i];

			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				auto pG = pGain + i;
				const auto cond = pCond[i];
				auto g = *pG;
				if (cond>0) {
					g *= incr;
					if (g > capHigh) g = capHigh;
				} else if (cond < 0) {
					g *= decr;
					if (g < capLow) g = capLow;
				}
				*pG = g;
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
			//auto pCond = _get_thread_temp_raw_storage(dataCnt);

			auto ppCond = _get_thread_temp_storage_ptrs_head(dataCnt);
			auto pdW = dLdW.dataAsVec();
			const auto prevdW = prevdLdW.dataAsVec();
			auto pGain = ILRGain.dataAsVec();

			m_threads.run([pdW, prevdW, pGain, decr, incr, capLow, capHigh, ppCond](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto cnt = r.cnt();
				auto pCond = ppCond[r.tid()];
				const auto pW = pdW + ofs;
				const auto prW = prevdW + ofs;
				const auto pGn = pGain + ofs;

				for (numel_cnt_t i = 0; i < cnt; ++i) pCond[i] = pW[i] * prW[i];

				for (numel_cnt_t i = 0; i < cnt; ++i) {
					auto pG = pGn + i;
					const auto cond = pCond[i];
					auto g = *pG;
					if (cond > 0) {
						g *= incr;
						if (g > capHigh) g = capHigh;
					} else if (cond < 0) {
						g *= decr;
						if (g < capLow) g = capLow;
					}
					*pG = g;
				}

				for (numel_cnt_t i = 0; i < cnt; ++i) pW[i] *= pGn[i];
			},dataCnt);
		}

		//////////////////////////////////////////////////////////////////////////
		//apply momentum vW = momentum.*vW + dW
		void apply_momentum(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			if (vW.numel() < Thresholds_t::apply_momentum) {
				apply_momentum_st(vW, momentum, dW);
			}else apply_momentum_mt(vW, momentum, dW);
		}
		void apply_momentum_st(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			NNTL_ASSERT(vW.size() == dW.size());
			NNTL_ASSERT(!vW.empty() && !dW.empty());

			const auto dataCnt = vW.numel();
			const auto pV = vW.dataAsVec();
			const auto pdW = dW.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				pV[i] = momentum*pV[i] + pdW[i];
			}
		}
		void apply_momentum_mt(realmtx_t& vW, const real_t momentum, const realmtx_t& dW)noexcept {
			NNTL_ASSERT(vW.size() == dW.size());
			NNTL_ASSERT(!vW.empty() && !dW.empty());

			const auto pV = vW.dataAsVec();
			const auto pdW = dW.dataAsVec();
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
				evMulC_ip_st_naive(A, b);
			} else evMulC_ip_mt_naive(A, b);
		}
		void evMulC_ip_st_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel()>0);
			ievMulC_ip_st_naive(A.dataAsVec(), A.numel(), b);
		}
		void ievMulC_ip_st_naive(real_t* ptrA, const numel_cnt_t dataCnt, const real_t b)noexcept {
			const auto ptrAE = ptrA + dataCnt;
			while (ptrA != ptrAE)  *ptrA++ *= b;
		}
		void evMulC_ip_mt_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel()>0);
			ievMulC_ip_mt_naive(A.dataAsVec(), A.numel(), b);
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
				evMulC_ip_Anb_st_naive(A, b);
			} else evMulC_ip_Anb_mt_naive(A, b);
		}
		void evMulC_ip_Anb_st_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
			ievMulC_ip_st_naive(A.dataAsVec(), A.numel_no_bias(), b);
		}
		void evMulC_ip_Anb_mt_naive(realmtx_t& A, const real_t b)noexcept {
			NNTL_ASSERT(!A.empty() && A.numel_no_bias() > 0);
			ievMulC_ip_mt_naive(A.dataAsVec(), A.numel_no_bias(), b);
		}


		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise multiplication A = A.*B
		void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evMul_ip) {
				evMul_ip_st_naive(A, B);
			}else evMul_ip_mt_naive(A, B);
		}
		void evMul_ip_st_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size() == B.size());
			ievMul_ip_st_naive(A.dataAsVec(), B.dataAsVec(), A.numel());
		}
		void ievMul_ip_st_naive(real_t* ptrA, const real_t*ptrB, numel_cnt_t dataCnt) noexcept{
			//for (numel_cnt_t i = 0; i < dataCnt; ++i) ptrA[i] *= ptrB[i];
			const bool bOdd = dataCnt & 1;
			if (bOdd) --dataCnt;
			for (numel_cnt_t i = 0; i < dataCnt; i+=2) {
				ptrA[i] *= ptrB[i];
				const auto j = i + 1;
				ptrA[j] *= ptrB[j];
			}
			if (bOdd) ptrA[dataCnt] *= ptrB[dataCnt];
		}
		void evMul_ip_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size() == B.size());
			ievMul_ip_mt_naive(A.dataAsVec(), B.dataAsVec(), A.numel());
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
				evMul_ip_Anb_st_naive(A, B);
			} else evMul_ip_Anb_mt_naive(A, B);
		}
		void evMul_ip_Anb_st_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size_no_bias() == B.size());
			ievMul_ip_st_naive(A.dataAsVec(), B.dataAsVec(), B.numel());
		}
		void evMul_ip_Anb_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			A.assert_storage_does_not_intersect(B);
			NNTL_ASSERT(A.size_no_bias() == B.size());
			ievMul_ip_mt_naive(A.dataAsVec(), B.dataAsVec(), B.numel());
		}
		

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise addition A = A+B
		void evAdd_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evAdd_ip) {
				evAdd_ip_st(A, B);
			} else evAdd_ip_mt(A, B);
		}
		void evAdd_ip_st(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty());
			const auto pA = A.dataAsVec();
			const auto dataCnt = A.numel();
			const auto pB = B.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] += pB[i];
		}
		void evAdd_ip_mt(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size() && !A.empty() && !B.empty());
			const auto pA = A.dataAsVec();
			const auto pB = B.dataAsVec();
			m_threads.run([pA, pB](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) pA[i] += pB[i];
			}, A.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		//inplace elementwise subtraction A = A-B
		void evSub_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			if (A.numel() < Thresholds_t::evSub_ip) {
				evSub_ip_st_naive(A, B);
			}else evSub_ip_mt_naive(A, B);
		}
		void evSub_ip_st_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());
			
			const auto dataCnt = A.numel();
			const auto pA = A.dataAsVec();
			const auto pB = B.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pA[i] -= pB[i];
		}
		void evSub_ip_mt_naive(realmtx_t& A, const realmtx_t& B)noexcept {
			NNTL_ASSERT(A.size() == B.size());

			const auto pA = A.dataAsVec();
			const auto pB = B.dataAsVec();
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
				evSub_st_naive(A, B, C);
			} else evSub_mt_naive(A, B, C);
		}
		void evSub_st_naive(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			NNTL_ASSERT(A.size() == B.size() && A.size()==C.size());

			const auto dataCnt = A.numel();
			const auto pA = A.dataAsVec(), pB = B.dataAsVec();
			const auto pC = C.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) pC[i] = pA[i] - pB[i];
		}
		void evSub_mt_naive(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			NNTL_ASSERT(A.size() == B.size() && A.size() == C.size());

			const auto pA = A.dataAsVec(), pB = B.dataAsVec();
			const auto pC = C.dataAsVec();
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
				evMulC_ip_Sub_ip_st(vW, momentum, W);
			}else evMulC_ip_Sub_ip_mt(vW, momentum, W);
		}
		void evMulC_ip_Sub_ip_st(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
			NNTL_ASSERT(vW.size() == W.size() && !vW.empty() && !W.empty()); //NNTL_ASSERT(momentum > real_t(0.0) && momentum < real_t(1.0));
			auto pV = vW.dataAsVec();
			const auto pVE = pV + vW.numel();
			auto pW = W.dataAsVec();
			while (pV != pVE) {
				const auto v = *pV * momentum;
				*pV++ = v;
				*pW++ -= v;
			}
		}
		void evMulC_ip_Sub_ip_mt(realmtx_t& vW, const real_t momentum, realmtx_t& W)noexcept {
			NNTL_ASSERT(vW.size() == W.size() && !vW.empty() && !W.empty()); //NNTL_ASSERT(momentum > real_t(0.0) && momentum < real_t(1.0));

			auto pVf = vW.dataAsVec();
			auto pWf = W.dataAsVec();
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
			},vW.numel());
		}
		
		//////////////////////////////////////////////////////////////////////////
		//elementwise squaring dest = src.^2;
		void evSquare(realmtx_t& dest, const realmtx_t& src)noexcept {
			if (src.numel() < Thresholds_t::evSquare) {
				evSquare_st(dest, src);
			}else evSquare_mt(dest, src);
		}
		void evSquare_st(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.dataAsVec();
			auto pD = dest.dataAsVec();
			const auto dataCnt = src.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto s = pS[i];
				pD[i] = s*s;
			}
		}
		void evSquare_mt(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.dataAsVec();
			auto pD = dest.dataAsVec();
			m_threads.run([pS,pD](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto s = pS[i];
					pD[i] = s*s;
				}
			},src.numel());
		}
		//////////////////////////////////////////////////////////////////////////
		//finding elementwise absolute values dest = .abs(src);
		void evAbs(realmtx_t& dest, const realmtx_t& src)noexcept {
			if (src.numel() < Thresholds_t::evAbs) {
				evAbs_st(dest, src);
			} else evAbs_mt(dest, src);
		}
		void evAbs_st(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.dataAsVec();
			auto pD = dest.dataAsVec();
			const auto dataCnt = src.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i)  pD[i] = abs(pS[i]);
		}
		void evAbs_mt(realmtx_t& dest, const realmtx_t& src)noexcept {
			NNTL_ASSERT(dest.size() == src.size());

			const auto pS = src.dataAsVec();
			auto pD = dest.dataAsVec();
			m_threads.run([pS, pD](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i)  pD[i] = abs(pS[i]);
			}, src.numel());
		}


		//////////////////////////////////////////////////////////////////////////
		//C = A * B, - matrix multiplication
		void mMulAB_C(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			NNTL_ASSERT(acols == B.rows() && A.rows() == C.rows() && B.cols() == C.cols());

			b_OpenBLAS::gemm(false, false, A.rows(), C.cols(), acols, real_t(1.0), A.dataAsVec(), A.rows(), B.dataAsVec(), B.rows(),
				real_t(0.0), C.dataAsVec(), C.rows());
		}
		//////////////////////////////////////////////////////////////////////////
		//matrix multiplication C(no bias) = A * B` (B transposed). C could have emulated biases (they will be left untouched)
		void mMulABt_Cnb(const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto ccols = C.cols_no_bias();
			NNTL_ASSERT(A.cols() == B.cols() && A.rows() == C.rows() && B.rows() == ccols);

			b_OpenBLAS::gemm(false, true, A.rows(), ccols, A.cols(), real_t(1.0), A.dataAsVec(), A.rows(), B.dataAsVec(), ccols,
				real_t(0.0), C.dataAsVec(), C.rows());
		}
		//////////////////////////////////////////////////////////////////////////
		//C = a*(A` * B) - matrix multiplication of transposed A times B with result normalization
		void mScaledMulAtB_C(real_t alpha, const realmtx_t& A, const realmtx_t& B, realmtx_t& C)noexcept {
			A.assert_storage_does_not_intersect(B);
			A.assert_storage_does_not_intersect(C);
			B.assert_storage_does_not_intersect(C);
			const auto acols = A.cols();
			const auto arows = A.rows();
			NNTL_ASSERT(arows == B.rows() && acols == C.rows() && B.cols() == C.cols());

			b_OpenBLAS::gemm(true, false, acols, B.cols(), arows, alpha, A.dataAsVec(), arows, B.dataAsVec(), arows,
				real_t(0.0), C.dataAsVec(), acols);
		}

		//////////////////////////////////////////////////////////////////////////
		// sigmoid function
		//////////////////////////////////////////////////////////////////////////
		void sigm(realmtx_t& srcdest) noexcept {
			if (srcdest.numel() < Thresholds_t::sigm) {
				sigm_st_naive(srcdest);
			}else sigm_mt_naive(srcdest);
		}
		// Remember to ignore biases!
		void sigm_st_naive(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());
			const auto dataCnt = srcdest.numel_no_bias();
			auto ptr = srcdest.dataAsVec();
			for (range_t i = 0; i < dataCnt; ++i) {
				ptr[i] = real_t(1.0) / (real_t(1.0) + std::exp(-ptr[i]));
			}
		}
		void sigm_mt_naive(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());

			auto ptr = srcdest.dataAsVec();

			m_threads.run([ptr](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (range_t i = ofs; i < im; ++i) {
					ptr[i] = real_t(1.0) / (real_t(1.0) + std::exp(-ptr[i]));
				}
			}, srcdest.numel_no_bias());
		}

		//////////////////////////////////////////////////////////////////////////
		// d(sigm)/d(arg) - sigmoid derivative df = f.*(1-f), where fValue is activation value (used in no_bias version)
		void dsigm(const realmtx_t& fValue, realmtx_t& df) noexcept {
			if (fValue.numel_no_bias() < Thresholds_t::dsigm) {
				dsigm_st_naive(fValue, df);
			} else dsigm_mt_naive(fValue, df);
		}
		void dsigm_st_naive(const realmtx_t& fValue, realmtx_t& df) noexcept {
			fValue.assert_storage_does_not_intersect(df);
			NNTL_ASSERT(fValue.size_no_bias() == df.size());

			const auto dataCnt = fValue.numel_no_bias();
			auto ptrF = fValue.dataAsVec();
			auto ptrDF = df.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt;++i) {
				const auto f = ptrF[i];
				ptrDF[i] = f*(real_t(1.0) - f);
			}
		}
		void dsigm_mt_naive(const realmtx_t& fValue, realmtx_t& df) noexcept {
			fValue.assert_storage_does_not_intersect(df);
			NNTL_ASSERT(fValue.size_no_bias() == df.size());

			auto ptrF = fValue.dataAsVec();
			auto ptrDF = df.dataAsVec();

			m_threads.run([ptrF, ptrDF](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (range_t i = ofs; i < im; ++i) {
					const auto f = ptrF[i];
					ptrDF[i] = f*(real_t(1.0) - f);
				}
			}, fValue.numel_no_bias());
		}
		

		//////////////////////////////////////////////////////////////////////////
		//calculates derivative of quadratic loss function for sigm neurons wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		//////////////////////////////////////////////////////////////////////////
		//dL/dZ = (err===a-y)*a*(1-a)
		// because activations comes from the output layer, expecting no biases there
		void dSigmQuadLoss_dZ(const realmtx_t& activations, const realmtx_t& data_y, realmtx_t& dLdZ) {
			if (activations.numel() < Thresholds_t::dSigmQuadLoss_dZ) {
				dSigmQuadLoss_dZ_st_naive(activations, data_y, dLdZ);
			}else dSigmQuadLoss_dZ_mt_naive(activations, data_y, dLdZ);
		}
		//usually error is defined as diffrence between data_y and last layer activation, i.e. nn.e=y-nn.a{n}, but
		//that will lead to necessity of negation of error in back propagation algorithm. To get rid of that negation,
		// we'll define error as nn.a{n}-y. This won't bother loss calculation, because it is either squares error
		// (conventional quadratic loss function) or doesn't use that error definition at all (crossentropy error)
		void dSigmQuadLoss_dZ_st_naive(const realmtx_t& activations, const realmtx_t& data_y, realmtx_t& dLdZ) {
			NNTL_ASSERT(!activations.emulatesBiases());
			NNTL_ASSERT(activations.size() == data_y.size() && activations.size() == dLdZ.size());
			activations.assert_storage_does_not_intersect(data_y);
			activations.assert_storage_does_not_intersect(dLdZ);
			data_y.assert_storage_does_not_intersect(dLdZ);

			const auto ptrAct = activations.dataAsVec(), ptrDataY=data_y.dataAsVec();
			auto ptrdLdZ = dLdZ.dataAsVec();
			const auto dataCnt = activations.numel();

			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto a = ptrAct[i];
				ptrdLdZ[i] = (a - ptrDataY[i])*a*(real_t(1.0) - a);
			}
		}
		void dSigmQuadLoss_dZ_mt_naive(const realmtx_t& activations, const realmtx_t& data_y, realmtx_t& dLdZ) {
			NNTL_ASSERT(!activations.emulatesBiases());
			NNTL_ASSERT(activations.size() == data_y.size() && activations.size() == dLdZ.size());
			activations.assert_storage_does_not_intersect(data_y);
			activations.assert_storage_does_not_intersect(dLdZ);
			data_y.assert_storage_does_not_intersect(dLdZ);

			const auto ptrAct = activations.dataAsVec(), ptrDataY = data_y.dataAsVec();
			auto ptrdLdZ = dLdZ.dataAsVec();
			const auto dataCnt = activations.numel();

			m_threads.run([ptrAct, ptrDataY, ptrdLdZ](const par_range_t& r) {
				const auto ofs = r.offset();
				const numel_cnt_t im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto a = ptrAct[i];
					ptrdLdZ[i] = (a - ptrDataY[i])*a*(real_t(1.0) - a);
				}
			}, dataCnt);
		}

		//////////////////////////////////////////////////////////////////////////
		//ReLU
		void relu(realmtx_t& srcdest) noexcept {
			if (srcdest.numel() < Thresholds_t::relu) {
				relu_st_naive(srcdest);
			} else relu_mt_naive(srcdest);
		}
		void relu_st_naive(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());

			const auto dataCnt = srcdest.numel();
			auto pV = srcdest.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				auto p = pV + i;
				if (*p < real_t(0.0))  *p = real_t(0.0);
			}
		}
		void relu_mt_naive(realmtx_t& srcdest) noexcept {
			NNTL_ASSERT(!srcdest.empty());

			auto pV = srcdest.dataAsVec();
			m_threads.run([pV](const par_range_t& r) {
				const auto ofs = r.offset();
				const numel_cnt_t im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					auto p = pV + i;
					if (*p < real_t(0.0)) *p = real_t(0.0);
				}
			}, srcdest.numel());
		}

		//////////////////////////////////////////////////////////////////////////
		// d(ReLU)/dZ
		void drelu(const realmtx_t& fValue, realmtx_t& df) noexcept {
			if (df.numel() < Thresholds_t::drelu) {
				drelu_st_naive(fValue, df);
			} else drelu_mt_naive(fValue, df);
		}
		void drelu_st_naive(const realmtx_t& fValue, realmtx_t& df) noexcept {
			fValue.assert_storage_does_not_intersect(df);
			NNTL_ASSERT(fValue.size_no_bias() == df.size());

			const auto dataCnt = fValue.numel_no_bias();
			const auto ptrF = fValue.dataAsVec();
			const auto ptrDF = df.dataAsVec();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				ptrDF[i] = ptrF[i]>0 ? real_t(1.0) : real_t(0.0);
			}
		}
		void drelu_mt_naive(const realmtx_t& fValue, realmtx_t& df) noexcept {
			fValue.assert_storage_does_not_intersect(df);
			NNTL_ASSERT(fValue.size_no_bias() == df.size());

			const auto ptrF = fValue.dataAsVec();
			const auto ptrDF = df.dataAsVec();
			m_threads.run([ptrF, ptrDF](const par_range_t& r) {
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (range_t i = ofs; i < im; ++i) {
					ptrDF[i] = ptrF[i]>0 ? real_t(1.0) : real_t(0.0);
				}
			}, fValue.numel_no_bias());
		}


		//////////////////////////////////////////////////////////////////////////
		//loss functions
		//////////////////////////////////////////////////////////////////////////
		real_t loss_quadratic(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_quadratic) {
				return loss_quadratic_st_naive(activations, data_y);
			} else return loss_quadratic_mt_naive(activations, data_y);
		}
		real_t loss_quadratic_st_naive(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
			const auto dataCnt = activations.numel();
			const auto ptrA = activations.dataAsVec(), ptrY = data_y.dataAsVec();
			real_t ql = 0;
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const real_t e = ptrA[i] - ptrY[i];
				ql += e*e;
			}
			return ql / (2 * activations.rows());
		}
		real_t loss_quadratic_mt_naive(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());

			const auto pA = activations.dataAsVec();
			const auto pY = data_y.dataAsVec();

			real_t ql = m_threads.reduce([pA,pY](const par_range_t& r)->real_t {
				const auto ofs = r.offset();
				const numel_cnt_t im = ofs + r.cnt();
				real_t ret = 0;
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const real_t e = pA[i] - pY[i];
					ret += e*e;
				}
				return ret;
			}, [](const real_t* ptr, const range_t cnt)->real_t {
				real_t ret = ptr[0];
				for (numel_cnt_t i = 1; i < cnt; ++i) ret += ptr[i];
				return ret;
			}, activations.numel());

			return ql / (2 * activations.rows());
		}

		// cross entropy function for sigmoid (applicable ONLY for binary data_y and sigmoid activation function)
		// L = -y*log(a)-(1-y)log(1-a), dL/dz = dL/dA * dA/dZ = (a-y)
		real_t loss_sigm_xentropy(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			if (activations.numel() < Thresholds_t::loss_sigm_xentropy) {
				return loss_sigm_xentropy_st_naivepart(activations, data_y);
			} else return loss_sigm_xentropy_mt_naivepart(activations, data_y);
		}
		real_t loss_sigm_xentropy_st_naivepart(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			NNTL_ASSERT(activations.size() == data_y.size() && !activations.empty() && !data_y.empty());
			const auto dataCnt = activations.numel();
			const auto ptrA = activations.dataAsVec(), ptrY = data_y.dataAsVec();
			constexpr auto log_zero = math::real_ty_limits<real_t>::log_almost_zero;
			real_t ql = 0;
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto y = ptrY[i];
				auto a = ptrA[i];
				NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
				NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

				if (y > real_t(0.0)) {
					ql += (a == real_t(0.0) ? log_zero : log(a));
				} else {
					const auto oma = real_t(1.0) - a;
					ql += (oma == real_t(0.0) ? log_zero : log(oma));
				}
				NNTL_ASSERT(!isnan(ql));
			}
			return -ql / activations.rows();
		}
		real_t loss_sigm_xentropy_mt_naivepart(const realmtx_t& activations, const realmtx_t& data_y)noexcept {
			const auto ptrA = activations.dataAsVec();
			const auto ptrY = data_y.dataAsVec();

			real_t ql = m_threads.reduce([ptrA, ptrY](const par_range_t& r)->real_t {
				const auto ofs = r.offset();
				const numel_cnt_t im = ofs + r.cnt();
				constexpr auto log_zero = math::real_ty_limits<real_t>::log_almost_zero;
				real_t ret = 0;
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto y = ptrY[i];
					auto a = ptrA[i];
					NNTL_ASSERT(y == real_t(0.0) || y == real_t(1.0));
					NNTL_ASSERT(a >= real_t(0.0) && a <= real_t(1.0));

					if (y > real_t(0.0)) {
						ret += (a == real_t(0.0) ? log_zero : log(a));
					} else {
						const auto oma = real_t(1.0) - a;
						ret += (oma == real_t(0.0) ? log_zero : log(oma));
					}
					NNTL_ASSERT(!isnan(ret));
				}
				return ret;
			}, [](const real_t* ptr, const range_t cnt)->real_t {
				real_t ret = ptr[0];
				for (numel_cnt_t i = 1; i < cnt; ++i) ret += ptr[i];
				return ret;
			}, activations.numel());
			return -ql / activations.rows();
		}

		//////////////////////////////////////////////////////////////////////////
		//gradient application procedures
		void RMSProp_Hinton(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			if (dW.numel() < Thresholds_t::RMSProp_Hinton) {
				RMSProp_Hinton_st(dW, rmsF, learningRate, emaDecay, numericStabilizer);
			}else RMSProp_Hinton_mt(dW, rmsF, learningRate, emaDecay, numericStabilizer);
		}
		void RMSProp_Hinton_st(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//TODO: this implementation probably isn't vectorized well

			auto pdW = dW.dataAsVec();
			auto prmsF = rmsF.dataAsVec();
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

			//TODO: this implementation probably isn't vectorized well

			auto pdW = dW.dataAsVec();
			auto prmsF = rmsF.dataAsVec();
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
				RMSProp_Graves_st(dW, rmsF, rmsG, learningRate, emaDecay, numericStabilizer);
			} else RMSProp_Graves_mt(dW, rmsF, rmsG, learningRate, emaDecay, numericStabilizer);
		}
		void RMSProp_Graves_st(realmtx_t& dW, realmtx_t& rmsF, realmtx_t& rmsG, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size() && rmsF.size()==rmsG.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//TODO: this implementation probably isn't vectorized well

			auto pdW = dW.dataAsVec();
			auto prmsF = rmsF.dataAsVec();
			auto prmsG = rmsG.dataAsVec();
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

			//TODO: this implementation probably isn't vectorized well

			auto pdW = dW.dataAsVec();
			auto prmsF = rmsF.dataAsVec();
			auto prmsG = rmsG.dataAsVec();
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
				RProp_st(dW, learningRate);
			} else RProp_mt(dW, learningRate);
		}
		void RProp_st(realmtx_t& dW, const real_t learningRate)noexcept {
			auto p = dW.dataAsVec();
			const auto pE = p + dW.numel();
			//TODO: verify vectorization
			while (p != pE) {
				*p++ = learningRate*math::sign(*p);
			}
		}
		void RProp_mt(realmtx_t& dW, const real_t learningRate)noexcept {
			auto pW = dW.dataAsVec();
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
				ModProp_st(dW, rmsF, learningRate, emaDecay, numericStabilizer);
			} else ModProp_mt(dW, rmsF, learningRate, emaDecay, numericStabilizer);
		}
		void ModProp_st(realmtx_t& dW, realmtx_t& rmsF, const real_t learningRate,
			const real_t emaDecay, const real_t numericStabilizer)noexcept
		{
			NNTL_ASSERT(dW.size() == rmsF.size());
			NNTL_ASSERT(emaDecay > 0 && emaDecay < 1);
			NNTL_ASSERT(numericStabilizer > 0 && numericStabilizer < 1);

			//TODO: this implementation probably isn't vectorized well

			auto pdW = dW.dataAsVec();
			auto prmsF = rmsF.dataAsVec();
			const auto _1_emaDecay = 1 - emaDecay;
			const auto dataCnt = dW.numel();
			for (numel_cnt_t i = 0; i < dataCnt; ++i) {
				const auto pW = pdW + i;
				const auto pF = prmsF + i;
				const auto w = *pW;
				const auto ema = (*pF)*emaDecay + abs(w)*_1_emaDecay;
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

			//TODO: this implementation probably isn't vectorized well

			auto pdW = dW.dataAsVec();
			auto prmsF = rmsF.dataAsVec();
			m_threads.run([pdW, prmsF, learningRate, emaDecay, numericStabilizer](const par_range_t& r) {
				const auto _1_emaDecay = 1 - emaDecay;
				const auto ofs = r.offset();
				const auto im = ofs + r.cnt();
				for (numel_cnt_t i = ofs; i < im; ++i) {
					const auto pW = pdW + i;
					const auto pF = prmsF + i;
					const auto w = *pW;
					const auto ema = (*pF)*emaDecay + abs(w)*_1_emaDecay;
					*pF = ema;
					*pW = learningRate*(w / (ema + numericStabilizer));
				}
			}, dW.numel());
		}


	public:
		real_t* _get_thread_temp_raw_storage(const numel_cnt_t maxDataSize)noexcept {
			_assert_thread_storage_allocated(maxDataSize);
			return &m_threadTempRawStorage[0];
		}
	protected:
		void _assert_thread_storage_allocated(const numel_cnt_t maxDataSize)const noexcept {
			NNTL_ASSERT(m_minTempStorageSize >= maxDataSize);
			NNTL_ASSERT(m_minPerThreadTempStorageSize > 0);
			NNTL_ASSERT(m_threadTempRawStorage.size() >= m_minTempStorageSize);
			NNTL_ASSERT(m_threadTempRawStorage.size() >= m_threadTempRawStoragePtrs.size()*m_minPerThreadTempStorageSize);
		}

		const thread_temp_storage_ptrs_t& _get_thread_temp_storage_ptrs(const numel_cnt_t maxDataSize)noexcept {
			_assert_thread_storage_allocated(maxDataSize);
			return m_threadTempRawStoragePtrs;
		}
		real_t**const _get_thread_temp_storage_ptrs_head(const numel_cnt_t maxDataSize)noexcept {
			_assert_thread_storage_allocated(maxDataSize);
			return &m_threadTempRawStoragePtrs[0];
		}
	};


}
}
