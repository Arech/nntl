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

#include <type_traits>

#pragma warning(push, 3)
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#pragma warning(pop)

namespace nntl {
namespace utils {
namespace mtx2Normal {
	//mtx2Normal is a collection of helper routines that performs given matrix normalization to given std and mean.
	//Works on either a whole matrix, or individual matrix columns. Matrix is processed in batches
	// See how to use it in LsuvExt.h

	template<typename DataT, typename StatsT = DataT>
	struct Settings { //: public math::smatrix_td {
		typedef DataT datat_t; //this is a type of data to normalize
		typedef StatsT statsdata_t; //this is a type of statistics (could be more precise, for example)

		//scale is ~std(), central ~mean(), but not necessary (depends on functor passed into function)
		statsdata_t targetScale{ statsdata_t(1.0) }, targetCentral{ statsdata_t(0.0) };
		statsdata_t ScaleTolerance{ statsdata_t(.01) }, CentralTolerance{ statsdata_t(.01) };

		unsigned maxTries{ 20 };
		unsigned emitDbgAfterTry{ 15 };

		bool bCentralNormalize{ true };
		bool bScaleNormalize{ true };

		bool bVerbose{ true };
		bool bOnBadStatsBreak{ true };

		bool bSayIfScaleSpecial{ true };

		bool bShouldNormalize()const noexcept {
			return (bCentralNormalize || bScaleNormalize) && maxTries > 0;
		}
	};

	template<typename StatsT>
	struct ScaleCentralData {
		typedef StatsT stats_t;

		stats_t sc;
		stats_t ofs;
		bool bScale, bOffset;

		ScaleCentralData()noexcept : sc(stats_t(1.)), ofs(stats_t(0.)), bScale(false), bOffset(false) {}
		ScaleCentralData(const stats_t s, const stats_t o, const bool bS, const bool bO)noexcept : sc(s), ofs(o), bScale(bS), bOffset(bO) {}


		/* ok, but doesn't help with assigning into boost::variant2<>. Had to use as_type<>()
		 *template<typename T, bool c = ::std::is_same<stats_t, T>::value>
		::std::enable_if_t<c, ScaleCentralData<stats_t>&> operator=(const ScaleCentralData<T>& other)noexcept {
			sc = other.sc;
			ofs = other.ofs;
			bScale = other.bScale;
			bOffset = other.bOffset;
			return *this;
		}

		template<typename T, bool c = ::std::is_same<stats_t, T>::value>
		::std::enable_if_t<!c, ScaleCentralData<stats_t>&> operator=(const ScaleCentralData<T>& other)noexcept {
			sc = static_cast<stats_t>(other.sc);
			ofs = static_cast<stats_t>(other.ofs);
			bScale = other.bScale;
			bOffset = other.bOffset;
			return *this;
		}*/


		template<typename T, bool c = ::std::is_same<stats_t, T>::value>
		::std::enable_if_t<c, const ScaleCentralData<stats_t>&> as_type()const noexcept {
			return *this;
		}
		template<typename T, bool c = ::std::is_same<stats_t, T>::value>
		::std::enable_if_t<!c, ScaleCentralData<T>> as_type()const noexcept {
			return ScaleCentralData<T>(static_cast<T>(sc), static_cast<T>(ofs), bScale, bOffset);
		}

		//////////////////////////////////////////////////////////////////////////
		// Note that scale & offset is expected in form x1=sc*x0 + ofs
		// For the form x1 = sc*(x0+ofs) we'd need different aggregation formula

		//scale only. Note that if there's an offset, it must be rescaled too
		void _upd_scale(const stats_t v)noexcept {
			sc *= v;
			ofs *= v; //if it doesn't exist (!bOffset) then ofs==0
			bScale = true;
		}
		//offset only. Offset doesn't affect the scale if any.
		void _upd_ofs(const stats_t v)noexcept {
			ofs += v;
			bOffset = true;
		}
		//note that if one to apply sequential update of scale and offset, the scale MUST be updated first.

		//simultaneous update
		void _upd_both(const stats_t vSc, const stats_t vOfs)noexcept {
			sc *= vSc;
			ofs = vSc*ofs + vOfs;
			bScale = bOffset = true;
		}

		//////////////////////////////////////////////////////////////////////////
		// for consistence of update statistics when source and destination types may be different
		// (for example in case when we measure in double, by apply in run-time to float we'd better to keep the stats to apply in float)
		template<typename T, bool c = ::std::is_same<stats_t, T>::value>
		::std::enable_if_t<c> _upd_all_typed(const ScaleCentralData<T>& other)noexcept {
			if (other.bScale) _upd_scale(other.sc);
			if (other.bOffset) _upd_ofs(other.ofs);
		}
		template<typename T, bool c = ::std::is_same<stats_t, T>::value>
		::std::enable_if_t<!c> _upd_all_typed(const ScaleCentralData<T>& other)noexcept {
			static_assert(sizeof(T) > sizeof(stats_t), "");//strange if not
			
			ScaleCentralData<T> tmp(sc, ofs, bScale, bOffset);

			if (other.bScale) tmp._upd_scale(other.sc);
			if (other.bOffset) tmp._upd_ofs(other.ofs);

			sc = static_cast<stats_t>(tmp.sc);
			ofs = static_cast<stats_t>(tmp.ofs);
			bScale = tmp.bScale;
			bOffset = tmp.bOffset;
		}

		//calculating with the same type without switching 
		template<typename T>
		void _upd_all_typed(const T s, const T o, const bool bS, const bool bO)noexcept {
			_upd_all_typed(ScaleCentralData<T>(s, o, bS, bO));
		}


		void reset()noexcept {
			sc = stats_t(1.);
			ofs = stats_t(0.);
			bScale = bOffset = false;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// collection of routines to actually update a matrix. Probably there should be no non-static members/types
	struct upd_mtx_scale_central {
		template<typename StatsT>
		using ScaleCentralData_tpl = ScaleCentralData<StatsT>;

		template<typename StatsT>
		using ScaleCentralVector_tpl = ::std::vector<ScaleCentralData_tpl<StatsT>>;
		
		// #supportsBatchInRow
		template<typename RealT, typename iMathT, typename StatsT>
		static void whole(iMathT& iM, ::nntl::math::smatrix<RealT>& m, const ScaleCentralData_tpl<StatsT>& st)noexcept
		{
			const auto mul = static_cast<RealT>(st.sc), ofs = static_cast<RealT>(st.ofs);
			const auto bOfs = st.bOffset;
			if (st.bScale) {
				if (bOfs) {
					iM.evMulCAddC_ip_nb(m, mul, ofs);
				} else {//scale only
					iM.evMulC_ip_nb(m, mul);
				}
			} else if (bOfs){//offset only	
				iM.evAddC_ip_nb(m, ofs);
			}
		}

		// doesn't support matrices in bBatchInRow() mode at this moment, but definitely should in future.
		template<typename RealT, typename iMathT, typename StatsT>
		static void batchwise(iMathT& iM, ::nntl::math::smatrix<RealT>& m, const ScaleCentralVector_tpl<StatsT>& allSt)noexcept {
			NNTL_ASSERT(m.bBatchInColumn());

			iM.ithreads().run([&iM, &allSt, &m](const auto& pr)noexcept {
				batchwise_st(iM, m, math::s_vec_range(pr), allSt);
			}, m.sample_size());
		}

	protected:
		template<typename RealT, typename iMathT, typename StatsT>
		static void batchwise_st(iMathT& iM, ::nntl::math::smatrix<RealT>& m, const ::nntl::math::s_vec_range& idxR
			, const ScaleCentralVector_tpl<StatsT>& st)noexcept
		{
			NNTL_ASSERT(m.bBatchInColumn());

			const auto colEnd = idxR.elmEnd;
			NNTL_ASSERT(colEnd <= st.size());
			NNTL_ASSERT(st.size() == m.cols_no_bias());
			vec_len_t colIdx = idxR.elmBegin;
			auto pM = m.colDataAsVec(colIdx);
			const auto ldm = static_cast<ptrdiff_t>(m.ldim());
			const auto mRows = static_cast<ptrdiff_t>(m.rows());

			for (; colIdx < colEnd; ++colIdx) {
				const auto& colSt = st[colIdx];

				const RealT vSc = static_cast<RealT>(colSt.sc), vOfs = static_cast<RealT>(colSt.ofs);

				//performing actual data normalization
				const bool bScale = colSt.bScale && !::std::isnan(vSc) && ::std::isfinite(vSc);
				const bool bOfs = colSt.bOffset && !::std::isnan(vOfs) && ::std::isfinite(vOfs);

				if (bScale | bOfs) {
					const auto pME = pM + mRows;
					if (bScale) {
						if (bOfs) {
							iM.vMulCAddC_ip_st(pM, pME, vSc, vOfs);
						} else {
							iM.vMulC_ip_st(pM, pME, vSc);
						}
					} else {
						iM.vAddC_ip_st(pM, pME, vOfs);
					}
				}

				pM += ldm;
			}
		}
	};


	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	
	//parameterless base !
	struct _FNorm_stats_var_mean_base {
		//////////////////////////////////////////////////////////////////////////
		//you probably don't want to redefine types of statistics, but it's possible in derived classes
		typedef ::boost::accumulators::tag::lazy_variance scale_measure_t; //note that this is biased variance estimator.
		typedef ::boost::accumulators::tag::mean central_measure_t;

		typedef upd_mtx_scale_central mtx_update_t;

		template<typename AccT>
		static typename AccT::sample_type get_central(const AccT& acc) noexcept {
			return ::boost::accumulators::extract_result<central_measure_t>(acc);
		}
	};

	template<bool bAdjustForSampleVar>
	struct _FNorm_stats_var_mean : public _FNorm_stats_var_mean_base {
		template<typename AccT, bool c = bAdjustForSampleVar>
		static ::std::enable_if_t<c, typename AccT::sample_type> get_scale(const AccT& acc, bool& bScaleSpecial) noexcept {
			typedef typename AccT::interm_statsdata_t interm_statsdata_t;

			const interm_statsdata_t n = static_cast<interm_statsdata_t>(::boost::accumulators::count(acc));
			NNTL_ASSERT(n > interm_statsdata_t(1.));
			interm_statsdata_t unbiasVar = n / (n - interm_statsdata_t(1.0));
			const auto r = static_cast<typename AccT::sample_type>(::std::sqrt(unbiasVar*::boost::accumulators::extract_result<scale_measure_t>(acc)));
			bScaleSpecial = (r == typename AccT::sample_type(0.));
			return r;
		}

		template<typename AccT, bool c = bAdjustForSampleVar>
		static ::std::enable_if_t<!c, typename AccT::sample_type> get_scale(const AccT& acc, bool& bScaleSpecial) noexcept {
			const auto r = ::std::sqrt(::boost::accumulators::extract_result<scale_measure_t>(acc));
			bScaleSpecial = (r == typename AccT::sample_type(0.));
			return r;
		}
	};

	namespace _impl {
		//helper to delay self_t use until it is actually known
		template<typename StatsDataT, typename IntermStatsT, typename StatsFuncT>
		struct make_acc : public ::boost::accumulators::accumulator_set<StatsDataT, ::boost::accumulators::stats<
			typename StatsFuncT::scale_measure_t, typename StatsFuncT::central_measure_t>>
		{
		private:
			typedef ::boost::accumulators::accumulator_set<StatsDataT, ::boost::accumulators::stats<
				typename StatsFuncT::scale_measure_t, typename StatsFuncT::central_measure_t>> _base_t;

		public:
			typedef IntermStatsT interm_statsdata_t;

		public:
			template<typename ...ArgsT>
			make_acc(ArgsT... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
		};
	}

	template<typename FinalT, typename DataT, typename StatsFuncT, typename StatsT>
	struct _FNorm_base : public StatsFuncT {
		typedef DataT datat_t;
		typedef StatsFuncT StatsFunctor_t;
		typedef StatsT statsdata_t;
		//to squeeze cheaply some additional precision while using float
		typedef ::std::conditional_t < (sizeof(statsdata_t) < sizeof(ext_real_t)), ext_real_t, statsdata_t > interm_statsdata_t;
		typedef ::nntl::math::smatrix<datat_t> datamtx_t;

		typedef FinalT self_t;
		NNTL_METHODS_SELF_CHECKED((::std::is_base_of<_FNorm_base<FinalT, DataT, StatsFuncT, StatsT>, FinalT>::value)
			, "FinalT must derive from _FNorm_base<FinalT, RealT, StatsFuncT, StatsT>");

		typedef _impl::make_acc<statsdata_t, interm_statsdata_t, StatsFunctor_t> Accum_t;

		//raising some necessary StatsFunctor_t definitions into this class context
		using StatsFunctor_t::mtx_update_t;
		using StatsFunctor_t::get_scale;
		using StatsFunctor_t::get_central;

		//////////////////////////////////////////////////////////////////////////
		// Resets object state to be ready to walk over the matrix.
		// Returns how many iterations (batch counts) must be done to satisfy preferred arguments
		nntl_interface numel_cnt_t prepareToWalk()noexcept;
		
		//returns a pointer to matrix data. Matrix must have at least 1 element (and more than 1 over all batches)
		//walk() is not required to obey batchIdx, it's just a convenience argument. The only requirement is that
		// the whole matrix must be walked over with all batches.
		// Note that the matrix returned MUST have samples in rows and batches in columns
		nntl_interface __declspec(restrict) const datamtx_t* __restrict walk(numel_cnt_t batchIdx)noexcept;

	};

	template<typename FinalT, typename DataT, typename StatsFuncT = _FNorm_stats_var_mean<true>, typename StatsT = DataT>
	struct _FNorm_whole_base : public _FNorm_base<FinalT, DataT, StatsFuncT, StatsT> {
		//if a corresponding flag allows use of its variable, the variable is guaranteed to be !nan && finite
		// scaleVal is also guaranteed to be !bScaleSpecial (see get_scale())
		nntl_interface void normalize_whole(const statsdata_t scaleVal, const statsdata_t centralVal, const bool bScale, const bool bCentral)noexcept;
	};

	//////////////////////////////////////////////////////////////////////////
	// FNormT& FNorm is a functor that perform iteration over data and implements actual data normalization
	template<typename SettsT, typename FNormT>
	bool normalize_whole(const SettsT& Setts, FNormT& FNorm)noexcept {
		typedef typename SettsT::statsdata_t statsdata_t;
		typedef typename SettsT::datat_t datat_t;
		typedef typename FNormT::Accum_t  Accum_t;

		static_assert(::std::is_same<datat_t, typename FNormT::datat_t>::value && ::std::is_same<statsdata_t, typename FNormT::statsdata_t>::value, "");

		const bool bNormScale = Setts.bScaleNormalize, bNormCentral = Setts.bCentralNormalize;
		if (!(bNormScale | bNormCentral) | (Setts.maxTries <= 0)) return true;

		unsigned tryIdx = 0;
		for (; tryIdx < Setts.maxTries; ++tryIdx) {
			Accum_t acc;
			const numel_cnt_t batchesCnt = FNorm.prepareToWalk();
			NNTL_ASSERT(batchesCnt > 0);

			//walking over all dataset
			for (numel_cnt_t bidx = 0; bidx < batchesCnt; ++bidx) {
				const auto pAct = FNorm.walk(bidx);
				NNTL_ASSERT(pAct->bBatchInColumn());//we care here, because we don't want the bias row to spoil stats!
				applyAccumulator(pAct->begin(), pAct->end_no_bias(), acc);
			}

			bool bScaleSpecial;
			const auto scaleVal = FNorm.get_scale(acc, bScaleSpecial);
			const auto centralVal = FNorm.get_central(acc);

			const bool bCentralGood = !::std::isnan(centralVal) && ::std::isfinite(centralVal);
			const bool bScaleGood = !::std::isnan(scaleVal) && ::std::isfinite(scaleVal);

			const bool bScaleIsOk = (::std::abs(scaleVal - Setts.targetScale) < Setts.ScaleTolerance);
			const bool bCentralIsOk = (::std::abs(centralVal - Setts.targetCentral) < Setts.CentralTolerance);

			const bool bDoScale = !bScaleIsOk & bScaleGood & bNormScale & !bScaleSpecial;
			const bool bDoCentral = !bCentralIsOk & bCentralGood & bNormCentral;

			if (Setts.bVerbose) {
				STDCOUTL("#" << tryIdx << " scale (\"std\") = " << scaleVal
					<< (bScaleIsOk ? " ok!" : (bNormScale ? " BAD" : " #BAD, but no change allowed")) << ::std::endl
					<< "#" << tryIdx << " central (\"mean\") = " << centralVal
					<< (bCentralIsOk ? " ok!" : (bNormCentral ? " BAD" : " #BAD, but no change allowed")));
			}
			if (bScaleSpecial & static_cast<bool>(Setts.bSayIfScaleSpecial)) {
				STDCOUTL("  * note, scale has special value = " << scaleVal << ", skipping its modification");
			}

			if (!bCentralGood | !bScaleGood) {
				NNTL_ASSERT(!"got invalid statistics");
				STDCOUTL("*** got invalid statistics");
				if (Setts.bOnBadStatsBreak) return false;
			}

			if (bDoScale | bDoCentral)
				FNorm.normalize_whole(scaleVal, centralVal, bDoScale, bDoCentral);

			if ((!bDoScale | bScaleIsOk) & (!bDoCentral | bCentralIsOk)) break;
		}

		return tryIdx < Setts.maxTries;
	};

	template<typename FinalT, typename DataT, typename StatsFuncT = _FNorm_stats_var_mean<true>, typename StatsT = DataT>
	struct _FNorm_cw_base : public _FNorm_base<FinalT, DataT, StatsFuncT, StatsT> {
		nntl_interface vec_len_t total_cols()const noexcept;

		//forward args to iThreads.run();
		template<typename ...ArgsT>
		nntl_interface void iThreads_run(ArgsT&&... args)noexcept;

		template<typename AccT>
		nntl_probably_force_inline statsdata_t cw_get_scale(const vec_len_t colIdx, const AccT& acc, bool& bScaleSpecial)const noexcept {
			NNTL_UNREF(colIdx);
			return get_self().get_scale(acc, bScaleSpecial);
		}

		template<typename AccT>
		nntl_probably_force_inline statsdata_t cw_get_central(const vec_len_t colIdx, const AccT& acc)const noexcept {
			NNTL_UNREF(colIdx);
			return get_self().get_central(acc);
		}

		static constexpr void cw_begin(const unsigned /*tryIdx*/) noexcept{}
		static constexpr void cw_end() noexcept{}

		nntl_interface void normalize_cw(const vec_len_t colIdx, const statsdata_t scaleVal, const statsdata_t centralVal
			, const bool bScale, const bool bCentral)noexcept;
	};

	template<typename SettsT, typename FNormT>
	bool normalize_cw(const SettsT& Setts, FNormT& FNorm)noexcept {
		typedef typename SettsT::statsdata_t statsdata_t;
		typedef typename SettsT::datat_t datat_t;
		typedef typename FNormT::Accum_t  Accum_t;

		static_assert(::std::is_same<datat_t, typename FNormT::datat_t>::value && ::std::is_same<statsdata_t, typename FNormT::statsdata_t>::value, "");

		const vec_len_t totalCols = FNorm.total_cols();
		const bool bNormScale = Setts.bScaleNormalize, bNormCentral = Setts.bCentralNormalize;
		
		if (!(bNormScale | bNormCentral) | (Setts.maxTries <= 0)) return true;

		::std::vector<bool> bColumnOk(totalCols, false);
		::std::vector<Accum_t> Accums(totalCols);

		unsigned tryIdx = 0;
		for (; tryIdx < Setts.maxTries; ++tryIdx) {
			for (vec_len_t colIdx = 0; colIdx < totalCols; ++colIdx) Accums[colIdx] = {};//resetting

			const numel_cnt_t batchesCnt = FNorm.prepareToWalk();
			NNTL_ASSERT(batchesCnt > 0);

			//walking over all dataset
			for (numel_cnt_t bidx = 0; bidx < batchesCnt; ++bidx) {
				const auto pAct = FNorm.walk(bidx);
				NNTL_ASSERT(pAct->bBatchInColumn());//and we really care here!

				FNorm.iThreads_run([pAct, pAccs = &Accums](const auto& pr)noexcept {
					const ptrdiff_t batchSize = static_cast<ptrdiff_t>(pAct->rows());
					const ptrdiff_t lda = static_cast<ptrdiff_t>(pAct->ldim());
					vec_len_t colIdx = static_cast<vec_len_t>(pr.offset());

					auto pA = pAct->colDataAsVec(colIdx--);
					const auto pAE = pAct->colDataAsVec(static_cast<vec_len_t>(pr.end()));
					while (pA != pAE) {
						applyAccumulator(pA, pA + batchSize, (*pAccs)[++colIdx]);
						pA += lda;
					}
					NNTL_ASSERT(pA <= pAct->end_no_bias());
				}, totalCols);
			}

			statsdata_t scaleSum(0), centrSum(0);
			bool bAllScaleOk = true, bAllCentrOk = true, bEmitNL = false;
			bool bShowScaleOk = true, bShowCentralOk = true;
			bool bScaleSpecial;

			FNorm.cw_begin(tryIdx);

			for (vec_len_t colIdx = 0; colIdx < totalCols; ++colIdx) {
				const auto scaleVal = FNorm.cw_get_scale(colIdx, Accums[colIdx], bScaleSpecial);
				const auto centralVal = FNorm.cw_get_central(colIdx, Accums[colIdx]);

				scaleSum += scaleVal;
				centrSum += centralVal;

				if (bColumnOk[colIdx])
					continue;

				const bool bCentralGood = !::std::isnan(centralVal) && ::std::isfinite(centralVal);
				const bool bScaleGood = !::std::isnan(scaleVal) && ::std::isfinite(scaleVal);

				const bool bScaleIsOk = (::std::abs(scaleVal - Setts.targetScale) < Setts.ScaleTolerance);
				const bool bCentralIsOk = (::std::abs(centralVal - Setts.targetCentral) < Setts.CentralTolerance);

				const bool bDoScale = (!bScaleIsOk) & bScaleGood & bNormScale & (!bScaleSpecial);
				const bool bDoCentral = (!bCentralIsOk) & bCentralGood & bNormCentral;

				const bool bTreatScaleOK = (!bDoScale | bScaleIsOk);
				const bool bTreakCentralOK = (!bDoCentral | bCentralIsOk);

				bShowScaleOk &= bScaleIsOk;
				bShowCentralOk &= bCentralIsOk;

				bAllScaleOk &= bTreatScaleOK;
				bAllCentrOk &= bTreakCentralOK;

				if (!bCentralGood | !bScaleGood) {
					NNTL_ASSERT(!"got invalid statistics");
					STDCOUTL("*** got invalid statistics");
					if (Setts.bOnBadStatsBreak) {
						FNorm.cw_end();
						return false;
					}
				}
				if (bScaleSpecial & static_cast<bool>(Setts.bSayIfScaleSpecial)) {
					STDCOUTL("  * note, column#"<< colIdx << " scale has special value = " << scaleVal << ", skipping its modification");
				}

				if (bDoScale | bDoCentral)
					FNorm.normalize_cw(colIdx, scaleVal, centralVal, bDoScale, bDoCentral);

				const bool bOk = bTreatScaleOK & bTreakCentralOK;
				bColumnOk[colIdx] = bOk;

				if (!bOk & (tryIdx >= Setts.emitDbgAfterTry)) {
					bEmitNL = true;
					STDCOUT(" " << colIdx << "(" << centralVal << ", " << scaleVal << ")");
				}
			}
			if (bEmitNL) STDCOUT(::std::endl);

			FNorm.cw_end();

			if (Setts.bVerbose) {
				STDCOUTL("#" << tryIdx << " AVG scale (\"std\") = " << scaleSum / totalCols
					<< (bShowScaleOk ? " ok!" : (bNormScale ? " BAD" : " #BAD, but no change allowed")) << ::std::endl
					<< "#" << tryIdx << " AVG central (\"mean\") = " << centrSum / totalCols
					<< (bShowCentralOk? " ok!" : (bNormCentral ? " BAD" : " #BAD, but no change allowed")));
			}

			if (bAllScaleOk & bAllCentrOk) break;
		}

		return tryIdx < Setts.maxTries;
	};

	//////////////////////////////////////////////////////////////////////////
	template<typename DataT, typename AccT>
	static nntl_probably_force_inline void applyAccumulator(const DataT* __restrict pBegin
		, const DataT* __restrict const pEnd, AccT& acc)noexcept
	{
		while (pBegin < pEnd) {
			acc(*pBegin++);
		}
	};

}
}
}