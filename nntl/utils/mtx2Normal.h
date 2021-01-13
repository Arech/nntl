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
	};

	//////////////////////////////////////////////////////////////////////////
	namespace _impl {
		//helper to delay self_t use until it is actually known
		template<typename ST>
		struct make_acc : public ::boost::accumulators::accumulator_set<typename ST::statsdata_t, ::boost::accumulators::stats<
			typename ST::scale_measure_t, typename ST::central_measure_t>>
		{
		private:
			typedef ::boost::accumulators::accumulator_set<typename ST::statsdata_t, ::boost::accumulators::stats<
				typename ST::scale_measure_t, typename ST::central_measure_t>> _base_t;
		public:
			template<typename ...ArgsT>
			make_acc(ArgsT... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
		};
	}

	template<typename FinalT, typename DataT, bool bAdjustForSampleVar = true, typename StatsT = DataT>
	struct _FNorm_base {
		typedef DataT datat_t;
		typedef StatsT statsdata_t;
		//to squeeze cheaply some additional precision while using float
		typedef ::std::conditional_t < (sizeof(statsdata_t) < sizeof(ext_real_t)), ext_real_t, statsdata_t > interm_statsdata_t;
		typedef ::nntl::math::smatrix<datat_t> datamtx_t;

		typedef FinalT self_t;
		NNTL_METHODS_SELF_CHECKED((::std::is_base_of<_FNorm_base<FinalT, DataT, bAdjustForSampleVar, StatsT>, FinalT>::value)
			, "FinalT must derive from _FNorm_base<FinalT, RealT, bAdjustForSampleVar, StatsT>");

		//////////////////////////////////////////////////////////////////////////
		//you probably don't want to redefine types of statistics, but anyway
		typedef ::boost::accumulators::tag::lazy_variance scale_measure_t; //note that this is biased variance estimator.
		typedef ::boost::accumulators::tag::mean central_measure_t;

		typedef _impl::make_acc<self_t> Accum_t;

		template<typename T>
		static ::std::enable_if_t<::std::is_floating_point<T>::value> die_check_fpvar(const T v)noexcept {
			if (::std::isnan(v) || !::std::isfinite(v)) {
				NNTL_ASSERT(!"Ops, bad floating point value!");
				//ok to die
			#pragma warning(push)
			#pragma warning(disable:4297)//function assumed not to throw
				throw ::std::runtime_error("Ops, bad floating point value passed to die_check_fpvar");
			#pragma warning(pop)
			}
		}
		template<typename T>
		static constexpr ::std::enable_if_t<!::std::is_floating_point<T>::value> die_check_fpvar(const T)noexcept {}

		// Resets object state to be ready to walk over the matrix.
		// Returns how many iterations (batch counts) must be done to satisfy preferred arguments
		nntl_interface numel_cnt_t prepareToWalk()noexcept;
		
		//returns a pointer to matrix data. Matrix must have at least 1 element (and more than 1 over all batches)
		//walk() is not required to obey batchIdx, it's just a convenience argument. The only requirement is that
		// the whole matrix must be walked over with all batches.
		// Note that the matrix returned MUST have samples in rows and batches in columns
		nntl_interface __declspec(restrict) const datamtx_t* __restrict walk(numel_cnt_t batchIdx)noexcept;

		template<typename AccT, bool c = bAdjustForSampleVar>
		static ::std::enable_if_t<c, statsdata_t> get_scale(const AccT& acc, bool& bScaleSpecial) noexcept {
			const interm_statsdata_t n = static_cast<interm_statsdata_t>(::boost::accumulators::count(acc));
			NNTL_ASSERT(n > interm_statsdata_t(1.));
			interm_statsdata_t unbiasVar = n / (n - interm_statsdata_t(1.0));
			const auto r = static_cast<statsdata_t>(::std::sqrt(unbiasVar*::boost::accumulators::extract_result<typename self_t::scale_measure_t>(acc)));
			bScaleSpecial = (r == statsdata_t(0.));
			return r;
		}

		template<typename AccT, bool c = bAdjustForSampleVar>
		static ::std::enable_if_t<!c, statsdata_t> get_scale(const AccT& acc, bool& bScaleSpecial) noexcept {
			const auto r = ::std::sqrt(::boost::accumulators::extract_result<typename self_t::scale_measure_t>(acc));
			bScaleSpecial = (r == statsdata_t(0.));
			return r;
		}

		template<typename AccT>
		static statsdata_t get_central(const AccT& acc) noexcept {
			return ::boost::accumulators::extract_result<typename self_t::central_measure_t>(acc);
		}
	};

	template<typename FinalT, typename DataT, bool bAdjustForSampleVar = true, typename StatsT = DataT>
	struct _FNorm_whole_base : public _FNorm_base<FinalT, DataT, bAdjustForSampleVar, StatsT> {
		//perform change scale operation over the whole data and return scaling factor
		nntl_interface void change_scale(const statsdata_t scaleVal)noexcept;
		nntl_interface void change_central(const statsdata_t centralVal)noexcept;
		nntl_interface void change_both(const statsdata_t scaleVal, const statsdata_t centralVal)noexcept;
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
		if (!(bNormScale || bNormCentral) || Setts.maxTries <= 0) return true;

		unsigned tryIdx = 0;
		for (; tryIdx < Setts.maxTries; ++tryIdx) {
			Accum_t acc;
			const numel_cnt_t batchesCnt = FNorm.prepareToWalk();
			NNTL_ASSERT(batchesCnt > 0);

			//walking over all dataset
			for (numel_cnt_t bidx = 0; bidx < batchesCnt; ++bidx) {
				const auto pAct = FNorm.walk(bidx);
				NNTL_ASSERT(pAct->bBatchesInColumns());
				applyAccumulator(pAct->begin(), pAct->end_no_bias(), acc);
			}

			bool bScaleSpecial;
			const auto scaleVal = FNorm.get_scale(acc, bScaleSpecial);
			const auto centralVal = FNorm.get_central(acc);

			const bool bCentralGood = !::std::isnan(centralVal) && ::std::isfinite(centralVal);
			const bool bScaleGood = !::std::isnan(scaleVal) && ::std::isfinite(scaleVal);

			const bool bScaleIsOk = (::std::abs(scaleVal - Setts.targetScale) < Setts.ScaleTolerance);
			const bool bCentralIsOk = (::std::abs(centralVal - Setts.targetCentral) < Setts.CentralTolerance);

			const bool bDoScale = !bScaleIsOk && bScaleGood && bNormScale && !bScaleSpecial;
			const bool bDoCentral = !bCentralIsOk && bCentralGood && bNormCentral;

			if (Setts.bVerbose) {
				STDCOUTL("#" << tryIdx << " scale (\"std\") = " << scaleVal
					<< (bScaleIsOk ? " ok!" : (bNormScale ? " BAD" : " #BAD, but no change allowed")) << ::std::endl
					<< "#" << tryIdx << " central (\"mean\") = " << centralVal
					<< (bCentralIsOk ? " ok!" : (bNormCentral ? " BAD" : " #BAD, but no change allowed")));
			}
			if (bScaleSpecial && Setts.bSayIfScaleSpecial) {
				STDCOUTL("  * note, scale has special value = " << scaleVal << ", skipping its modification");
			}

			if (!bCentralGood || !bScaleGood) {
				NNTL_ASSERT(!"got invalid statistics");
				STDCOUTL("*** got invalid statistics");
				if (Setts.bOnBadStatsBreak) return false;
			}

			if (bDoScale || bDoCentral) {
				if (bDoScale) {
					if (bDoCentral) {
						FNorm.change_both(scaleVal, centralVal);
					} else {//scale only
						FNorm.change_scale(scaleVal);
					}
				} else {//do central only
					FNorm.change_central(centralVal);
				}
			}

			if ((!bDoScale || bScaleIsOk) && (!bDoCentral || bCentralIsOk)) break;
		}

		return tryIdx < Setts.maxTries;
	};

	template<typename FinalT, typename DataT, bool bAdjustForSampleVar = true, typename StatsT = DataT>
	struct _FNorm_cw_base : public _FNorm_base<FinalT, DataT, bAdjustForSampleVar, StatsT> {
		nntl_interface vec_len_t total_cols()const noexcept;

		//forward args to iThreads.run();
		template<typename ...ArgsT>
		nntl_interface void iThreads_run(ArgsT&&... args)noexcept;

		template<typename AccT, bool c = bAdjustForSampleVar>
		nntl_probably_force_inline ::std::enable_if_t<c, statsdata_t> cw_get_scale(const vec_len_t colIdx, const AccT& acc, bool& bScaleSpecial)const noexcept {
			NNTL_UNREF(colIdx);
			return get_self().get_scale(acc, bScaleSpecial);
		}

		template<typename AccT, bool c = bAdjustForSampleVar>
		nntl_probably_force_inline ::std::enable_if_t<!c, statsdata_t> cw_get_scale(const vec_len_t colIdx, const AccT& acc, bool& bScaleSpecial)const noexcept {
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

		//perform change scale operation over the column colIdx of original data
		nntl_interface void cw_change_scale(const vec_len_t colIdx, const statsdata_t scaleVal)noexcept;
		nntl_interface void cw_change_central(const vec_len_t colIdx, const statsdata_t centralVal)noexcept;
		nntl_interface void cw_change_both(const vec_len_t colIdx, const statsdata_t scaleVal, const statsdata_t centralVal)noexcept;
	};

	template<typename SettsT, typename FNormT>
	bool normalize_cw(const SettsT& Setts, FNormT& FNorm)noexcept {
		typedef typename SettsT::statsdata_t statsdata_t;
		typedef typename SettsT::datat_t datat_t;
		typedef typename FNormT::Accum_t  Accum_t;

		static_assert(::std::is_same<datat_t, typename FNormT::datat_t>::value && ::std::is_same<statsdata_t, typename FNormT::statsdata_t>::value, "");

		const vec_len_t totalCols = FNorm.total_cols();
		const bool bNormScale = Setts.bScaleNormalize, bNormCentral = Setts.bCentralNormalize;
		
		if (!(bNormScale || bNormCentral) || Setts.maxTries <= 0) return true;

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
				NNTL_ASSERT(pAct->bBatchesInColumns());

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

				const bool bDoScale = !bScaleIsOk && bScaleGood && bNormScale && !bScaleSpecial;
				const bool bDoCentral = !bCentralIsOk && bCentralGood && bNormCentral;

				const bool bTreatScaleOK = (!bDoScale || bScaleIsOk);
				const bool bTreakCentralOK = (!bDoCentral || bCentralIsOk);

				bShowScaleOk &= bScaleIsOk;
				bShowCentralOk &= bCentralIsOk;

				bAllScaleOk &= bTreatScaleOK;
				bAllCentrOk &= bTreakCentralOK;

				if (!bCentralGood || !bScaleGood) {
					NNTL_ASSERT(!"got invalid statistics");
					STDCOUTL("*** got invalid statistics");
					if (Setts.bOnBadStatsBreak) {
						FNorm.cw_end();
						return false;
					}
				}
				if (bScaleSpecial && Setts.bSayIfScaleSpecial) {
					STDCOUTL("  * note, column#"<< colIdx << " scale has special value = " << scaleVal << ", skipping its modification");
				}

				if (bDoScale || bDoCentral) {
					if (bDoScale) {
						if (bDoCentral) {
							FNorm.cw_change_both(colIdx, scaleVal, centralVal);
						} else {//do scale only
							FNorm.cw_change_scale(colIdx, scaleVal);
						}
					} else {//do central only
						FNorm.cw_change_central(colIdx, centralVal);
					}
				}

				const bool bOk = bTreatScaleOK && bTreakCentralOK;
				bColumnOk[colIdx] = bOk;

				if (!bOk && tryIdx >= Setts.emitDbgAfterTry) {
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

			if (bAllScaleOk && bAllCentrOk) break;
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