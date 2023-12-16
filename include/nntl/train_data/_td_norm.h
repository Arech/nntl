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

#include <vector>

#include "_i_train_data.h"

#include "../utils/mtx2Normal.h"

//note: the code will probably be refactored in some way. All normalization-related code should probably be made independent of TD itself.

namespace nntl {

	template<typename DataT, typename StatsT = DataT>
	struct TDNormBaseSettings : public utils::mtx2Normal::Settings<DataT, StatsT> {
		vec_len_t batchSize{ -1 };//see walk_over_set() batchSize argument
		vec_len_t batchCount{ 0 };//0 means gather stats over whole dataset, else this count of batches
	};

	template<typename DataT, typename StatsT = DataT>
	struct TDNormSettings : public TDNormBaseSettings<DataT, StatsT> {
		bool bColumnwise{ false };//columns instead of whole matrix
	};

	//////////////////////////////////////////////////////////////////////////
	namespace _impl {
		
		// Note that it supports only smatrix::bBatchInColumn() mode
		
		template<typename XT, typename StatsT, typename StatsFuncT = utils::mtx2Normal::_FNorm_stats_var_mean<true>>
		class td_norm : public virtual DataSetsId {
		public:
			//////////////////////////////////////////////////////////////////////////
			//typedefs
			typedef td_norm<XT, StatsT, StatsFuncT> td_norm_self_t;
			
			typedef XT x_t;
			typedef StatsT stats_t;
			typedef StatsFuncT StatsFunctor_t;

			typedef math::smatrix<x_t> x_mtx_t;
			
			typedef TDNormSettings<x_t, stats_t> NormalizationSettings_t;
			typedef TDNormBaseSettings<x_t, stats_t> NormBaseSettings_t;
			
			//typedef utils::mtx2Normal::ScaleCentralData<stats_t> ScaleCentralData_t;
			//typedef ::std::vector<ScaleCentralData_t> vector_stats_t;

		//protected:

			template<typename TdT, typename CommonDataT>
			struct _NormBase {
				typedef TdT td_t;
				typedef NormBaseSettings_t Setts_t;
				typedef CommonDataT common_data_t;

				td_t& m_td;
				const Setts_t& m_Setts;
				const common_data_t& m_CD;

				_NormBase(td_t& td, const Setts_t& S, const common_data_t& cd) noexcept
					: m_td(td), m_Setts(S), m_CD(cd) {}

				// Returns how many iterations (batch counts) must be done to satisfy preferred arguments
				numel_cnt_t prepareToWalk()noexcept {
					const auto dataBatchCnt = m_td.walk_over_set(train_set_id, m_CD, m_Setts.batchSize, flag_exclude_dataY);
					NNTL_ASSERT(dataBatchCnt > 0);
					return ::std::min(dataBatchCnt, m_Setts.batchCount ? numel_cnt_t(m_Setts.batchCount) : dataBatchCnt);
				}
				//returns a pointer to matrix data. Matrix must have at least 1 element (and more than 1 over all batches)
				//walk() is not required to obey batchIdx, it's just a convenience argument. The only requirement is that
				// the whole matrix must be walked over with all batches.
				// I don't get how well MSVC works with restrict keyword (probably there are some bugs), so leaving this "crowded" ver with two kwrds
				__declspec(restrict) const x_mtx_t* __restrict walk(numel_cnt_t batchIdx)noexcept {
					m_td.next_subset(batchIdx, m_CD);
					NNTL_ASSERT(m_td.batchX().bBatchInColumn()); //mtx2Normal algos currently doesn't support the other mode
					return &m_td.batchX();
				}
			};

			template<typename FinalT, typename TdT, typename CommonDataT>
			class _Norm_whole
				: public utils::mtx2Normal::_FNorm_whole_base<FinalT, x_t, StatsFunctor_t, stats_t>
				, protected _NormBase<TdT, CommonDataT>
			{
				typedef utils::mtx2Normal::_FNorm_whole_base<FinalT, x_t, StatsFunctor_t, stats_t> _base_t;
				typedef _NormBase<TdT, CommonDataT> _nbase_t;

				//typedef ScaleCentralData_t indstat_t;
				typedef typename StatsFunctor_t::mtx_update_t::template ScaleCentralData_tpl<stats_t> indstat_t;

			public:
				using _nbase_t::walk;
				using _nbase_t::prepareToWalk;

				indstat_t m_Stat;

			protected:
				//for sequential update scale MUST be updated first
				void _upd_scale(const statsdata_t v)noexcept {
					//die_check_fpvar(v);
					m_Stat._upd_scale(v);
				}
				void _upd_ofs(const statsdata_t v)noexcept {
					//die_check_fpvar(v);
					m_Stat._upd_ofs(v);
				}

			public:
				template<typename ...ArgsT>
				_Norm_whole(ArgsT&&... args)noexcept : _nbase_t(::std::forward<ArgsT>(args)...) {}
				
				void _do_normalize_whole(const indstat_t& st)noexcept {
					//note that there may be several walks over a dataset to calculate proper stats,
					//so we can't pass cumulative m_Stat here instead of these vars which apply concretely to a single pass.
					m_td._fix_trainX_whole<typename self_t::mtx_update_t>(m_CD, st);
				}

				void normalize_whole(statsdata_t scaleVal, statsdata_t centralVal, const bool bScale, const bool bCentral)noexcept {
					if (bScale) {
						scaleVal = m_Setts.targetScale / scaleVal;
						get_self()._upd_scale(scaleVal);
					}else scaleVal = statsdata_t(1.);

					if (bCentral) {
						centralVal = (m_Setts.targetCentral - centralVal)*scaleVal;
						get_self()._upd_ofs(centralVal);
					}
					
					get_self()._do_normalize_whole(indstat_t(scaleVal, centralVal, bScale, bCentral));
				}
			};

			template<typename TdT, typename CommonDataT>
			class Norm_whole final : public _Norm_whole<Norm_whole<TdT, CommonDataT>, TdT, CommonDataT> {
				typedef _Norm_whole<Norm_whole<TdT, CommonDataT>, TdT, CommonDataT> _base_t;
			public:
				template<typename ...ArgsT>
				Norm_whole(ArgsT&&... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
			};

			//////////////////////////////////////////////////////////////////////////

			template<typename FinalT, typename TdT, typename CommonDataT>
			class _Norm_cw
				: public utils::mtx2Normal::_FNorm_cw_base<FinalT, x_t, StatsFunctor_t, stats_t>
				, protected _NormBase<TdT, CommonDataT>
			{
				typedef utils::mtx2Normal::_FNorm_cw_base<FinalT, x_t, StatsFunctor_t, stats_t> _base_t;
				typedef _NormBase<TdT, CommonDataT> _nbase_t;

			public:
				//typedef vector_stats_t fullstats_t;
				typedef typename StatsFunctor_t::mtx_update_t::template ScaleCentralVector_tpl<stats_t> fullstats_t;

			public:
				fullstats_t m_Stats;

			protected:
				fullstats_t m_iterStats;
				int iterations{ 0 };

			public:
				using _nbase_t::walk;
				using _nbase_t::prepareToWalk;

				template<typename ...ArgsT>
				_Norm_cw(ArgsT&&... args)noexcept : _nbase_t(::std::forward<ArgsT>(args)...) {
					const auto wid = total_cols();
					m_Stats.resize(wid);
					m_iterStats.resize(wid);
				}

			protected:
				//for sequential update scale MUST be updated first
				void _upd_scale(const vec_len_t c, const statsdata_t v)noexcept {
					//die_check_fpvar(v);
					m_Stats[c]._upd_scale(v);
					m_iterStats[c]._upd_scale(v);
				}
				void _upd_ofs(const vec_len_t c, const statsdata_t v)noexcept {
					//die_check_fpvar(v);
					m_Stats[c]._upd_ofs(v);
					m_iterStats[c]._upd_ofs(v);
				}

			public:
				vec_len_t total_cols()const noexcept { return m_td.xWidth(); }

				//forward args to iThreads.run();
				template<typename ...ArgsT>
				void iThreads_run(ArgsT&&... args)noexcept {
					return m_CD.get_iThreads().run(::std::forward<ArgsT>(args)...);
				}

				//resetting iteration statistics
				void cw_begin(const unsigned /*tryIdx*/) noexcept {
					for (auto& e : m_iterStats) e.reset();
				}
				//actually applying updates on iteration end
				void cw_end() noexcept {
					get_self()._do_normalize_cw();

					if (++iterations > 2) STDCOUTL("There were " << iterations << " iterations, please double check the normalization works correctly and delete that notice then!");
				}

				void _do_normalize_cw()noexcept {
					m_td._fix_trainX_cw<typename self_t::mtx_update_t>(m_CD, m_iterStats);
				}

				void normalize_cw(const vec_len_t colIdx, statsdata_t scaleVal, const statsdata_t centralVal
					, const bool bScale, const bool bCentral)noexcept
				{
					if (bScale) {
						NNTL_ASSERT(scaleVal != statsdata_t(0));
						scaleVal = m_Setts.targetScale / scaleVal;
						get_self()._upd_scale(colIdx, scaleVal);
					} else scaleVal = statsdata_t(1.);

					if (bCentral) {
						get_self()._upd_ofs(colIdx, (m_Setts.targetCentral - centralVal)*scaleVal);
					}
				}
			};

			template<typename TdT, typename CommonDataT>
			class Norm_cw final : public _Norm_cw<Norm_cw<TdT, CommonDataT>, TdT, CommonDataT> {
				typedef _Norm_cw<Norm_cw<TdT, CommonDataT>, TdT, CommonDataT> _base_t;
			public:
				template<typename ...ArgsT>
				Norm_cw(ArgsT&&... args)noexcept : _base_t(::std::forward<ArgsT>(args)...) {}
			};

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////

		public:
			//performs X data normalization using given settings. Statistics are gathered using train_x dataset and then applied
			//to every other X dataset.
			// Dataset MUST be properly initialized for inference
			// Support only bBatchInColumn()==true underlying X data matrices!
			// td must provide the following 4 functions in order to suit for the algo:
			// 
			// template<typename MtxUpdT, typename CommonDataT>
			// void _fix_trainX_whole(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st);
			// 
			// template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			// bool _fix_dataX_whole(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st);
			// 
			// template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			// void _fix_trainX_cw(const CommonDataT& cd, const typename MtxUpdT::ScaleCentralVector_tpl<StatsT>& allSt);
			// 
			// template<typename MtxUpdT, typename CommonDataT, typename StatsT>
			// bool _fix_dataX_cw(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralVector_tpl<StatsT>& allSt);
			//
			// see their implementation and comments in _impl::_td_base<> and _inmem_train_data<>. Also transf_train_data<>
			// 
			template<typename TdT, typename CommonDataT>
			static bool normalize_data(TdT& td, const CommonDataT& cd, const NormalizationSettings_t& Setts)noexcept {
				return Setts.bColumnwise
					? normalize_data_cw(td, cd, Setts)
					: normalize_data_whole(td, cd, Setts);
			}
			
			// Support only bBatchInColumn()==true underlying X data matrices!
			template<typename TdT, typename CommonDataT>
			static bool normalize_data_whole(TdT& td, const CommonDataT& cd, const NormBaseSettings_t& Setts)noexcept {
				if (!Setts.bShouldNormalize()) {
					//STDCOUTL("Hey, set proper flags, nothing to do now!");
					return true;
				}

				vec_len_t bs = Setts.batchSize < 0 ? 0 : Setts.batchSize;
				if (!td.is_initialized4inference(bs)) {
					NNTL_ASSERT(!"Initialize with at least init4inference() first");
					STDCOUTL("Initialize with at least init4inference() first");
					return false;
				}

				STDCOUTL("Doing train_x normalization over the whole data...");

				Norm_whole<TdT, CommonDataT> fn(td, Setts, cd);
				bool r = utils::mtx2Normal::normalize_whole(Setts, fn);
				if (r) {
					NNTL_ASSERT(fn.m_Stat.bScale || fn.m_Stat.bOffset);
					r = td._fix_dataX_whole<decltype(fn)::mtx_update_t>(cd, fn.m_Stat);
				}
				return r;
			}

			// Support only bBatchInColumn()==true underlying X data matrices!
			template<typename TdT, typename CommonDataT>
			static bool normalize_data_cw(TdT& td, const CommonDataT& cd, const NormBaseSettings_t& Setts)noexcept {
				if (!Setts.bShouldNormalize()) {
					//STDCOUTL("Hey, set proper flags, nothing to do now!");
					return true;
				}

				vec_len_t bs = Setts.batchSize < 0 ? 0 : Setts.batchSize;
				if (!td.is_initialized4inference(bs)) {
					NNTL_ASSERT(!"Initialize with at least init4inference() first");
					STDCOUTL("## Initialize with at least init4inference() first");
					return false;
				}
				STDCOUTL("Doing train_x normalization columnwise...");

				Norm_cw<TdT, CommonDataT> fn(td, Setts, cd);
				bool r = utils::mtx2Normal::normalize_cw(Setts, fn);
				if (r) r = td._fix_dataX_cw<decltype(fn)::mtx_update_t>(cd, fn.m_Stats);
				return r;
			}
		};
	}
}
