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

#include "_i_train_data.h"

#include "_td_norm.h"

namespace nntl {
namespace _impl {

	// If not mentioned explicitly in a function comment, every member function of the class #supportsBatchInRow (at least it should)
	template<typename FinalPolymorphChild, typename XT, typename YT>
	class _td_base : public _i_train_data<XT, YT> {
	public:
		typedef FinalPolymorphChild self_t;
		NNTL_METHODS_SELF_CHECKED((::std::is_base_of<_i_train_data, FinalPolymorphChild>::value)
			, "FinalPolymorphChild must derive from _i_train_data<FinalPolymorphChild>");

	protected:
		x_mtx_t* m_pCurBatchX{ nullptr };
		y_mtx_t* m_pCurBatchY{ nullptr };

		vec_len_t m_maxFPropSize{ 0 }, m_maxTrainBatchSize{ 0 }, m_curBatchSize{ 0 };
		data_set_id_t m_curDataset2Walk{ invalid_set_id }; // set to non invalid_set_id value only for inferencing

	public:
		//////////////////////////////////////////////////////////////////////////
		// default implementation, redefine in derived class if necessary
		data_set_id_t datasets_count()const noexcept { return 2; }

		static DatasetNamingFunc_t get_dataset_naming_function() noexcept {
			return [](data_set_id_t dataSetId, char* pName, unsigned siz) {
				switch (dataSetId) {
				case train_set_id:
					::strcpy_s(pName, siz, "train");
					break;

				case test_set_id:
					::strcpy_s(pName, siz, "test");
					break;

				default:
					::sprintf_s(pName, siz, "!Unknown!%d", dataSetId);
					break;
				}
			};
		}

		bool isSuitableForOutputOf(neurons_count_t n)const noexcept { return n == get_self().yWidth(); }

		//////////////////////////////////////////////////////////////////////////
		//convenience wrappers around dataset_samples_count()
		numel_cnt_t trainset_samples_count()const noexcept { return get_self().dataset_samples_count(train_set_id); }
		numel_cnt_t testset_samples_count()const noexcept { return get_self().dataset_samples_count(test_set_id); }

		//////////////////////////////////////////////////////////////////////////

		//convenience wrappers around walk_over_set(). Always uses maxFPropSize returned from init*() as a batch size
		template<typename CommonDataT>
		numel_cnt_t walk_over_train_set(const CommonDataT& cd)noexcept {
			NNTL_ASSERT(get_self().is_initialized4inference());
			return get_self().walk_over_set(train_set_id, cd);
		}
		template<typename CommonDataT>
		numel_cnt_t walk_over_test_set(const CommonDataT& cd)noexcept {
			NNTL_ASSERT(get_self().is_initialized4inference());
			return get_self().walk_over_set(test_set_id, cd);
		}

		//////////////////////////////////////////////////////////////////////////

		bool is_initialized4inference()const noexcept {
			NNTL_ASSERT(!get_self().empty());
			return m_maxFPropSize > 0;
		}

		bool is_initialized4train()const noexcept {
			NNTL_ASSERT(!get_self().empty());
			return m_maxTrainBatchSize > 0 && m_maxFPropSize > 0;
		}

		//returns the values that had corresponding vars on init*() exit
		vec_len_t get_maxFPropSize()const noexcept { return m_maxFPropSize; }
		vec_len_t get_maxTrainBatchSize()const noexcept { return m_maxTrainBatchSize; }

		//redefine in derived if needed.
// 		vec_len_t get_orig_maxFPropSize()const noexcept { return m_maxFPropSize; }
// 		vec_len_t get_orig_maxTrainBatchSize()const noexcept { return m_maxTrainBatchSize; }

		void deinit4all()noexcept {
			m_pCurBatchX = nullptr;
			m_pCurBatchY = nullptr;
			m_maxTrainBatchSize = m_maxFPropSize = m_curBatchSize = 0;
			m_curDataset2Walk = invalid_set_id;
		}

		//for training & inference mode to return
		// - corresponding training set batches for given on_next_epoch(epochIdx) and on_next_batch(batchIdx) (if they were called just before)
		//		Each batch always contain the same amount of rows equal to the one was set to cd.get_cur_batch_size() during on_next_epoch()
		// - corresponding batches of data if walk_over_set()/next_subset() were called before
		//		Each batch contain at max maxFPropSize rows (set in init4inference())
		// Note that once the mode was changed, the caller MUST assume that underlying data, returned by these functions, became a junk
		const x_mtx_t& batchX()const noexcept { NNTL_ASSERT(m_pCurBatchX); return *m_pCurBatchX; }
		const y_mtx_t& batchY()const noexcept { NNTL_ASSERT(m_pCurBatchY); return *m_pCurBatchY; }
	protected:
		x_mtx_t& batchX_mutable()noexcept { NNTL_ASSERT(m_pCurBatchX); return *m_pCurBatchX; }
		y_mtx_t& batchY_mutable()noexcept { NNTL_ASSERT(m_pCurBatchY); return *m_pCurBatchY; }

	public:
		//////////////////////////////////////////////////////////////////////////

		numel_cnt_t biggest_samples_count()const noexcept {
			const auto tds = get_self().datasets_count();
			numel_cnt_t r = 0;
			for (data_set_id_t i = 0; i < tds; ++i) {
				r = ::std::max(r, get_self().dataset_samples_count(i));
			}
			return r;
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////// 
		// data normalization related stuff
		// Note, it supports bBatchInColumn() matrix mode only for now!

		typedef utils::mtx2Normal::_FNorm_stats_var_mean<true> default_StatsFuncT;

		template<typename ST, typename StatsFuncT>
		using NormalizerF_tpl = _impl::td_norm<x_t, ST, StatsFuncT>;
		
		template<typename StatsT, typename CommonDataT, typename StatsFuncT = typename self_t::default_StatsFuncT
			, typename NormF = self_t::template NormalizerF_tpl<StatsT, StatsFuncT>>
		bool normalize_data(const CommonDataT& cd, const typename NormF::NormalizationSettings_t& Setts, const NormF& n = NormF())noexcept {
			static_assert(::std::is_same<StatsT, typename NormF::stats_t>::value, "");
			NNTL_ASSERT(!get_self().empty());

			if (Setts.bShouldNormalize()) {
				if (Setts.batchSize > 0 && get_self().get_maxFPropSize() > 0 && get_self().get_maxFPropSize() < Setts.batchSize) {
					NNTL_ASSERT(!"Improper initialization");
					STDCOUTL("Improper initialization");
					return false;
				}
				const bool bDoInit = !get_self().is_initialized4inference();
				if (bDoInit) {
					vec_len_t mfs = Setts.batchSize < 0 ? 0 : Setts.batchSize;
					const auto ec = get_self().init4inference(mfs);
					if (nnet_errors_t::Success != ec) {
						STDCOUTL("Failed to initialize TD for normalization: " << _nnet_errs::get_error_str(ec));
						return false;
					}
				}

				bool r = n.normalize_data(get_self(), cd, Setts);

				if (bDoInit) get_self().deinit4all();

				if (!r) {
					STDCOUTL("Normalization failed");
					return false;
				}
			} else STDCOUTL("Skipping dataset normalization");
			return true;
		}

		// normalization support functions

		//redefine in derived class to override
		template<typename MtxUpdT, typename CommonDataT, typename StatsT>
		void _do_fix_dataX_whole(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st)noexcept
		{
			const auto dc = get_self().datasets_count();
			for (data_set_id_t dsi = test_set_id; dsi < dc; ++dsi) {
				get_self()._fix_trainX_whole<MtxUpdT>(cd, st, dsi);
			}
		}
		template<typename MtxUpdT, typename CommonDataT, typename StatsT>
		void _do_fix_dataX_cw(const CommonDataT& cd
			, const typename MtxUpdT::template ScaleCentralVector_tpl<StatsT>& allSt)noexcept
		{
			const auto dc = get_self().datasets_count();
			for (data_set_id_t dsi = test_set_id; dsi < dc; ++dsi) {
				get_self()._fix_trainX_cw<MtxUpdT>(cd, allSt, dsi);
			}
		}

		template<typename MtxUpdT, typename CommonDataT, typename StatsT>
		bool _fix_dataX_whole(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralData_tpl<StatsT>& st)noexcept {
			STDCOUT("TrainX dataset stats: ");
			if (st.bOffset) {
				STDCOUT(st.ofs);
			} else STDCOUT("n/a");
			STDCOUT(", ");
			if (st.bScale) {
				STDCOUT(st.sc);
			} else STDCOUT("n/a");
			STDCOUTL(". Going to update other datasets...");

			//checks for nan/finiteness is not necessary btw, should be guaranteed
			const bool bScale = st.bScale && !::std::isnan(st.sc) && ::std::isfinite(st.sc);
			const bool bOfs = st.bOffset && !::std::isnan(st.ofs) && ::std::isfinite(st.ofs);

			if (!bScale && !bOfs) {
				STDCOUTL("Failed to obtain statistics on train_x, cancelling normalization");
				return false;
			}

			get_self()._do_fix_dataX_whole<MtxUpdT>(cd, st);
			return true;
		}

		template<typename MtxUpdT, typename CommonDataT, typename StatsT>
		bool _fix_dataX_cw(const CommonDataT& cd, const typename MtxUpdT::template ScaleCentralVector_tpl<StatsT>& allSt)noexcept {
			const auto siz = allSt.size();
			StatsT sumSc{ StatsT(0.) }, sumCentr{ StatsT(0.) };
			int nSc{ 0 }, nOfs{ 0 }, nSkip{ 0 };

			for (size_t idx = 0; idx < siz; ++idx) {
				const auto& ist = allSt[idx];
				const auto vSc = ist.sc, vOfs = ist.ofs;

				const bool bScale = ist.bScale && !::std::isnan(vSc) && ::std::isfinite(vSc);
				const bool bOfs = ist.bOffset && !::std::isnan(vOfs) && ::std::isfinite(vOfs);

				if (bScale || bOfs) {
					if (bScale) {
						++nSc;
						sumSc += vSc;
					}
					if (bOfs) {
						++nOfs;
						sumCentr += vOfs;
					}
				} else {
					STDCOUTL("** Note, row " << idx << " doesn't have normalization data at all! Double check it's ok");
					++nSkip;
				}
			}

			STDCOUTL("Average train_x stats: " << (sumCentr / nOfs) << ", " << (sumSc / nSc) << ". " << nSkip << " rows skipped." \
				" Applying to the rest datasets...");

			if (nSkip == siz) {
				STDCOUTL("## nothing to apply!");
				return false;
			}

			get_self()._do_fix_dataX_cw<MtxUpdT>(cd, allSt);
			return true;
		}
	};

}
}
