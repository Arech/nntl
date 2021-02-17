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

namespace nntl {
namespace math {

	//functors to use with mcwFindKOrdered, member functions must be static
	//Note that it's possible to speedup even more by utilizing the same idea as in ::bbost::sort::spreadsort::float_sort
	// https://www.boost.org/doc/libs/1_75_0/libs/sort/doc/html/sort/single_thread/spreadsort/sort_hpp/float_sort.html
	// i.e. compare floats as integers, but that's a bit too much for now, leave for #TODO

	template<typename T>
	struct Order_BiggestDistinct {
		static constexpr T most_extreme()noexcept { return ::std::numeric_limits<T>::lowest(); }
		static constexpr bool first_better(const T& vNew, const T& vOld)noexcept { return vNew > vOld; }
	};
	template<typename T>
	struct Order_Biggest {
		static constexpr T most_extreme()noexcept { return ::std::numeric_limits<T>::lowest(); }
		static constexpr bool first_better(const T& vNew, const T& vOld)noexcept { return vNew >= vOld; }
	};

	template<typename T>
	struct Order_SmallestDistinct {
		static constexpr T most_extreme()noexcept { return ::std::numeric_limits<T>::max(); }
		static constexpr bool first_better(const T& vNew, const T& vOld)noexcept { return vNew < vOld; }
	};
	template<typename T>
	struct Order_Smallest {
		static constexpr T most_extreme()noexcept { return ::std::numeric_limits<T>::max(); }
		static constexpr bool first_better(const T& vNew, const T& vOld)noexcept { return vNew <= vOld; }
	};

	//////////////////////////////////////////////////////////////////////////
	//support structure for mcwFindKOrdered()
	//helps to determine proper datatype of helper/cache/result matrix and it's size.
	// The idea behind is to keep the computation result and temporarily data as close in memory as possible
	// Traverse hlprmtx_t only using this helper functions
	template<typename SrcT, typename IdxT = vec_len_t>
	struct mcwFindKOrdered_hlpr_base {
		typedef mcwFindKOrdered_hlpr_base<SrcT> hlpr_base_t;

		static_assert(::std::is_arithmetic<SrcT>::value, "");

		static_assert(::std::is_same<SrcT, float>::value || ::std::is_same<SrcT, double>::value
			, "Tested only with float/double, test with new type and proper data sizes first");

		typedef SrcT src_value_t;
		typedef IdxT idxs_t;

		static constexpr bool bSourceTFirst = (sizeof(src_value_t) >= sizeof(idxs_t));

		//biggest of SrcT and idxs_t while keeping a floating-point`ness (b/c it usually has stricter memory alignment rules).
		typedef ::std::conditional_t< bSourceTFirst
			, src_value_t
			, ::std::conditional_t< ::std::is_floating_point<src_value_t>::value, typename int_t_limits<idxs_t>::similar_FP_t, idxs_t>
		> value_t;

		typedef smatrix<src_value_t> srcmtx_t;

		typedef smatrix<value_t> hlprmtx_t;

		// 			class hlprmtx_t : public math::smatrix<value_t> {
		// 				typedef math::smatrix<value_t> _base_t;
		// 			public:
		// 				vec_len_t k{0};
		// 
		// 				template<typename... AT>
		// 				hlprmtx_t(AT&&... a)noexcept : _base_t(::std::forward<AT>(a)...) {}
		// 			};

	protected:
		typedef ::std::conditional_t<bSourceTFirst, src_value_t, idxs_t> first_t;
		typedef ::std::conditional_t<bSourceTFirst, idxs_t, src_value_t> second_t;

	public:

		//first k elements of hlprmtx are always the elements of biggest/strictest type
		//so we only need to find how many elements of the biggest type needed to store k elements of smallest type
		template<bool bSS = (sizeof(first_t) == sizeof(second_t))>
		static ::std::enable_if_t<bSS, vec_len_t> hlprmtx_numel_for_k(const vec_len_t k)noexcept {
			return 2 * k;
		}
		template<bool bSS = (sizeof(first_t) == sizeof(second_t))>
		static ::std::enable_if_t<!bSS, vec_len_t> hlprmtx_numel_for_k(const vec_len_t k)noexcept {
			const auto scndTypeElms = static_cast<vec_len_t>((k * sizeof(second_t) + sizeof(first_t) - 1) / sizeof(first_t));
			NNTL_ASSERT(scndTypeElms <= k);
			const auto r = k + scndTypeElms;
			NNTL_ASSERT(k == get_k_for_hlpr_rows(r));
			return r;
		}


		template<bool bSS = (sizeof(first_t) == sizeof(second_t))>
		static ::std::enable_if_t<bSS, vec_len_t> get_k_for_hlpr_rows(const vec_len_t hlpr_rows)noexcept {
			NNTL_ASSERT(0 == hlpr_rows % 2);
			return hlpr_rows / 2;
		}

		template<bool bSS = (sizeof(first_t) == sizeof(second_t))>
		static ::std::enable_if_t<!bSS, vec_len_t> get_k_for_hlpr_rows(const vec_len_t hlpr_rows)noexcept {
			// numerator is max possible bytes taken, while denominator is a real bytes taken by pair
			return static_cast<vec_len_t>((hlpr_rows * sizeof(first_t)) / (sizeof(first_t) + sizeof(second_t)));
		}

		static vec_len_t get_k(const hlprmtx_t& hlpr)noexcept {
			return get_k_for_hlpr_rows(hlpr.rows());
			//return hlpr.k;
		}

		static smatrix_td::mtx_size_t hlprmtx_size_for(const vec_len_t k, const vec_len_t cols_nb)noexcept {
			return smatrix_td::mtx_size_t(hlprmtx_numel_for_k(k), cols_nb);
			// 				const auto nek = hlprmtx_numel_for_k(k);
			// 				static constexpr size_t _CLm1 = 64 - 1;
			// 				return math::smatrix_td::mtx_size_t( static_cast<vec_len_t>((nek + _CLm1) &(~_CLm1)), cols_nb);
			// 				//tried to make row size a multiple of cache line length to reduce false sharing, but it gave strange results:
			// 				//minimum run-time became less indeed. However, total time of multiple runs became consistently longer and
			// 				//what's even more strange - other sibling algos became slower. HW is weird.
		}
		static bool is_hlprmtx_fine_for(const hlprmtx_t& hlpr, const srcmtx_t& src)noexcept {
			return !hlpr.empty() && !hlpr.emulatesBiases() && (get_k(hlpr) > 0) && src.cols_no_bias() == hlpr.cols();
		}

		// each hlpr column consists of first [k] elements of bigger/equal to src_value_t type and then [k] elements of rest type
		// value_t is biggest of idxs_t/src_value_t so there should be no alignment issues.
		template<bool bF = bSourceTFirst>
		static constexpr ::std::enable_if_t<bF, idxs_t*__restrict> get_idxs_ptr_from_column_ptr(value_t*const pHlprColumn, const vec_len_t k) noexcept {
			return reinterpret_cast<idxs_t*>(pHlprColumn + k);
		}
		template<bool bF = bSourceTFirst>
		static constexpr ::std::enable_if_t<!bF, idxs_t*__restrict> get_idxs_ptr_from_column_ptr(value_t*const pHlprColumn, const vec_len_t /*k*/) noexcept {
			return reinterpret_cast<idxs_t*>(pHlprColumn);
		}
		static constexpr const idxs_t*__restrict get_idxs_ptr_from_column_ptr(const value_t*const pHlprColumn, const vec_len_t k) noexcept {
			return get_idxs_ptr_from_column_ptr(const_cast<value_t*const>(pHlprColumn), k);
		}

		template<bool bF = bSourceTFirst>
		static ::std::enable_if_t<bF, src_value_t*__restrict> get_cache_ptr_from_column_ptr(value_t*const pHlprColumn, const vec_len_t /*k*/)noexcept {
			return reinterpret_cast<src_value_t*>(pHlprColumn);
		}
		template<bool bF = bSourceTFirst>
		static ::std::enable_if_t<!bF, src_value_t*__restrict> get_cache_ptr_from_column_ptr(value_t*const pHlprColumn, const vec_len_t k)noexcept {
			return reinterpret_cast<src_value_t*>(pHlprColumn + k);
		}
		static const src_value_t*__restrict get_cache_ptr_from_column_ptr(const value_t*const pHlprColumn, const vec_len_t k)noexcept {
			return get_cache_ptr_from_column_ptr(const_cast<value_t*const>(pHlprColumn), k);
		}
	};
	
	template<typename SrcT, template<class>class OrderTpl>
	class mcwFindKOrdered_hlpr : public mcwFindKOrdered_hlpr_base<SrcT> {
		typedef mcwFindKOrdered_hlpr_base<SrcT> _base_t;

	public:
		typedef OrderTpl<SrcT> OrderFunctor_t;
		typedef mcwFindKOrdered_hlpr<SrcT, OrderTpl> mcwFindKOrdered_hlpr_t;
		
		//mainly for debug use
		static bool is_hlprmtx_ok(const hlprmtx_t& hlpr)noexcept {
			const auto k = get_k(hlpr), colms = hlpr.cols();
			const auto ldH = hlpr.ldim();
			auto pH = hlpr.data();
			const auto pHE = hlpr.end();
			while (pH != pHE) {
				const auto pV = get_cache_ptr_from_column_ptr(pH, k);
				for (vec_len_t r = 1; r < k; ++r) {
					//if (pV[r - 1] >= pV[r]) {
					if (!OrderFunctor_t::first_better(pV[r], pV[r - 1])) {
						NNTL_ASSERT(!"Incoherent data!");
						return false;
					}
				}
				pH += ldH;
			}
			return true;
		}

		//////////////////////////////////////////////////////////////////////////
		// aggregation over many batches(source matrices) support
		struct Aggregator {
		protected:
			struct DatasetElement {
				numel_cnt_t batchIdx;
				src_value_t val;
				vec_len_t idx;
			};
			//DatasetElement must be POD type to be placed into smatrix
			smatrix<DatasetElement> m_allData;

		public:
			Aggregator()noexcept {}

			bool export_values(srcmtx_t& vals)const noexcept {
				NNTL_ASSERT(!m_allData.empty());

				if (!vals.empty()) vals.clear();
				vals.dont_emulate_biases();
				vals.set_batchInRow(false);

				const auto k = m_allData.rows(), colmn = m_allData.cols();
				if (!vals.resize(k, colmn)) return false;

				for (vec_len_t c = 0; c < colmn; ++c) {
					const auto pDSE = m_allData.colDataAsVec(c);
					const auto pV = vals.colDataAsVec(c);
					for (vec_len_t r = 0; r < k; ++r) {
						//changing the ordering to descending
						//NNTL_ASSERT(r == 0 || pDSE[r].val > pDSE[r - 1].val);//sanity checks
						NNTL_ASSERT(r == 0 || OrderFunctor_t::first_better(pDSE[r].val, pDSE[r - 1].val));//sanity checks	
						NNTL_ASSERT(pDSE[r].batchIdx >= 0 && pDSE[r].idx >= 0);//sanity checks
						pV[r] = pDSE[k - r - 1].val;
					}
				}
				return true;
			}

			::std::vector<bool> batches_in_use(const numel_cnt_t maxBatches)const noexcept {
				NNTL_ASSERT(!m_allData.empty());

				::std::vector<bool> bf(maxBatches, false);
				const auto ld = m_allData.ldim();
				const auto k = m_allData.rows();
				auto pDSE = m_allData.data();
				const auto pDE = m_allData.end();
				while (pDSE != pDE) {
					for (vec_len_t r = 0; r < k; ++r) {
						//NNTL_ASSERT(r == 0 || pDSE[r].val > pDSE[r - 1].val);//sanity checks
						NNTL_ASSERT(r == 0 || OrderFunctor_t::first_better(pDSE[r].val, pDSE[r - 1].val));//sanity checks
						NNTL_ASSERT(pDSE[r].batchIdx >= 0 && pDSE[r].idx >= 0);//sanity checks
						const auto bi = pDSE[r].batchIdx;
						NNTL_ASSERT(bi < maxBatches);
						bf[bi] = true;
					}
					pDSE += ld;
				}
				return bf;
			}

			//execute f for every found DatasetElement with given batch index
			template<typename F>
			bool for_batchIdx(const numel_cnt_t bi, F&& f)const noexcept {
				NNTL_ASSERT(!m_allData.empty());
				NNTL_ASSERT(bi >= 0);
				const auto k = m_allData.rows(), colmn = m_allData.cols();
				for (vec_len_t c = 0; c < colmn; ++c) {
					const auto pDSE = m_allData.colDataAsVec(c);
					for (vec_len_t r = 0; r < k; ++r) {
						//NNTL_ASSERT(r == 0 || pDSE[r].val > pDSE[r - 1].val);//sanity checks
						NNTL_ASSERT(r == 0 || OrderFunctor_t::first_better(pDSE[r].val, pDSE[r - 1].val));//sanity checks
						NNTL_ASSERT(pDSE[r].batchIdx >= 0 && pDSE[r].idx >= 0);//sanity checks
																			   //changing the ordering to descending
						const auto hr = k - r - 1;
						if (bi == pDSE[hr].batchIdx) {
							if (!::std::forward<F>(f)(c, r, pDSE[hr].val, pDSE[hr].idx))
								return false;
						}
					}
				}
				return true;
			}

			bool aggregate(const hlprmtx_t& hlpr, const numel_cnt_t batchIdx)noexcept {
				NNTL_ASSERT(batchIdx >= 0);
				const vec_len_t k = _base_t::get_k(hlpr), colmn = hlpr.cols();
				if (batchIdx <= 0) {
					const auto properSize = smatrix_td::mtx_size_t(k, colmn);
					if (m_allData.size() != properSize) {
						m_allData.resize(properSize, tag_noBias());
						if (m_allData.isAllocationFailed()) return false;
					}
					for (vec_len_t c = 0; c < colmn; ++c) {
						const auto pH = hlpr.colDataAsVec(c);
						const auto pIdxs = _base_t::get_idxs_ptr_from_column_ptr(pH, k);
						const auto pVals = _base_t::get_cache_ptr_from_column_ptr(pH, k);
						const auto pDSE = m_allData.colDataAsVec(c);
						for (vec_len_t hr = 0; hr < k; ++hr) {
							//cache are written in ascending order, i.e. biggest - last
							//NNTL_ASSERT(hr + 1 >= k || pVals[hr] < pVals[hr + 1]);
							NNTL_ASSERT(hr + 1 >= k || OrderFunctor_t::first_better(pVals[hr + 1], pVals[hr]));
							auto& dse = pDSE[hr];
							dse.batchIdx = 0;
							dse.val = pVals[hr];
							dse.idx = pIdxs[hr];
						}
					}
				} else {
					NNTL_ASSERT(m_allData.size() == smatrix_td::mtx_size_t(k, colmn));
					for (vec_len_t c = 0; c < colmn; ++c) {
						const auto pH = hlpr.colDataAsVec(c);
						const auto pIdxs = _base_t::get_idxs_ptr_from_column_ptr(pH, k);
						const auto pVals = _base_t::get_cache_ptr_from_column_ptr(pH, k);
						const auto pDSE = m_allData.colDataAsVec(c);
						//invalid cache index (<0) signifies the values left from previous batches. If the index is valid,
						//the value is from current batch.
						// We should traverse cache now to properly aggregate it into m_allData
						for (vec_len_t hr = 0; hr < k; ++hr) {
							//cache are written in ascending order, i.e. biggest - last
							//NNTL_ASSERT(hr + 1 >= k || pVals[hr] < pVals[hr + 1]);
							NNTL_ASSERT(hr + 1 >= k || OrderFunctor_t::first_better(pVals[hr + 1], pVals[hr + 1]));
							const auto nV = pVals[hr];
							const auto nI = pIdxs[hr];
							// new value at hr index could only be equal to or bigger than stored in m_allData.
							if (nI < 0) {
								//invalid index means that the value nV is left from old batch.
								//NNTL_ASSERT(nV >= pDSE[hr].val);
								NNTL_ASSERT(!OrderFunctor_t::first_better(pDSE[hr].val, nV));
								if (nV == pDSE[hr].val) {
									//nothing else should be done here. Every higher indexes must be the same, b/c any difference
									//will lead to shifting values
								#ifdef NNTL_DEBUG
									for (vec_len_t r = hr + 1; r < k; ++r) {
										NNTL_ASSERT(pDSE[r].val == pVals[r]);
									}
								#endif // NNTL_DEBUG
									break;
								} else {
									// We must find corresponding entry in range [hr+1..k) and move it to the current position.
									vec_len_t r = hr + 1;
									for (; r < k; ++r) {
										const auto& dse = pDSE[r];
										//NNTL_ASSERT(nV >= dse.val);
										NNTL_ASSERT(!OrderFunctor_t::first_better(dse.val, nV));
										if (nV == dse.val) {
											//moving dse to the hr index
											pDSE[hr] = dse;
											break;
										}
									}
									NNTL_ASSERT(r < k || !"DatasetElement not found! WTF?!");
								}
							} else {
								//valid index means that the value nV is from the current batch. We must store it under hr index
								auto& dse = pDSE[hr];
								//NNTL_ASSERT(nV > dse.val);
								NNTL_ASSERT(OrderFunctor_t::first_better(nV, dse.val));
								dse.batchIdx = batchIdx;
								dse.val = nV;
								dse.idx = nI;
							}
						}
					}
				}
				return true;
			}
		};
	};

}
}
