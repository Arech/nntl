/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

//bwlist is a base class that implements machinery necessary to create a black or whitelist of layers by their ID or LayerID

#include "vector_conditions.h"

namespace nntl {
namespace utils {

	template<typename RealT>
	class _bwlist {
	public:
		typedef _bwlist self_t;
		NNTL_METHODS_SELF();

		typedef ::std::vector<layer_type_id_t> layer_types_list_t;
		typedef ::std::vector<layer_index_t> layer_idx_list_t;

	protected:
		vector_conditions m_dumpLayerCond;
		layer_types_list_t m_layerTypes;
		layer_idx_list_t m_layerIdxs;

		bool m_bWhiteList;

	private:
		self_ref_t _addToList(const layer_type_id_t& t)noexcept {
			m_layerTypes.push_back(t);
			return get_self();
		}
		self_ref_t _addToList(const layer_index_t& lidx)noexcept {
			m_layerIdxs.push_back(lidx);
			return get_self();
		}

		template<typename LtlT>
		::std::enable_if_t< ::std::is_same<layer_type_id_t, ::std::remove_cv_t<typename LtlT::value_type>>::value, self_ref_t>
			_addToList(LtlT&& tl)noexcept {
			NNTL_ASSERT(!m_layerTypes.size());
			m_layerTypes = ::std::forward<LtlT>(tl);
			return get_self();
		}
		template<typename LilT>
		::std::enable_if_t< ::std::is_same<layer_index_t, ::std::remove_cv_t<typename LilT::value_type>>::value, self_ref_t>
			_addToList(LilT&& lil)noexcept {
			NNTL_ASSERT(!m_layerIdxs.size());
			m_layerIdxs = ::std::forward<LilT>(lil);
			return get_self();
		}

		void _checksetWhitelist()noexcept {
			if (!isWhiteList()) {
				NNTL_ASSERT(!m_layerTypes.size());
				NNTL_ASSERT(!m_layerIdxs.size());
				setWhiteList();
			}
		}
		void _checksetBlacklist()noexcept {
			if (isWhiteList()) {
				NNTL_ASSERT(!m_layerTypes.size());
				NNTL_ASSERT(!m_layerIdxs.size());
				setBlackList();
			}
		}

	protected:
		void _bwlist_init(const size_t totalLayers)noexcept {
			m_dumpLayerCond.clear()
				.resize(totalLayers, !m_bWhiteList);
		}
		void _bwlist_updateLayer(const layer_index_t lIdx, const layer_type_id_t layerTypeId)noexcept {
			NNTL_ASSERT(lIdx < m_dumpLayerCond.size());

			if (
				(m_layerTypes.size() && ::std::any_of(m_layerTypes.begin(), m_layerTypes.end(), [layerTypeId](const layer_type_id_t& e)->bool {
				return e == layerTypeId;
			})) || (m_layerIdxs.size() && ::std::any_of(m_layerIdxs.begin(), m_layerIdxs.end(), [lIdx](const layer_index_t& e)->bool {
				return e == lIdx;
			}))
				) {
				m_dumpLayerCond.set(lIdx, m_bWhiteList);
			}
		}

		const bool _bwlist_layerAllowed(const layer_index_t lIdx)const noexcept {
			return m_dumpLayerCond(lIdx);
		}

	public:
		~_bwlist()noexcept {}
		_bwlist(const bool bWhitelistByDefault = false)noexcept : m_bWhiteList(bWhitelistByDefault) {}

		self_ref_t resetLists()noexcept {
			m_dumpLayerCond.clear();
			m_layerTypes.clear();
			m_layerIdxs.clear();
			return get_self();
		}
		self_ref_t setMode(const bool bWhiteList)noexcept {
			m_bWhiteList = bWhiteList;
			return resetLists();
		}
		const bool getMode()const noexcept { return m_bWhiteList; }
		self_ref_t setWhiteList()noexcept { return setMode(true); }
		self_ref_t setBlackList()noexcept { return setMode(false); }
		const bool isWhiteList()const noexcept { return getMode(); }
		const bool isBlackList()const noexcept { return !getMode(); }

		self_ref_t whitelist(const layer_type_id_t& p)noexcept {
			_checksetWhitelist();
			return _addToList(p);
		}
		template<typename LtlT>
		::std::enable_if_t< ::std::is_same<layer_type_id_t, ::std::remove_cv_t<typename LtlT::value_type>>::value, self_ref_t>
			whitelist(LtlT&& p)noexcept {
			_checksetWhitelist();
			return _addToList(::std::forward<LtlT>(p));
		}
		self_ref_t whitelist(const layer_index_t& p)noexcept {
			_checksetWhitelist();
			return _addToList(p);
		}
		template<typename LilT>
		::std::enable_if_t< ::std::is_same<layer_index_t, ::std::remove_cv_t<typename LilT::value_type>>::value, self_ref_t>
			whitelist(LilT&& p)noexcept {
			_checksetWhitelist();
			return _addToList(::std::forward<LilT>(p));
		}

		self_ref_t blacklist(const layer_type_id_t& p)noexcept {
			_checksetBlacklist();
			return _addToList(p);
		}
		template<typename LtlT>
		::std::enable_if_t< ::std::is_same<layer_type_id_t, ::std::remove_cv_t<typename LtlT::value_type>>::value, self_ref_t>
			blacklist(LtlT&& p)noexcept {
			_checksetBlacklist();
			return _addToList(::std::forward<LtlT>(p));
		}
		self_ref_t blacklist(const layer_index_t& p)noexcept {
			_checksetBlacklist();
			return _addToList(p);
		}
		template<typename LilT>
		::std::enable_if_t< ::std::is_same<layer_index_t, ::std::remove_cv_t<typename LilT::value_type>>::value, self_ref_t>
			blacklist(LilT&& p)noexcept {
			_checksetBlacklist();
			return _addToList(::std::forward<LilT>(p));
		}
	};

}
}