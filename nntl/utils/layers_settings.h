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

//this files defines an object to store some settings for possibly any layer in a layer stack.
//It's not meant to be very efficient and fast

#include <map>
//#include <forward_list>
#include <list>
#include <algorithm>

namespace nntl {
namespace utils {

	template<typename SettsT>
	class layer_settings {
	public:
		typedef SettsT setts_t;

	private:
		typedef layer_settings self_t;
		NNTL_TYPEDEFS_SELF();

	protected:
		//typedef ::std::forward_list<setts_t> layer_setts_keeper_t;
		typedef ::std::list<setts_t> layer_setts_keeper_t;
		typedef ::std::map<layer_index_t, typename layer_setts_keeper_t::iterator> layer_setts_map_t;

	protected:
		layer_setts_keeper_t m_keeper;
		layer_setts_map_t m_map;


	public:
		~layer_settings()noexcept{}
		template<typename ST>
		layer_settings(ST&& def)noexcept {
			m_keeper.push_front(::std::forward<ST>(def));
		}
		layer_settings()noexcept {
			m_keeper.emplace_front(setts_t());
		}

		const setts_t& get_default()const noexcept {
			return m_keeper.front();
		}

		template<typename ST>
		self_ref_t set_default(ST&& def)noexcept {
			if (!m_keeper.empty())
				m_keeper.pop_front();
			m_keeper.push_front(::std::forward<ST>(def));
		}

		const setts_t& get(const layer_index_t idx)const noexcept {
			return 0 == m_map.count(idx) 
				? get_default() 
				: *(m_map.at(idx));
		}
		const setts_t& operator[](const layer_index_t idx)const noexcept { return get(idx); }

		template<typename _F>
		void for_each(_F&& f)noexcept {
			NNTL_ASSERT(!m_keeper.empty());
			::std::for_each(m_keeper.begin(), m_keeper.end(), ::std::forward<_F>(f));
		}

		template<typename ST>
		self_ref_t add(const layer_index_t idx, ST&& sett)noexcept {
			NNTL_ASSERT(!m_keeper.empty());
			if (0 != m_map.count(idx)) {
				//removing from the m_keeper
				m_keeper.erase(m_map.at(idx));
			}

			m_map[idx] = m_keeper.insert(m_keeper.end(), ::std::forward<ST>(sett));
			return *this;
		}

		//too much doings with unused function
/*

		template<typename LIdxsSeqT, typename ST>
		::std::enable_if_t<::std::is_same<layer_index_t, ::std::remove_cv_t<typename LIdxsSeqT::value_type>>::value, self_ref_t>
			add(const LIdxsSeqT& idxsSeq, ST&& sett)noexcept
		{
			NNTL_ASSERT(!m_keeper.empty());
#ifdef NNTL_DEBUG
			for (const auto i : idxsSeq) {
				NNTL_ASSERT(0 == m_map.count(i));
				if (0 != m_map.count(i)) {
					m_keeper.erase(m_map.at(i));
					//#BUGBUG must store erased iterator from m_map.at(i) somewhere to prevent double erasing.
					// Other i's from idxsSeq references might have the same iterator.
				}
			}
#endif // NNTL_DEBUG

			//m_keeper.insert_after(m_keeper.begin(), ::std::forward<ST>(sett));
			//const setts_t* pSett = &(*::std::next(m_keeper.begin()));
			//auto &pSett = ::std::next(m_keeper.begin());

			auto pSett = m_keeper.insert(m_keeper.end(), ::std::forward<ST>(sett));
			for (const auto i : idxsSeq) {
				m_map[i] = pSett;
			}
			return *this;
		}*/


	};

}
}