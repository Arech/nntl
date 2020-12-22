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

// probably not finished yet

#include <vector>
#include <algorithm>

#include "../serialization/serialization.h"

//seq_data stores a set of sequences for classification

namespace nntl {

	template<typename RealT>
	class seq_data {
	public:
		typedef RealT real_t;
		typedef RealT value_type;

		typedef math::smatrix_deform<real_t> realmtxdef_t;

	protected:
		typedef ::std::vector<realmtxdef_t> seqSet_t;
		
	protected:
		::std::vector<seqSet_t> m_seqStor;
		::std::vector<vec_len_t> m_classSeqCnt;


		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int) {
			/*ar & serialization::make_nvp("train_x", m_train_x);
			ar & serialization::make_nvp("train_y", m_train_y);
			ar & serialization::make_nvp("test_x", m_test_x);
			ar & serialization::make_nvp("test_y", m_test_y);
			NNTL_ASSERT(absorbsion_will_succeed(m_train_x, m_train_y, m_test_x, m_test_y));*/
			// 			STDCOUTL("serialize_training_parameters is " << ::std::boolalpha
			// 				<< utils::binary_option(ar, serialization::serialize_training_parameters) << ::std::noboolalpha);
		}

		//!! copy constructor not needed
		seq_data(const seq_data& other)noexcept = delete;
		//!!assignment is not needed
		seq_data& operator=(const seq_data& rhs) noexcept = delete;

	public:
		~seq_data()noexcept {}
		seq_data()noexcept {}

		//empty if there's at least one unfilled matrix
		bool empty()const noexcept {
			NNTL_ASSERT(m_seqStor.size() == m_classSeqCnt.size());
			return m_seqStor.empty() || m_classSeqCnt.empty()
				|| ::std::any_of(m_seqStor.cbegin(), m_seqStor.cend(), [](const auto& e)noexcept
			{
				return e.empty() || ::std::any_of(e.cbegin(), e.cend(), [](const auto& m)noexcept {return m.empty(); });
			});
		}

		const ::std::vector<vec_len_t>& get_class_seq_count()const noexcept { return m_classSeqCnt; }

		bool is_clear()const noexcept {
			NNTL_ASSERT(m_seqStor.size() == m_classSeqCnt.size());
			return m_seqStor.empty() && m_classSeqCnt.empty();
		}

		void clear()noexcept {
			NNTL_ASSERT(m_seqStor.size() == m_classSeqCnt.size());
			m_seqStor.clear();
			m_classSeqCnt.clear();
			m_seqStor.shrink_to_fit();
			m_classSeqCnt.shrink_to_fit();
		}

		void describe()const noexcept {
			if (is_clear()) {
				STDCOUTL("#### There's no data stored inside!");
			} else {
				const auto cc = m_seqStor.size();
				STDCOUTL("There's a sequence set for " << cc << " classes:");
				for (size_t i = 0; i < cc; ++i) {
					const auto ss = m_seqStor[i].size();
					NNTL_ASSERT(ss > 0);
					if (ss) {
						STDCOUTL("\tClass " << i << " has " << ss << " sequences of different length");
						for (size_t j = 0; j < ss; ++j) {
							const auto& mtxEl = m_seqStor[i][j];
							if (mtxEl.empty()) {
								STDCOUTL("\t\t#### Sequence set " << j << " is empty!");
							} else {
								STDCOUTL("\t\tSequence set " << j << " has size " << mtxEl.rows() << " x " << mtxEl.cols());
							}
						}
					} else {
						STDCOUTL("\t#### Class " << i << " is empty!");
					}
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////
		//set of functions for creation and population seq_data.
		// not expected to be called outside of readers such as bin_file
	public:
		bool _prealloc_classes(::std::vector<vec_len_t>&& vClsSeqCnt)noexcept {
			NNTL_ASSERT(is_clear());
			if (!is_clear() || vClsSeqCnt.size() == 0) return false;

			bool bRet = true;
			try {
				m_classSeqCnt = ::std::move(vClsSeqCnt);
				const auto s = m_classSeqCnt.size();
				m_seqStor.reserve(s);

				for (size_t i = 0; i < s; ++i) {
					NNTL_ASSERT(m_classSeqCnt[i] > 0);
					m_seqStor.emplace_back(static_cast<size_t>(m_classSeqCnt[i]));
				}
			} catch (const ::std::exception&) {
				bRet = false;
			}
			return bRet;
		}

		realmtxdef_t* _next_mtx_for_class(const size_t classId)noexcept {
			NNTL_ASSERT(!is_clear());
			if (classId < m_seqStor.size() && !m_seqStor[classId].empty()) {
				//looking for first empty matrix in m_seqStor[classId] subvector
				for (auto& mtxEl : m_seqStor[classId]) {
					if (mtxEl.empty()) return &mtxEl;
				}
			}
			return nullptr;
		}
		


	};

}