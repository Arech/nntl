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


namespace nntl {
namespace utils {

	template<typename RealT>
	class dataHolder : public nntl::math::smatrix_td {
	public:
		typedef RealT real_t;
		typedef nntl::math::smatrix<real_t> realmtx_t;
		typedef nntl::math::smatrix_deform<real_t> realmtxdef_t;

	protected:
		const realmtx_t* m_pDataX;
		const realmtx_t* m_pDataY;

		realmtxdef_t m_batchX, m_batchY;

		::std::vector<vec_len_t> m_rowsIdxs;
		::std::vector<vec_len_t>::const_iterator m_next;
		
		bool m_bBatchSizeFixed;

	public:
		~dataHolder()noexcept {
			deinit();
		}
		dataHolder()noexcept : m_pDataX(nullptr), m_pDataY(nullptr) {
			m_batchX.will_emulate_biases();
			m_batchY.dont_emulate_biases();
		}
		
		const ::std::vector<vec_len_t>::const_iterator curBatchIdxs()const noexcept {
			NNTL_ASSERT(m_rowsIdxs.size() > 0 && m_next<=m_rowsIdxs.cend());
			return m_next - curBatchSize();
		}
		const ::std::vector<vec_len_t>::const_iterator& curBatchIdxsEnd()const noexcept {
			NNTL_ASSERT(m_rowsIdxs.size() > 0 && m_next <= m_rowsIdxs.cend());
			return m_next;
		}

		const realmtx_t& batchX()const noexcept {
			NNTL_ASSERT(!m_batchX.empty() && m_batchX.numel() > 0);
			return m_batchX;
		}
		const realmtx_t& batchY()const noexcept {
			NNTL_ASSERT(m_pDataY && !m_batchY.empty() && m_batchY.numel() > 0);
			return m_batchY;
		}

		void deinit()noexcept {
			m_batchX.clear();
			m_batchY.clear();
			m_rowsIdxs.clear();
			m_pDataX = nullptr;
			m_pDataY = nullptr;
		}
		void init(const vec_len_t maxBatchSize, const bool bBatchSizeFixed, const realmtx_t& data_x, const realmtx_t* pData_y = nullptr)noexcept {
			NNTL_ASSERT(maxBatchSize > 0 && maxBatchSize <= data_x.rows() && (!pData_y || data_x.rows() == pData_y->rows()));
			NNTL_ASSERT(data_x.emulatesBiases() && (!pData_y || !pData_y->emulatesBiases()));
			NNTL_ASSERT(m_batchX.empty() && m_batchY.empty());
			NNTL_ASSERT(m_batchX.emulatesBiases() && !m_batchY.emulatesBiases());

			m_pDataX = &data_x;
			m_pDataY = pData_y;
			m_bBatchSizeFixed = bBatchSizeFixed;

			if (bBatchSizeFixed && maxBatchSize==data_x.rows()) {
				//dropping const just to use useExternalStorage() api. Anyway, we're not going to allow modification of batch_x
				m_batchX.useExternalStorage(const_cast<realmtx_t&>(data_x));
				if (pData_y) m_batchY.useExternalStorage(const_cast<realmtx_t&>(*pData_y));
			} else {
				m_batchX.resize(maxBatchSize, data_x.cols_no_bias());
				if (pData_y) m_batchY.resize(maxBatchSize, pData_y->cols());
			}

			m_rowsIdxs.resize(data_x.rows());
			::std::iota(m_rowsIdxs.begin(), m_rowsIdxs.end(), 0);
			m_next = m_rowsIdxs.cend();
		}

		bool isFullBatch()const noexcept {
			return m_batchX.rows() == m_pDataX->rows();
		}
		bool isBatchAliasToData()const noexcept {
			return m_bBatchSizeFixed && isFullBatch();
		}

		void prepateToBatchSize(const vec_len_t batchSize)noexcept {
			if (isBatchAliasToData()) {
				NNTL_ASSERT(batchSize == m_batchX.rows());
			} else {
				m_batchX.deform_rows(batchSize);
				if (m_pDataY) m_batchY.deform_rows(batchSize);
			}
		}
		const vec_len_t curBatchSize()const noexcept { return m_batchX.rows(); }

		template<typename iRngT, typename iMathT>
		void nextBatch(iRngT& iR, iMathT& iM)noexcept {
			if (isBatchAliasToData()) {
				//nothing to do here
			} else {
				NNTL_ASSERT(m_rowsIdxs.size());
				if (m_next >= m_rowsIdxs.cend() - curBatchSize()) {
					m_next = m_rowsIdxs.cbegin();
					::std::random_shuffle(m_rowsIdxs.begin(), m_rowsIdxs.end(), iR);
				}
				iM.mExtractRows(*m_pDataX, m_next, m_batchX);
				if (m_pDataY) iM.mExtractRows(*m_pDataY, m_next, m_batchY);

				m_next += curBatchSize();
			}
		}
	};

}
}
