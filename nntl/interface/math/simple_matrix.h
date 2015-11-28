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

#include <utility>
//#include <memory>
#include "../../_defs.h"


//TODO: Consider replacing generic memcpy/memcmp/memset and others similar generic functions with
//their versions from Agner Fox's asmlib. TEST if it really helps!!!!

namespace nntl {
namespace math {

	// wrapper class to store vectors/matrices in column-major ordering
	// Almost sure, that developing this class is like reinventing the wheel. But I don't know where to look for
	// better implementations, therefore it might be faster to write the class myself than to try to find an
	// alternative an apply it to my needs.
	

	template <typename T_>
	class simple_matrix {
	public:
		typedef T_ value_type;
		typedef value_type* value_ptr_t;
		typedef const value_type* cvalue_ptr_t;

		//typedef simple_matrix<value_type> self_t;

		//rows/cols type. int should be enought. If not, redifine to smth bigger
		typedef uint32_t vec_len_t;
		typedef uint64_t numel_cnt_t;
		typedef std::pair<const vec_len_t, const vec_len_t> mtx_size_t;
		
		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// no std::unique_ptr here because of exception unsafety.
		value_ptr_t m_pData;
		vec_len_t m_rows, m_cols;

		// TODO: probably it's better to turn m_bEmulateBiases and m_bDontManageStorage variables into template parameters,
		// because it looks
		// like we don't need to change them during lifetime of an object. It'll help to make class a little faster and
		// array of objects will require significantly less memory ( N x default alignment, which is 8 or even 16 bytes)
		// That could lead to another speedup due to faster access to a class members within first 256 bytes of member space

		bool m_bEmulateBiases;// off by default. Turn on before filling(resizing) matrix for X data storage. Will append
							  // a last column prefilled with ones to emulate neuron biases.

		bool m_bDontManageStorage;// off by default. Flag to support external memory management and useExternalStorage() functionality

	protected:
		void _realloc() noexcept {
			if (m_bDontManageStorage) {
				NNTL_ASSERT(!"Hey! WTF? You cant manage matrix memory in m_bDontManageStorage mode!");
				abort();
			} else {
				delete[] m_pData;
				if (m_rows > 0 && m_cols > 0) {
					m_pData = new(std::nothrow) value_type[numel()];
				} else {
					m_rows = 0;
					m_cols = 0;
					m_pData = nullptr;
				}
			}
		}
		void _free()noexcept {
			if (! m_bDontManageStorage) delete[] m_pData;
			m_pData = nullptr;
			m_rows = 0;
			m_cols = 0;
		}

	public:
		~simple_matrix()noexcept {
			_free();
		}

		simple_matrix() noexcept : m_pData(nullptr), m_rows(0), m_cols(0), m_bEmulateBiases(false), m_bDontManageStorage(false){};
		
		simple_matrix(const vec_len_t rows, const vec_len_t cols, const bool _bEmulBias=false) noexcept : m_pData(nullptr),
			m_rows(rows), m_cols(cols), m_bEmulateBiases(_bEmulBias), m_bDontManageStorage(false)
		{
			NNTL_ASSERT(rows > 0 && cols > 0);
			if (m_bEmulateBiases) m_cols++;
			_realloc();
			if (m_bEmulateBiases) set_biases();
		}
		simple_matrix(const mtx_size_t& msize, const bool _bEmulBias = false) noexcept : m_pData(nullptr),
			m_rows(msize.first), m_cols(msize.second), m_bEmulateBiases(_bEmulBias), m_bDontManageStorage(false)
		{
			NNTL_ASSERT(m_rows > 0 && m_cols > 0);
			if (m_bEmulateBiases) m_cols++;
			_realloc();
			if (m_bEmulateBiases) set_biases();
		}

		simple_matrix(simple_matrix&& src)noexcept : m_pData(nullptr), m_rows(src.m_rows), m_cols(src.m_cols),
			m_bEmulateBiases(src.m_bEmulateBiases), m_bDontManageStorage(src.m_bDontManageStorage)
		{
			m_pData = src.m_pData;
			src.m_pData = nullptr;
			src.m_rows = 0;
			src.m_cols = 0;
		}

		simple_matrix& operator=(simple_matrix&& rhs) noexcept {
			if (this!=&rhs) {
				_free();
				m_pData = rhs.m_pData;
				m_rows = rhs.m_rows;
				m_cols = rhs.m_cols;

				rhs.m_pData = nullptr;
				rhs.m_rows = 0;
				rhs.m_cols = 0;

				m_bEmulateBiases = rhs.m_bEmulateBiases;
				m_bDontManageStorage = rhs.m_bDontManageStorage;
			}
			return *this;
		}
		

		//!! copy constructor not needed
		simple_matrix(const simple_matrix& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		simple_matrix& operator=(const simple_matrix& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

		//class object copying is a costly procedure, therefore making a special member function for it and will use only in special cases
		const bool cloneTo(simple_matrix& dest)const noexcept {
			if (dest.m_bDontManageStorage) {
				if (dest.m_rows != m_rows || dest.m_cols != m_cols) return false;
			} else {
				dest.m_bEmulateBiases = false;//to make resize() work correctly if dest.m_bEmulateBiases==true
				if (!dest.resize(m_rows, m_cols)) return false;
			}
			dest.m_bEmulateBiases = m_bEmulateBiases;
			memcpy(dest.m_pData, m_pData, byte_size());
			NNTL_ASSERT(*this == dest);
			return true;
		}
		const bool operator==(const simple_matrix& rhs)const noexcept {
			//TODO: this is bad implementation, but it's enough cause we're gonna use it for testing only.
			return m_bEmulateBiases==rhs.m_bEmulateBiases && size() == rhs.size()
				&& 0 == memcmp(m_pData, rhs.m_pData, byte_size());
		}

		const bool isAllocationFailed()const noexcept {
			return nullptr == m_pData && m_rows>0 && m_cols>0;
		}

		const bool bDontManageStorage()const noexcept { return m_bDontManageStorage; }

		//////////////////////////////////////////////////////////////////////////
		void set_biases()const noexcept {
			NNTL_ASSERT(m_bEmulateBiases);
			// filling last column with ones to emulate biases
			const auto ne = numel();
			std::fill(&m_pData[ne - m_rows], &m_pData[ne], value_type(1.0));
		}
		const bool emulatesBiases()const noexcept { return m_bEmulateBiases; }
		void will_emulate_biases()noexcept {
			NNTL_ASSERT(empty() && m_rows == 0 && m_cols == 0);
			//for example, we can use bEmulateBiases-enabled matrix of neuron activations as destination of matrix product operation
			// (prev layer activations times weights) as well as biased source of data for the next layer
			m_bEmulateBiases = true;
		}
		void dont_emulate_biases()noexcept {
			NNTL_ASSERT(empty());
			m_bEmulateBiases = false;
		}
		void assert_biases_ok()noexcept {
#ifdef NNTL_DEBUG
			NNTL_ASSERT(emulatesBiases());
			
			const auto ne = numel();
			auto pS = &m_pData[ne - m_rows];
			const auto pE = m_pData + ne;
			while (pS != pE) {
				NNTL_ASSERT(*pS++ == value_type(1.0) || !"Bias check failed!");
			}
#endif // NNTL_DEBUG
		}

		//////////////////////////////////////////////////////////////////////////
		const vec_len_t rows() const noexcept { return m_rows; }
		const vec_len_t cols() const noexcept { return m_cols; }

		// ***if biases are emulated, then actual columns number is one greater, than data cols number
		const vec_len_t cols_no_bias() const noexcept { 
			if (m_bEmulateBiases && m_cols>0) {
				return m_cols - 1;
			}else return m_cols;
		}

		const mtx_size_t size()const noexcept { return mtx_size_t(m_rows, m_cols); }
		const mtx_size_t size_no_bias()const noexcept { 
			if (m_bEmulateBiases && m_cols > 0) {
				return mtx_size_t(m_rows, m_cols-1);
			}else return size();
		}
		static constexpr numel_cnt_t sNumel(vec_len_t r, vec_len_t c)noexcept { return static_cast<numel_cnt_t>(r)*static_cast<numel_cnt_t>(c); }
		static constexpr numel_cnt_t sNumel(const mtx_size_t s)noexcept { return sNumel(s.first, s.second); }
		const numel_cnt_t numel()const noexcept { return sNumel(m_rows,m_cols); }
		const numel_cnt_t numel_no_bias()const noexcept { 
			if (m_bEmulateBiases && m_cols>0) {
				return sNumel(m_rows,m_cols-1);
			} else return numel();
		}
		const numel_cnt_t byte_size()const noexcept {
			return numel()*sizeof(value_type);
		}
		const numel_cnt_t byte_size_no_bias()const noexcept {
			return numel_no_bias()*sizeof(value_type);
		}

		const bool empty()const noexcept { return nullptr == m_pData; }

		value_ptr_t dataAsVec()noexcept {
			NNTL_ASSERT(!empty() && m_cols>0 && m_rows>0);
			return m_pData;
		}
		cvalue_ptr_t dataAsVec()const noexcept {
			NNTL_ASSERT(!empty() && m_cols>0 && m_rows>0);
			return m_pData;
		}
		value_ptr_t colDataAsVec(vec_len_t c)noexcept {
			NNTL_ASSERT(!empty() && m_cols>0 && m_rows>0 && c <= m_cols);
			return &m_pData[c*m_rows];
		}
		cvalue_ptr_t colDataAsVec(vec_len_t c)const noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0 && c <= m_cols);
			return &m_pData[c*m_rows];
		}

		// get/set are for non performance critical code!
		void set(const vec_len_t r, const vec_len_t c, const value_type v)noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			m_pData[ r + c*m_rows ] = v;
		}
		const value_type get(const vec_len_t r, const vec_len_t c)const noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			return m_pData[r + c*m_rows];
		}

		void clear()noexcept {
			_free();
			m_bDontManageStorage = false;
		}
		const bool resize(const simple_matrix& m)noexcept {
			NNTL_ASSERT(!m.empty() && m.rows() > 0 && m.cols() > 0);
			NNTL_ASSERT(emulatesBiases() == m.emulatesBiases());
			return resize(m.rows(), m.cols());
		}
		const bool resize(const mtx_size_t& sz)noexcept { return resize(sz.first, sz.second); }
		const bool resize(const vec_len_t r, vec_len_t c) noexcept {
			NNTL_ASSERT(!m_bDontManageStorage);
			NNTL_ASSERT(r > 0 && c > 0);
			if (r <= 0 || c <= 0) return false;

			if (m_bEmulateBiases) c++;

			if (r == m_rows && c == m_cols) return true;

			m_rows = r;
			m_cols = c;

			_realloc();

			const bool bRet = nullptr != m_pData;
			if (m_bEmulateBiases && bRet) set_biases();
			return bRet;
		}

		void zeros()noexcept {
			NNTL_ASSERT(!empty());
			memset(m_pData, 0, byte_size_no_bias());
		}
		void ones()noexcept {
			NNTL_ASSERT(!empty());
			std::fill(m_pData, m_pData + numel(), value_type(1.0));
		}

		//ATTN: _i_math implementation should have a faster variants of this function.
		//extract number=cnt rows by their indexes, specified by sequential iterator begin, into allocated dest matrix
		template<typename SeqIt>
		void extractRows(SeqIt begin, const numel_cnt_t cnt, simple_matrix<value_type>& dest)const noexcept {
			NNTL_ASSERT(!dest.empty());
			static_assert(std::is_same<vec_len_t, SeqIt::value_type>::value, "Iterator should point to vec_len_t data");

			const numel_cnt_t destRows = dest.rows(), destCols = dest.cols(), thisRows = m_rows, thisCols = m_cols;
			NNTL_ASSERT(destCols == thisCols && destRows >= cnt && cnt <= thisRows);

			//TODO: accessing row data, defined by SeqIt begin in sequential order could provide some performance gains. However
			//it requires the content of [begin,begin+cnt) to be sorted. Therefore, testing is required to decide whether it's all worth it

			const auto pThis = m_pData, pDest = dest.m_pData;
			for (numel_cnt_t c = 0; c < destCols; ++c) {
				SeqIt pRI = begin;
				auto destCur = &pDest[c*destRows];
				const auto destEnd = destCur + cnt;
				const auto thisBeg = &pThis[c*thisRows];
				//for (numel_cnt_t ri = 0; ri < cnt; ++ri) {
					//pDest[ri + ofsDest] = pThis[*pRI + ofsThis];
				while(destCur!=destEnd){
					//*destCur++ = pThis[*pRI++ + ofsThis];
					*destCur++ = *(thisBeg + *pRI++);
				}
			}

		}

		//extract number=cnt rows by their indexes, specified by sequential iterator begin, into allocated dest matrix
		template<typename SeqIt>
		void extractRows_slow(SeqIt begin, const numel_cnt_t cnt, simple_matrix<value_type>& dest)const noexcept {
			NNTL_ASSERT(!dest.empty());
			static_assert(std::is_same<vec_len_t, SeqIt::value_type>::value, "Iterator should point to vec_len_t data");

			const numel_cnt_t destRows = dest.rows(), destCols = dest.cols(), thisRows=m_rows,thisCols=m_cols;
			NNTL_ASSERT(destCols == thisCols && destRows >= cnt && cnt<=thisRows);

			//TODO: accessing row data, defined by SeqIt begin in sequential order could provide some performance gains. However
			//it requires the content of [begin,begin+cnt) to be sorted. Therefore, testing is required to decide whether it's all worth it
			
			auto pThis = m_pData, pDest = dest.m_pData;
			
			//walk over this matrix in colmajor order - that PROBABLY would work better due to cache coherency
			//TODO: test whether it is really better
			for (numel_cnt_t c = 0; c < destCols; ++c) {
				SeqIt pRI = begin;
				const auto ofsDest = c*destRows, ofsThis = c*thisRows;
				for (numel_cnt_t ri = 0; ri < cnt;++ri) {
					pDest[ri + ofsDest] = pThis[*pRI + ofsThis];
					++pRI;
				}
			}
		}


		const bool useExternalStorage(value_ptr_t ptr, const simple_matrix& sizeLikeThis)noexcept {
			return useExternalStorage(ptr, sizeLikeThis.rows(), sizeLikeThis.cols(), sizeLikeThis.emulatesBiases());
		}
		const bool useExternalStorage(value_ptr_t ptr, vec_len_t rows, vec_len_t cols, bool bEmulateBiases = false)noexcept {
			NNTL_ASSERT(ptr != nullptr && rows > 0 && cols>0);
			_free();
			m_bDontManageStorage = true;
			m_pData = ptr;
			m_rows = rows;
			m_cols = cols;
			m_bEmulateBiases = bEmulateBiases;

			//usually, external storage is used with shared memory to store temporary computations, therefore data won't survive between
			// usages and there is no need to prefill it with biases. m_bEmulateBiases flag is helpful for routines like numel_no_bias()
			//if (bEmulateBiases) _fill_biases();
			return true;
		}

		const bool useInternalStorage(vec_len_t rows=0, vec_len_t cols=0, bool bEmulateBiases = false)noexcept {
			_free();
			m_bDontManageStorage = false;
			m_bEmulateBiases = bEmulateBiases;
			if (rows > 0 && cols > 0) {
				return resize(rows, cols);
			} else return true;
		}

		void assert_storage_does_not_intersect(const simple_matrix& m)const noexcept {
			NNTL_ASSERT(this != &m);
			NNTL_ASSERT(!empty() && !m.empty());
			//nonstrict nonequality it necessary here, because &[numel()] references element past the end of the allocated array
			NNTL_ASSERT( &m.m_pData[m.numel()] <= m_pData || &m_pData[numel()] <= m.m_pData);
		}		
	};

	//this class will allow to reuse the same storage for matrix of smaller size
	template <typename T_>
	class simple_matrix_deformable : public simple_matrix<T_> {
	private:
		typedef simple_matrix<T_> _base_class;

	protected:
#ifdef NNTL_DEBUG
		numel_cnt_t m_maxSize;
#endif // NNTL_DEBUG

	public:
		~simple_matrix_deformable()noexcept {}
		simple_matrix_deformable()noexcept : _base_class() {}
		simple_matrix_deformable(const vec_len_t rows, const vec_len_t cols, const bool _bEmulBias = false)noexcept : _base_class(rows, cols, _bEmulBias) {
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		void clear()noexcept {
#ifdef NNTL_DEBUG
			m_maxSize = 0;
#endif // NNTL_DEBUG
			_base_class::clear();
		}

		const bool cloneFrom(const simple_matrix& src)noexcept {
			_free();
			auto r = src.cloneTo(*this);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		const bool resize(const vec_len_t rows, vec_len_t cols)noexcept {
			_free();
			const auto r = _base_class::resize(rows, cols);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		const bool resize(const simple_matrix& m)noexcept {
			_free();
			const auto r = _base_class::resize(m);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		const bool resize(const numel_cnt_t ne)noexcept {
			NNTL_ASSERT(ne > 0);
			_free();
			auto ptr = new(std::nothrow) value_type[ne];
			if (nullptr == ptr)return false;
			if (!useExternalStorage(ptr, ne)) {
				delete[] ptr;
				return false;
			}
			m_bDontManageStorage = false;
			return true;
		}

		const bool useExternalStorage(value_ptr_t ptr, vec_len_t rows, vec_len_t cols, bool bEmulateBiases = false)noexcept {
			const auto r = _base_class::useExternalStorage(ptr, rows, cols, bEmulateBiases);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}
		const bool useExternalStorage(value_ptr_t ptr, const simple_matrix& sizeLikeThis)noexcept {
			return useExternalStorage(ptr, sizeLikeThis.rows(), sizeLikeThis.cols(), sizeLikeThis.emulatesBiases());
		}
		const bool useExternalStorage(value_ptr_t ptr, numel_cnt_t cnt, bool bEmulateBiases = false)noexcept {
			const auto r = _base_class::useExternalStorage(ptr, 1, 1, bEmulateBiases);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = cnt;
#endif // NNTL_DEBUG
			return r;
		}


		// ALWAYS run your code in debug mode first.
		// ALWAYS think about biases - when they should be or should not be set/restored
		// You've been warned
		vec_len_t deform_cols(vec_len_t c)noexcept {
			NNTL_ASSERT(m_maxSize >= sNumel(m_rows, c));
			NNTL_ASSERT(!empty());
			auto ret = m_cols;
			m_cols = c;
			return ret;
		}
		void hide_last_col()noexcept {
			NNTL_ASSERT(!empty() && m_cols>=2);
			--m_cols;
		}
		void restore_last_col()noexcept {
			NNTL_ASSERT(!empty() && m_cols >= 1);
			NNTL_ASSERT(m_maxSize >= sNumel(m_rows, m_cols+1));
			++m_cols;
		}

		// use deform_rows() with extreme care on col-major (default!!!) data!!!
		vec_len_t deform_rows(vec_len_t r)noexcept {
			NNTL_ASSERT(m_maxSize >= sNumel(r, m_cols));
			NNTL_ASSERT(!empty());
			auto ret = m_rows;
			m_rows = r;
			return ret;
		}
		void deform(vec_len_t r, vec_len_t c)noexcept {
			NNTL_ASSERT(m_maxSize >= sNumel(r, c));
			NNTL_ASSERT(!empty());
			m_rows = r;
			m_cols = c;
		}

		void deform_like(const simple_matrix& m)noexcept { deform(m.rows(), m.cols()); }
		void deform_like_no_bias(const simple_matrix& m)noexcept { deform(m.rows(), m.cols_no_bias()); }
	};
	
}
}