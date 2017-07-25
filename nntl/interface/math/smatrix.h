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

#include <utility>
//#include <memory>
#include "../../_defs.h"
#include "../../common.h"

#include "../threads/parallel_range.h"

//#TODO: Consider replacing generic memcpy/memcmp/memset and others similar generic functions with
//their versions from Agner Fox's asmlib. TEST if it really helps!!!!

namespace nntl {
namespace math {

	// types that don't rely on matrix value_type
	struct smatrix_td {
		//rows/cols type. int should be enought. If not, redifine to smth bigger
		//typedef uint32_t vec_len_t;
		typedef neurons_count_t vec_len_t;
		
		//#todo: size_t should be here?
		typedef uint64_t numel_cnt_t;
		
		typedef std::pair<const vec_len_t, const vec_len_t> mtx_size_t;
		typedef std::pair<vec_len_t, vec_len_t> mtx_coords_t;
	};

	//////////////////////////////////////////////////////////////////////////
	// wrapper class to store vectors/matrices in column-major ordering
	template <typename T_>
	class smatrix : public smatrix_td {
	public:
		typedef T_ value_type;
		typedef value_type* value_ptr_t;
		typedef const value_type* cvalue_ptr_t;
		
		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// no std::unique_ptr here because of exception unsafety.
		value_ptr_t m_pData;
		vec_len_t m_rows, m_cols;

		// #TODO: probably it's better to turn m_bEmulateBiases and m_bDontManageStorage variables into template parameters,
		// because it looks like sometimes we don't need to change them during lifetime of an object. It'll help to make class a little faster and
		// array of objects will require significantly less memory ( N x default alignment, which is 8 or even 16 bytes)
		// That could lead to another speedup due to faster access to a class members within first 256 bytes of member space

		bool m_bEmulateBiases;// off by default. Turn on before filling(resizing) matrix for X data storage. This will append
		// an additional last column prefilled with ones to emulate neuron biases. Hence m_cols will be 1 greater, than specified
		// to resize() operation. However, if you're going to use external memory management, then a call to useExternalStorage() 
		// should contain _col, that takes additional bias column into account (i.e. a call to useExternalStorage() should should
		// provide the function with a final memory bytes available)
		//

		bool m_bDontManageStorage;// off by default. Flag to support external memory management and useExternalStorage() functionality

		bool m_bHoleyBiases; //off by default. This flag is for the test_biases_ok() function use only. When it is on,
		//it is ok for a bias to also have a value of zero. We need this flag to support gating layers, that should completely
		// blackout data samples including biases. If we are to remove it, test_biases_ok() would fail assestions on gated layers.
		// This flag doesn't require m_bEmulateBiases flag, because matrix could have biases without this flag set in some cases

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
		~smatrix()noexcept {
			_free();
		}

		smatrix() noexcept : m_pData(nullptr), m_rows(0), m_cols(0)
			, m_bEmulateBiases(false), m_bDontManageStorage(false), m_bHoleyBiases(false){};
		
		smatrix(const vec_len_t _rows, const vec_len_t _cols, const bool _bEmulBias=false) noexcept : m_pData(nullptr),
			m_rows(_rows), m_cols(_cols), m_bEmulateBiases(_bEmulBias), m_bDontManageStorage(false), m_bHoleyBiases(false)
		{
			NNTL_ASSERT(_rows > 0 && _cols > 0);
			if (m_bEmulateBiases) ++m_cols;
			_realloc();
			if (m_bEmulateBiases) set_biases();
		}
		smatrix(const mtx_size_t& msize, const bool _bEmulBias = false) noexcept : m_pData(nullptr),
			m_rows(msize.first), m_cols(msize.second), m_bEmulateBiases(_bEmulBias), m_bDontManageStorage(false), m_bHoleyBiases(false)
		{
			NNTL_ASSERT(m_rows > 0 && m_cols > 0);
			if (m_bEmulateBiases) ++m_cols;
			_realloc();
			if (m_bEmulateBiases) set_biases();
		}

		//useExternalStorage(value_ptr_t ptr, const smatrix& sizeLikeThis) variation
		smatrix(value_ptr_t ptr, const smatrix& sizeLikeThis)noexcept : m_pData(nullptr), m_bDontManageStorage(false){
			useExternalStorage(ptr, sizeLikeThis);
			m_bHoleyBiases = sizeLikeThis.isHoleyBiases();
		}

		smatrix(value_ptr_t ptr, const mtx_size_t& sizeLikeThis, const bool bEmulateBiases = false, const bool _bHoleyBiases = false)noexcept
			: m_pData(nullptr), m_bDontManageStorage(false)
		{
			useExternalStorage(ptr, sizeLikeThis, bEmulateBiases, _bHoleyBiases);
		}
		smatrix(value_ptr_t ptr, const vec_len_t& r, const vec_len_t& c, const bool bEmulateBiases = false, const bool _bHoleyBiases = false)noexcept
			: m_pData(nullptr), m_bDontManageStorage(false)
		{
			useExternalStorage(ptr, r, c, bEmulateBiases, _bHoleyBiases);
		}

		struct tag_useExternalStorageNoBias {};
		//useExternalStorage_no_bias(value_ptr_t ptr, const smatrix& sizeLikeThis) variation
		smatrix(value_ptr_t ptr, const smatrix& sizeLikeThisNoBias, const tag_useExternalStorageNoBias&)noexcept 
			: m_pData(nullptr), m_bDontManageStorage(false)
		{
			useExternalStorage_no_bias(ptr, sizeLikeThisNoBias, sizeLikeThisNoBias.isHoleyBiases());
		}

		//////////////////////////////////////////////////////////////////////////

		smatrix(smatrix&& src)noexcept : m_pData(src.m_pData), m_rows(src.m_rows), m_cols(src.m_cols),
			m_bEmulateBiases(src.m_bEmulateBiases), m_bDontManageStorage(src.m_bDontManageStorage), m_bHoleyBiases(src.m_bHoleyBiases)
		{
			src.m_pData = nullptr;
			src.m_rows = 0;
			src.m_cols = 0;
		}

		smatrix& operator=(smatrix&& rhs) noexcept {
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
				m_bHoleyBiases = rhs.m_bHoleyBiases;
			}
			return *this;
		}
		

		//!! copy constructor not needed
		smatrix(const smatrix& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		smatrix& operator=(const smatrix& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

		//class object copying is a costly procedure, therefore making a special member function for it and will use only in special cases
		const bool clone_to(smatrix& dest)const noexcept {
			if (dest.m_bDontManageStorage) {
				if (dest.m_rows != m_rows || dest.m_cols != m_cols) {
					NNTL_ASSERT(!"Wrong dest matrix!");
					return false;
				}
			} else {
				dest.m_bEmulateBiases = false;//to make resize() work correctly if dest.m_bEmulateBiases==true
				if (!dest.resize(m_rows, m_cols)) return false;
			}
			dest.m_bEmulateBiases = m_bEmulateBiases;
			return copy_to(dest);
		}

		//strips biases column from the source data
		const bool clone_to_no_bias(smatrix& dest)const noexcept {
			const auto cnb = cols_no_bias();
			if (dest.m_bDontManageStorage) {
				if (dest.m_rows != m_rows || dest.m_cols != cnb) {
					NNTL_ASSERT(!"Wrong dest matrix!");
					return false;
				}
			} else {				
				if (!dest.resize(m_rows, cnb)) {
					NNTL_ASSERT(!"resize failed!");
					return false;
				}
			}
			dest.m_bEmulateBiases = false;
			dest.m_bHoleyBiases = false;
			memcpy(dest.m_pData, m_pData, byte_size_no_bias());
			return true;
		}
		//similar to clone_to_no_bias(), however it doesn't change the destination bias-related flags. It just ignores their existence
		const bool copy_data_skip_bias(smatrix& dest)const noexcept {
			if (dest.rows() != rows() || dest.cols_no_bias() != cols_no_bias()) {
				NNTL_ASSERT(!"Wrong dest matrix!");
				return false;
			}
			memcpy(dest.m_pData, m_pData, byte_size_no_bias());
			return true;
		}

		const bool copy_to(smatrix& dest)const noexcept {
			if (dest.rows() != rows() || dest.cols() != cols() || dest.emulatesBiases() != emulatesBiases()) {
				NNTL_ASSERT(!"Wrong dest matrix!");
				return false;
			}
			memcpy(dest.m_pData, m_pData, byte_size());
			dest.m_bHoleyBiases = m_bHoleyBiases;
			//dest.m_bEmulateBiases = m_bEmulateBiases;
			NNTL_ASSERT(*this == dest);
			return true;
		}
		
		//////////////////////////////////////////////////////////////////////////

		const bool operator==(const smatrix& rhs)const noexcept {
			//TODO: this is bad implementation, but it's enough cause we're gonna use it for testing only.
			return m_bEmulateBiases == rhs.m_bEmulateBiases && size() == rhs.size() && m_bHoleyBiases == rhs.m_bHoleyBiases
				&& 0 == memcmp(m_pData, rhs.m_pData, byte_size());
		}
		const bool operator!=(const smatrix& rhs)const noexcept {
			return !operator==(rhs);
		}

		const bool isAllocationFailed()const noexcept {
			return nullptr == m_pData && m_rows>0 && m_cols>0;
		}

		const bool bDontManageStorage()const noexcept { return m_bDontManageStorage; }

		//////////////////////////////////////////////////////////////////////////
		void fill_column_with(const vec_len_t c, const value_type& v)noexcept {
			NNTL_ASSERT(!empty() && m_rows && m_cols);
			NNTL_ASSERT(c < m_cols);
			const auto pC = colDataAsVec(c);
			std::fill(pC, pC + m_rows, v);
		}

		void set_biases() noexcept {
			NNTL_ASSERT(m_bEmulateBiases);
			// filling last column with ones to emulate biases
			fill_column_with(m_cols - 1, value_type(1.0));
			m_bHoleyBiases = false;
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

		const bool isHoleyBiases()const noexcept { 
			NNTL_ASSERT(!empty());//this concept applies to non-empty matrices only
			return m_bHoleyBiases;
		}
		void copy_biases_from(const value_type* ptr)noexcept {
			NNTL_ASSERT(m_bEmulateBiases && !empty());
			memcpy(colDataAsVec(m_cols - 1), ptr, static_cast<numel_cnt_t>(rows())*sizeof(value_type));
			m_bHoleyBiases = true;
			NNTL_ASSERT(test_biases_holey());
		}
		void copy_biases_from(const smatrix& src)noexcept {
			NNTL_ASSERT(m_bEmulateBiases && !empty() && src.emulatesBiases() && !src.empty());

			m_bHoleyBiases = src.isHoleyBiases();
			if (m_bHoleyBiases) {
				memcpy(colDataAsVec(m_cols - 1), src.colDataAsVec(src.cols() - 1), static_cast<numel_cnt_t>(rows()) * sizeof(value_type));
			}else fill_column_with(m_cols - 1, value_type(1.0));			
			
			NNTL_ASSERT(test_biases_ok());
		}

// 		void _holey_biases()noexcept {
// 			NNTL_ASSERT(!empty());//this concept applies to non-empty matrices only
// 			m_bHoleyBiases = true;
// 		}

// 	protected:
// 		void set_holey_biases(const bool b)noexcept { 
// 			NNTL_ASSERT(!empty());//this concept applies to non-empty matrices only
// 			m_bHoleyBiases = b;
// 		}

	public:
		//function is expected to be called from context of NNTL_ASSERT macro.
		bool test_biases_ok()const noexcept {
			//NNTL_ASSERT(emulatesBiases());
			//if (!emulatesBiases()) return false;//not necessary test, because there's no restriction to have biases otherwise
			return m_bHoleyBiases ? test_biases_holey() : test_biases_strict();
		}
		bool test_biases_strict()const noexcept {
			NNTL_ASSERT(emulatesBiases() && !m_bHoleyBiases);
			const auto ne = numel();
			auto pS = &m_pData[ne - m_rows];
			const auto pE = m_pData + ne;
			bool cond = true;
			while (pS != pE) {
				const auto c = *pS++ == value_type(1.0);
				NNTL_ASSERT(c || !"Bias check failed!");
				cond = cond && c;
			}
			return cond;
		}
		bool test_biases_holey()const noexcept {
			NNTL_ASSERT(emulatesBiases() && m_bHoleyBiases);

			const auto ne = numel();
			auto pS = &m_pData[ne - m_rows];
			const auto pE = m_pData + ne;
			bool cond = true;
			while (pS != pE) {
				const auto v = *pS++;
				const auto c = ((v == value_type(1.0)) | (v == value_type(0.0)));
				NNTL_ASSERT(c || !"Holey bias check failed!");
				cond = cond && c;
			}
			return cond;
		}


		//debug only
		bool isBinary()const noexcept {
			auto pS = m_pData;
			const auto pE = end();
			bool cond = true;
			while (pS != pE) {
				const auto v = *pS++;
				const auto c = ((v == value_type(1.0)) | (v == value_type(0.0)));
				NNTL_ASSERT(c || !"Not a binary matrix!");
				cond = cond && c;
			}
			return cond;
		}

		//for testing only
		void breakWhenDenormal()const noexcept {
			enable_denormals();
			auto pS = m_pData;
			const auto pE = end();
			while (pS != pE) {
				if (std::fpclassify(*pS++) == FP_SUBNORMAL) {
					__debugbreak();
				}
			}
			global_denormalized_floats_mode();
		}

		//////////////////////////////////////////////////////////////////////////
		const vec_len_t rows() const noexcept { return m_rows; }
		const vec_len_t cols() const noexcept { return m_cols; }

		// ***if biases are emulated, then actual columns number is one greater, than data cols number
		const vec_len_t cols_no_bias() const noexcept { 
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return m_cols - 1;
			}else return m_cols;
		}

		const mtx_size_t size()const noexcept { return mtx_size_t(m_rows, m_cols); }
		const mtx_size_t size_no_bias()const noexcept { 
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return mtx_size_t(m_rows, m_cols-1);
			}else return size();
		}
		
		static constexpr numel_cnt_t sNumel(vec_len_t r, vec_len_t c)noexcept { 
			//NNTL_ASSERT(r && c); //sNumel is used in many cases where r==0 or c==0 is perfectly legal. No need to assert here.
			return static_cast<numel_cnt_t>(r)*static_cast<numel_cnt_t>(c); 
		}
		static constexpr numel_cnt_t sNumel(const mtx_size_t& s)noexcept { return sNumel(s.first, s.second); }

		const numel_cnt_t numel()const noexcept { 
			//NNTL_ASSERT(m_rows && m_cols);
			return sNumel(m_rows,m_cols); 
		}
		const numel_cnt_t numel_no_bias()const noexcept { 
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return sNumel(m_rows,m_cols-1);
			} else return numel();
		}

		//////////////////////////////////////////////////////////////////////////
		// triangular matrix support
		//returns the number of elements in a triangular matrix of size N. (Elements of the main diagonal are excluded)
		static constexpr numel_cnt_t sNumelTriangl(const vec_len_t& n)noexcept {
			return (static_cast<numel_cnt_t>(n)*(n-1)) / 2;
		}
		const numel_cnt_t numel_triangl()const noexcept {
			NNTL_ASSERT(rows() == cols());
			return sNumelTriangl(rows());
		}
		//returns (ri,ci) coordinates of the element with index k of upper/lower (toggled by template parameter bool bLowerTriangl) triangular matrix
		template<bool bLowerTriangl>
		static std::enable_if_t<!bLowerTriangl> sTrianglCoordsFromIdx(const vec_len_t& n, const numel_cnt_t& k, vec_len_t& ri, vec_len_t& ci) noexcept {
			NNTL_ASSERT(k <= sNumelTriangl(n));
			const auto _ci = static_cast<numel_cnt_t>(std::ceil((std::sqrt(static_cast<real_t>(8 * k + 9)) - 1) / 2));
			ci = static_cast<vec_len_t>(_ci);
			NNTL_ASSERT(ci <= n);
			ri = static_cast<vec_len_t>(k - _ci*(_ci - 1) / 2);
			NNTL_ASSERT(static_cast<int>(ri) >= 0 && ri < n);
		}
		template<bool bLowerTriangl>
		static std::enable_if_t<bLowerTriangl> sTrianglCoordsFromIdx(const vec_len_t& n, const numel_cnt_t& k, vec_len_t& ri, vec_len_t& ci) noexcept {
			const auto trNumel = sNumelTriangl(n);
			NNTL_ASSERT(k <= trNumel);
			const auto nm1 = n - 1;
			if (k==trNumel) {
				ri = 0;
				ci = nm1;
			} else {
				const auto _k = trNumel - k - 1;
				const auto _ci = static_cast<numel_cnt_t>(std::ceil((std::sqrt(static_cast<real_t>(8 * _k + 9)) - 1) / 2));
				NNTL_ASSERT(_ci < n);				
				ci = nm1 - static_cast<vec_len_t>(_ci);
				ri = nm1 - static_cast<vec_len_t>(_k - _ci*(_ci - 1) / 2);
				NNTL_ASSERT(static_cast<int>(ri) > 0 && ri < n);
			}
			
		}
		template<bool bLowerTriangl>
		void triangl_coords_from_idx(const numel_cnt_t& k, vec_len_t& ri, vec_len_t& ci)const noexcept {
			NNTL_ASSERT(rows() == cols());
			sTrianglCoordsFromIdx<bLowerTriangl>(rows(), k, ri, ci);
		}

		//////////////////////////////////////////////////////////////////////////

		const numel_cnt_t byte_size()const noexcept {
			return numel()*sizeof(value_type);
		}
		const numel_cnt_t byte_size_no_bias()const noexcept {
			return numel_no_bias()*sizeof(value_type);
		}

		const bool empty()const noexcept { return nullptr == m_pData; }

		//to conform std::vector API
		value_ptr_t data()noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0);
			return m_pData;
		}
		cvalue_ptr_t data()const noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0);
			return m_pData;
		}		

		//not a real iterators
		value_ptr_t begin()noexcept { return data(); }
		cvalue_ptr_t begin()const noexcept { return data(); }
		value_ptr_t end()noexcept { return data()+numel(); }
		cvalue_ptr_t end()const noexcept { return data()+numel(); }

		value_ptr_t end_no_bias()noexcept { return data() + numel_no_bias(); }
		cvalue_ptr_t end_no_bias()const noexcept { return data() + numel_no_bias(); }

		value_ptr_t colDataAsVec(vec_len_t c)noexcept {
			NNTL_ASSERT(!empty() && m_cols>0 && m_rows>0 && c <= m_cols);//non strict inequality c <= m_cols to allow reference 'after the last' column
			return m_pData + sNumel(m_rows, c);
		}
		cvalue_ptr_t colDataAsVec(vec_len_t c)const noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0 && c <= m_cols);//non strict inequality c <= m_cols to allow reference 'after the last' column
			return m_pData + sNumel(m_rows, c);
		}

		// get/set are for non performance critical code!
		void set(const vec_len_t r, const vec_len_t c, const value_type& v)noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			m_pData[ r + sNumel(m_rows, c)] = v;
		}
		const value_type& get(const vec_len_t r, const vec_len_t c)const noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			return m_pData[r + sNumel(m_rows, c)];
		}
		value_type& get(const vec_len_t r, const vec_len_t c) noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			return m_pData[r + sNumel(m_rows, c)];
		}
		const value_type& get(const mtx_coords_t& crd)const noexcept { return get(crd.first, crd.second); }
		value_type& get(const mtx_coords_t& crd) noexcept { return get(crd.first, crd.second); }

		void clear()noexcept {
			_free();
			//m_bEmulateBiases = false; //MUST NOT clear this flag; it defines a matrix mode and clear() just performs a cleanup. No mode change allowed here.
			m_bDontManageStorage = false;
			m_bHoleyBiases = false;//this is a kind of run-time state of the bias column. We've cleared the matrix, therefore clearing this state too.
		}
		const bool resize(const smatrix& m)noexcept {
			NNTL_ASSERT(!m.empty() && m.rows() > 0 && m.cols() > 0);
			NNTL_ASSERT(emulatesBiases() == m.emulatesBiases());
			return resize(m.rows(), m.cols());
		}
		const bool resize(const mtx_size_t& sz)noexcept { return resize(sz.first, sz.second); }
		const bool resize(const vec_len_t r, vec_len_t c) noexcept {
			NNTL_ASSERT(!m_bDontManageStorage);
			NNTL_ASSERT(r > 0 && c > 0);
			if (r <= 0 || c <= 0) {
				NNTL_ASSERT(!"Wrong row or col count!");
				return false;
			}

			if (m_bEmulateBiases) c++;

			if (r == m_rows && c == m_cols) return true;

			m_rows = r;
			m_cols = c;

			_realloc();

			const bool bRet = nullptr != m_pData;
			if (m_bEmulateBiases && bRet) set_biases();
			m_bHoleyBiases = false;
			return bRet;
		}

		void zeros()noexcept {
			NNTL_ASSERT(!empty());
			memset(m_pData, 0, byte_size_no_bias());
		}
		void ones()noexcept {
			NNTL_ASSERT(!empty());
			std::fill(m_pData, m_pData + numel(), value_type(1.0));
			m_bHoleyBiases = false;
		}

		// fills matrix with data from pSrc doing type conversion. Bias units left untouched.
		template<typename OtherBaseT>
		std::enable_if_t<!std::is_same<value_type, OtherBaseT>::value> fill_from_array_no_bias(const OtherBaseT*const pSrc)noexcept {
			static_assert(std::is_arithmetic<OtherBaseT>::value, "OtherBaseT must be a simple arithmetic data type");
			NNTL_ASSERT(!empty() && numel() > 0);
			const auto ne = numel_no_bias();
			const auto p = data();
			for (numel_cnt_t i = 0; i < ne; ++i) p[i] = static_cast<value_type>(pSrc[i]);
		}
		template<typename OtherBaseT>
		std::enable_if_t<std::is_same<value_type, OtherBaseT>::value> fill_from_array_no_bias(const OtherBaseT*const pSrc)noexcept {
			NNTL_ASSERT(!empty() && numel() > 0);
			memcpy(m_pData, pSrc, byte_size_no_bias());
		}


		void useExternalStorage_no_bias(value_ptr_t ptr, const smatrix& sizeLikeThisNoBias, bool bHBiases = false)noexcept {
			useExternalStorage(ptr, sizeLikeThisNoBias.rows(), sizeLikeThisNoBias.cols_no_bias(), false, bHBiases);
		}
		void useExternalStorage_no_bias(smatrix& src)noexcept {
			useExternalStorage(src.data(), src.rows(), src.cols_no_bias(), false, false);
		}
		void useExternalStorage(smatrix& src)noexcept {
			useExternalStorage(src.data(), src.rows(), src.cols_no_bias(), src.emulatesBiases(), src.isHoleyBiases());
		}
		void useExternalStorage(value_ptr_t ptr, const mtx_size_t& sizeLikeThis, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
			useExternalStorage(ptr, sizeLikeThis.first, sizeLikeThis.second, bEmulateBiases, bHBiases);
		}
		void useExternalStorage(value_ptr_t ptr, const smatrix& sizeLikeThis, bool bHBiases = false)noexcept {
			useExternalStorage(ptr, sizeLikeThis.rows(), sizeLikeThis.cols(), sizeLikeThis.emulatesBiases(),bHBiases);
		}
		//note that bEmulateBiases doesn't increment _cols count and doesn't fill biases if specified! This is done to make
		// sure that a caller knows that ptr will address enough space and to prevent unnecessary memory writes.
		// Also this allows to use matrix sizeLikeThis as an argument to useExternalStorage() call
		void useExternalStorage(value_ptr_t ptr, vec_len_t _rows, vec_len_t _cols, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
			NNTL_ASSERT(ptr && _rows > 0 && _cols>0);
			_free();
			m_bDontManageStorage = true;
			m_pData = ptr;
			m_rows = _rows;
			m_cols = _cols;
			m_bEmulateBiases = bEmulateBiases;
			m_bHoleyBiases = bHBiases;

			//usually, external storage is used with shared memory to store temporary computations, therefore data won't survive between
			// usages and there is no need to prefill it with biases. m_bEmulateBiases flag is helpful for routines like numel_no_bias()
			//if (bEmulateBiases) _fill_biases();
			//return true;
		}

		const bool useInternalStorage(vec_len_t _rows=0, vec_len_t _cols=0, bool bEmulateBiases = false)noexcept {
			_free();
			m_bDontManageStorage = false;
			m_bHoleyBiases = false;
			m_bEmulateBiases = bEmulateBiases;
			if (_rows > 0 && _cols > 0) {
				return resize(_rows, _cols);
			} else return true;
		}

		void assert_storage_does_not_intersect(const smatrix& m)const noexcept {
			NNTL_ASSERT(this != &m);
			NNTL_ASSERT(!empty() && !m.empty());
			//nonstrict nonequality it necessary here, because &[numel()] references element past the end of the allocated array
			NNTL_ASSERT( &m.m_pData[m.numel()] <= m_pData || &m_pData[numel()] <= m.m_pData);
		}

	};

	//////////////////////////////////////////////////////////////////////////
	//this class will allow to reuse the same storage for matrix of smaller size
	//////////////////////////////////////////////////////////////////////////
	template <typename T_>
	class smatrix_deform : public smatrix<T_> {
	private:
		typedef smatrix<T_> _base_class;

	protected:
#ifdef NNTL_DEBUG
		numel_cnt_t m_maxSize;
#endif // NNTL_DEBUG

	public:
		~smatrix_deform()noexcept {}
		smatrix_deform()noexcept : _base_class() {}
		smatrix_deform(const vec_len_t _rows, const vec_len_t _cols, const bool _bEmulBias = false)noexcept : _base_class(_rows, _cols, _bEmulBias) {
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		smatrix_deform(smatrix&& src)noexcept : _base_class(std::move(src)) {
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		smatrix_deform& operator=(smatrix&& rhs) noexcept {
			if (this != &rhs) {
				_base_class::operator =(std::move(rhs));
#ifdef NNTL_DEBUG
				m_maxSize = numel();
#endif // NNTL_DEBUG
			}
			return *this;
		}


		void clear()noexcept {
#ifdef NNTL_DEBUG
			m_maxSize = 0;
#endif // NNTL_DEBUG
			_base_class::clear();
		}

		const bool cloneFrom(const smatrix& src)noexcept {
			_free();
			auto r = src.clone_to(*this);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		void update_on_hidden_resize()noexcept {
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		const bool resize(const vec_len_t _rows, vec_len_t _cols)noexcept {
			_free();
			const auto r = _base_class::resize(_rows, _cols);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		const bool resize(const smatrix& m)noexcept {
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
			if (nullptr == ptr) {
				NNTL_ASSERT(!"Memory allocation failed!");
				return false;
			}
			useExternalStorage(ptr, ne);
			m_bDontManageStorage = false;
			return true;
		}

		void useExternalStorage(value_ptr_t ptr, vec_len_t _rows, vec_len_t _cols, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
			_base_class::useExternalStorage(ptr, _rows, _cols, bEmulateBiases,bHBiases);
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}
		void useExternalStorage(value_ptr_t ptr, const smatrix& sizeLikeThis, bool bHBiases = false)noexcept {
			useExternalStorage(ptr, sizeLikeThis.rows(), sizeLikeThis.cols(), sizeLikeThis.emulatesBiases(), bHBiases);
		}
		void useExternalStorage(value_ptr_t ptr, numel_cnt_t cnt, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
			NNTL_ASSERT(cnt < std::numeric_limits<vec_len_t>::max());
			_base_class::useExternalStorage(ptr, static_cast<vec_len_t>(cnt), 1, bEmulateBiases, bHBiases);
#ifdef NNTL_DEBUG
			m_maxSize = cnt;
#endif // NNTL_DEBUG
		}

		void useExternalStorage(smatrix& src)noexcept {
			_base_class::useExternalStorage(src);
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		void useExternalStorage_no_bias(smatrix& src)noexcept {
			_base_class::useExternalStorage_no_bias(src);
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
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

		bool hide_biases()noexcept {
			bool hasBiases = emulatesBiases();
			if (hasBiases) {
				hide_last_col();
				m_bEmulateBiases = false;
			}
			return hasBiases;
		}
		void restore_biases()noexcept { //should only be called iff hide_biases() returned true
			NNTL_ASSERT(!emulatesBiases());
			restore_last_col();
			m_bEmulateBiases = true;
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

		void deform_like(const smatrix& m)noexcept { deform(m.rows(), m.cols()); }
		void deform_like_no_bias(const smatrix& m)noexcept { deform(m.rows(), m.cols_no_bias()); }
	};

	//////////////////////////////////////////////////////////////////////////
	// helper class to define matrix elements range
	//////////////////////////////////////////////////////////////////////////
	//template<typename T_>
	class s_elems_range : public smatrix_td {
	public:
		typedef threads::parallel_range<numel_cnt_t> par_range_t;

	public:
		const numel_cnt_t elmEnd;
		const numel_cnt_t elmBegin;
		//const bool bInsideMT;

	public:
		~s_elems_range()noexcept {}

		template<typename _T>
		s_elems_range(const smatrix<_T>& A)noexcept : elmEnd(A.numel()), elmBegin(0){}

		s_elems_range(const par_range_t& pr)noexcept : elmEnd(pr.offset() + pr.cnt()), elmBegin(pr.offset()) {}

		s_elems_range(const numel_cnt_t eb, const numel_cnt_t ee)noexcept : elmEnd(ee), elmBegin(eb) {
			NNTL_ASSERT(elmEnd >= elmBegin);
		}

		const numel_cnt_t totalElements()const noexcept { return elmEnd - elmBegin; }
	};

	//////////////////////////////////////////////////////////////////////////
	// helper class to define matrix rows-cols range
	//////////////////////////////////////////////////////////////////////////
	// in general, it depends on how to use it, but usually the class doesn't specify a rectangular sub-block of a matrix,
	// but defines a range that starts and ends on the specified elements.
	class s_rowcol_range : public smatrix_td {
	public:
		const vec_len_t rowEnd;
		const vec_len_t rowBegin;
		const vec_len_t colEnd;
		const vec_len_t colBegin;

	public:
		~s_rowcol_range()noexcept {}

		template<typename T_>
		s_rowcol_range(const vec_len_t rb, const vec_len_t re, const smatrix<T_>& A)noexcept : rowEnd(re), rowBegin(rb), colEnd(A.cols()), colBegin(0) {
			NNTL_ASSERT(rowEnd >= rowBegin);
		}

		template<typename T_>
		s_rowcol_range(const smatrix<T_>& A, const vec_len_t cb, const vec_len_t ce)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(ce), colBegin(cb) {
			NNTL_ASSERT(colEnd >= colBegin);
		}

		template<typename T_>
		s_rowcol_range(const smatrix<T_>& A)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(A.cols()), colBegin(0) {}

		const vec_len_t totalRows()const noexcept { return rowEnd - rowBegin; }
		const vec_len_t totalCols()const noexcept { return colEnd - colBegin; }

		template<typename T_>
		const bool can_apply(const smatrix<T_>& A)const noexcept {
			const auto r = A.rows(), c = A.cols();
			NNTL_ASSERT(rowBegin <= rowEnd && rowBegin < r && rowEnd <= r);
			NNTL_ASSERT(colBegin <= colEnd && colBegin < c && colEnd <= c);
			return rowBegin < r && rowEnd <= r
				&& colBegin < c && colEnd <= c;
		}
	};
/*
	template<typename T_>
	class s_rowcol_range {
	public:
		typedef T_ value_type;
		typedef smatrix<value_type> simplemtx_t;
		typedef typename simplemtx_t::vec_len_t vec_len_t;

	public:
		const vec_len_t rowEnd;
		const vec_len_t rowBegin;
		const vec_len_t colEnd;
		const vec_len_t colBegin;

	public:
		~s_rowcol_range()noexcept {}
		s_rowcol_range(const vec_len_t rb, const vec_len_t re, const simplemtx_t& A)noexcept : rowEnd(re), rowBegin(rb), colEnd(A.cols()), colBegin(0) {
			NNTL_ASSERT(rowEnd >= rowBegin);
		}
		s_rowcol_range(const simplemtx_t& A, const vec_len_t cb, const vec_len_t ce)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(ce), colBegin(cb) {
			NNTL_ASSERT(colEnd >= colBegin);
		}
		s_rowcol_range(const simplemtx_t& A)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(A.cols()), colBegin(0) {}

		const vec_len_t totalRows()const noexcept { return rowEnd - rowBegin; }
		const vec_len_t totalCols()const noexcept { return colEnd - colBegin; }

		const bool can_apply(const simplemtx_t& A)const noexcept {
			const auto r = A.rows(), c = A.cols();
			NNTL_ASSERT(rowBegin <= rowEnd && rowBegin < r && rowEnd <= r);
			NNTL_ASSERT(colBegin <= colEnd && colBegin < c && colEnd <= c);
			return rowBegin < r && rowEnd <= r
				&& colBegin < c && colEnd <= c;
		}
	};*/
	
}
}
