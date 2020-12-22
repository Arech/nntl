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

#include <utility>
//#include <intrin.h>

#include "../../_defs.h"
#include "../../common.h"
#include "_base.h"
#include "../threads/parallel_range.h"

//#TODO: Consider replacing generic memcpy/memcmp/memset and others similar generic functions with
//their versions from Agner Fox's asmlib. TEST if it really helps!!!!

namespace nntl {
namespace math {

	struct tag_noBias {};

	// types that don't rely on matrix value_type
	struct smatrix_td {
		//rows/cols type. int should be enough. If not, redefine to smth bigger
		//typedef uint32_t vec_len_t;
		// And note! To allow efficient vectorization of for() loops, their loop variable MUST be of a signed type, because
		//signed overflow is an "undefined behavior" that allow compiler to skip an additional check each increment.
		// In contrast to this, unsigned overflow is well defined and for that reason for loops can't be properly vectorized in many cases.
		// So counter variables actually MUST be signed!
		//static_assert(::std::is_signed<neurons_count_t>::value, "Hey! neurons_count_t MUST be signed!");
		//typedef neurons_count_t vec_len_t;
		
		//typedef ::std::make_signed_t<size_t> numel_cnt_t;
		
		typedef ::std::pair<const vec_len_t, const vec_len_t> mtx_size_t;
		typedef ::std::pair<vec_len_t, vec_len_t> mtx_coords_t;

		typedef tag_noBias tag_noBias;

		/*typedef ::std::make_signed_t<vec_len_t> vec_len_idx_t;
		typedef ::std::make_signed_t<numel_cnt_t> numel_cnt_idx_t;

		//////////////////////////////////////////////////////////////////////////

		static constexpr vec_len_idx_t s_vec_len_2_idx(vec_len_t v)noexcept{
			return static_cast<vec_len_idx_t>(v);
		}*/

		//////////////////////////////////////////////////////////////////////////

		inline static constexpr numel_cnt_t sNumel(const vec_len_t r, const vec_len_t c)noexcept {
			return static_cast<numel_cnt_t>(r)*static_cast<numel_cnt_t>(c); 
			//return __emulu(r, c);
			//return __emul(r, c);
		}
		inline static constexpr numel_cnt_t sNumel(const mtx_size_t s)noexcept { return sNumel(s.first, s.second); }

		//////////////////////////////////////////////////////////////////////////
		//debug/NNTL_ASSERT use is preferred
		template<typename T = value_type>
		static ::std::enable_if_t<::std::is_floating_point<T>::value, bool>
		_isBinaryVec(const T* pData, const numel_cnt_t ne, const bool bNonBinIsOK = false) noexcept {
			NNTL_UNREF(bNonBinIsOK);//it's used just to trigger assert and nothing more
			typedef typename real_t_limits<T>::similar_FWI_t similar_FWI_t;

			const auto _one = similar_FWI_one<T>();
			const auto _poszero = similar_FWI_pos_zero<T>();
			const auto _negzero = similar_FWI_neg_zero<T>();
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			auto ptr = reinterpret_cast<const similar_FWI_t*>(pData);

			const auto pE = ptr + ne;
			similar_FWI_t cond = similar_FWI_t(1);
			while (ptr != pE) {//vectorizes
				const auto v = *ptr++;
				//we must make sure that binary zero is an actual unsigned(positive) zero
				const similar_FWI_t c = ((v == _one) | (v == _poszero) | (v == _negzero));
				NNTL_ASSERT(bNonBinIsOK || c || !"Not a binary vector!");
				cond = cond & c;
			}
			return !!cond;
		}
		template<typename T = value_type>
		static ::std::enable_if_t<::std::is_integral<T>::value, bool> _isBinaryVec(const T* ptr, const numel_cnt_t ne, const bool bNonBinIsOK = false) noexcept {
			return _isBinaryStrictVec(ptr, ne, bNonBinIsOK);
		}
		//////////////////////////////////////////////////////////////////////////
		template<typename T = value_type>
		static ::std::enable_if_t<::std::is_floating_point<T>::value, bool> _isBinaryStrictVec(const T* pData, const numel_cnt_t ne, const bool bNonBinIsOK = false) noexcept {
			typedef typename real_t_limits<T>::similar_FWI_t similar_FWI_t;

			const auto _one = similar_FWI_one<T>();
			const auto _zero = similar_FWI_pos_zero<T>();
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			auto ptr = reinterpret_cast<const similar_FWI_t*>(pData);

			const auto pE = ptr + ne;
			similar_FWI_t cond = similar_FWI_t(1);
			while (ptr != pE) {//vectorizes
				const auto v = *ptr++;
				//we must make sure that binary zero is an actual unsigned(positive) zero
				const similar_FWI_t c = ((v == _one) | (v == _zero));
				NNTL_ASSERT(bNonBinIsOK || c || !"Not a binary vector!");
				cond = cond & c;
			}
			return !!cond;
		}
		template<typename T = value_type>
		static ::std::enable_if_t<::std::is_integral<T>::value, bool> _isBinaryStrictVec(const T* ptr, const numel_cnt_t ne, const bool bNonBinIsOK = false) noexcept {
			const auto _one = T(1);
			const auto _zero = T(0);

			const auto pE = ptr + ne;
			T cond = T(1);
			while (ptr != pE) {//should vectorize, but better check for every type needed
				const auto v = *ptr++;
				//we must make sure that binary zero is an actual unsigned(positive) zero
				const T c = ((v == _one) | (v == _zero));
				NNTL_ASSERT(bNonBinIsOK || c || !"Not a binary vector!");
				cond = cond & c;
			}
			return !!cond;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// wrapper class to store vectors/matrices in column-major ordering
	// #TODO note that using char as T_ template parameter may lead to strict aliasing related performance penalties
	// in many cases where accessing data via char* will be used. Probably should make here some type substituting voodoo to get
	// rid of char if char is used ?
	template <typename T_>
	class smatrix : public smatrix_td {
		static_assert(::std::is_pod<T_>::value, "Matrix type must be POD type for proper mem allocation");
		//non-pod type would require constructor/destructor code to run during new/delete

	public:
		typedef T_ value_type;
		typedef value_type* value_ptr_t;
		typedef const value_type* cvalue_ptr_t;
		
		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		// no ::std::unique_ptr here because of exception unsafety.
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

		bool m_bHoleyBiases; //off by default. This flag is for the test_biases_ok() function and some optimizations use only. When it is on,
		//it is ok for a bias to also have a value of zero. We need this flag to support gating layers, that should completely
		// blackout data samples including biases. If we are to remove it, test_biases_ok() would fail assestions on gated layers.
		// This flag doesn't require m_bEmulateBiases flag, because matrix could have biases without this flag set in some cases

	protected:
		void _realloc() noexcept {
			if (m_bDontManageStorage) {
				NNTL_ASSERT(!"Hey! WTF? You cant manage matrix memory in m_bDontManageStorage mode!");
				abort();
			} else {
				//delete[] m_pData;
				_aligned_free(m_pData);
				if (m_rows > 0 && m_cols > 0) {
					//m_pData = new(::std::nothrow) value_type[numel()];
					//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
					//#TODO: NNTL_CFG_DEFAULT_FP_PTR_ALIGN is too strict for non-floating point data
					m_pData = reinterpret_cast<value_type*>(_aligned_malloc(byte_size(), utils::mem_align_for<value_type>()));
					NNTL_ASSERT(m_pData);
				} else {
					m_rows = 0;
					m_cols = 0;
					m_pData = nullptr;
				}
			}
		}
		void _free()noexcept {
			if (!m_bDontManageStorage) {
				//delete[] m_pData;
				_aligned_free(m_pData);
			}
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
			useExternalStorage(ptr, sizeLikeThis, sizeLikeThis.emulatesBiases() ? sizeLikeThis.isHoleyBiases() : false);
		}

		smatrix(value_ptr_t ptr, const mtx_size_t& sizeLikeThis, const bool bEmulateBiases = false, const bool _bHoleyBiases = false)noexcept
			: m_pData(nullptr), m_bDontManageStorage(false)
		{
			useExternalStorage(ptr, sizeLikeThis, bEmulateBiases, _bHoleyBiases);
		}
		smatrix(value_ptr_t ptr, const vec_len_t r, const vec_len_t c, const bool bEmulateBiases = false, const bool _bHoleyBiases = false) noexcept
			: m_pData(nullptr), m_bDontManageStorage(false)
		{
			useExternalStorage(ptr, r, c, bEmulateBiases, _bHoleyBiases);
		}

		//useExternalStorage_no_bias(value_ptr_t ptr, const smatrix& sizeLikeThis) variation
		smatrix(value_ptr_t ptr, const smatrix& sizeLikeThisNoBias, const tag_noBias&)noexcept : m_pData(nullptr), m_bDontManageStorage(false)
		{
			useExternalStorage_no_bias(ptr, sizeLikeThisNoBias);
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
		
		//////////////////////////////////////////////////////////////////////////
		//!! copy constructor not needed
		smatrix(const smatrix& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		smatrix& operator=(const smatrix& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

		//////////////////////////////////////////////////////////////////////////

		smatrix submatrix_cols_no_bias(const vec_len_t colStart, const vec_len_t numCols)noexcept {
			NNTL_ASSERT(numCols);
			NNTL_ASSERT(colStart + numCols <= cols_no_bias());
			return smatrix(colDataAsVec(colStart), rows(), numCols, false);
		}

		//////////////////////////////////////////////////////////////////////////
		
		//class object copying is a costly procedure, therefore making a special member function for it and will use only in special cases
		bool clone_to(smatrix& dest)const noexcept {
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
		bool clone_to_no_bias(smatrix& dest)const noexcept {
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
		bool copy_data_skip_bias(smatrix& dest)const noexcept {
			if (dest.rows() != rows() || dest.cols_no_bias() != cols_no_bias()) {
				NNTL_ASSERT(!"Wrong dest matrix!");
				return false;
			}
			memcpy(dest.m_pData, m_pData, byte_size_no_bias());
			return true;
		}

		bool copy_to(smatrix& dest)const noexcept {
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

		bool operator==(const smatrix& rhs)const noexcept {
			//TODO: this is bad implementation, but it's enough cause we're gonna use it for testing only.
			return m_bEmulateBiases == rhs.m_bEmulateBiases && size() == rhs.size() && m_bHoleyBiases == rhs.m_bHoleyBiases
				&& 0 == memcmp(m_pData, rhs.m_pData, byte_size());
		}
		bool operator!=(const smatrix& rhs)const noexcept {
			return !operator==(rhs);
		}

		bool isAllocationFailed()const noexcept {
			return nullptr == m_pData && m_rows>0 && m_cols>0;
		}

		inline bool bDontManageStorage()const noexcept { return m_bDontManageStorage; }

		//////////////////////////////////////////////////////////////////////////
		void fill_column_with(const vec_len_t c, const value_type v)noexcept {
			NNTL_ASSERT(!empty() && m_rows && m_cols);
			NNTL_ASSERT(c < m_cols);
// 			const auto pC = colDataAsVec(c);
// 			::std::fill(pC, pC + m_rows, v); //doesn't get vectorized!
			_fill_elements(colDataAsVec(c), m_rows, v);
		}

	protected:
		//::std::fill(m_pData, m_pData + numel(), value_type(1.0)); //doesn't get vectorized!
		void _fill_elements(const value_ptr_t pBegin, const numel_cnt_t n, const value_type v)noexcept {
			for (numel_cnt_t i = 0; i < n; ++i) pBegin[i] = v;
		}

	public:
		void set_biases() noexcept {
			NNTL_ASSERT(m_bEmulateBiases);
			// filling last column with ones to emulate biases
			fill_column_with(m_cols - 1, value_type(1.0));
			m_bHoleyBiases = false;
		}
		inline bool emulatesBiases()const noexcept { return m_bEmulateBiases; }
		void will_emulate_biases()noexcept {
			NNTL_ASSERT(empty() && m_rows == 0 && m_cols == 0);
			m_bEmulateBiases = true;
		}
		void dont_emulate_biases()noexcept {
			NNTL_ASSERT(empty() && m_rows == 0 && m_cols == 0);
			m_bEmulateBiases = false;
		}
		void emulate_biases(const bool b)noexcept {
			NNTL_ASSERT(empty() && m_rows == 0 && m_cols == 0);
			m_bEmulateBiases = b;
		}

		bool isHoleyBiases()const noexcept { 
			NNTL_ASSERT(!empty() && emulatesBiases());//this concept applies to non-empty matrices with a bias column only
			return m_bHoleyBiases;
		}
		bool hasHoleyBiases()const noexcept {//almost the same as isHoleyBiases() but doesn't require the matrix to have biases
			NNTL_ASSERT(!empty());//this concept applies to non-empty matrices only
			return m_bEmulateBiases && m_bHoleyBiases;
		}
		void holey_biases(const bool b)noexcept {
			NNTL_ASSERT(emulatesBiases());
			m_bHoleyBiases = b;
		}

		void copy_biases_from(const value_type* ptr, const vec_len_t countNonZeros = 0)noexcept {
			NNTL_ASSERT(m_bEmulateBiases && !empty() && rows() >= countNonZeros);
			NNTL_ASSERT(_isBinaryVec(ptr, rows()));
			memcpy(bias_column(), ptr, static_cast<numel_cnt_t>(rows())*sizeof(value_type));
			m_bHoleyBiases = countNonZeros != rows();
		}
		void copy_biases_from(const smatrix& src)noexcept {
			NNTL_ASSERT(m_bEmulateBiases && !empty() && src.emulatesBiases() && !src.empty());
			NNTL_ASSERT(src.rows() == rows());

			m_bHoleyBiases = src.isHoleyBiases();
			if (m_bHoleyBiases) {
				NNTL_ASSERT(_isBinaryVec(src.bias_column(), src.rows()));
				::std::memcpy(bias_column(), src.bias_column(), static_cast<numel_cnt_t>(rows()) * sizeof(value_type));
			}else fill_column_with(m_cols - 1, value_type(1.0));
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//function is expected to be called from context of NNTL_ASSERT macro.
		//debug/NNTL_ASSERT only
		bool test_biases_ok()const noexcept {
			return m_bHoleyBiases ? test_biases_holey() : test_biases_strict();
		}
		//debug/NNTL_ASSERT only
		bool test_biases_strict()const noexcept {
			NNTL_ASSERT(emulatesBiases() && !m_bHoleyBiases);
			const auto ne = numel();
			auto pS = &m_pData[ne - m_rows];
			const auto pE = m_pData + ne;
			int cond = 1;
			while (pS != pE) {
				const int c = *pS++ == value_type(1.0);
				NNTL_ASSERT(c || !"Bias check failed!");
				cond = cond & c;
			}
			return !!cond;
		}
		//debug/NNTL_ASSERT use only
		bool test_biases_holey()const noexcept {
			NNTL_ASSERT(emulatesBiases() && m_bHoleyBiases);
			return _isBinaryVec(bias_column(), rows());
		}

		//debug/NNTL_ASSERT use only
		template<bool b = ::std::is_floating_point<value_type>::value>
		::std::enable_if_t<b,bool> test_noNaNs()const noexcept {
			const auto ne = numel();
			auto pS = m_pData;
			const auto pE = m_pData + ne;
			int cond = 0;
			while (pS != pE) {
				const auto v = *pS++;
				const int c = ::std::isnan(v);
				NNTL_ASSERT(!c || !"NaN check failed!");
				cond = cond | c;
			}
			return !cond;
		}
		template<bool b = ::std::is_floating_point<value_type>::value>
		constexpr ::std::enable_if_t<!b, bool> test_noNaNs()const noexcept { return true; }

		//////////////////////////////////////////////////////////////////////////
		//debug/NNTL_ASSERT use is preferred
		bool _isBinary(const bool bNonBinIsOK = false)const noexcept {
			return _isBinaryVec<value_type>(m_pData, numel(), bNonBinIsOK);
		}
		bool _isBinaryStrict(const bool bNonBinIsOK = false)const noexcept {
			return _isBinaryStrictVec<value_type>(m_pData, numel(), bNonBinIsOK);
		}
		bool _isBinaryStrictNoBias(const bool bNonBinIsOK = false)const noexcept {
			return _isBinaryStrictVec<value_type>(m_pData, numel_no_bias(), bNonBinIsOK);
		}
		
		//debug/NNTL_ASSERT only
		template<typename _T> struct isAlmostBinary_eps {};
		template<> struct isAlmostBinary_eps<float> { static constexpr float eps = float(1e-8); };
		template<> struct isAlmostBinary_eps<double> { static constexpr double eps = 1e-16; };
		//to account numeric problems
		bool _isAlmostBinary(const value_type eps/* = isAlmostBinary_eps<value_type>::eps*/)const noexcept {
			auto pS = m_pData;
			const auto pE = end();
			int cond = 1;
			while (pS != pE) {
				const auto v = *pS++;
				const int c = (::std::abs(v - value_type(1.0)) < eps | ::std::abs(v) < eps);
				NNTL_ASSERT(c || !"Not a (almost) binary matrix!");
				cond = cond & c;
			}
			return !!cond;
		}

		//for testing/debugging only
		void _breakWhenDenormal()const noexcept {
			enable_denormals();
			auto pS = m_pData;
			const auto pE = end();
			while (pS != pE) {
				if (::std::fpclassify(*pS++) == FP_SUBNORMAL) {
					__debugbreak();
				}
			}
			global_denormalized_floats_mode();
		}

		//////////////////////////////////////////////////////////////////////////
		inline vec_len_t rows() const noexcept { return m_rows; }
		inline vec_len_t cols() const noexcept { return m_cols; }

		//leading matrix dimension
		//it's true for an old code (to be updated), but in new code NEVER assume ldim==rows, always use ldim()
		inline vec_len_t ldim() const noexcept { return m_rows; }

		// ***if biases are emulated, then actual columns number is one greater, than data cols number
		inline vec_len_t cols_no_bias() const noexcept {
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return m_cols - 1;
			}else return m_cols;
		}
		inline vec_len_t cols(const bool bNoBias) const noexcept {
			if (bNoBias && m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return m_cols - 1;
			} else return m_cols;
		}

		mtx_size_t size()const noexcept { return mtx_size_t(m_rows, m_cols); }
		mtx_size_t transposed_size()const noexcept { return mtx_size_t(m_cols, m_rows); }

		mtx_size_t size_no_bias()const noexcept { 
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return mtx_size_t(m_rows, m_cols-1);
			}else return size();
		}
		mtx_size_t transposed_size_no_bias()const noexcept {
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return mtx_size_t(m_cols - 1, m_rows);
			} else return size();
		}

		inline numel_cnt_t numel()const noexcept {
			//NNTL_ASSERT(m_rows && m_cols);
			return sNumel(m_rows,m_cols); 
		}
		inline numel_cnt_t numel_no_bias()const noexcept {
			if (m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return sNumel(m_rows,m_cols-1);
			} else return numel();
		}
		inline numel_cnt_t numel(const bool bNoBias)const noexcept {
			if (bNoBias && m_bEmulateBiases) {
				NNTL_ASSERT(m_cols > 0);
				return sNumel(m_rows, m_cols - 1);
			} else return numel();
		}

		//////////////////////////////////////////////////////////////////////////
		// triangular matrix support
		//returns the number of elements in a triangular matrix of size N. (Elements of the main diagonal are excluded)
		static constexpr numel_cnt_t sNumelTriangl(const vec_len_t n)noexcept {
			//return (static_cast<numel_cnt_t>(n)*(n-1)) / 2;
			return __emulu(n, n - 1) / 2;
		}
		numel_cnt_t numel_triangl()const noexcept {
			NNTL_ASSERT(rows() == cols());
			return sNumelTriangl(rows());
		}
		//returns (ri,ci) coordinates of the element with index k of upper/lower (toggled by template parameter bool bLowerTriangl) triangular matrix
		template<bool bLowerTriangl>
		static ::std::enable_if_t<!bLowerTriangl> sTrianglCoordsFromIdx(const vec_len_t n, const numel_cnt_t k, vec_len_t& ri, vec_len_t& ci) noexcept {
			NNTL_UNREF(n);
			NNTL_ASSERT(k <= sNumelTriangl(n));
			const auto _ci = static_cast<numel_cnt_t>(::std::ceil((::std::sqrt(static_cast<double>(8 * k + 9)) - 1) / 2));
			ci = static_cast<vec_len_t>(_ci);
			NNTL_ASSERT(ci <= n);
			ri = static_cast<vec_len_t>(k - _ci*(_ci - 1) / 2);
			NNTL_ASSERT(static_cast<int>(ri) >= 0 && ri < n);
		}
		template<bool bLowerTriangl>
		static ::std::enable_if_t<bLowerTriangl> sTrianglCoordsFromIdx(const vec_len_t n, const numel_cnt_t k, vec_len_t& ri, vec_len_t& ci) noexcept {
			const auto trNumel = sNumelTriangl(n);
			NNTL_ASSERT(k <= trNumel);
			const auto nm1 = n - 1;
			if (k==trNumel) {
				ri = 0;
				ci = nm1;
			} else {
				const auto _k = trNumel - k - 1;
				const auto _ci = static_cast<numel_cnt_t>(::std::ceil((::std::sqrt(static_cast<double>(8 * _k + 9)) - 1) / 2));
				NNTL_ASSERT(_ci < n);				
				ci = nm1 - static_cast<vec_len_t>(_ci);
				ri = nm1 - static_cast<vec_len_t>(_k - _ci*(_ci - 1) / 2);
				NNTL_ASSERT(static_cast<int>(ri) > 0 && ri < n);
			}
			
		}
		template<bool bLowerTriangl>
		void triangl_coords_from_idx(const numel_cnt_t k, vec_len_t& ri, vec_len_t& ci)const noexcept {
			NNTL_ASSERT(rows() == cols());
			sTrianglCoordsFromIdx<bLowerTriangl>(rows(), k, ri, ci);
		}

		//////////////////////////////////////////////////////////////////////////

		inline numel_cnt_t byte_size()const noexcept {
			return numel()*sizeof(value_type);
		}
		inline numel_cnt_t byte_size_no_bias()const noexcept {
			return numel_no_bias()*sizeof(value_type);
		}

		inline bool empty()const noexcept { return nullptr == m_pData; }

		//to conform ::std::vector API
		inline value_ptr_t data()noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0);
			return m_pData;
		}
		inline cvalue_ptr_t data()const noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0);
			return m_pData;
		}		

		/*value_type& operator[](numel_cnt_t elmIdx)noexcept {
			NNTL_ASSERT(elmIdx < numel());
			return data()[elmIdx];
		}
		value_type operator[](numel_cnt_t elmIdx)const noexcept {
			NNTL_ASSERT(elmIdx < numel());
			return data()[elmIdx];
		}*/

		//not a real iterators, just pointers
		inline value_ptr_t begin()noexcept { return data(); }
		inline cvalue_ptr_t begin()const noexcept { return data(); }

		inline value_ptr_t end()noexcept { return data()+numel(); }
		inline cvalue_ptr_t end()const noexcept { return data()+numel(); }
		inline value_ptr_t end(const bool bNoBias)noexcept { return data() + numel(bNoBias); }
		inline cvalue_ptr_t end(const bool bNoBias)const noexcept { return data() + numel(bNoBias); }

		inline value_ptr_t end_no_bias()noexcept { return data() + numel_no_bias(); }
		inline cvalue_ptr_t end_no_bias()const noexcept { return data() + numel_no_bias(); }

		inline value_ptr_t colDataAsVec(vec_len_t c)noexcept {
			NNTL_ASSERT(!empty() && m_cols>0 && m_rows>0 && c <= m_cols);//non strict inequality c <= m_cols to allow reference 'after the last' column
			return m_pData + sNumel(m_rows, c);
		}
		inline cvalue_ptr_t colDataAsVec(vec_len_t c)const noexcept {
			NNTL_ASSERT(!empty() && m_cols > 0 && m_rows > 0 && c <= m_cols);//non strict inequality c <= m_cols to allow reference 'after the last' column
			return m_pData + sNumel(m_rows, c);
		}

		inline value_ptr_t bias_column()noexcept {
			NNTL_ASSERT(emulatesBiases());
			return colDataAsVec(m_cols - 1);
		}
		inline cvalue_ptr_t bias_column()const noexcept {
			NNTL_ASSERT(emulatesBiases());
			return colDataAsVec(m_cols - 1);
		}

		//////////////////////////////////////////////////////////////////////////
		// get/set are for non performance critical code!
		inline void set(const vec_len_t r, const vec_len_t c, const value_type& v)noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			m_pData[ r + sNumel(ldim(), c)] = v;
		}
		inline const value_type& get(const vec_len_t r, const vec_len_t c)const noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			return m_pData[r + sNumel(ldim(), c)];
		}
		inline value_type& get(const vec_len_t r, const vec_len_t c) noexcept {
			NNTL_ASSERT(!empty());
			NNTL_ASSERT(r < m_rows && c < m_cols);
			return m_pData[r + sNumel(ldim(), c)];
		}
		inline const value_type& get(const mtx_coords_t crd)const noexcept { return get(crd.first, crd.second); }
		inline value_type& get(const mtx_coords_t crd) noexcept { return get(crd.first, crd.second); }

		void clear()noexcept {
			_free();
			//m_bEmulateBiases = false; //MUST NOT clear this flag; it defines a matrix mode and clear() just performs a cleanup. No mode change allowed here.
			m_bDontManageStorage = false;
			m_bHoleyBiases = false;//this is a kind of run-time state of the bias column. We've cleared the matrix, therefore clearing this state too.
		}

		template<typename T>
		bool resize(const smatrix<T>& m)noexcept {
			NNTL_ASSERT(!m.empty() && m.rows() > 0 && m.cols() > 0);
			NNTL_ASSERT(emulatesBiases() == m.emulatesBiases());
			return resize(m.rows(), m.cols());
		}
		bool resize(const mtx_size_t sz)noexcept { return resize(sz.first, sz.second); }
		bool resize(const vec_len_t r, vec_len_t c) noexcept {
			NNTL_ASSERT(!m_bDontManageStorage);
			NNTL_ASSERT(r > 0 && c > 0);
			if (r <= 0 || c <= 0) {
				NNTL_ASSERT(!"Wrong row or col count!");
				return false;
			}

			if (m_bEmulateBiases) ++c;

			if (r == m_rows && c == m_cols) return true;

			m_rows = r;
			m_cols = c;

			_realloc();

			const bool bRet = nullptr != m_pData;
			if (m_bEmulateBiases && bRet) set_biases();
			m_bHoleyBiases = false;
			return bRet;
		}

		void fill(const value_type v)noexcept {
			NNTL_ASSERT(!empty());
			//::std::fill(m_pData, m_pData + numel(), value_type(1.0)); //doesn't get vectorized!
			_fill_elements(m_pData, numel(), v);
		}
		void zeros()noexcept {
			NNTL_ASSERT(!empty());
			::std::memset(m_pData, 0, byte_size_no_bias());
		}
		void ones()noexcept {
			fill(value_type(1));
			m_bHoleyBiases = false;
		}
		void nans()noexcept {
			fill(::std::numeric_limits<value_type>::quiet_NaN());
			m_bHoleyBiases = false;
		}
		void nans_no_bias()noexcept {
			NNTL_ASSERT(!empty());
			//::std::fill(m_pData, m_pData + numel(), value_type(1.0)); //doesn't get vectorized!
			_fill_elements(m_pData, numel_no_bias(), ::std::numeric_limits<value_type>::quiet_NaN());
			m_bHoleyBiases = false;
		}

		// fills matrix with data from pSrc doing type conversion. Bias units left untouched.
		template<typename OtherBaseT>
		::std::enable_if_t<!::std::is_same<value_type, OtherBaseT>::value> fill_from_array_no_bias(const OtherBaseT*const pSrc)noexcept {
			static_assert(::std::is_arithmetic<OtherBaseT>::value, "OtherBaseT must be a simple arithmetic data type");
			NNTL_ASSERT(!empty() && numel() > 0);
			const auto ne = numel_no_bias();
			const auto p = data();
			for (numel_cnt_t i = 0; i < ne; ++i) p[i] = static_cast<value_type>(pSrc[i]);
		}
		template<typename OtherBaseT>
		::std::enable_if_t<::std::is_same<value_type, OtherBaseT>::value> fill_from_array_no_bias(const OtherBaseT*const pSrc)noexcept {
			NNTL_ASSERT(!empty() && numel() > 0);
			memcpy(m_pData, pSrc, byte_size_no_bias());
		}


		//full matrix
		void useExternalStorage_no_bias(smatrix& src)noexcept {
			useExternalStorage(src.data(), src.rows(), src.cols_no_bias(), false, false);
		}
		void useExternalStorage(smatrix& src)noexcept {
			const auto b = src.emulatesBiases();
			useExternalStorage(src.data(), src.rows(), src.cols(), b, b && src.isHoleyBiases());
		}
		//matrix part, column set [cBeg, cBeg+totCols)
		void useExternalStorage_no_bias(smatrix& src, vec_len_t cBeg, vec_len_t totCols)noexcept {
			NNTL_ASSERT(src.end_no_bias() >= src.colDataAsVec(cBeg + totCols));
			useExternalStorage(src.colDataAsVec(cBeg), src.rows(), totCols, false, false);
		}

		//raw pointers
		void useExternalStorage_no_bias(value_ptr_t ptr, const smatrix& sizeLikeThisNoBias)noexcept {
			useExternalStorage(ptr, sizeLikeThisNoBias.rows(), sizeLikeThisNoBias.cols_no_bias(), false, false);
		}
		void useExternalStorage(value_ptr_t ptr, const mtx_size_t sizeLikeThis, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
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

	protected:
		//special version for smatrix_deform use
		void _useExternalStorage(value_ptr_t ptr, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
			NNTL_ASSERT(ptr);
			_free();
			m_bDontManageStorage = true;
			m_pData = ptr;
			m_rows = m_cols = 0;
			m_bEmulateBiases = bEmulateBiases;
			m_bHoleyBiases = bHBiases;

			//usually, external storage is used with shared memory to store temporary computations, therefore data won't survive between
			// usages and there is no need to prefill it with biases. m_bEmulateBiases flag is helpful for routines like numel_no_bias()
			//if (bEmulateBiases) _fill_biases();
			//return true;
		}

	public:
		bool useInternalStorage(vec_len_t _rows=0, vec_len_t _cols=0, bool bEmulateBiases = false)noexcept {
			_free();
			m_bDontManageStorage = false;
			m_bHoleyBiases = false;
			m_bEmulateBiases = bEmulateBiases;
			if (_rows > 0 && _cols > 0) {
				return resize(_rows, _cols);
			} else return true;
		}

		template<typename VT2, bool b = ::std::is_same<value_type, VT2>::value>
		::std::enable_if_t<b> assert_storage_does_not_intersect(const smatrix<VT2>& m)const noexcept {
			NNTL_UNREF(m);
			NNTL_ASSERT(this != &m);
			NNTL_ASSERT(!empty() && !m.empty());
			//nonstrict nonequality it necessary here, because &[numel()] references element past the end of the allocated array
			NNTL_ASSERT( &m.m_pData[m.numel()] <= m_pData || &m_pData[numel()] <= m.m_pData);
		}
		template<typename VT2, bool b = ::std::is_same<value_type, VT2>::value>
		constexpr ::std::enable_if_t<!b> assert_storage_does_not_intersect(const smatrix<VT2>& m)const noexcept {}
	};

	//////////////////////////////////////////////////////////////////////////
	
	namespace _impl {
		template<class T>
		using is_smatrix_derived = ::std::is_base_of<smatrix<typename T::value_type>, T>;
	}

	template<typename T>
	using is_smatrix = ::std::conditional_t<utils::has_value_type<T>::value, _impl::is_smatrix_derived<T>, ::std::false_type>;

	//////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////
	//this class will allow to reuse the same storage for matrix of smaller size
	//////////////////////////////////////////////////////////////////////////
	template <typename T_>
	class smatrix_deform : public smatrix<T_> {
	private:
		typedef smatrix<T_> _base_class;

#ifdef NNTL_DEBUG
	protected:
		numel_cnt_t m_maxSize = 0;

	public:
		bool _isOkToDeform(vec_len_t r, vec_len_t c_noBias, bool bEmulBias)const noexcept {
			return m_maxSize >= sNumel(r, c_noBias + vec_len_t(bEmulBias));
		}
#endif // NNTL_DEBUG
	public:

		~smatrix_deform()noexcept {}
		smatrix_deform()noexcept : _base_class() {}
		smatrix_deform(const vec_len_t _rows, const vec_len_t _cols, const bool _bEmulBias = false)noexcept
			: _base_class(_rows, _cols, _bEmulBias)
		{
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		//note that ne MUST include bias column if bEmulateBiases was set to true! (same convention as useExternalStorage())
		smatrix_deform(const numel_cnt_t ne, const bool bEmulateBiases = false, const bool _bHoleyBiases = false) noexcept
			: _base_class()
		{
			resize(ne, &bEmulateBiases, &_bHoleyBiases);
		}

		smatrix_deform(value_ptr_t ptr, const vec_len_t r, const vec_len_t c, const bool bEmulateBiases = false, const bool _bHoleyBiases = false) noexcept
			: _base_class(ptr, r, c, bEmulateBiases, _bHoleyBiases)
		{
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}

		smatrix_deform(value_ptr_t ptr, const smatrix_deform& sizeLikeThis) noexcept
			: _base_class(ptr, sizeLikeThis)
		{
#ifdef NNTL_DEBUG
			m_maxSize = sizeLikeThis.m_maxSize;
#endif // NNTL_DEBUG
		}

		//////////////////////////////////////////////////////////////////////////

		smatrix_deform(smatrix&& src)noexcept : _base_class(::std::move(src)) {
#ifdef NNTL_DEBUG
			m_maxSize = numel();
#endif // NNTL_DEBUG
		}
		smatrix_deform(smatrix_deform&& src)noexcept : _base_class(::std::move(src)) {
#ifdef NNTL_DEBUG
			m_maxSize = src.m_maxSize;
#endif // NNTL_DEBUG
		}

		smatrix_deform(const smatrix_deform& other) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		smatrix_deform& operator=(const smatrix_deform& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

		smatrix_deform& operator=(smatrix&& rhs) noexcept {
			if (this != &rhs) {
				_base_class::operator =(::std::move(rhs));
#ifdef NNTL_DEBUG
				m_maxSize = numel();
#endif // NNTL_DEBUG
			}
			return *this;
		}
		smatrix_deform& operator=(smatrix_deform&& rhs) noexcept {
			if (this != &rhs) {
				_base_class::operator =(::std::move(rhs));
#ifdef NNTL_DEBUG
				m_maxSize = rhs.m_maxSize;
				rhs.m_maxSize = 0;
#endif // NNTL_DEBUG
			}
			return *this;
		}

		smatrix_deform submatrix_cols_no_bias(const vec_len_t colStart, const vec_len_t numCols)noexcept {
			NNTL_ASSERT(numCols);
			NNTL_ASSERT(colStart + numCols <= cols_no_bias());
			return smatrix_deform(colDataAsVec(colStart), rows(), numCols, false);
		}

		void clear()noexcept {
#ifdef NNTL_DEBUG
			m_maxSize = 0;
#endif // NNTL_DEBUG
			_base_class::clear();
		}

		bool cloneFrom(const smatrix& src)noexcept {
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

		bool resize(const vec_len_t _rows, vec_len_t _cols)noexcept {
			_free();
			const auto r = _base_class::resize(_rows, _cols);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		bool resize(const smatrix& m)noexcept {
			_free();
			const auto r = _base_class::resize(m);
#ifdef NNTL_DEBUG
			if (r) m_maxSize = numel();
#endif // NNTL_DEBUG
			return r;
		}

		bool resize(const mtx_size_t s)noexcept {
			return resize(s.first, s.second);
		}

		//note that ne MUST include bias column if *pbEmulateBiases was set to true! (same convention as useExternalStorage())
		//NOTE you MUST call deform(r,c) before using resized with routine matrix!
		bool resize(const numel_cnt_t ne, const bool*const pbEmulateBiases = nullptr, const bool*const pbHBiases = nullptr)noexcept {
			NNTL_ASSERT(ne > 0);
			_free();
			//auto ptr = new(::std::nothrow) value_type[ne];
			//#todo for C++17 must change to ::std::launder(reinterpret_cast< ... 
			value_type* ptr = reinterpret_cast<value_type*>(_aligned_malloc(sizeof(value_type)*ne, utils::mem_align_for<value_type>()));
			if (nullptr == ptr) {
				NNTL_ASSERT(!"Memory allocation failed!");
				return false;
			}
			//default flag values according to defaults of useExternalStorage()
			useExternalStorage(ptr, ne, pbEmulateBiases ? *pbEmulateBiases : false, pbHBiases ? *pbHBiases : false);
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

		//warning, if you change default flag values here in func definition, change in resize(const numel_cnt_t ne) also!
		void useExternalStorage(value_ptr_t ptr, numel_cnt_t cnt, bool bEmulateBiases = false, bool bHBiases = false)noexcept {
			_base_class::_useExternalStorage(ptr, bEmulateBiases, bHBiases);
#ifdef NNTL_DEBUG
			m_maxSize = cnt;
#else
			NNTL_UNREF(cnt);
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
		vec_len_t deform_cols(const vec_len_t c)noexcept {
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

		bool hide_last_n_cols(const vec_len_t n)noexcept {
			NNTL_ASSERT(!empty() && m_cols > n + emulatesBiases());
			bool hasBiases = emulatesBiases();
			m_cols -= n + hasBiases;
			m_bEmulateBiases = false;
			return hasBiases;
		}
		void restore_last_n_cols(const bool bThereWasBiases, const vec_len_t n)noexcept {
			NNTL_ASSERT(!empty() && m_cols >= 1);
			NNTL_ASSERT(m_maxSize >= sNumel(m_rows, m_cols + n + bThereWasBiases));
			m_cols += n + bThereWasBiases;
			m_bEmulateBiases = bThereWasBiases;
		}


		bool hide_biases()noexcept {
			bool hasBiases = emulatesBiases();
			if (hasBiases) {
				hide_last_col();
				m_bEmulateBiases = false;
			}
			return hasBiases;
		}
		void restore_biases(const bool bHideBiasesReturned)noexcept { //should only be called iff hide_biases() returned true
			NNTL_ASSERT(!emulatesBiases());
			if (bHideBiasesReturned) {
				restore_last_col();
				m_bEmulateBiases = true;
			}
		}


		//no asserts, just flag value. Better use for testing purposes only
		bool holeyBiases_flag()const noexcept { return m_bHoleyBiases; }
		//////////////////////////////////////////////////////////////////////////
		//note that the following functions change flags ONLY; they does NOT affect stored data
		void _drop_biases()noexcept { m_bEmulateBiases = false; }
		void _enforce_biases()noexcept { m_bEmulateBiases = true; }
		
		//////////////////////////////////////////////////////////////////////////
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
		
		template<typename _T>
		void deform_like(const smatrix<_T>& m)noexcept { deform(m.rows(), m.cols()); }
		template<typename _T>
		void deform_like_no_bias(const smatrix<_T>& m)noexcept { deform(m.rows(), m.cols_no_bias()); }
	};

	//////////////////////////////////////////////////////////////////////////
	// helper class to define matrix elements range
	//////////////////////////////////////////////////////////////////////////
	template<typename T>
	class st_range : public smatrix_td {
	public:
		typedef threads::parallel_range<numel_cnt_t> par_range_t;
		typedef T value_type;

	public:
		const value_type elmEnd;
		const value_type elmBegin;

	public:
		~st_range()noexcept {}

		template<typename _T>
		st_range(const smatrix<_T>& A)noexcept 
			: elmEnd( static_cast<value_type>(A.numel()) ), elmBegin(0)
		{
			NNTL_ASSERT(A.numel() < ::std::numeric_limits<value_type>::max());
		}

		template<typename _T>
		st_range(const smatrix<_T>& A, const tag_noBias&)noexcept
			: elmEnd( static_cast<value_type>(A.numel_no_bias())), elmBegin(0)
		{
			NNTL_ASSERT(A.numel_no_bias() < ::std::numeric_limits<value_type>::max());
		}

		st_range(const par_range_t& pr)noexcept
			: elmEnd(static_cast<value_type>(pr.offset() + pr.cnt())), elmBegin(static_cast<value_type>(pr.offset()))
		{
			NNTL_ASSERT(pr.offset() < ::std::numeric_limits<value_type>::max());
			NNTL_ASSERT(pr.cnt() < ::std::numeric_limits<value_type>::max());
			NNTL_ASSERT((pr.offset() + pr.cnt()) < ::std::numeric_limits<value_type>::max());
		}

		st_range(const value_type eb, const value_type ee)noexcept : elmEnd(ee), elmBegin(eb) {
			NNTL_ASSERT(elmEnd >= elmBegin);
		}

		value_type totalElements()const noexcept { return elmEnd - elmBegin; }
	};

	class s_elems_range : public st_range<numel_cnt_t> {
		typedef st_range<numel_cnt_t> base_class_t;
	public:
		template<typename ...ArgsT>
		s_elems_range(ArgsT&&... a) : base_class_t(::std::forward<ArgsT>(a)...){}
	};

	class s_vec_range : public st_range<vec_len_t> {
		typedef st_range<vec_len_t> base_class_t;
	public:
		template<typename ...ArgsT>
		s_vec_range(ArgsT&&... a) : base_class_t(::std::forward<ArgsT>(a)...) {}
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
		s_rowcol_range(const vec_len_t rb, const vec_len_t re, const smatrix<T_>& A)noexcept 
			: rowEnd(re), rowBegin(rb), colEnd(A.cols()), colBegin(0)
		{
			NNTL_ASSERT(rowEnd >= rowBegin);
		}

		template<typename T_>
		s_rowcol_range(const smatrix<T_>& A, const vec_len_t cb, const vec_len_t ce)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(ce), colBegin(cb) {
			NNTL_ASSERT(colEnd >= colBegin);
		}

		template<typename T_>
		s_rowcol_range(const smatrix<T_>& A)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(A.cols()), colBegin(0) {}

		template<typename T_>
		s_rowcol_range(const smatrix<T_>& A, const tag_noBias&)noexcept : rowEnd(A.rows()), rowBegin(0), colEnd(A.cols_no_bias()), colBegin(0) {}

		vec_len_t totalRows()const noexcept { return rowEnd - rowBegin; }
		vec_len_t totalCols()const noexcept { return colEnd - colBegin; }

		template<typename T_>
		bool can_apply(const smatrix<T_>& A)const noexcept {
			const auto r = A.rows(), c = A.cols();
			NNTL_ASSERT(rowBegin <= rowEnd && rowBegin < r && rowEnd <= r);
			NNTL_ASSERT(colBegin <= colEnd && colBegin < c && colEnd <= c);
			return rowBegin < r && rowEnd <= r
				&& colBegin < c && colEnd <= c;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// helper functions to easily use matrix or vector
	template<typename T>
	numel_cnt_t Numel(const smatrix<T>& c)noexcept { return c.numel(); }

	template<typename T>
	numel_cnt_t Numel(const ::std::vector<T>& c)noexcept { return conform_sign(c.size()); }

	template<typename T, ::std::size_t N>
	constexpr numel_cnt_t Numel(const ::std::array<T,N>& c)noexcept { return conform_sign(c.size()); }

	//compares whether comparands have size() or at least same numel
	template<typename T1, typename T2>
	bool IsSameSizeNumel(const ::std::vector<T1>& c1, const ::std::vector<T2>& c2)noexcept { return c1.size() == c2.size(); }
	template<typename T1, typename T2>
	bool IsSameSizeNumel(const smatrix<T1>& c1, const ::std::vector<T2>& c2)noexcept { return c1.numel() == Numel(c2); }
	template<typename T1, typename T2>
	bool IsSameSizeNumel(const ::std::vector<T1>& c1, const smatrix<T2>& c2)noexcept { return c2.numel() == Numel(c1); }
	template<typename T1, typename T2>
	bool IsSameSizeNumel(const smatrix<T1>& c1, const smatrix<T2>& c2)noexcept { return c1.size() == c2.size(); }
}
}
