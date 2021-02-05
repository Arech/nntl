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

#include <limits>

#include "_defs.h"

namespace nntl {
	
	//typename for referring layer numbers, indexes, counts and so on. *640k* 256 layers should enough for everyone :-D
	//Damn, I've never thought, it'll not be enought so soon!)
	// must be unsigned
	typedef ::std::uint16_t layer_index_t;

	static constexpr layer_index_t invalid_layer_index = ::std::numeric_limits<layer_index_t>::max();

	// must be unsigned -- What the hell? Why "must"?
	// It's unlikely we ever get stuck into the signed int32 limit, but signed integers allows to perform loop vectorization better
	// If we'd ever stuck int INT_MAX, just set to int64
	typedef ::std::int32_t neurons_count_t;
	typedef neurons_count_t vec_len_t;

	typedef ::std::conditional_t<::std::is_signed<neurons_count_t>::value, ::std::make_signed_t<size_t>, size_t> numel_cnt_t;

	//by convention layer_type_id_t can't be zero
	typedef ::std::uint64_t layer_type_id_t;

	//real_t with extended precision for some temporarily calculations
	typedef double ext_real_t;

	//thread id must be in range [0,workers_count())
	//worker threads should have par_range_t::tid>=1. tid==0 is reserved to main thread.
	// If scheduler will launch less than workers_count() threads to process task, 
	// then maximum tid must be equal to <scheduled workers count>+1 (+1 refers to a main thread, that's also
	// used in scheduling)
	// Making signed to ease working with neurons_count_t/vec_len_t type that is signed.
	typedef neurons_count_t thread_id_t;

	//see also NNTL_STRING macro
	// by now some code such as file-related functions in _supp{} may be bounded to char only in strchar_t
	//TODO: Don't think it is good idea to solve it now.
	// OBSOLETTE, use char instead
	typedef char strchar_t;

	namespace utils {
		//////////////////////////////////////////////////////////////////////////
		//https://bitbucket.org/martinhofernandes/wheels/src/default/include/wheels/meta/type_traits.h%2B%2B?fileviewer=file-view-default#cl-161
		//! Tests if T is a specialization of Template
		template <typename T, template <typename...> class Template>
		struct is_specialization_of : ::std::false_type {};
		template <template <typename...> class Template, typename... Args>
		struct is_specialization_of<Template<Args...>, Template> : ::std::true_type {};

		// 		template <typename T>
		// 		using is_tuple2 = is_specialization_of<T, ::std::tuple>;
		
		template< class, class = ::std::void_t<> >
		struct has_value_type : ::std::false_type { };
		// specialization recognizes types that do have a nested ::options_t member:
		template< class T >
		struct has_value_type<T, ::std::void_t<typename T::value_type>> : ::std::true_type {};


		template< class, class = ::std::void_t<> >
		struct has_x_t : ::std::false_type { };
		template< class T >
		struct has_x_t<T, ::std::void_t<typename T::x_t>> : ::std::true_type {};

		template< class, class = ::std::void_t<> >
		struct has_y_t : ::std::false_type { };
		template< class T >
		struct has_y_t<T, ::std::void_t<typename T::y_t>> : ::std::true_type {};

		template<typename T>
		using has_x_t_and_y_t = ::std::conjunction<has_x_t<T>, has_y_t<T>>;
	}

	//////////////////////////////////////////////////////////////////////////

	template<typename T>
	::std::conditional_t<::std::is_unsigned<neurons_count_t>::value, ::std::make_unsigned_t<T>, ::std::make_signed_t<T>>
		conform_sign(T v)
	#ifndef NNTL_DEBUG
		noexcept
	#endif // NNTL_DEBUG
	{
		static_assert(::std::is_unsigned<neurons_count_t>::value || ::std::is_unsigned<T>::value, "WTF? It's for unsigned types only!");
		typedef ::std::conditional_t<::std::is_unsigned<neurons_count_t>::value, ::std::make_unsigned_t<T>, ::std::make_signed_t<T>> conf_T;

	#ifdef NNTL_DEBUG
		if (v > static_cast<T>(::std::numeric_limits<conf_T>::max())) {
			NNTL_ASSERT(!"Failed to convert to signed");
			throw ::std::overflow_error("Failed to convert to signed");
		}
	#endif
		return static_cast<conf_T>(v);
	}

	struct BatchSizes {
		vec_len_t maxBS{ 0 }; //max batch size for any mode
		vec_len_t maxTrainBS{ 0 }; //max batch size for training only 

		BatchSizes()noexcept{}

		BatchSizes(const vec_len_t mbs, const vec_len_t tbs)noexcept : maxBS(mbs), maxTrainBS(tbs) {
			NNTL_ASSERT(mbs > 0 && (0 == tbs || (tbs > 0 && mbs >= tbs)));
		}

		//no assertions here
		BatchSizes(const BatchSizes& o)noexcept : maxBS(o.maxBS), maxTrainBS(o.maxTrainBS) {}
		BatchSizes(BatchSizes&& o)noexcept : maxBS(o.maxBS), maxTrainBS(o.maxTrainBS) {}
		
	public:
		void set(const vec_len_t mbs, const vec_len_t tbs)noexcept {
			NNTL_ASSERT(mbs > 0 && (0 == tbs || (tbs > 0 && mbs >= tbs)));
			maxBS = mbs;
			maxTrainBS = tbs;
		}
		void set_from(const BatchSizes& o)noexcept {
			NNTL_ASSERT(o.isValid());
			_set_from(o);
		}
	protected:
		void _set_from(const BatchSizes& o)noexcept {
			maxBS = o.maxBS;
			maxTrainBS = o.maxTrainBS;
		}

	public:
		BatchSizes& operator=(const BatchSizes& o)noexcept {
			if (this != &o) {
				set_from(o);
			}
			return *this;
		}
		//move usually perform compiler when operating over higher-level structures that may contain uninitalized/empty/non valid BatchSizes
		//so skipping assertions here
		BatchSizes& operator=(BatchSizes&& o)noexcept {
			_set_from(o);
			return *this;
		}

		bool isValid()const noexcept {
			NNTL_ASSERT((maxBS == 0 && maxTrainBS == 0) || (maxBS > 0 && (0 == maxTrainBS || (maxTrainBS > 0 && maxBS >= maxTrainBS))));
			return maxBS > 0;
		}

		bool isValidForTraining()const noexcept {
			NNTL_ASSERT(maxBS >= maxTrainBS);
			return maxBS > 0 && maxTrainBS > 0;
		}

		vec_len_t biggest()const noexcept {
			NNTL_ASSERT(isValid());
			//return ::std::max(maxBatchSize, maxTrainingBatchSize);
			return maxBS;
		}

		vec_len_t max_bs4mode(const bool bTrainingMode)const noexcept {
			NNTL_ASSERT(isValid());
			NNTL_ASSERT(!bTrainingMode || maxTrainBS > 0);
			return bTrainingMode ? maxTrainBS : biggest();
		}

		void clear()noexcept {
			maxBS = maxTrainBS = 0;
		}

		bool operator==(const BatchSizes& o)const noexcept {
			return maxBS == o.maxBS && maxTrainBS == o.maxTrainBS;
		}
		bool operator!=(const BatchSizes& o)const noexcept {
			return maxBS != o.maxBS || maxTrainBS != o.maxTrainBS;
		}

		BatchSizes operator*(const vec_len_t mult)const noexcept {
			NNTL_ASSERT(isValid());
			NNTL_ASSERT(mult > 0);
			return BatchSizes(maxBS*mult, maxTrainBS*mult);
		}

		BatchSizes operator/(const vec_len_t divis)const noexcept {
			NNTL_ASSERT(isValid());
			NNTL_ASSERT(divis > 0 && 0 == maxBS%divis);
			return BatchSizes(maxBS / divis, maxTrainBS / divis);
		}
	};
}