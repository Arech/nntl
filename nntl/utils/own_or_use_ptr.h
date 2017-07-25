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

	template<typename _T, bool _bOwning>
	class _own_or_use_ptr {
	public:
		typedef typename ::std::remove_pointer<_T>::type value_t;
		typedef value_t* value_ptr_t;
		typedef value_t& value_ref_t;

		static_assert(_bOwning != ::std::is_pointer<_T>::value, "WTF?");

	public:
		static constexpr bool bOwning = _bOwning;

	protected:
		value_ptr_t m_ptr;

	public:
		~_own_or_use_ptr()noexcept { release(); }
		_own_or_use_ptr()noexcept : m_ptr(nullptr) {}

		//!! copy constructor not needed
		_own_or_use_ptr(const _own_or_use_ptr& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		_own_or_use_ptr& operator=(const _own_or_use_ptr& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

// 		_own_or_use_ptr(_own_or_use_ptr&& other)noexcept {
// 			m_ptr = other.m_ptr;
// 			other.m_ptr = nullptr;
// 		}
// 		_own_or_use_ptr& operator=(_own_or_use_ptr&& other) noexcept {
// 			m_ptr = other.m_ptr;
// 			other.m_ptr = nullptr;
// 		}

		const bool empty()const noexcept { return m_ptr == nullptr; }

		operator value_ptr_t()const noexcept { NNTL_ASSERT(!empty()); return m_ptr; }
		value_ptr_t operator->()const noexcept { NNTL_ASSERT(!empty()); return m_ptr; }
		value_ptr_t ptr()const noexcept { NNTL_ASSERT(!empty()); return m_ptr; }

		operator value_ref_t()const noexcept { NNTL_ASSERT(!empty()); return *m_ptr; }
		value_ref_t get()const noexcept { NNTL_ASSERT(!empty()); return *m_ptr; }


		void release()noexcept {
			if (m_ptr) {
				if (bOwning) delete m_ptr;
				m_ptr = nullptr;
			}
		}
	};

	template<typename _T>
	class own_or_use_ptr : public _own_or_use_ptr<_T, true> {
	private:
		typedef _own_or_use_ptr<_T, true> _base_class;
	public:
		typedef own_or_use_ptr<_T> type;
		static_assert(!::std::is_pointer<_T>::value, "WTF?");

		~own_or_use_ptr()noexcept {}
		own_or_use_ptr()noexcept {
			m_ptr = new(::std::nothrow) value_t;
		}
// 		own_or_use_ptr(own_or_use_ptr&& other)noexcept : _base_class(::std::move(other)) {}
// 		_own_or_use_ptr& operator=(_own_or_use_ptr&& other) noexcept;
	};

	template<typename _T>
	class own_or_use_ptr<_T*> : public _own_or_use_ptr<_T*, false> {
	private:
		typedef _own_or_use_ptr<_T*, false> _base_class;
	public:
		static_assert(!::std::is_pointer<_T>::value, "WTF?");
		typedef own_or_use_ptr<_T*> type;

		~own_or_use_ptr()noexcept {}
		own_or_use_ptr()noexcept;//no {} by intent

		own_or_use_ptr(_T* ptr)noexcept {
			m_ptr = ptr;
		}
		//own_or_use_ptr(own_or_use_ptr&& other)noexcept : _base_class(::std::move(other)) {}
	};


	template <typename ptr_t, typename = typename ::std::enable_if< ::std::is_pointer<ptr_t>::value, bool>::type>
	inline constexpr own_or_use_ptr<ptr_t> make_own_or_use_ptr(ptr_t ptr)noexcept {
		return own_or_use_ptr<ptr_t>(ptr);
	}

	template <typename nonptr_t, typename = typename ::std::enable_if<!::std::is_pointer<nonptr_t>::value, bool>::type>
	inline constexpr own_or_use_ptr<nonptr_t> make_own_or_use_ptr()noexcept {
		return own_or_use_ptr<nonptr_t>();
	}


	template<typename _T>
	using own_or_use_ptr_t = typename own_or_use_ptr<_T>::type;

}
}