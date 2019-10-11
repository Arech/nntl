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

#include <stack>

namespace nntl {
namespace utils {

	//////////////////////////////////////////////////////////////////////////
	//helper to store current layer index. Maximum depth is hardcoded into _maxDepth. It is checked only during DEBUG builds,
	// though it is ok if the actual depth is bigger, because container would just reallocate its memory to suit new needs
	template<typename IdxT, IdxT defaultVal, size_t _maxDepth>
	class layer_idx_keeper : private ::std::stack<IdxT, ::std::vector<IdxT>> {
	private:
		typedef ::std::stack<IdxT, ::std::vector<IdxT>> _base_class_t;
	public:
		typedef IdxT value_t;
		//typedef const value_t& cref_value_t;
		static constexpr size_t maxDepth = _maxDepth;
		static constexpr value_t default_value = defaultVal;
		//static const value_t default_value = defaultVal;

	public:
		~layer_idx_keeper()noexcept {
			NNTL_ASSERT(0 == size());
		}
		layer_idx_keeper()noexcept {
			c.reserve(maxDepth);
		}

		void push(const value_t& v)noexcept {
			NNTL_ASSERT(size() < maxDepth);
			_base_class_t::push(v);//#exceptions STL
		}
		void push(value_t&& v)noexcept {
			NNTL_ASSERT(size() < maxDepth);
			_base_class_t::push(::std::move(v));//#exceptions STL
		}

		void pop()noexcept {
			NNTL_ASSERT(size());
			_base_class_t::pop();
		}
		value_t top()const noexcept {
			return _base_class_t::size() ? _base_class_t::top() : default_value;
		}

		/*const value_t& top_unsafe()const noexcept {
			return _base_class_t::top();
		}
		value_t top_n(const size_type& n)const noexcept {
			return _base_class_t::size()>=n ? *::std::prev(_base_class_t::_Get_container().end(), n) : default_value;
		}
		const value_t& top_n_unsafe(const size_type& n)const noexcept {
			return *::std::prev(_base_class_t::_Get_container().end(), n);
		}*/

		operator value_t()const noexcept {
			return top();
		}

		//checks whether the previous entry is the same as current
		bool bUpperLayerDifferent()const noexcept {
			return _base_class_t::size() < 2 || _base_class_t::top() != *::std::prev(_base_class_t::_Get_container().end(), 2);
		}

		size_type nestingLevel()const noexcept {
			size_type nl = 0;
			const auto s = _base_class_t::size();
			const auto t = _base_class_t::top();
			for (size_type i = 2; i <= s; ++i) {
				if (t != ::std::prev(_base_class_t::_Get_container().end(), i)) {
					break;
				}
				++nl;
			}
			return nl;
		}

		size_type size()const noexcept {
			return _base_class_t::size();
		}
	};
}
}
