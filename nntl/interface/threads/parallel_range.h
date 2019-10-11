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

namespace nntl {
namespace threads {

	template<typename range_type>
	class parallel_range {
	public:
		typedef range_type range_t;
		static_assert(::std::is_pod<range_t>::value, "Expecting template parameter T to be plain data type");

		//thread id must be in range [0,workers_count())
		//worker threads should have par_range_t::tid>=1. tid==0 is reserved to main thread.
		// If scheduler will launch less than workers_count() threads to process task, 
		// then maximum tid must be equal to <scheduled workers count>+1 (+1 refers to a main thread, that's also
		// used in scheduling)
		//typedef unsigned int thread_id_t;

	protected:
		range_t m_cnt;
		range_t m_offset;
		const thread_id_t m_tid;

	public:
		~parallel_range()noexcept {}
		parallel_range()noexcept:m_cnt(0), m_offset(0), m_tid(0) {}
		parallel_range(range_t c)noexcept : m_cnt(c), m_offset(0), m_tid(0) {}
		parallel_range(range_t ofs, range_t c, thread_id_t t)noexcept : m_cnt(c), m_offset(ofs), m_tid(t) {}

		parallel_range& offset(range_t o)noexcept { m_offset = o; return *this; }
		parallel_range& cnt(range_t c)noexcept { m_cnt = c; return *this; }

		range_t offset()const noexcept { return m_offset; }
		range_t cnt()const noexcept { return m_cnt; }
		thread_id_t tid()const noexcept { return m_tid; }
	};

}
}