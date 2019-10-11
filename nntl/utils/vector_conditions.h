/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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

#include <vector>

namespace nntl {

	//this is to define some flags for each epoch (currently it is whether to launch nn evaluation after an epoch or not)
	class vector_conditions {
	public:
		typedef vector_conditions self_t;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		//TODO: vector may throw exceptions...
		::std::vector<bool> m_flgEvalPerf;

	public:
		~vector_conditions()noexcept {}
		vector_conditions()noexcept {}
		vector_conditions(size_t maxEpoch, const bool& defVal=true)noexcept:m_flgEvalPerf(maxEpoch, defVal) {
			NNTL_ASSERT(maxEpoch > 0);
			verbose(maxEpoch - 1);
		}
		vector_conditions(size_t maxEpoch,size_t stride)noexcept : m_flgEvalPerf(maxEpoch, false) {
			NNTL_ASSERT(maxEpoch > 0 && stride>0);
			verbose(stride, maxEpoch, stride);
			verbose(maxEpoch - 1);
		}
		vector_conditions(size_t maxEpoch, size_t startsAt, size_t stride)noexcept : m_flgEvalPerf(maxEpoch, false) {
			NNTL_ASSERT(maxEpoch > 0 && startsAt <= maxEpoch && stride > 0);
			verbose(startsAt, maxEpoch, stride);
			verbose(maxEpoch - 1);
		}
		vector_conditions(vector_conditions&& src)noexcept : m_flgEvalPerf(::std::move(src.m_flgEvalPerf)) {}

		//!! copy constructor not needed
		vector_conditions(const vector_conditions& other)noexcept = delete;
		//!!assignment is not needed
		vector_conditions& operator=(const vector_conditions& rhs) noexcept = delete;

		self_t& resize(size_t maxEpoch, const bool& defVal = false)noexcept {
			m_flgEvalPerf.resize(maxEpoch, defVal);
			return *this;
		}
		self_t& clear()noexcept {
			m_flgEvalPerf.clear();
			return *this;
		}

		size_t size()const noexcept { return m_flgEvalPerf.size(); }
		size_t maxEpoch()const noexcept { return size(); }

		//using () instead of [] because can't (and don't need to) return reference
		const bool operator()(size_t e)const noexcept { return m_flgEvalPerf[e]; }

		self_t& set(size_t i, const bool v)noexcept { m_flgEvalPerf[i] = v; return *this; }
		self_t& set(const size_t _beg, const size_t _end, const size_t stride, const bool v)noexcept {
			NNTL_ASSERT(stride > 0 && _beg > 0 && _end >= _beg);
			for (size_t i = _beg - 1; i < _end; i += stride) m_flgEvalPerf[i] = v;
			return *this;
		}

		self_t& verbose(size_t i)noexcept { return set(i, true); }
		self_t& verbose(size_t _beg, size_t _end, size_t stride = 1)noexcept { return set(_beg, _end, stride, true); }

		self_t& silence(size_t i)noexcept { return set(i, false); }
		self_t& silence(size_t _beg, size_t _end, size_t stride = 1)noexcept { return set(_beg, _end, stride, false); }
	};

}