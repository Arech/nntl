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

#include <vector>

namespace nntl {

	//this is to define some flags for each epoch (currently it is whether to launch nn evaluation after an epoch or not)
	class nnet_cond_epoch_eval {
	public:
		typedef nnet_cond_epoch_eval self_t;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		//TODO: vector may throw exceptions...
		std::vector<bool> m_flgEvalPerf;

	public:
		~nnet_cond_epoch_eval()noexcept {}
		nnet_cond_epoch_eval(size_t maxEpoch)noexcept:m_flgEvalPerf(maxEpoch, true) {
			NNTL_ASSERT(maxEpoch > 0);
		}
		nnet_cond_epoch_eval(size_t maxEpoch,size_t stride)noexcept : m_flgEvalPerf(maxEpoch, false) {
			NNTL_ASSERT(maxEpoch > 0);
			for (size_t i = 0; i < maxEpoch; i+=stride)  m_flgEvalPerf[i] = true;
			m_flgEvalPerf[maxEpoch - 1] = true;
		}
		nnet_cond_epoch_eval(size_t maxEpoch, size_t startsAt, size_t stride)noexcept : m_flgEvalPerf(maxEpoch, false) {
			NNTL_ASSERT(maxEpoch > 0 && startsAt<=maxEpoch);
			for (size_t i = startsAt; i < maxEpoch; i += stride)  m_flgEvalPerf[i] = true;
			m_flgEvalPerf[maxEpoch - 1] = true;
		}
		nnet_cond_epoch_eval(nnet_cond_epoch_eval&& src)noexcept : m_flgEvalPerf(std::move(src.m_flgEvalPerf)) {}

		//!! copy constructor not needed
		nnet_cond_epoch_eval(const nnet_cond_epoch_eval& other)noexcept = delete;
		//!!assignment is not needed
		nnet_cond_epoch_eval& operator=(const nnet_cond_epoch_eval& rhs) noexcept = delete;


		size_t maxEpoch()const noexcept { return m_flgEvalPerf.size(); }

		//using () instead of [] because can't (and don't need to) return reference
		const bool operator()(size_t e)const noexcept { return m_flgEvalPerf[e]; }

		self_t& verbose(size_t e)noexcept { m_flgEvalPerf[e] = true; return *this; }
	};

}