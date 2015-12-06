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

// rng helper, that converts uniform distribution to normal (gaussian) distribution
// This code for non-performance critical use only!

#include <random>

namespace nntl {
namespace rng {

	template <typename iRng>
	struct distr_normal_naive {
	public:
		typedef iRng iRng_t;
		typedef typename iRng_t::real_t real_t;
		typedef typename iRng_t::realmtx_t realmtx_t;

	protected:
		iRng_t& m_iR;
		std::normal_distribution<real_t> m_distr;

	public:
		~distr_normal_naive() {}
		distr_normal_naive(iRng_t& iR, real_t mn = real_t(0.0), real_t stdev = real_t(1.0))noexcept
			: m_iR(iR), m_distr(mn, stdev) {}

		void gen_vector(real_t* ptr, const size_t n)noexcept {
			const auto pE = ptr + n;
			while (ptr != pE) *ptr++ = m_distr(m_iR);
		}

		void gen_matrix(realmtx_t& m)noexcept {
			NNTL_ASSERT(!m.empty() && m.numel() > 0);
			gen_vector(m.dataAsVec(), m.numel());
		}
	};

}
}
