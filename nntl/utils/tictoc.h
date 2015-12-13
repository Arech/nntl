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

#include <algorithm>
#include "../utils/chrono.h"

namespace nntl {
namespace utils {

	class tictoc {
	public:
		typedef std::chrono::nanoseconds duration_t;

	public:
		std::chrono::steady_clock::time_point m_tStart;
		duration_t m_dFirstRun, m_dBestRun, m_dAllRun;
		uint64_t m_repeats;

	public:
		~tictoc()noexcept {};
		tictoc()noexcept { reset(); };

		void reset()noexcept {
			m_repeats = 0;
			m_dFirstRun = duration_t(0);
			m_dBestRun = duration_t(std::numeric_limits< duration_t::rep >::max());
			m_dAllRun = duration_t(0);
		}

		void tic()noexcept {
			m_tStart = std::chrono::steady_clock::now();
		}

		void toc()noexcept {
			const auto dur = std::chrono::steady_clock::now() - m_tStart;
			if (0 == m_repeats) m_dFirstRun = dur;
			m_dBestRun = std::min(dur, m_dBestRun);
			m_dAllRun += dur;
			++m_repeats;
		}

		std::string to_string() const {
			return m_repeats > 0
				? nntl::utils::duration_readable(m_dFirstRun) + " "
				+ nntl::utils::duration_readable(m_dBestRun) + " "
				+ nntl::utils::duration_readable(m_dAllRun, m_repeats)
				: "never run!";
		}

		void say(const char* desr = "?", const char* spc = ":\t")const {
			std::cout << desr << spc << to_string() << std::endl;
		}
	};

	inline std::ostream& operator<<(std::ostream& os, const tictoc& tt) {
		os << tt.to_string();
		return os;
	}

}
}