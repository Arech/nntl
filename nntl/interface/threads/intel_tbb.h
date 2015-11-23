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

#include "../threads.h"

#define TBB_USE_EXCEPTIONS 0

//TODO: how to link TBB in MSVC2015 project??? There are no appropriate binaries in latest (4.3-4.4) distributions

#pragma warning(disable: 6297 6001)
#include <tbb/tbb.h>
#pragma warning(default: 6297 6001)

namespace nntl {
	namespace threads {

		class IntelTBB : public _i_threads {
			//!! copy constructor not needed
			IntelTBB(const IntelTBB& other)noexcept = delete;
			//!!assignment is not needed
			IntelTBB& operator=(const IntelTBB& rhs) noexcept = delete;

		protected:
			typedef std::function<void(float_t_*, numel_cnt_t)> func_t;

		public:
			~IntelTBB()noexcept {

			}
			IntelTBB()noexcept {

			}

			template<typename Func>
			void run(Func&& F, float_t_* ptr, const numel_cnt_t cnt) noexcept {
				tbb::parallel_for(tbb::blocked_range<numel_cnt_t>(0, cnt), [=, &F](const tbb::blocked_range<size_t>& r) {
					F(&ptr[r.begin()], r.size());
				});
			}

		protected:

		protected:


		};

	}
}
