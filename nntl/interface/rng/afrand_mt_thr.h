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

//#include "../../../_extern/agner.org/AF_randomc_h/random.h"

#ifndef NNTL_OVERRIDE_AFRANDOM_MT_THRESHOLDS

namespace nntl {
namespace rng {

	namespace _impl {

		template<typename AgnerFogRNG, typename real_t>
		struct AFRAND_MT_THR {};

		template<> struct AFRAND_MT_THR<AFog::CRandomMersenne, double> {
			static constexpr size_t bnd_gen_vector = 1650;
			static constexpr size_t bnd_gen_vector_gtz = 1645;
			static constexpr size_t bnd_gen_vector_norm = 1640;

			static constexpr size_t bnd_bernoulli_vector = 1500;
			static constexpr size_t bnd_normal_vector = 1000;
		};

		template<> struct AFRAND_MT_THR<AFog::CRandomSFMT0, double> {
			static constexpr size_t bnd_gen_vector = 2640;
			static constexpr size_t bnd_gen_vector_gtz = 2630;
			static constexpr size_t bnd_gen_vector_norm = 2620;

			static constexpr size_t bnd_bernoulli_vector = 2500;
			static constexpr size_t bnd_normal_vector = 2000;
		};

		//insanely strange RNG
		template<> struct AFRAND_MT_THR<AFog::CRandomSFMT1, double> {
			static constexpr size_t bnd_gen_vector = 3000;
			static constexpr size_t bnd_gen_vector_gtz = 3000;
			static constexpr size_t bnd_gen_vector_norm = 3000;

			static constexpr size_t bnd_bernoulli_vector = 2900;
			static constexpr size_t bnd_normal_vector = 2300;
		};

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		template<> struct AFRAND_MT_THR<AFog::CRandomMersenne, float> {
			static constexpr size_t bnd_gen_vector = 2900;//1600;// 2900;
			static constexpr size_t bnd_gen_vector_gtz = 2900;// 1600;//2900;
			static constexpr size_t bnd_gen_vector_norm = 2900;// 1600;// 2900;

			static constexpr size_t bnd_bernoulli_vector = 1700;
			static constexpr size_t bnd_normal_vector = 175;
		};

		template<> struct AFRAND_MT_THR<AFog::CRandomSFMT0, float> {
			static constexpr size_t bnd_gen_vector = 4200;//2620;//  4200;
			static constexpr size_t bnd_gen_vector_gtz = 4200;//2620;// 4200;
			static constexpr size_t bnd_gen_vector_norm = 4200;// 2620;// 4200;

			static constexpr size_t bnd_bernoulli_vector = 2250;
			static constexpr size_t bnd_normal_vector = 175;
		};

		template<> struct AFRAND_MT_THR<AFog::CRandomSFMT1, float> {
			static constexpr size_t bnd_gen_vector = 4400;// 3000;//4400;
			static constexpr size_t bnd_gen_vector_gtz = 4400;//3000;//4400;
			static constexpr size_t bnd_gen_vector_norm = 4400;//3000;//  4400;

			static constexpr size_t bnd_bernoulli_vector = 2200;
			static constexpr size_t bnd_normal_vector = 180;
		};
	}

}
}

#endif
