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

#include "_defs.h"
#include "common.h"

namespace nntl {
namespace activation {

	//class defines interface for activation functions. It's intended to be used as a parent class only
	//usually, _i_activation implementation class is nothing more than a thunk into iMath, which contains efficient code
	class _i_activation {
		_i_activation() = delete;
		~_i_activation() = delete;
	public:
		typedef nntl::math_types::floatmtx_ty floatmtx_t;
		typedef floatmtx_t::value_type float_t_;

		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		nntl_interface static void f(floatmtx_t& srcdest, iMath& m) noexcept;
		
		//f derivative, fValue is used in no_bias version!
		template <typename iMath>
		nntl_interface static void df(const floatmtx_t& fValue, floatmtx_t& df, iMath& m) noexcept;
	};


	//for use in output layer activations
	class _i_activation_loss {
		~_i_activation_loss() = delete;
		_i_activation_loss() = delete;
	public:
		//loss function
		template <typename iMath>
		nntl_interface static _i_activation::float_t_ loss(const _i_activation::floatmtx_t& activations, const _i_activation::floatmtx_t& data_y, iMath& m)noexcept;

		//loss function derivative wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		template <typename iMath>
		nntl_interface static void dLdZ(const _i_activation::floatmtx_t& activations, const _i_activation::floatmtx_t& data_y,
			_i_activation::floatmtx_t& dLdZ, iMath& m)noexcept;
		//we glue into single function calculation of dL/dA and dA/dZ. The latter is in fact calculated by _i_activation::df(), but if
		//we'll calculate dL/dZ in separate functions, then we can't make some optimizations
	};


	//////////////////////////////////////////////////////////////////////////
	/*class identity : public _i_activation {
		identity() = delete;
		~identity() = delete;
	public:
		template <typename iMath>
		static void f(floatmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
		};
		template <typename iMath>
		static void df(floatmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
		}
	};*/

	//////////////////////////////////////////////////////////////////////////
	//sigmoids
	class sigm : public _i_activation {
		sigm() = delete;
		~sigm() = delete;
	public:
		template <typename iMath>
		static void f(floatmtx_t& srcdest, iMath& m) noexcept{
			static_assert( std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math" );
			m.sigm(srcdest);
		};
		template <typename iMath>
		static void df(const floatmtx_t& fValue, floatmtx_t& df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			m.dsigm(fValue, df);//fValue is used in no_bias version!
		}
	};

	class sigm_quad_loss : public sigm, public _i_activation_loss {
		sigm_quad_loss() = delete;
		~sigm_quad_loss() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const floatmtx_t& activations, const floatmtx_t& data_y, floatmtx_t& dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			m.dSigmQuadLoss_dZ(activations, data_y, dLdZ);
		}

		template <typename iMath>
		static float_t_ loss(const floatmtx_t& activations, const floatmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_quadratic(activations, data_y);
		}
	};

	class sigm_xentropy_loss : public sigm, public _i_activation_loss {
		sigm_xentropy_loss() = delete;
		~sigm_xentropy_loss() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const floatmtx_t& activations, const floatmtx_t& data_y, floatmtx_t& dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			//dL/dz = dL/dA * dA/dZ = (a-y)
			m.evSub(activations, data_y, dLdZ);
		}

		template <typename iMath>
		static float_t_ loss(const floatmtx_t& activations, const floatmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_sigm_xentropy(activations, data_y);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//ReLU
	class relu : public _i_activation {
		relu() = delete;
		~relu() = delete;
	public:
		template <typename iMath>
		static void f(floatmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			m.relu(srcdest);
		};

		template <typename iMath>
		static void df(const floatmtx_t& fValue, floatmtx_t& df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math, iMath>::value, "iMath should implement math::_i_math");
			m.drelu(fValue, df);//fValue is used in no_bias version!
		}
	};

}
}