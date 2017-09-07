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

#include "../_defs.h"
#include "../common.h"
#include "../weights_init.h"

namespace nntl {
namespace activation {

	template<typename ActT, typename TestActT>
	using is_type_of = ::std::is_base_of<ActT, TestActT>;

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	template<typename RealT, typename WeightsInitT>
	class _i_function : public math::smatrix_td
	{
	public:
		typedef RealT real_t;
		
		typedef WeightsInitT weights_scheme_t;

		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;		

		//apply f to each srcdest matrix element to compute activation values. The biases (if any) must be left untouched!
		template <typename iMath>
		nntl_interface static void f(realmtx_t& srcdest, iMath& m) noexcept;

		//get requirements on temporary memory size needed to calculate f() over matrix act (need it for memory
		// preallocation algorithm of iMath). This is default version. Override in derived class if need something more
		// #todo we should probably split output to fprop() only and fprop()+bprop() versions like we're doing in _i_layer::init()
		// #todo probably should adopt a _i_loss_addendum initialization scheme here
		template <typename iMath>
		static numel_cnt_t needTempMem(const realmtx_t& act, iMath& ) noexcept {
			return act.numel();
		}

		//////////////////////////////////////////////////////////////////////////
		//to support state full activations this functions must be overridden
		static constexpr bool act_init()noexcept{ return true; }
		static constexpr void act_deinit()noexcept {}
	};

	//class defines interface for activation functions. It's intended to be used as a parent class only
	//usually, _i_activation implementation class is nothing more than a thunk into iMath, which contains efficient code
	template<typename RealT, typename WeightsInitT>
	class _i_activation : public _i_function<RealT, WeightsInitT> {
	public:
		//computes activation function derivative by using its value.
		//i.e. computes y' based on y value ( not the x-value, where y=y(x) )
		template <typename iMath>
		nntl_interface static void df(realmtx_t& f_df, iMath& m) noexcept;

		//to support linear layers
		template <typename iMath>
		static void dIdentity(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(::std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_UNREF(m);
			NNTL_ASSERT(!f_df.emulatesBiases());
			//m.dIdentity(f_df);
			f_df.ones();
		}

		static constexpr real_t act_scaling_coeff()noexcept {
			return real_t(1.);
		}
	};


	//for use in an output layer activations
	template<typename RealT>
	class _i_activation_loss {
	public:
		//loss function
		template <typename iMath>
		nntl_interface static RealT loss(const typename iMath::realmtx_t& activations, const typename iMath::realmtx_t& data_y, iMath& m)noexcept;

		//loss function derivative wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		template <typename iMath>
		nntl_interface static void dLdZ(const typename iMath::realmtx_t& data_y,
			IN OUT typename iMath::realmtx_t& act_dLdZ, iMath& m)noexcept;
		//we glue into single function calculation of dL/dA and dA/dZ. The latter is in fact calculated by _i_activation::df(), but if
		//we'll calculate dL/dZ in separate functions, then we can't make some optimizations

		//to support linear layers
		template <typename iMath>
		nntl_interface static void dLdZIdentity(const typename iMath::realmtx_t& data_y,
			IN OUT typename iMath::realmtx_t& act_dLdZ, iMath& m) noexcept;
	};

	template<typename RealT>
	class _i_quadratic_loss : public _i_activation_loss<RealT> {
	public:
		template <typename iMath>
		static void dLdZIdentity(const typename iMath::realmtx_t& data_y,
			IN OUT typename iMath::realmtx_t& act_dLdZ, iMath& m) noexcept
		{
			static_assert(::std::is_base_of<math::_i_math<RealT>, iMath>::value, "iMath should implement math::_i_math");
			// L = 1/2 * (a-y)^2; dL/dZ = dL/dA * dA/dZ;
			// dA/dZ == 1;
			// ==> dL/dZ = dL/dA = (a-y);
			NNTL_ASSERT(data_y.size() == act_dLdZ.size());
			NNTL_ASSERT(!data_y.emulatesBiases() && !act_dLdZ.emulatesBiases());
			m.evSub_ip(act_dLdZ, data_y);
		}
	};

	template<typename RealT>
	class _i_xentropy_loss : public _i_activation_loss<RealT> {
	public:
		template <typename iMath>
		static void dLdZIdentity(const typename iMath::realmtx_t& data_y,
			IN OUT typename iMath::realmtx_t& act_dLdZ, iMath& m) noexcept
		{
			static_assert(::std::is_base_of<math::_i_math<RealT>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!data_y.emulatesBiases() && !act_dLdZ.emulatesBiases());
			m.dIdentityXEntropyLoss_dZ(data_y, act_dLdZ);
		}
	};


}
}
