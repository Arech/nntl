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

#include "_defs.h"
#include "common.h"
#include "interfaces.h"

namespace nntl {
namespace activation {

	template<typename RealT>
	class _i_function {
		_i_function() = delete;
		~_i_function() = delete;
	public:
		typedef RealT real_t;

		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;
		typedef typename realmtx_t::numel_cnt_t numel_cnt_t;
		typedef typename realmtx_t::vec_len_t vec_len_t;

		//apply f to each srcdest matrix element to compute activation values. The biases (if any) must be left untouched!
		template <typename iMath>
		nntl_interface static void f(realmtx_t& srcdest, iMath& m) noexcept;

		//get requirements on temporary memory size needed to calculate f() over matrix act (need it for memory
		// preallocation algorithm of iMath). This is default version. Override in derived class if need something more
		template <typename iMath>
		static numel_cnt_t needTempMem(const realmtx_t& act, iMath& m) noexcept {
			return act.numel();
		}
		
	};

	//class defines interface for activation functions. It's intended to be used as a parent class only
	//usually, _i_activation implementation class is nothing more than a thunk into iMath, which contains efficient code
	template<typename RealT>
	class _i_activation : public _i_function<RealT> {
		_i_activation() = delete;
		~_i_activation() = delete;
	public:

		//computes activation function derivative by using its value.
		//i.e. computes y' based on y value ( not the x-value, where y=y(x) )
		template <typename iMath>
		nntl_interface static void df(realmtx_t& f_df, iMath& m) noexcept;
	};


	//for use in output layer activations
	template<typename RealT>
	class _i_activation_loss {
		~_i_activation_loss() = delete;
		_i_activation_loss() = delete;
	public:
		//loss function
		template <typename iMath>
		nntl_interface static typename _i_activation<RealT>::real_t loss(const typename _i_activation<RealT>::realmtx_t& activations, const typename _i_activation<RealT>::realmtx_t& data_y, iMath& m)noexcept;

		//loss function derivative wrt total neuron input Z (=Aprev_layer*W), dL/dZ
		template <typename iMath>
		nntl_interface static void dLdZ(const typename _i_activation<RealT>::realmtx_t& data_y,
			IN OUT typename _i_activation<RealT>::realmtx_t& act_dLdZ, iMath& m)noexcept;
		//we glue into single function calculation of dL/dA and dA/dZ. The latter is in fact calculated by _i_activation::df(), but if
		//we'll calculate dL/dZ in separate functions, then we can't make some optimizations
	};


	//////////////////////////////////////////////////////////////////////////
	//sigmoid
	template<typename RealT=d_interfaces::real_t, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>>
	class sigm : public _i_activation<RealT> {
		sigm() = delete;
		~sigm() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;

	public:
		//apply f to each srcdest matrix element to compute activation values. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept{
			static_assert( std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math" );
			m.sigm(srcdest);
		};
		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dsigm(f_df);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>>
	class sigm_quad_loss : public sigm<RealT, WeightsInitScheme>, public _i_activation_loss<RealT> {
		sigm_quad_loss() = delete;
		~sigm_quad_loss() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.dSigmQuadLoss_dZ(data_y, act_dLdZ);
		}

		template <typename iMath>
		static real_t loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_quadratic(activations, data_y);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>>
	class sigm_xentropy_loss : public sigm<RealT, WeightsInitScheme>, public _i_activation_loss<RealT> {
		sigm_xentropy_loss() = delete;
		~sigm_xentropy_loss() = delete;
	public:
		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			//dL/dz = dL/dA * dA/dZ = (a-y)
			//m.evSub(activations, data_y, dLdZ);
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			m.evSub_ip(act_dLdZ, data_y);
		}

		template <typename iMath>
		static real_t loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_sigm_xentropy(activations, data_y);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// SoftMax (for output layer only - it's easier to get dL/dL than dA/dL for SoftMax)
	// TODO: which weight initialization scheme is better for SoftMax?
	// TODO: may be it's worth to implement SoftMax activation for hidden layers, i.e. make a dA/dZ implementation
	template<typename RealT, typename WeightsInitScheme = weights_init::Martens_SI_sigm<>>
	class softmax_xentropy_loss : public _i_function<RealT>, public _i_activation_loss<RealT> {
		softmax_xentropy_loss() = delete;
		~softmax_xentropy_loss() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtxdef_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.softmax(srcdest);
		};
		//get requirements on temporary memory size needed to calculate f() over matrix act (need it for memory
		// preallocation algorithm of iMath).
		template <typename iMath>
		static numel_cnt_t needTempMem(const realmtx_t& act, iMath& m) noexcept {
			return m.softmax_needTempMem(act);
		}


		template <typename iMath>
		static void dLdZ(const realmtx_t& data_y, realmtx_t& act_dLdZ, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			//SoftMax dL/dZ = dL/dA * dA/dZ = (a-y)
			//m.evSub(activations, data_y, dLdZ);
			NNTL_ASSERT(!act_dLdZ.emulatesBiases() && !data_y.emulatesBiases());
			m.evSub_ip(act_dLdZ, data_y);
		}

		template <typename iMath>
		static real_t loss(const realmtx_t& activations, const realmtx_t& data_y, iMath& m)noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			return m.loss_softmax_xentropy(activations, data_y);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	//ReLU
	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class relu : public _i_activation<RealT> {
		relu() = delete;
		~relu() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.relu(srcdest);
		};

		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.drelu(f_df);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// Leaky Relu
	template<typename RealT, size_t LeakKInv100 = 10000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class leaky_relu : public _i_activation<RealT> {
		leaky_relu() = delete;
		~leaky_relu() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;
		static constexpr real_t LeakK = real_t(100.0) / real_t(LeakKInv100);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath>
		static void f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.leakyrelu(srcdest, LeakK);
		};

		template <typename iMath>
		static void df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.dleakyrelu(f_df, LeakK);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using leaky_relu_1000 = leaky_relu<RealT, 100000, WeightsInitScheme>;
	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using leaky_relu_100 = leaky_relu<RealT, 10000, WeightsInitScheme>;
	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using very_leaky_relu_5p5 = leaky_relu<RealT, 550, WeightsInitScheme>;

	//////////////////////////////////////////////////////////////////////////
	// ELU
	template<typename RealT, size_t Alpha1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class elu : public _i_activation<RealT> {
		elu() = delete;
		~elu() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;
		static constexpr real_t Alpha = real_t(Alpha1e3) / real_t(1000.0);
		static constexpr bool bIsUnitAlpha = (Alpha1e3 == 1000);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha>
		static std::enable_if_t<!bUnitAlpha> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elu(srcdest, Alpha);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha>
		static std::enable_if_t<bUnitAlpha> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elu_unitalpha(srcdest);
		};

		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha>
		static std::enable_if_t<!bUnitAlpha> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delu(f_df, Alpha);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha>
		static std::enable_if_t<bUnitAlpha> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delu_unitalpha(f_df);
		}
	};

	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elu_ua = elu<RealT, 1000, WeightsInitScheme>;


	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//ELogU : log(x+1)/log(b) | x>0,  alpha*(exp(x)-1) | x<0
	template<typename RealT, size_t Alpha1e3 = 1000, size_t B1e6 = 2000000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	class elogu : public _i_activation<RealT> {
		elogu() = delete;
		~elogu() = delete;
	public:
		typedef WeightsInitScheme weights_scheme;
		static constexpr real_t Alpha = real_t(Alpha1e3) / real_t(1000.0);
		static constexpr bool bIsUnitAlpha = (Alpha1e3 == 1000);

		static constexpr real_t B = real_t(B1e6) / real_t(1000000.0);
		static constexpr bool bIsNaturalB = (B1e6 == 2718281);

	public:
		//apply f to each srcdest matrix element. The biases (if any) must be left untouched!
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<!bUnitAlpha && !bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu(srcdest, Alpha, B);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<bUnitAlpha && !bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu_ua(srcdest, B);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<!bUnitAlpha && bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu_nb(srcdest, Alpha);
		};
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<bUnitAlpha && bNatB> f(realmtx_t& srcdest, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			m.elogu_ua_nb(srcdest);
		};

		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<!bUnitAlpha && !bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu(f_df, Alpha, B);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<bUnitAlpha && !bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu_ua(f_df, B);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<!bUnitAlpha && bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu_nb(f_df, Alpha);
		}
		template <typename iMath, bool bUnitAlpha = bIsUnitAlpha, bool bNatB = bIsNaturalB>
		static std::enable_if_t<bUnitAlpha && bNatB> df(realmtx_t& f_df, iMath& m) noexcept {
			static_assert(std::is_base_of<math::_i_math<real_t>, iMath>::value, "iMath should implement math::_i_math");
			NNTL_ASSERT(!f_df.emulatesBiases());
			m.delogu_ua_nb(f_df);
		}
	};

	template<typename RealT, size_t B1e6 = 2000000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elogu_ua = elogu<RealT, 1000, B1e6, WeightsInitScheme>;
	template<typename RealT, size_t Alpha1e3 = 1000, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elogu_nb = elogu<RealT, Alpha1e3, 2718281, WeightsInitScheme>;
	template<typename RealT, typename WeightsInitScheme = weights_init::He_Zhang<>>
	using elogu_ua_nb = elogu<RealT, 1000, 2718281, WeightsInitScheme>;
}
}