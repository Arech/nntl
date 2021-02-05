/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "dropout.h"

#include "../_SNN_common.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	// Default parameters corresponds to a "standard" SELU with stable fixed point at (0,1)
	// Alpha-Dropout (arxiv:1706.02515 "Self-Normalizing Neural Networks", by Günter Klambauer et al.) is a modification
	// of the Inverse Dropout technique (see the Dropout<> class) that modifies original activations A to A2 by first setting their
	// values to the (-Alpha*Lambda) value with a probability (1-p) (leaving as it is with probability p),
	// and then performing the affine transformation A3 := a.*A2 + b, where a and b are scalar functions of p and SELU parameters.
	// 
	// Here is how it's implemented:
	// 1. During the _dropout_apply() phase we construct dropoutMask and mtxB matrices by setting their values to:
	// {dropoutMask <- 0, mtxB <- (a*(-Alpha*Lambda) + b) } with probability (1-p) and
	// {dropoutMask <- a, mtxB <- b } with probability p
	// Then we compute the post-dropout activations A3 <- A.*dropoutMask + mtxB
	// 2. For the _dropout_restoreScaling() phase we obtain almost original activations by doing A <- (A3-mtxB) ./ a.
	// Dropped out values will have a value of zero. dL/dA scaling is the same as for the inverted dropout:
	// dL/dA = dL/dA .* dropoutMask
	template<typename RealT
		, int64_t Alpha1e9 = 0, int64_t Lambda1e9 = 0, int fpMean1e6 = 0, int fpVar1e6 = 1000000
		//, ADCorr corrType = ADCorr::no
	>
	struct AlphaDropout 
		: public _impl::_dropout_base<RealT>
		, public _impl::SNN_td<RealT, Alpha1e9, Lambda1e9, fpMean1e6, fpVar1e6> //, corrType>
	{
	private:
		typedef _impl::_dropout_base<RealT> _base_class_t;

	public:
		using _base_class_t::real_t;

		//this flag means that the dropout algorithm doesn't change the activation value if it is zero.
		// for example, it is the case of classical dropout (it drops values to zeros), but not the case of AlphaDropout
		static constexpr bool bDropoutIsZeroStable = false;

	protected:
		real_t m_a, m_b, m_mbDropVal;

	protected:
		AlphaDropout()noexcept : _base_class_t(), m_a(real_t(0)), m_b(real_t(0)), m_mbDropVal(real_t(0))
		{}

		template<class Archive>
		void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {
			if (bDropout()) {
				if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
					ar & NNTL_SERIALIZATION_NVP(m_a);
					ar & NNTL_SERIALIZATION_NVP(m_b);
				}
			}

			_base_class_t::_dropout_serialize(ar, version);
		}

		template<typename CommonDataT>
		bool _dropout_init(const neurons_count_t neurons_cnt, const CommonDataT& CD, const _impl::_layer_init_data<CommonDataT>& lid)noexcept {
			NNTL_ASSERT(neurons_cnt);
			if (!_base_class_t::_dropout_init(neurons_cnt, CD, lid))  return false;

			if (CD.is_training_possible()) {
				//condition means if (there'll be a training session) and (we're going to use dropout)
				//to get rid of initial nan
				dropoutPercentActive(m_dropoutPercentActive);
			}
			return true;
		}

		template<typename CommonDataT>
		void _dropout_apply(realmtx_t& activations, const CommonDataT& CD) noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(real_t(0.) < m_dropoutPercentActive && m_dropoutPercentActive < real_t(1.));

			if (CD.is_training_mode()) {
				//must make dropoutMask and apply it
				NNTL_ASSERT(m_dropoutMask.size() == activations.size_no_bias());
				NNTL_ASSERT(m_dropoutMask.size() == m_origActivations.size());
				NNTL_ASSERT(m_a && m_b && m_mbDropVal);

				_dropout_saveActivations(activations);
				CD.iRng().gen_matrix_norm(m_dropoutMask);

				auto& _iI = CD.iInspect();
				_iI.fprop_preDropout(activations, m_dropoutPercentActive, m_dropoutMask);

				CD.iMath().make_alphaDropout(activations, m_dropoutPercentActive, m_a, m_b, m_mbDropVal, m_dropoutMask);

				_iI.fprop_postDropout(activations, m_dropoutMask);
			}
		}

	public:
		void dropoutPercentActive(const real_t dpa)noexcept {
			_base_class_t::dropoutPercentActive(dpa);
			if (bDropout()) {
				NNTL_ASSERT(m_dropoutMask.size() == m_origActivations.size());
				calc_coeffs<ext_real_t>(dpa, /*m_origActivations.cols(),*/ m_a, m_b, m_mbDropVal);
			}
		}
	};

}
