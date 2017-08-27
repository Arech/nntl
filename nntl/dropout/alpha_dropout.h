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
		, ADCorr corrType = ADCorr::no
	>
	struct AlphaDropout 
		: public _impl::_dropout_base<RealT>
		, public _impl::SNN_td<RealT, Alpha1e9, Lambda1e9, fpMean1e6, fpVar1e6, corrType>
	{
	private:
		typedef _impl::_dropout_base<RealT> _base_class_t;

	public:
		using _base_class_t::real_t;

	protected:
		realmtxdef_t m_mtxB;
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

				if (utils::binary_option<true>(ar, serialization::serialize_dropout_mask))
					ar & NNTL_SERIALIZATION_NVP(m_mtxB);
			}

			_base_class_t::_dropout_serialize(ar, version);
		}

		template<typename CommonDataT>
		bool _dropout_init(const neurons_count_t neurons_cnt, const CommonDataT& CD)noexcept {
			NNTL_ASSERT(neurons_cnt);
			if (!_base_class_t::_dropout_init(neurons_cnt, CD))  return false;

			if (CD.is_training_possible()) {
				//condition means if (there'll be a training session) and (we're going to use dropout)
				NNTL_ASSERT(!m_mtxB.emulatesBiases());
				//resize to the biggest possible size during training
				if (!m_mtxB.resize(CD.training_batch_size(), neurons_cnt))//!bDropoutWorksAtEvaluationToo !!!!
					return false;

				NNTL_ASSERT(m_mtxB.size() == m_dropoutMask.size());

				//to get rid of initial nan
				dropoutPercentActive(m_dropoutPercentActive);
			}
			return true;
		}

		void _dropout_deinit() noexcept {
			m_mtxB.clear();
			_base_class_t::_dropout_deinit();
		}

		template<typename CommonDataT>
		void _dropout_on_batch_size_change(const CommonDataT& CD) noexcept {
			_base_class_t::_dropout_on_batch_size_change(CD);
			if (CD.is_training_mode() && bDropout()) {
				NNTL_ASSERT(!m_mtxB.empty());
				m_mtxB.deform_rows(CD.get_cur_batch_size());
				NNTL_ASSERT(m_mtxB.size() == m_dropoutMask.size());
			}
		}

// 		template<typename iMathT, typename iRngT, typename iInspectT>
// 		void _dropout_apply(realmtx_t& activations, const bool bTrainingMode, iMathT& iM, iRngT& iR, iInspectT& _iI) noexcept {
		template<typename CommonDataT>
		void _dropout_apply(realmtx_t& activations, const CommonDataT& CD) noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(real_t(0.) < m_dropoutPercentActive && m_dropoutPercentActive < real_t(1.));

			if (CD.is_training_mode()) {
				//must make dropoutMask and apply it
				NNTL_ASSERT(m_dropoutMask.size() == activations.size_no_bias());
				NNTL_ASSERT(m_mtxB.size() == m_dropoutMask.size());
				NNTL_ASSERT(m_a && m_b && m_mbDropVal);

				CD.iRng().gen_matrix_norm(m_dropoutMask);

				auto& _iI = CD.iInspect();
				_iI.fprop_preDropout(activations, m_dropoutPercentActive, m_dropoutMask);

				CD.iMath().make_alphaDropout(activations, m_dropoutPercentActive, m_a, m_b, m_mbDropVal, m_dropoutMask, m_mtxB);

				_iI.fprop_postDropout(activations, m_dropoutMask);
			}
		}

		//For the _dropout_restoreScaling() phase we obtain almost original activations by doing A <- (A3-mtxB) ./ a.
		// Dropped out values will have a value of zero. dL/dA scaling is the same as for the inverted dropout:
		// dL/dA = dL/dA .* dropoutMask
		//template<typename iMathT, typename iInspectT>
		template<typename CommonDataT>
		void _dropout_restoreScaling(realmtx_t& dLdA, realmtx_t& activations, const CommonDataT& CD)noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(m_dropoutMask.size() == dLdA.size());

			auto& iM = CD.iMath();
			iM.evMul_ip(dLdA, m_dropoutMask);

			auto& _iI = CD.iInspect();
			_iI.bprop_preCancelDropout(activations, m_dropoutPercentActive);
			iM.evSubMtxMulC_ip_nb(activations, m_mtxB, static_cast<real_t>(ext_real_t(1.) / m_a));
			_iI.bprop_postCancelDropout(activations);
		}

	public:
		void dropoutPercentActive(const real_t dpa)noexcept {
			_base_class_t::dropoutPercentActive(dpa);
			if (bDropout()) {
				calc_coeffs<ext_real_t>(dpa, m_mtxB.cols(), m_a, m_b, m_mbDropVal);
			}
		}
	};

}
