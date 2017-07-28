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

//This file defines wrapper around classical Dropout
//Layer classes will be derived from this classes using public inheritance, so code wisely

#include "_base.h"

namespace nntl {

	namespace _impl {

		template<typename RealT>
		class _dropout_base : public _impl::_No_Dropout_at_All<RealT> {
		public:
			typedef math::smatrix_deform<real_t> realmtxdef_t;

		public:
			//DON'T redefine in derived classes!
			static constexpr bool bDropoutWorksAtEvaluationToo = false;

		protected:
			//////////////////////////////////////////////////////////////////////////
			//vars

			//matrix of dropped out neuron activations, used when 1>m_dropoutPercentActive>0
			realmtxdef_t m_dropoutMask;//<batch_size rows> x <m_neurons_cnt cols> (must not have a bias column)
			real_t m_dropoutPercentActive;//probability of keeping unit active

		protected:
			~_dropout_base()noexcept {}
			_dropout_base()noexcept : m_dropoutPercentActive(real_t(1.)) {}

			template<class Archive>
			void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {
				if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
					ar & NNTL_SERIALIZATION_NVP(m_dropoutPercentActive);
				}

				if (bDropout() && utils::binary_option<true>(ar, serialization::serialize_dropout_mask))
					ar & NNTL_SERIALIZATION_NVP(m_dropoutMask);
			}

			bool _dropout_init(const bool isTrainingPossible, const vec_len_t max_batch_size, const neurons_count_t neurons_cnt)noexcept {
				NNTL_ASSERT(neurons_cnt);
				if (isTrainingPossible) {
					NNTL_ASSERT(max_batch_size);
					//we don't check bDropout() here because assume that if the dropout enabled, it'll be used
					//even if now it's disabled.
					NNTL_ASSERT(!m_dropoutMask.emulatesBiases());
					//resize to the biggest possible size during training
					if (!m_dropoutMask.resize(max_batch_size, neurons_cnt)) return false;
				}
				return true;
			}

			void _dropout_deinit()noexcept {
				m_dropoutMask.clear();
				//we mustn't clear settings here
			}

			void _dropout_on_batch_size_change(const bool isTrainingMode, const vec_len_t batchSize)noexcept {
				NNTL_ASSERT(batchSize);
				if (isTrainingMode && bDropout()) {
					NNTL_ASSERT(!m_dropoutMask.empty());
					m_dropoutMask.deform_rows(batchSize);
				}
			}

		public:
			bool bDropout()const noexcept { return m_dropoutPercentActive < real_t(1.); }

			real_t dropoutPercentActive()const noexcept { return m_dropoutPercentActive; }

			void dropoutPercentActive(const real_t dpa)noexcept {
				NNTL_ASSERT(real_t(0.) <= dpa && dpa <= real_t(1.));
				m_dropoutPercentActive = (dpa <= real_t(+0.) || dpa > real_t(1.)) ? real_t(1.) : dpa;
			}
		};
	}


	//////////////////////////////////////////////////////////////////////////
	//this is a wrapper around "classical" dropout technique invented by Jeff Hinton (actually, it's a modification
	// that is called "inverse dropout". It doesn't require special handling during evaluation step)
	// Here is what it does during training: imagine you have an additional dropout layer over the current layer. Then:
	//
	// for the fprop() - it makes a dropoutMask matrix that containts either a 0 with probability (1-p),
	// or a 1/p with probability p. Then it just multiplies the layer activations elementwise by the dropoutMask
	// values (_dropout_apply() phase)
	// 
	// for the bprop() - the dropout layer has to update dL/dA derivative that will be passed further to the original layer to 
	// reflect the scaling (by the 1/p) occured during fprop(). This is done by multipling dL/dA by the same
	// dropoutMask (_dropout_restoreScaling() phase).
	// 
	// And there's another trick involved in _dropout_restoreScaling(), that has to deal with the
	// implementation fact that we don't really
	// have a separate storage for the pre-dropout activations (A), as well as for pre-activation (Z) values.
	// We just overwrite pre-dropout matrix with a new post-dropout values, and are doing the same for pre-activation while
	// computing activation values.
	// Therefore, just because we compute dA/dZ based on function values itself (instead of function arguments)
	// we have to 'revert' post-dropout activations to its pre-dropout state at some moment before computing dA/dZ.
	// This is also done during _dropout_restoreScaling() phase by multiplying post-dropout A values by p - by doing so,
	// we get almost the original pre-dropout activation matrix but with some values just set to 0. Even if 
	// the calculation of dA/dZ using the function value of zero will return a non-zero derivative
	// - the final dL/dZ derivative will still be correct, because by the chain rule:
	// dL/dZ = dA/dZ .* dL/dA /*for the original layer*/
	//       = dA/dZ .* dL/dA /*for the upper imaginary dropout layer*/ .* dropoutMask
	// i.e. elementwise application of the dropoutMask at the final step also removes "invalid" dA/dZ values computed from
	// dropped out zeros.
	template<typename RealT>
	class Dropout : public _impl::_dropout_base<RealT> {
	private:
		typedef _impl::_dropout_base<RealT> _base_class_t;

	protected:
		~Dropout()noexcept {}
		Dropout()noexcept : _base_class_t() {}

		template<typename iMathT, typename iRngT, typename iInspectT>
		void _dropout_apply(realmtx_t& activations, const bool bTrainingMode, iMathT& iM, iRngT& iR, iInspectT& _iI)noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(real_t(0.) < m_dropoutPercentActive && m_dropoutPercentActive < real_t(1.));

			if (bTrainingMode) {
				//must make dropoutMask and apply it
				NNTL_ASSERT(m_dropoutMask.size() == activations.size_no_bias());
				iR.gen_matrix_norm(m_dropoutMask);

				_iI.fprop_preDropout(activations, m_dropoutPercentActive, m_dropoutMask);
				iM.make_dropout(activations, m_dropoutPercentActive, m_dropoutMask);
				_iI.fprop_postDropout(activations, m_dropoutMask);
			}
		}

		//this function restores zeros (dropped out values) and a proper scaling of dL/dA as well as undoes the scaling
		//of activation values
		template<typename iMathT, typename iInspectT>
		void _dropout_restoreScaling(realmtx_t& dLdA, realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {
			NNTL_ASSERT(bDropout());

			NNTL_ASSERT(m_dropoutMask.size() == dLdA.size());
			iM.evMul_ip(dLdA, m_dropoutMask);

			_iI.bprop_preCancelDropout(activations, m_dropoutPercentActive);
			iM.evMulC_ip_Anb(activations, m_dropoutPercentActive);
			_iI.bprop_postCancelDropout(activations);
		}
	};

}