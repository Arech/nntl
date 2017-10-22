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
			//DON'T redefine in derived classes! Search for uses and double check if you want to change it
			static constexpr bool bDropoutWorksAtEvaluationToo = false;

		protected:
			//////////////////////////////////////////////////////////////////////////
			//vars

			//matrix of dropped out neuron activations, used when 1>m_dropoutPercentActive>0
			realmtxdef_t m_dropoutMask;//<batch_size rows> x <m_neurons_cnt cols> (must not have a bias column)
			realmtxdef_t m_origActivations;//<batch_size rows> x <m_neurons_cnt cols> (must not have a bias column)

			real_t m_dropoutPercentActive;//probability of keeping unit active

		protected:
			~_dropout_base()noexcept {}
			_dropout_base()noexcept : m_dropoutPercentActive(real_t(1.)) {}

			template<class Archive>
			void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {
				NNTL_UNREF(version);
				if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
					ar & NNTL_SERIALIZATION_NVP(m_dropoutPercentActive);
				}

				if (bDropout() && utils::binary_option<true>(ar, serialization::serialize_dropout_mask)) {
					ar & NNTL_SERIALIZATION_NVP(m_dropoutMask);
					ar & NNTL_SERIALIZATION_NVP(m_origActivations);
				}
			}

			template<typename CommonDataT>
			bool _dropout_init(const neurons_count_t neurons_cnt, const CommonDataT& CD)noexcept {
				NNTL_ASSERT(neurons_cnt);
				static_assert(!bDropoutWorksAtEvaluationToo, "Next if() depends on it");
				if (CD.is_training_possible()) {
					const auto max_batch_size = CD.training_batch_size();//!bDropoutWorksAtEvaluationToo !!!!
					NNTL_ASSERT(max_batch_size);
					//we don't check bDropout() here because assume that if the dropout enabled, it'll be used
					//even if now it's disabled.
					NNTL_ASSERT(!m_dropoutMask.emulatesBiases());
					//resize to the biggest possible size during training
					if (!m_dropoutMask.resize(max_batch_size, neurons_cnt)) return false;

					NNTL_ASSERT(!m_origActivations.emulatesBiases());
					if (!m_origActivations.resize(max_batch_size, neurons_cnt)) return false;

					CD.iRng().preinit_additive_norm(m_dropoutMask.numel());
				}
				return true;
			}

			void _dropout_deinit()noexcept {
				m_dropoutMask.clear();
				m_origActivations.clear();
				//we mustn't clear settings here
			}

			template<typename CommonDataT>
			void _dropout_on_batch_size_change(const CommonDataT& CD)noexcept {
				if (CD.is_training_mode() && bDropout()) {
					const auto bs = CD.get_cur_batch_size();

					NNTL_ASSERT(!m_dropoutMask.empty());
					m_dropoutMask.deform_rows(bs);

					NNTL_ASSERT(!m_origActivations.empty());
					m_origActivations.deform_rows(bs);
				}
			}

			void _dropout_saveActivations(const realmtx_t& curAct)noexcept {
				NNTL_ASSERT(bDropout());
				NNTL_ASSERT(curAct.size_no_bias() == m_origActivations.size());
				const auto c = curAct.copy_data_skip_bias(m_origActivations);
				NNTL_ASSERT(c);
			}
			void _dropout_restoreActivations(realmtx_t& Act)const noexcept {
				NNTL_ASSERT(bDropout());
				NNTL_ASSERT(Act.size_no_bias() == m_origActivations.size());
				const auto c = m_origActivations.copy_data_skip_bias(Act);
				NNTL_ASSERT(c);
			}

			template<typename CommonDataT>
			bool _dropout_has_original_activations(const CommonDataT& CD) const noexcept {
				return (bDropoutWorksAtEvaluationToo || CD.is_training_mode()) && bDropout();
			}

			const realmtxdef_t& _dropout_get_original_activations()const noexcept {
				NNTL_ASSERT(bDropout());
				NNTL_ASSERT(!m_origActivations.empty() && !m_origActivations.emulatesBiases());
#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
				NNTL_ASSERT(m_origActivations.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK
				return m_origActivations;
			}

			//this function restores zeros (dropped out values) and a proper scaling of dL/dA as well as undoes the scaling
			//of activation values
			// 		template<typename iMathT, typename iInspectT>
			// 		void _dropout_restoreScaling(realmtx_t& dLdA, realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {
			template<typename CommonDataT>
			void _dropout_restoreScaling(realmtx_t& dLdA, realmtx_t& activations, const CommonDataT& CD)noexcept {
				NNTL_ASSERT(bDropout() && CD.is_training_mode());
				NNTL_ASSERT(m_dropoutMask.size() == dLdA.size());
				NNTL_ASSERT(m_dropoutMask.size() == m_origActivations.size());

				auto& _iI = CD.iInspect();
				_iI.bprop_preCancelDropout(dLdA, activations, m_dropoutPercentActive);

				CD.iMath().evMul_ip(dLdA, m_dropoutMask);
				_dropout_restoreActivations(activations);

				_iI.bprop_postCancelDropout(dLdA, activations);
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

	public:
		//this flag means that the dropout algorithm doesn't change the activation value if it is zero.
		// for example, it is the case of classical dropout (it drops values to zeros), but not the case of AlphaDropout
		static constexpr bool bDropoutIsZeroStable = true;

	protected:
		~Dropout()noexcept {}
		Dropout()noexcept : _base_class_t() {}
		
		template<typename CommonDataT>
		void _dropout_apply(realmtx_t& activations, const CommonDataT& CD) noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(real_t(0.) < m_dropoutPercentActive && m_dropoutPercentActive < real_t(1.));

			if (CD.is_training_mode()) {
				//must make dropoutMask and apply it
				NNTL_ASSERT(m_dropoutMask.size() == activations.size_no_bias());
				NNTL_ASSERT(m_dropoutMask.size() == m_origActivations.size());

				_dropout_saveActivations(activations);
				CD.iRng().gen_matrix_norm(m_dropoutMask);

				auto& _iI = CD.iInspect();
				_iI.fprop_preDropout(activations, m_dropoutPercentActive, m_dropoutMask);

				CD.iMath().make_dropout(activations, m_dropoutPercentActive, m_dropoutMask);

				_iI.fprop_postDropout(activations, m_dropoutMask);
			}
		}
	};

}