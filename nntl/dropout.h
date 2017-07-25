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

//This file defines wrappers around different dropout implementations
//Layer classes will be derived from this classes using public inheritance, so code wisely


#include "interface/math/smatrix.h"

namespace nntl {

	namespace _impl {
		//this class would be used internally to remove Dropout API completely, as well as a base class for all Dropout classes
		template<typename RealT>
		struct _No_Dropout_at_All : public math::smatrix_td {
			typedef RealT real_t;
			typedef math::smatrix<real_t> realmtx_t;
		};
	}

	//this class defines a dropout class interface as well as dummy class that removes dropout
	template<typename RealT>
	class NoDropout : public _impl::_No_Dropout_at_All<RealT> {
	protected:
		template<class Archive>
		static constexpr void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {}

		static constexpr bool _dropout_init(const vec_len_t training_batch_size, const neurons_count_t neurons_cnt)noexcept {
			return true;
		}
		static constexpr void _dropout_deinit() noexcept {}
		static constexpr void _dropout_on_batch_size_change(const vec_len_t batchSize) noexcept {}

		template<typename iMathT, typename iRngT, typename iInspectT>
		static constexpr void _dropout_apply(realmtx_t& activations, const bool bTrainingMode
			, iMathT& iM, iRngT& iR, iInspectT& _iI) noexcept {}

		template<typename iMathT, typename iInspectT>
		static constexpr void _dropout_restoreScaling(realmtx_t& dLdZ, realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {}

	public:
		static constexpr bool bDropout() noexcept { return false; }
		static constexpr real_t dropoutPercentActive() noexcept { return real_t(1.); }
		static constexpr void dropoutPercentActive(const real_t dpa) noexcept {
			if (dpa != real_t(1.) || dpa != real_t(0.)) {
				STDCOUTL("Trying to set dropout rate for a non-dropout available class");
			}
		}

		// if it returns a real pointer, then an activation value is considered dropped out iff corresponding mask value is zero
		static constexpr const realmtx_t* _dropout_get_mask() noexcept { return nullptr; }
	};

	template<typename DropoutT>
	using is_dummy_dropout_class = ::std::disjunction<
		::std::is_same < DropoutT, NoDropout<typename DropoutT::real_t>>
		, ::std::is_same < DropoutT, _impl::_No_Dropout_at_All<typename DropoutT::real_t>>
	>;

	//////////////////////////////////////////////////////////////////////////
	namespace _impl {

		template<typename RealT>
		class _dropout_base : public _impl::_No_Dropout_at_All<RealT> {
		public:
			typedef math::smatrix_deform<real_t> realmtxdef_t;

		protected:
			//////////////////////////////////////////////////////////////////////////
			//vars

			//matrix of dropped out neuron activations, used when 1>m_dropoutPercentActive>0
			realmtxdef_t m_dropoutMask;//<batch_size rows> x <m_neurons_cnt cols> (must not have a bias column)
			real_t m_dropoutPercentActive;//probability of keeping unit active

		protected:
			~_dropout_base()noexcept {}
			_dropout_base()noexcept : m_dropoutPercentActive(real_t(1.)){}

			template<class Archive>
			void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {
				if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
					ar & NNTL_SERIALIZATION_NVP(m_dropoutPercentActive);
				}

				if (bDropout() && utils::binary_option<true>(ar, serialization::serialize_dropout_mask))
					ar & NNTL_SERIALIZATION_NVP(m_dropoutMask);
			}

			bool _dropout_init(const vec_len_t training_batch_size, const neurons_count_t neurons_cnt)noexcept {
				//NNTL_ASSERT(get_self().has_common_data());
				if (training_batch_size > 0) {
					//condition means if (there'll be a training session) and (we're going to use dropout)
					NNTL_ASSERT(!m_dropoutMask.emulatesBiases());
					//resize to the biggest possible size during training
					if (!m_dropoutMask.resize(training_batch_size, neurons_cnt)) return false;
				}
				return true;
			}

			void _dropout_deinit()noexcept {
				m_dropoutMask.clear();
			}

			void _dropout_on_batch_size_change(const vec_len_t batchSize)noexcept {
				if (bDropout()) {
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

			const realmtx_t* _dropout_get_mask()const noexcept { return &m_dropoutMask; }
			//real_t _dropout_activations_scaleInverse()const noexcept { return m_dropoutPercentActive; }
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
			} else {
				//only applying dropoutPercentActive -- we don't need to do this since make_dropout() implements so called 
				// "inverse dropout" that doesn't require this step
				//_Math.evMulC_ip_Anb(m_activations, real_t(1.0) - m_dropoutPercentActive);
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
	template<typename RealT, int64_t Alpha1e9 = 0, int64_t Lambda1e9 = 0, int fpMean1e6=0, int fpVar1e6=1000000>
	struct AlphaDropout : public _impl::_dropout_base<RealT> {
	private:
		typedef _impl::_dropout_base<RealT> _base_class_t;

	public:
		static constexpr ext_real_t AlphaExt = Alpha1e9 ? ext_real_t(Alpha1e9) / ext_real_t(1e9) : ext_real_t(1.6732632423543772848170429916717);
		static constexpr ext_real_t LambdaExt = Lambda1e9 ? ext_real_t(Lambda1e9) / ext_real_t(1e9) : ext_real_t(1.0507009873554804934193349852946);

		static constexpr ext_real_t AlphaExt_t_LambdaExt = AlphaExt*LambdaExt;
		static constexpr ext_real_t Neg_AlphaExt_t_LambdaExt = -AlphaExt_t_LambdaExt;

		static constexpr real_t Alpha = real_t(AlphaExt);
		static constexpr real_t Lambda = real_t(LambdaExt);
		static constexpr real_t Alpha_t_Lambda = real_t(AlphaExt*LambdaExt);
		static constexpr real_t Neg_Alpha_t_Lambda = -Alpha_t_Lambda;

		static constexpr ext_real_t FixedPointMeanExt = ext_real_t(fpMean1e6) / ext_real_t(1e6);
		static constexpr ext_real_t FixedPointVarianceExt = ext_real_t(fpVar1e6) / ext_real_t(1e6);

		static constexpr real_t FixedPointMean = real_t(FixedPointMeanExt);
		static constexpr real_t FixedPointVariance = real_t(FixedPointVarianceExt);

	protected:
		realmtxdef_t m_mtxB;
		real_t m_a, m_b, m_mbDropVal;

	protected:
		AlphaDropout()noexcept : _base_class_t() , m_a(real_t(0)), m_b(real_t(0)), m_mbDropVal(real_t(0))
		{}

		template<class Archive>
		void _dropout_serialize(Archive & ar, const unsigned int version) noexcept {
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_a);
				ar & NNTL_SERIALIZATION_NVP(m_b);
			}

			if (bDropout() && utils::binary_option<true>(ar, serialization::serialize_dropout_mask))
				ar & NNTL_SERIALIZATION_NVP(m_mtxB);

			_base_class_t::_dropout_serialize(ar, version);
		}

		bool _dropout_init(const vec_len_t training_batch_size, const neurons_count_t neurons_cnt)noexcept {
			if (!_base_class_t::_dropout_init(training_batch_size, neurons_cnt))  return false;

			if (training_batch_size > 0) {
				//condition means if (there'll be a training session) and (we're going to use dropout)
				NNTL_ASSERT(!m_mtxB.emulatesBiases());
				//resize to the biggest possible size during training
				if (!m_mtxB.resize(training_batch_size, neurons_cnt)) return false;
				
				NNTL_ASSERT(m_mtxB.size() == m_dropoutMask.size());
			}
			return true;
		}

		void _dropout_deinit() noexcept {
			m_mtxB.clear();
			_base_class_t::_dropout_deinit();
		}

		void _dropout_on_batch_size_change(const vec_len_t batchSize) noexcept {
			_base_class_t::_dropout_on_batch_size_change(batchSize);
			if (bDropout()) {
				NNTL_ASSERT(!m_mtxB.empty());
				m_mtxB.deform_rows(batchSize);
				NNTL_ASSERT(m_mtxB.size() == m_dropoutMask.size());
			}
		}

		template<typename iMathT, typename iRngT, typename iInspectT>
		void _dropout_apply(realmtx_t& activations, const bool bTrainingMode, iMathT& iM, iRngT& iR, iInspectT& _iI) noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(real_t(0.) < m_dropoutPercentActive && m_dropoutPercentActive < real_t(1.));

			if (bTrainingMode) {
				//must make dropoutMask and apply it
				NNTL_ASSERT(m_dropoutMask.size() == activations.size_no_bias());
				NNTL_ASSERT(m_mtxB.size() == m_dropoutMask.size());
				NNTL_ASSERT(m_a && m_b && m_mbDropVal);

				iR.gen_matrix_norm(m_dropoutMask);

				_iI.fprop_preDropout(activations, m_dropoutPercentActive, m_dropoutMask);

				iM.make_alphaDropout(activations, m_dropoutPercentActive, m_a, m_b, m_mbDropVal, m_dropoutMask, m_mtxB);

				_iI.fprop_postDropout(activations, m_dropoutMask);
			}
		}
		
		//For the _dropout_restoreScaling() phase we obtain almost original activations by doing A <- (A3-mtxB) ./ a.
		// Dropped out values will have a value of zero. dL/dA scaling is the same as for the inverted dropout:
		// dL/dA = dL/dA .* dropoutMask
		template<typename iMathT, typename iInspectT>
		void _dropout_restoreScaling(realmtx_t& dLdA, realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {
			NNTL_ASSERT(bDropout());		
			NNTL_ASSERT(m_dropoutMask.size() == dLdA.size());
			iM.evMul_ip(dLdA, m_dropoutMask);

			_iI.bprop_preCancelDropout(activations, m_dropoutPercentActive);
			iM.evSubMtxMulC_ip_nb(activations, m_mtxB, real_t(1.)/m_a);
			_iI.bprop_postCancelDropout(activations);
		}

	public:
		void dropoutPercentActive(const real_t dpa)noexcept {
			_base_class_t::dropoutPercentActive(dpa);
			if (bDropout()) {
				//calculating a and b vars
				const ext_real_t dropProb = ext_real_t(1) - m_dropoutPercentActive;
				const ext_real_t amfpm = Neg_AlphaExt_t_LambdaExt - FixedPointMeanExt;

				const ext_real_t aExt = ::std::sqrt(FixedPointVarianceExt / (m_dropoutPercentActive*(dropProb*amfpm*amfpm + FixedPointVarianceExt)));
				NNTL_ASSERT(aExt && !isnan(aExt) && isfinite(aExt));
				m_a = static_cast<real_t>(aExt);
				NNTL_ASSERT(isfinite(m_a));
				
				const ext_real_t bExt = FixedPointMeanExt - m_a*(m_dropoutPercentActive*FixedPointMeanExt + dropProb*Neg_AlphaExt_t_LambdaExt);
				NNTL_ASSERT(bExt && !isnan(bExt) && isfinite(bExt));
				m_b = static_cast<real_t>(bExt);
				NNTL_ASSERT(isfinite(m_b));

				m_mbDropVal = static_cast<real_t>(aExt * Neg_AlphaExt_t_LambdaExt + bExt);
			}
		}
	};

}
