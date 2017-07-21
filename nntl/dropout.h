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
		//this class would be used internall to remove Dropout API completely, as well as a base class for all Dropout classes
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
		static constexpr void _dropout_cancelScaling(realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {}

		template<typename iMathT, typename iInspectT>
		static constexpr void _dropout_restoreScaling(realmtx_t& dLdZ, iMathT& iM, iInspectT& _iI)noexcept {}

	public:
		static constexpr bool bDropout() noexcept { return false; }
		static constexpr real_t dropoutPercentActive() noexcept { return real_t(1.); }
		static constexpr void dropoutPercentActive(const real_t dpa) noexcept {
			if (dpa != real_t(1.) || dpa != real_t(0.)) {
				STDCOUTL("Trying to set dropout rate for a non-dropout available class");
			}
		}

	};

	template<typename DropoutT>
	using is_dummy_dropout_class = std::disjunction<
		std::is_same < DropoutT, NoDropout<typename DropoutT::real_t>>
		, std::is_same < DropoutT, _impl::_No_Dropout_at_All<typename DropoutT::real_t>>
	>;

	//////////////////////////////////////////////////////////////////////////
	//this is a wrapper around "classical" dropout technique invented by Jeff Hinton (actually, it's a modification
	// that is called "inverse dropout". It doesn't require special handling during evaluation step)
	template<typename RealT>
	class Dropout : public _impl::_No_Dropout_at_All<RealT> {
	private:
		typedef math::smatrix_deform<real_t> realmtxdef_t;

		//////////////////////////////////////////////////////////////////////////
		//vars
		
		//matrix of dropped out neuron activations, used when 1>m_dropoutPercentActive>0
		realmtxdef_t m_dropoutMask;//<batch_size rows> x <m_neurons_cnt cols> (must not have a bias column)
		real_t m_dropoutPercentActive;//probability of keeping unit active

		//////////////////////////////////////////////////////////////////////////
		//methods
	protected:
		~Dropout()noexcept {}
		Dropout()noexcept : m_dropoutPercentActive(real_t(1.)) {}

		template<class Archive>
		void _dropout_serialize(Archive & ar, const unsigned int version) noexcept{
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_dropoutPercentActive);
			}
			
			if (bDropout() && utils::binary_option<true>(ar, serialization::serialize_dropout_mask)) ar & NNTL_SERIALIZATION_NVP(m_dropoutMask);
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

		//we must undo the scaling of activations done by inverted dropout in order to obtain correct activation values
		template<typename iMathT, typename iInspectT>
		void _dropout_cancelScaling(realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {
			NNTL_ASSERT(bDropout());
			_iI.bprop_preCancelDropout(activations, m_dropoutPercentActive);
			iM.evMulC_ip_Anb(activations, m_dropoutPercentActive);
			_iI.bprop_postCancelDropout(activations);
		}

		//this function restores zeros (dropped out values) and a proper scaling of dLdZ
		template<typename iMathT, typename iInspectT>
		void _dropout_restoreScaling(realmtx_t& dLdZ, iMathT& iM, iInspectT& _iI)noexcept {
			NNTL_ASSERT(bDropout());
			NNTL_ASSERT(m_dropoutMask.size() == dLdZ.size());
			iM.evMul_ip(dLdZ, m_dropoutMask);
		}

	public:
		bool bDropout()const noexcept { return m_dropoutPercentActive < real_t(1.); }

		real_t dropoutPercentActive()const noexcept { return m_dropoutPercentActive; }

		void dropoutPercentActive(const real_t dpa)noexcept {
			NNTL_ASSERT(real_t(0.) <= dpa && dpa <= real_t(1.));
			m_dropoutPercentActive = (dpa <= real_t(+0.) || dpa > real_t(1.)) ? real_t(1.) : dpa;
			//m_bDoDropout = dpa < real_t(1.);
// 			if (!get_self()._check_init_dropout()) {
// 				NNTL_ASSERT(!"Failed to init dropout, probably no memory");
// 				abort();
// 			}
		}

	};



	//////////////////////////////////////////////////////////////////////////
	template<typename RealT>
	struct AlphaDropout : public math::smatrix_td {
	public:
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;

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
		static constexpr void _dropout_cancelScaling(realmtx_t& activations, iMathT& iM, iInspectT& _iI)noexcept {}

		template<typename iMathT, typename iInspectT>
		static constexpr void _dropout_restoreScaling(realmtx_t& dLdZ, iMathT& iM, iInspectT& _iI)noexcept {}

	public:
		static constexpr bool bDropout() noexcept { return false; }
		static constexpr real_t dropoutPercentActive() noexcept { return real_t(1.); }
		static constexpr void dropoutPercentActive(const real_t dpa) noexcept {
			if (dpa != real_t(1.) || dpa != real_t(0.)) {
				STDCOUTL("Trying to set dropout rate for a non-dropout available class");
			}
		}

	};

}
