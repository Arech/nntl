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

#include "../utils/mixins.h"

namespace nntl {
namespace GW { //GW namespace is for grad_works mixins and other stuff, that helps to implement gradient processing voodooo things

	template<typename RealT>
	struct ILR_range {
		typedef RealT real_t;

		real_t mulDecr, mulIncr, capLow, capHigh;

		ILR_range()noexcept:mulDecr(real_t(0.0)), mulIncr(real_t(0.0)), capLow(real_t(0.0)), capHigh(real_t(0.0)) {}
		ILR_range(const real_t decr, const real_t incr, const real_t cLow, const real_t cHigh)noexcept : mulDecr(decr), mulIncr(incr), capLow(cLow), capHigh(cHigh) {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
		}
		void set(const real_t decr, const real_t incr, const real_t cLow, const real_t cHigh)noexcept {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
			mulDecr = decr;
			mulIncr = incr;
			capLow = cLow;
			capHigh = cHigh;
		}
		void clear()noexcept { set(real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)); }
		const bool bUseMe()const noexcept {
			NNTL_ASSERT((mulDecr > real_t(0.0) && mulIncr > real_t(0.0) && capLow > real_t(0.0) && capHigh > real_t(0.0))
				|| (mulDecr == real_t(0.0) && mulIncr == real_t(0.0) && capLow == real_t(0.0) && capHigh == real_t(0.0)));
			return mulDecr > real_t(0.0);
		}

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & NNTL_SERIALIZATION_NVP(mulDecr);
			ar & NNTL_SERIALIZATION_NVP(mulIncr);
			ar & NNTL_SERIALIZATION_NVP(capLow);
			ar & NNTL_SERIALIZATION_NVP(capHigh);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// dummy mixin that silently disables ILR
	template<typename _FC, typename RealT, size_t MixinIdx>
	class AILR_dummy : private math::smatrix_td {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;

	public:
		enum OptsList {
			opts_total = 0
		};

	protected:

		//this dummy functions required by the root to be defined
		template<class Archive>
		static constexpr void ILR_serialize(Archive & ar, const unsigned int version) noexcept {}

		static constexpr bool ILR_init(const mtx_size_t& weightsSize) noexcept { return true; }
		static constexpr void ILR_deinit() noexcept {}
		static constexpr void ILR_apply(const bool bFirstRun, realmtx_t& dLdW, const realmtx_t& Vw) noexcept {}

	public:
		//extension functions (that aren't required by a root) probably shouldn't be defined at all.
		// 
		//static constexpr bool use_individual_learning_rates()const noexcept { return false; }
		//static constexpr bool applyILRToMomentum()const noexcept { return false; }
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//Adaptive Individual Learning Rates (or just ILR)
	template<typename _FC, typename RealT, size_t MixinIdx>
	class AILR : private math::smatrix_td {
	private:
		typedef _FC self_t;
		NNTL_METHODS_SELF();
		NNTL_METHODS_MIXIN_OPTIONS(MixinIdx);

		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;

	//public:
	private:
		ILR_range<real_t> m_ILR;

		realmtx_t m_ILRGain, m_prevdLdW;

	public:
		enum OptsList {
			f_UseILR = 0,
			f_ApplyILRToMomentum,//Geoffrey Hinton said for momentum method, that it's good to calculate
				// individual learning rates based on agreement in signs of accumulated momentum velocity and current gradient value.
				// However, this may lead to vanishing gradient gains and very small gradient value, when accumulated momentum
				// velocity was pretty big. It would require significantly more time to decrease and reverse the velocity with
				// a very small gradient. It may be good sometimes, but sometimes for some data it may be bad
				// therefore it may be beneficial to calculate IRL based on agreement in signs of current and previous gradient
				// (like in "no momentum" version)
			opts_total
		};

	protected:

		const bool _ILR_use_momentum()const noexcept {
			return get_self().use_momentums() & get_opt(f_ApplyILRToMomentum);
		}

		template<class Archive>
		void ILR_serialize(Archive & ar, const unsigned int version) {
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & m_ILR;//dont serialize as struct for ease of use in matlab
			}
			if (utils::binary_option<true>(ar, serialization::serialize_grad_works_state)) {
				if (use_individual_learning_rates()) {
					ar & NNTL_SERIALIZATION_NVP(m_ILRGain);
					if (!_ILR_use_momentum()) ar & NNTL_SERIALIZATION_NVP(m_prevdLdW);
				}
			}
		}

		const bool ILR_init(const mtx_size_t& weightsSize) noexcept {
			if (use_individual_learning_rates()) {
				if (!m_ILRGain.resize(weightsSize))return false;
				if (!_ILR_use_momentum() && !m_prevdLdW.resize(weightsSize))return false;
				m_ILRGain.ones();
			}
			return true;
		}
		void ILR_deinit()noexcept {
			m_ILRGain.clear();
			m_prevdLdW.clear();
			//m_ILR.clear();//we shouldn't clear this variable, as it contains only settings but not a run-time data
		}

		void ILR_apply(const bool bFirstRun, realmtx_t& dLdW, const realmtx_t& Vw)noexcept {
			if (use_individual_learning_rates()) {
				const auto bUseVelocity = _ILR_use_momentum();
				NNTL_ASSERT(bUseVelocity || m_prevdLdW.size() == dLdW.size());
				if (!bFirstRun) {
					const auto& prevdLdW = bUseVelocity ? Vw : m_prevdLdW;
					auto& iI = get_self().get_iInspect();
					iI.apply_grad_preILR(dLdW, prevdLdW, m_ILRGain);
					get_self().get_iMath().apply_ILR(dLdW, prevdLdW, m_ILRGain, m_ILR.mulDecr, m_ILR.mulIncr, m_ILR.capLow, m_ILR.capHigh);
					iI.apply_grad_postILR(dLdW, m_ILRGain);
				}
				if (!bUseVelocity) dLdW.cloneTo(m_prevdLdW);
			}
		}

	public:
		const bool use_individual_learning_rates()const noexcept { return get_opt(f_UseILR); }
		const bool applyILRToMomentum()const noexcept { return get_opt(f_ApplyILRToMomentum); }

		self_ref_t set_ILR(const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh) noexcept {
			m_ILR.set(decr, incr, capLow, capHigh);
			set_opt(f_UseILR, m_ILR.bUseMe());
			return get_self();
		}
		self_ref_t set_ILR(const GW::ILR_range<real_t>& ilr) noexcept {
			m_ILR = ilr;
			set_opt(f_UseILR, m_ILR.bUseMe());
			return get_self();
		}
		self_ref_t clear_ILR() noexcept {
			m_ILR.clear();
			set_opt(f_UseILR, false);
			return get_self();
		}
		//static constexpr bool defApplyILR2MomentumVelocity = false;//true value for this setting may prevent setups with ILR,
		//dropout & momentum to learn correctly.
		self_ref_t applyILRToMomentum(const bool b)noexcept {
			set_opt(f_ApplyILRToMomentum, b);
			if (!get_self().use_momentums() | !b) {
				if (m_ILRGain.rows() > 0 && m_ILRGain.cols() > 0 && !m_prevdLdW.resize(m_ILRGain)) {
					//#TODO: correct error return
					STDCOUTL("Failed to resize m_prevdLdW");
					abort();
				}
			}
			return get_self();
		}

	};

}
}