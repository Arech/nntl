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

namespace nntl {

	namespace _impl {

		template<typename i_math_t_>
		struct grad_works_init {
			typedef i_math_t_ i_math_t;
			static_assert(std::is_base_of<math::_i_math, i_math_t>::value, "i_math_t type should be derived from _i_math");

			typedef math_types::floatmtx_ty floatmtx_t;
			typedef floatmtx_t::mtx_size_t mtx_size_t;
			
			i_math_t* pMath;
			mtx_size_t weightsSize;

			grad_works_init(i_math_t* pM, const mtx_size_t& ws)noexcept:pMath(pM), weightsSize(ws) {}
		};
	}

	struct _i_grad_works {
		typedef math_types::floatmtx_ty floatmtx_t;
		typedef math_types::floatmtxdef_ty floatmtxdef_t;
		typedef floatmtx_t::value_type float_t_;
		typedef floatmtx_t::vec_len_t vec_len_t;
		typedef floatmtx_t::numel_cnt_t numel_cnt_t;
		typedef floatmtx_t::mtx_size_t mtx_size_t;

		template<typename grad_init_t>
		nntl_interface bool init(const grad_init_t& ind)noexcept;
		nntl_interface void deinit()noexcept;

		//should the learning rate be applied to dLdW before call to apply_grad()
		nntl_interface const bool pre_apply_learning_rate()const noexcept;

		nntl_interface void pre_training_fprop(floatmtx_t& weights)noexcept;

		//dLdW can have any values on output (use it for temporary calculations if needed)
		nntl_interface void apply_grad(floatmtxdef_t& weights, floatmtx_t& dLdW, float_t_ learningRate)noexcept;
	};

	struct ILR {
		typedef math_types::float_ty float_t_;

		float_t_ mulDecr, mulIncr, capLow, capHigh;

		ILR()noexcept:mulDecr(float_t_(0.0)), mulIncr(float_t_(0.0)), capLow(float_t_(0.0)), capHigh(float_t_(0.0)) {}
		ILR(float_t_ decr, float_t_ incr, float_t_ cLow, float_t_ cHigh)noexcept:mulDecr(decr), mulIncr(incr), capLow(cLow), capHigh(cHigh) {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
		}
		void set(float_t_ decr, float_t_ incr, float_t_ cLow, float_t_ cHigh)noexcept {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
			mulDecr = decr;
			mulIncr = incr;
			capLow = cLow;
			capHigh = cHigh;
		}
		void clear()noexcept { set(float_t_(0.0), float_t_(0.0), float_t_(0.0), float_t_(0.0)); }
		const bool bUseMe()const noexcept { return mulDecr > float_t_(0.0); }
	};


	// this class gathers all code about gradient application
	template<typename i_math_t>
	class grad_works : public _i_grad_works {
	public:
		typedef grad_works self_t;
		typedef i_math_t iMath_t;
		typedef _impl::grad_works_init<iMath_t> init_struct_t;

		enum GradType {
			ClassicalConstant,//classical method, just applying learning rate (done during bprop earlier) to dLdW to update weights
			RMSProp_Hinton,//RMSProp as described by prof.Hinton in his lecture #6
			RMSProp_Graves,//RMSProp modification by Alex Graves “Generating Sequences With Recurrent Neural Networks” (2013), equations (38)–(45)
				//"Graves’ implementation in particular seems to have introduced
				//the Gi terms into the RMS computation; these terms appear to
				//act as a sort of momentum for the RMS values." (from http://theanopt.readthedocs.org/en/latest/generated/theanopt.adaptive.RMSProp.html#theanopt.adaptive.RMSProp)
			RProp,//using only the sign of gradient as direction measure
			ModProp,//My own update to RMSProp (may be someone also invented it, have no idea)
			// It's like RMSProp, but divide dW by ema( abs(dW) ), instead of sqrt(ema(dW ^ 2)).
			// Works faster and...sometimes helps to learn weights when no other techniques works... Don't know why, may be it's data-related.
		};

	protected:
		iMath_t* m_pMath;
		float_t_ m_momentum;
		float_t_ m_emaDecay;
		float_t_ m_numericStabilizerEps;
		float_t_ m_maxWeightVecNorm;//coefficient of max-norm regularization ||W||2 <= c (see "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".2014)
		//hint: weights initialized from uniform distribution [-b,b]. It's second raw momentum is b^2/3, so the mean norm should
		//be about <row_vector_length>*b^2/3

		GradType m_type;
		bool m_bFirstRun; //flag to find out first call to apply_grad() after init()
		bool m_bNesterovMomentum;
		bool m_bApplyILRToMomentumVelocity;//Geoffrey Hinton said for momentum method, that it's good to calculate
		// individual learning rates based on agreement in signs of accumulated momentum velocity and current gradient value.
		// However, this may lead to vanishing gradient gains and very small gradient value, when accumulated momentum
		// velocity was pretty big. It would require significantly more time to decrease and reverse the velocity with
		// a very small gradient. It may be good sometimes, but sometimes for some data it may be bad
		// therefore it may be beneficial to calculate IRL based on agreement in signs of current and previous gradient
		// (like in "no momentum" version)
		bool m_bMaxWeightVecNormIgnoreBias;

		ILR m_ILR;

		floatmtx_t m_rmsF, m_rmsG, m_Vw, m_ILRGain, m_prevdLdW;

	public:
		~grad_works()noexcept {}

		//!! copy constructor not needed
		grad_works(const grad_works& other)noexcept = delete;
		//!!assignment is not needed
		grad_works& operator=(const grad_works& rhs) noexcept = delete;

		grad_works() noexcept : m_pMath(nullptr),m_momentum(0.0), m_bNesterovMomentum(true), m_emaDecay(0.9),
			m_numericStabilizerEps(.00001), m_maxWeightVecNorm(0.0),
			m_type(ClassicalConstant), m_bFirstRun(true), m_bApplyILRToMomentumVelocity(true), m_bMaxWeightVecNormIgnoreBias(false)
		{}

		//template<typename grad_init_t>
		bool init(const init_struct_t& ind)noexcept {
			//static_assert(std::is_base_of<_impl::grad_works_init, grad_init_t>::value, "Expecting here something like _impl::grad_works_init");

			//TODO: there must be some flag that prevents resetting of state data between distinct calls to nnet.train()
			//(which causes init/deinit cycle)

			if (use_momentums()) {
				if (!m_Vw.resize(ind.weightsSize)) return false;
				m_Vw.zeros();
			}

			if (m_type == RMSProp_Hinton || m_type == RMSProp_Graves || m_type == ModProp) {
				if (!m_rmsF.resize(ind.weightsSize))return false;
			}

			if (m_type == RMSProp_Graves) {
				if (!m_rmsG.resize(ind.weightsSize))return false;
			}

			if (use_individual_learning_rates()) {
				if (!m_ILRGain.resize(ind.weightsSize))return false;
				if ( (!use_momentums() || !m_bApplyILRToMomentumVelocity) && !m_prevdLdW.resize(ind.weightsSize))return false;
				m_ILRGain.ones();
			}

			m_pMath = ind.pMath;
			m_bFirstRun = true;
			return true;
		}

		void deinit() noexcept {
			m_pMath = nullptr;
			m_Vw.clear();
			m_rmsF.clear();
			m_rmsG.clear();
			m_ILRGain.clear();
			m_prevdLdW.clear();
			m_ILR.clear();
		}

		void pre_training_fprop(floatmtx_t& weights) noexcept {
			if (use_momentums() && m_bNesterovMomentum) {
				m_pMath->evSub_ip(weights, m_Vw);
			}
		}
		
		void apply_grad(floatmtxdef_t& weights, floatmtx_t& dLdW, float_t_ learningRate) noexcept {
			NNTL_ASSERT(m_pMath);
			NNTL_ASSERT(dLdW.size() == weights.size());

			const auto bFirstRun = m_bFirstRun;
			m_bFirstRun = false;

			//TODO: apply weight penalty L2 to dLdW

			switch (m_type) {
			case ClassicalConstant:
				m_pMath->evMulC_ip(dLdW, learningRate);
				break;

			case RMSProp_Hinton:
				if (bFirstRun) {
					m_pMath->evSquare(m_rmsF, dLdW);
					m_pMath->evMulC_ip(dLdW, learningRate);
				} else m_pMath->RMSProp_Hinton(dLdW, m_rmsF, learningRate, m_emaDecay, m_numericStabilizerEps);
				break;

			case RMSProp_Graves:
				if (bFirstRun) {
					m_pMath->evSquare(m_rmsF, dLdW);
					dLdW.cloneTo(m_rmsG);
					m_pMath->evMulC_ip(dLdW, learningRate);
				} else m_pMath->RMSProp_Graves(dLdW, m_rmsF, m_rmsG, learningRate, m_emaDecay, m_numericStabilizerEps);
				break;

			case RProp:
				m_pMath->RProp(dLdW,learningRate);
				break;

			case ModProp:
				if (bFirstRun) {
					m_pMath->evAbs(m_rmsF, dLdW);
					m_pMath->evMulC_ip(dLdW, learningRate);
				} else m_pMath->ModProp(dLdW, m_rmsF, learningRate, m_emaDecay, m_numericStabilizerEps);
				break;

			default:
				NNTL_ASSERT(!"WTF??");
				abort();
			}

			const auto bUseMomentums = use_momentums();
			if (use_individual_learning_rates()) {
				const auto bUseVelocity = bUseMomentums && m_bApplyILRToMomentumVelocity;
				if (!bFirstRun) {
					m_pMath->apply_ILR(dLdW, bUseVelocity ? m_Vw : m_prevdLdW, m_ILRGain, m_ILR.mulDecr, m_ILR.mulIncr, m_ILR.capLow, m_ILR.capHigh);
				}
				if(!bUseVelocity) dLdW.cloneTo(m_prevdLdW);
			}

			if (bUseMomentums) {
				NNTL_ASSERT(m_Vw.size() == dLdW.size());
				m_pMath->apply_momentum(m_Vw, m_momentum, dLdW);
				if (!m_bNesterovMomentum) m_Vw.cloneTo(dLdW);
			}

			m_pMath->evSub_ip(weights, dLdW);

			if (use_max_norm_regularization()) {
				if (m_bMaxWeightVecNormIgnoreBias) weights.hide_last_col();
				m_pMath->mCheck_normalize_rows(weights, m_maxWeightVecNorm);
				if (m_bMaxWeightVecNormIgnoreBias) weights.restore_last_col();
			}
		}
		
		//////////////////////////////////////////////////////////////////////////

		self_t& set_ILR(float_t_ decr, float_t_ incr, float_t_ capLow, float_t_ capHigh) noexcept {
			m_ILR.set(decr, incr, capLow, capHigh);
			return *this;
		}
		self_t& set_ILR(const ILR& ilr) noexcept {
			m_ILR = ilr;
			return *this;
		}
		self_t& set_momentum(float_t_ m, bool bApplyILRToMomentumVelocity = true)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			m_bNesterovMomentum = false;
			m_bApplyILRToMomentumVelocity = bApplyILRToMomentumVelocity;
			return *this;
		}
		self_t& set_nesterov_momentum(float_t_ m, bool bApplyILRToMomentumVelocity = true)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			m_bNesterovMomentum = true;
			m_bApplyILRToMomentumVelocity = bApplyILRToMomentumVelocity;
			return *this;
		}
		self_t& set_ema_decay(float_t_ c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_emaDecay = c;
			return *this;
		}
		self_t& set_numeric_stabilizer(float_t_ n)noexcept {
			NNTL_ASSERT(n >= 0 && n < float_t_(.1));
			m_numericStabilizerEps = n;
			return *this;
		}
		self_t& set_type(GradType gt)noexcept {
			m_type = gt;
			return *this;
		}

		self_t& set_weight_vector_max_norm2(const float_t_ mn, const bool bIgnoreBiasWeights=false)noexcept {
			NNTL_ASSERT(mn >= float_t_(0.0));
			m_maxWeightVecNorm = mn;
			m_bMaxWeightVecNormIgnoreBias = bIgnoreBiasWeights;
			return *this;
		}

		const bool use_max_norm_regularization()const noexcept { return m_maxWeightVecNorm > float_t_(0.0); }
		const bool use_individual_learning_rates()const noexcept { return m_ILR.bUseMe(); }
		const bool use_momentums()const noexcept { return m_momentum > float_t_(0.0); };

	protected:
		
	};

}