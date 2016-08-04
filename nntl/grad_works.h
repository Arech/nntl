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

#include <bitset>

namespace nntl {

	namespace _impl {

		template<typename i_math_t_>
		struct grad_works_init {
			typedef i_math_t_ i_math_t;
			//typedef math_types::real_ty real_t;
			typedef typename i_math_t::real_t real_t;
			typedef math::simple_matrix<real_t> realmtx_t;
			typedef typename realmtx_t::mtx_size_t mtx_size_t;

			static_assert(std::is_base_of<math::_i_math<real_t>, i_math_t>::value, "i_math_t type should be derived from _i_math");
			
			i_math_t* pMath;
			mtx_size_t weightsSize;

			grad_works_init(i_math_t* pM, const mtx_size_t& ws)noexcept:pMath(pM), weightsSize(ws) {}
		};

		//////////////////////////////////////////////////////////////////////////
		// numeric stabilizer eps value for double and float calculations
		template <typename real_t> struct NUM_STAB_EPS{};
		template<> struct NUM_STAB_EPS<float> { static constexpr double value = 1e-5; };
		template<> struct NUM_STAB_EPS<double> { static constexpr double value = 1e-9; };

	}

	struct _i_grad_works {
		typedef math_types::real_ty real_t;
		typedef math::simple_matrix<real_t> realmtx_t;
		typedef math::simple_matrix_deformable<real_t> realmtxdef_t;
		typedef typename realmtx_t::value_type real_t;
		typedef typename realmtx_t::vec_len_t vec_len_t;
		typedef typename realmtx_t::numel_cnt_t numel_cnt_t;
		typedef typename realmtx_t::mtx_size_t mtx_size_t;

		template<typename grad_init_t>
		nntl_interface bool init(const grad_init_t& ind)noexcept;
		nntl_interface void deinit()noexcept;

		nntl_interface auto set_learning_rate(const real_t learningRate)noexcept;
		nntl_interface const real_t learning_rate()const noexcept;

		nntl_interface void pre_training_fprop(realmtx_t& weights)noexcept;

		//dLdW can have any values on output (use it for temporary calculations if needed)
		nntl_interface void apply_grad(realmtxdef_t& weights, realmtxdef_t& dLdW)noexcept;

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		nntl_interface real_t lossAddendum(const realmtxdef_t& weights)const noexcept;

		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		nntl_interface bool hasLossAddendum()const noexcept;
	};

	struct ILR {
		typedef math_types::real_ty real_t;

		real_t mulDecr, mulIncr, capLow, capHigh;

		ILR()noexcept:mulDecr(real_t(0.0)), mulIncr(real_t(0.0)), capLow(real_t(0.0)), capHigh(real_t(0.0)) {}
		ILR(real_t decr, real_t incr, real_t cLow, real_t cHigh)noexcept:mulDecr(decr), mulIncr(incr), capLow(cLow), capHigh(cHigh) {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
		}
		void set(real_t decr, real_t incr, real_t cLow, real_t cHigh)noexcept {
			NNTL_ASSERT((decr > 0 && decr < 1 && incr > 1 && cHigh > cLow && cLow > 0) || (decr == 0 && incr == 0 && cHigh == 0 && cLow == 0));
			mulDecr = decr;
			mulIncr = incr;
			capLow = cLow;
			capHigh = cHigh;
		}
		void clear()noexcept { set(real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)); }
		const bool bUseMe()const noexcept { return mulDecr > real_t(0.0); }

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
		enum AlgFlags {
			f_FirstRun = 0,//flag to find out first call to apply_grad() after init()

			f_UseMomentum,
			f_UseNesterovMomentum,//Flag to turn on Nesterov Momentum, aka Nesterov Accelerated Gradient method.
			// The definition is (Sutskever, Martens et al. "On the importance of initialization and momentum in deep learning",2013):
			//		vW(t+1) = momentum*vW(t) - scaling*grad_Loss( W(t)+momentum*vW(t))
			//		W(t+1)  = W(t) + vW(t+1)
			// We'll change the sign of vW (for the compability with other code) and reorganize operations in the following way
			// (1)  vW`(t+1)= momentum*vW(t)
			// (2)  W`(t+1) = W(t) - momentum*vW(t)
			//				= W(t) - vW`(t+1)
			// (3)  vW(t+1) = momentum*vW(t) + scaling*grad_Loss( W(t)-momentum*vW(t))
			//				= vW`(t+1) + scaling*grad_Loss( W`(t+1) )
			// (4)  W(t+1)  = W(t) - vW(t+1) 
			//				= W(t) - vW`(t+1) - scaling*grad_Loss( W`(t+1) )
			//				= W`(t+1) - scaling * grad_Loss( W`(t+1) )
			// Steps (1)-(2) done during pre_training_fprop(), steps (3)-(4) - during apply_grad()

			f_UseILR,
			f_ApplyILRToMomentumVelocity,//Geoffrey Hinton said for momentum method, that it's good to calculate
			// individual learning rates based on agreement in signs of accumulated momentum velocity and current gradient value.
			// However, this may lead to vanishing gradient gains and very small gradient value, when accumulated momentum
			// velocity was pretty big. It would require significantly more time to decrease and reverse the velocity with
			// a very small gradient. It may be good sometimes, but sometimes for some data it may be bad
			// therefore it may be beneficial to calculate IRL based on agreement in signs of current and previous gradient
			// (like in "no momentum" version)

			f_UseMaxNorm,
			f_MaxNormRegIgnoreBias,//make max-norm regularizer to ignore bias weights

			f_UseL1,
			f_L1RegIgnoreBias,

			f_UseL2,
			f_L2RegIgnoreBias,

			f_LAST_Total
		};

		iMath_t* m_pMath;

		real_t m_learningRate;

		real_t m_momentum;
		real_t m_emaDecay;
		real_t m_numericStabilizerEps;
		real_t m_maxWeightVecNorm;//coefficient of max-norm regularization ||W||2 <= c (see "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".2014)
		//hint: weights initialized from uniform distribution [-b,b]. It's second raw momentum is b^2/3, so the mean norm should
		//be about <row_vector_length>*b^2/3

		real_t m_actualL2;//L2 (weight decay) regularizer coefficient
		real_t m_actualL1;//L1 regularizer coefficient

		GradType m_type;
		
		std::bitset<f_LAST_Total> m_flags;

		ILR m_ILR;

		realmtx_t m_rmsF, m_rmsG, m_Vw, m_ILRGain, m_prevdLdW;

		real_t m_L2;//L2 (weight decay) regularizer coefficient
		real_t m_L1;//L1 regularizer coefficient

		//////////////////////////////////////////////////////////////////////////
		// some static constants to make code consistent
		static constexpr bool defApplyILR2MomentumVelocity = true;
		static constexpr bool defRegularizersIgnoresBiasWeights = true;


		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.

			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & m_ILR;//dont serialize as struct for ease of use in matlab
				ar & NNTL_SERIALIZATION_NVP(m_learningRate);
				ar & NNTL_SERIALIZATION_NVP(m_momentum);
				ar & NNTL_SERIALIZATION_NVP(m_emaDecay);
				ar & NNTL_SERIALIZATION_NVP(m_numericStabilizerEps);
				ar & NNTL_SERIALIZATION_NVP(m_maxWeightVecNorm);
				ar & NNTL_SERIALIZATION_NVP(m_L2);
				ar & NNTL_SERIALIZATION_NVP(m_L1);
				ar & NNTL_SERIALIZATION_NVP(m_type);
				ar & NNTL_SERIALIZATION_NVP(m_flags);
			}

			if (utils::binary_option<true>(ar, serialization::serialize_grad_works_state)) {
				if (m_type == RMSProp_Hinton || m_type == RMSProp_Graves || m_type == ModProp) ar & NNTL_SERIALIZATION_NVP(m_rmsF);
				if (m_type == RMSProp_Graves) ar & NNTL_SERIALIZATION_NVP(m_rmsG);

				if (use_momentums()) ar & NNTL_SERIALIZATION_NVP(m_Vw);
				if (use_individual_learning_rates()) {
					ar & NNTL_SERIALIZATION_NVP(m_ILRGain);
					if (!use_momentums() | !m_flags[f_ApplyILRToMomentumVelocity]) ar & NNTL_SERIALIZATION_NVP(m_prevdLdW);
				}
			}
		}

	protected:
		void _flags_default()noexcept {
			m_flags.set(f_FirstRun).set(f_UseNesterovMomentum).set(f_ApplyILRToMomentumVelocity);
		}

	public:
		~grad_works()noexcept {}

		//!! copy constructor not needed
		grad_works(const grad_works& other)noexcept = delete;
		//!!assignment is not needed
		grad_works& operator=(const grad_works& rhs) noexcept = delete;

		grad_works(const real_t lr) noexcept : m_pMath(nullptr), m_momentum(0.0), m_emaDecay(0.9),
			m_numericStabilizerEps(_impl::NUM_STAB_EPS<real_t>::value), m_maxWeightVecNorm(0.0)
			, m_L1(0.0), m_L2(0.0), m_actualL1(0.0), m_actualL2(0.0),
			m_type(ClassicalConstant)
		{
			set_learning_rate(lr);
			_flags_default();
		}

		//template<typename grad_init_t>
		bool init(const init_struct_t& ind)noexcept {
			//TODO: there must be some flag that prevents resetting of the data state between distinct calls to nnet.train()
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
				if ((!use_momentums() | !m_flags[f_ApplyILRToMomentumVelocity]) && !m_prevdLdW.resize(ind.weightsSize))return false;
				m_ILRGain.ones();
			}

			m_pMath = ind.pMath;
			m_flags.set(f_FirstRun);
			return true;
		}

		void deinit() noexcept {
			m_pMath = nullptr;
			
			//TODO: current cleanup code is not compatible with multiple nnet::train() calls

			m_Vw.clear();
			m_rmsF.clear();
			m_rmsG.clear();
			m_ILRGain.clear();
			m_prevdLdW.clear();
			//m_ILR.clear();//we shouldn't clear this variable, as it contains only settings but not a run-time data
			//_flags_default();//same for flags
		}

		void pre_training_fprop(realmtx_t& weights) noexcept {
			if (use_nesterov_momentum()) {
				// (1)  vW`(t+1)= momentum*vW(t)
				// (2)  W`(t+1) = W(t) - momentum*vW(t)
				//				= W(t) - vW`(t+1)
				m_pMath->evMulC_ip_Sub_ip(m_Vw, m_momentum, weights);
			}
		}

		void apply_grad(realmtxdef_t& weights, realmtxdef_t& dLdW) noexcept {
			NNTL_ASSERT(m_pMath);
			NNTL_ASSERT(dLdW.size() == weights.size());

			const bool bFirstRun = m_flags[f_FirstRun];
			m_flags.reset(f_FirstRun);

			//applying L1-L2 penalties
			// BTW: because we're working with batches that averages dL/dW over many data samples and, possibly, over many 
			// individual dropout masks, it is highly unlikely that any element of dLdW could have a value of zero
			// (zero dL/dW means that we've found an optimal value of weight, that minimizes/maximizes the loss function).
			// Therefore it is safe to assume, that weights tuning process is still incomplete here and we should apply other
			// weights changing mechanisms, such as L1 or L2 regularization (we shouldn't do it if dL/dW for a weight is zero,
			// because it'll drive the weight's value away from its optimal state).
			// 
			if (use_L1_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L1RegIgnoreBias];
				if (bIgnoreBiases) { dLdW.hide_last_col(); weights.hide_last_col(); }
				m_pMath->evAddScaledSign_ip(dLdW, m_actualL1, weights);
				if (bIgnoreBiases) { dLdW.restore_last_col(); weights.restore_last_col(); }
			}

			if (use_L2_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L2RegIgnoreBias];
				if (bIgnoreBiases) { dLdW.hide_last_col(); weights.hide_last_col(); }
				m_pMath->evAddScaled_ip(dLdW, m_actualL2, weights);
				if (bIgnoreBiases) { dLdW.restore_last_col(); weights.restore_last_col(); }
			}
			
			switch (m_type) {
			case ClassicalConstant:
				m_pMath->evMulC_ip(dLdW, m_learningRate);
				break;

			case RMSProp_Hinton:
				if (bFirstRun) {
					m_pMath->evSquare(m_rmsF, dLdW);
					m_pMath->evMulC_ip(dLdW, m_learningRate);
				} else m_pMath->RMSProp_Hinton(dLdW, m_rmsF, m_learningRate, m_emaDecay, m_numericStabilizerEps);
				break;

			case RMSProp_Graves:
				if (bFirstRun) {
					m_pMath->evSquare(m_rmsF, dLdW);
					dLdW.cloneTo(m_rmsG);
					m_pMath->evMulC_ip(dLdW, m_learningRate);
				} else m_pMath->RMSProp_Graves(dLdW, m_rmsF, m_rmsG, m_learningRate, m_emaDecay, m_numericStabilizerEps);
				break;

			case RProp:
				m_pMath->RProp(dLdW, m_learningRate);
				break;

			case ModProp:
				if (bFirstRun) {
					m_pMath->evAbs(m_rmsF, dLdW);
					m_pMath->evMulC_ip(dLdW, m_learningRate);
				} else m_pMath->ModProp(dLdW, m_rmsF, m_learningRate, m_emaDecay, m_numericStabilizerEps);
				break;

			default:
				NNTL_ASSERT(!"WTF??");
				STDCOUTL("*** " << NNTL_FUNCTION << ": Wrong type of optimizer specified!");
				abort();
			}

			const auto bUseMomentums = use_momentums();
			if (use_individual_learning_rates()) {
				const auto bUseVelocity = bUseMomentums & m_flags[f_ApplyILRToMomentumVelocity];
				if (!bFirstRun) {
					m_pMath->apply_ILR(dLdW, bUseVelocity ? m_Vw : m_prevdLdW, m_ILRGain, m_ILR.mulDecr, m_ILR.mulIncr, m_ILR.capLow, m_ILR.capHigh);
				}
				if (!bUseVelocity) {
					NNTL_ASSERT(dLdW.size() == m_prevdLdW.size());
					dLdW.cloneTo(m_prevdLdW);
				}
			}

			bool bApplydLdW2Weights = true;
			if (bUseMomentums) {
				NNTL_ASSERT(m_Vw.size() == dLdW.size());
				if (m_flags[f_UseNesterovMomentum]) {
					// (3)  vW(t+1) = momentum*vW(t) + scaling*grad_Loss( W(t)-momentum*vW(t))
					//				= vW`(t+1) + scaling*grad_Loss( W`(t+1) )
					m_pMath->evAdd_ip(m_Vw, dLdW);
					// (4)  W(t+1)  = W(t) - vW(t+1) 
					//				= W(t) - vW`(t+1) - scaling*grad_Loss( W`(t+1) )
					//				= W`(t+1) - scaling * grad_Loss( W`(t+1) )
					//bApplydLdW2Weights=true;
				} else {
					//Vw = momentum.*Vw + dW
					m_pMath->apply_momentum(m_Vw, m_momentum, dLdW);
					m_pMath->evSub_ip(weights, m_Vw);
					bApplydLdW2Weights = false;
				}
			}

			if (bApplydLdW2Weights) m_pMath->evSub_ip(weights, dLdW);

			if (use_max_norm_regularization()) {
				const bool bIgnoreBiases = m_flags[f_MaxNormRegIgnoreBias];
				if (bIgnoreBiases) weights.hide_last_col();
				m_pMath->mCheck_normalize_rows(weights, m_maxWeightVecNorm);
				if (bIgnoreBiases) weights.restore_last_col();
			}
		}

		//////////////////////////////////////////////////////////////////////////

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum(const realmtxdef_t& weights)const noexcept {
			real_t ret(0.0);

			//the only modification of weights we may use is stripping/restoring last (bias) column,
			//which is in fact not a modification from outside POV
			realmtxdef_t& _W = *(const_cast<realmtxdef_t*>(&weights));

			if (use_L1_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L1RegIgnoreBias];
				if (bIgnoreBiases) _W.hide_last_col();
				ret += m_actualL1 * m_pMath->vSumAbs(_W);
				if (bIgnoreBiases) _W.restore_last_col();
			}

			if (use_L2_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L2RegIgnoreBias];
				if (bIgnoreBiases) _W.hide_last_col();
				ret += m_actualL2*real_t(.5) * m_pMath->vSumSquares(_W);
				if (bIgnoreBiases) _W.restore_last_col();
			}

			return ret;
		}
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return use_L1_regularization() | use_L2_regularization(); }

		//////////////////////////////////////////////////////////////////////////

		self_t& set_learning_rate(const real_t learningRate)noexcept {
			m_learningRate = learningRate;
			m_actualL1 = math::sign(m_learningRate)*m_L1;
			m_actualL2 = math::sign(m_learningRate)*m_L2;
			return *this;
		}
		const real_t learning_rate()const noexcept { return m_learningRate; }

		self_t& set_ILR(real_t decr, real_t incr, real_t capLow, real_t capHigh) noexcept {
			m_ILR.set(decr, incr, capLow, capHigh);
			m_flags[f_UseILR] = m_ILR.bUseMe();
			return *this;
		}
		self_t& set_ILR(const ILR& ilr) noexcept {
			m_ILR = ilr;
			m_flags[f_UseILR] = m_ILR.bUseMe();
			return *this;
		}
		self_t& set_momentum(real_t m, bool bApplyILRToMomentumVelocity = defApplyILR2MomentumVelocity)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			m_flags[f_UseMomentum] = m_momentum > real_t(0.0);
			m_flags.reset(f_UseNesterovMomentum);
			m_flags[f_ApplyILRToMomentumVelocity] = bApplyILRToMomentumVelocity;
			return *this;
		}
		self_t& set_nesterov_momentum(real_t m, bool bApplyILRToMomentumVelocity = defApplyILR2MomentumVelocity)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			m_flags[f_UseMomentum] = m_momentum > real_t(0.0);
			m_flags.set(f_UseNesterovMomentum);
			m_flags[f_ApplyILRToMomentumVelocity] = bApplyILRToMomentumVelocity;
			return *this;
		}
		self_t& set_ema_decay(real_t c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_emaDecay = c;
			return *this;
		}
		self_t& set_numeric_stabilizer(real_t n)noexcept {
			NNTL_ASSERT(n >= 0 && n < real_t(.1));
			m_numericStabilizerEps = n;
			return *this;
		}
		self_t& set_type(GradType gt)noexcept {
			m_type = gt;
			return *this;
		}

		self_t& set_max_norm(const real_t mn, const bool bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			NNTL_ASSERT(mn >= real_t(0.0));
			m_maxWeightVecNorm = mn;
			m_flags[f_UseMaxNorm] = m_maxWeightVecNorm > real_t(0.0);
			m_flags[f_MaxNormRegIgnoreBias] = bIgnoreBiasWeights;
			return *this;
		}
		//L1 is good for sparse signals
		self_t& set_L1(real_t l1, const bool bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			NNTL_ASSERT(l1 >= real_t(0.0));
			m_L1 = l1;
			m_actualL1 = math::sign(m_learningRate)*m_L1;
			m_flags[f_UseL1] = m_L1 > real_t(0.0);
			m_flags[f_L1RegIgnoreBias] = bIgnoreBiasWeights;
			return *this;
		}
		//L2 is just good)
		self_t& set_L2(real_t l2, const bool bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			NNTL_ASSERT(l2 >= real_t(0.0));
			m_L2 = l2;
			m_actualL2 = math::sign(m_learningRate)*m_L2;
			m_flags[f_UseL2] = m_L2 > real_t(0.0);
			m_flags[f_L2RegIgnoreBias] = bIgnoreBiasWeights;
			return *this;
		}

		const bool use_momentums()const noexcept { return m_flags[f_UseMomentum]; } // m_momentum > real_t(0.0);
		const bool nesterov_momentum()const noexcept { return m_flags[f_UseNesterovMomentum]; }
		const bool use_nesterov_momentum()const noexcept { return m_flags[f_UseMomentum] & m_flags[f_UseNesterovMomentum]; }
		const bool use_classical_momentum()const noexcept { return m_flags[f_UseMomentum] & (!m_flags[f_UseNesterovMomentum]); }

		const bool use_individual_learning_rates()const noexcept { return m_flags[f_UseILR]; }  // m_ILR.bUseMe(); }

		const bool use_max_norm_regularization()const noexcept { return m_flags[f_UseMaxNorm]; } // m_maxWeightVecNorm > real_t(0.0); }
		const bool use_L1_regularization()const noexcept { return m_flags[f_UseL1]; }
		const bool use_L2_regularization()const noexcept { return m_flags[f_UseL2]; }

	protected:

	};

}