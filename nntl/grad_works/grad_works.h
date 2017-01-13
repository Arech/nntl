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
#include "../common_nn_data.h"
#include "ILR.h"

namespace nntl {

	namespace _impl {
		//////////////////////////////////////////////////////////////////////////
		// numeric stabilizer eps value for double and float calculations
		template <typename real_t> struct NUM_STAB_EPS{};
		template<> struct NUM_STAB_EPS<float> { static constexpr float value = 1e-8f; };
		template<> struct NUM_STAB_EPS<double> { static constexpr double value = 1e-8; };

		//////////////////////////////////////////////////////////////////////////
		//grad_works interface - very unstable at this point / only core stable functions included
		template<typename RealT>
		struct _i_grad_works : public math::smatrix_td {
			typedef RealT real_t;
			typedef math::smatrix<real_t> realmtx_t;
			typedef math::smatrix_deform<real_t> realmtxdef_t;

			template<typename common_data_t>//actually common_data_t is well defined and templated here only for convenience
			nntl_interface bool init(const common_data_t& cd, const mtx_size_t& weightsSize)noexcept;
			nntl_interface void deinit()noexcept;

			nntl_interface auto learning_rate(const real_t learningRate)noexcept;
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
	}

	template<typename InterfacesT>
	class grad_works 
		: public _impl::_i_grad_works<typename InterfacesT::iMath_t::real_t>
		, public _impl::_common_data_consumer<InterfacesT>
	{
	public:
		typedef grad_works self_t;
		typedef _impl::common_nn_data<interfaces_t> common_data_t;

		using _impl::_common_data_consumer<InterfacesT>::real_t;

		//this definition should be local to a grad_works class definition. Other derivations of _impl::_i_grad_works could
		//have other optimizers implemented.
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
			
			Adam,//b1=.9, b2=.999, numStab=1e-8, learningRate=.001; --- remember to set numeric_stabilizer() correctly!!!
			AdaMax,// see Adam - A Method for Stochastic Optimization.1412.6980v8
		};

	protected:
		const bool _optimizerRequiresMatrixA()const noexcept {
			return m_type == RMSProp_Hinton || m_type == RMSProp_Graves || m_type == ModProp || m_type == Adam || m_type == AdaMax;
		}
		const bool _optimizerRequiresMatrixB()const noexcept {
			return m_type == RMSProp_Graves || m_type == Adam || m_type == AdaMax;
		}

	protected:
		enum AlgFlags {
			f_FirstRun = 0,//flag to find out first call to apply_grad() after init()

			f_UseMomentum,
			f_UseNesterovMomentum,//Flag to turn on Nesterov Momentum, aka Nesterov Accelerated Gradient method.
			// The definition is (Sutskever, Martens et al. "On the importance of initialization and momentum in deep learning",2013):
			//		vW(t+1) = momentum*vW(t) - scaling*grad_Loss( W(t)+momentum*vW(t))
			//		W(t+1)  = W(t) + momentum*vW(t) - scaling*grad_Loss( W(t)+momentum*vW(t)) = W(t) + vW(t+1)
			// We need to change the sign of vW (for the compability with other code) and reorganize operations in the following way
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
			f_ApplyILRToMomentum,//Geoffrey Hinton said for momentum method, that it's good to calculate
			// individual learning rates based on agreement in signs of accumulated momentum velocity and current gradient value.
			// However, this may lead to vanishing gradient gains and very small gradient value, when accumulated momentum
			// velocity was pretty big. It would require significantly more time to decrease and reverse the velocity with
			// a very small gradient. It may be good sometimes, but sometimes for some data it may be bad
			// therefore it may be beneficial to calculate IRL based on agreement in signs of current and previous gradient
			// (like in "no momentum" version)

			f_UseMaxNorm,
			f_NormIncludesBias,//if true, the max-norm parameter describes full norm of weight vector

			f_UseL1,
			f_L1RegIgnoreBias,

			f_UseL2,
			f_L2RegIgnoreBias,

			f_LAST_Total
		};
		
		real_t m_learningRate;

		real_t m_momentum;
		real_t m_optBeta1, m_optBeta2;
		real_t m_numericStabilizerEps;
		real_t m_WeightVecNormSqared;//coefficient of max-norm regularization ||W||2 <= c (see "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".2014)
		//hint: weights initialized from uniform distribution [-b,b]. It's second raw momentum is b^2/3, so the mean norm should
		//be about <row_vector_length>*b^2/3

		real_t m_actualL2;//L2 (weight decay) regularizer coefficient
		real_t m_actualL1;//L1 regularizer coefficient

		GradType m_type;
		
		std::bitset<f_LAST_Total> m_flags;

		GW::ILR_range<real_t> m_ILR;

		realmtx_t m_Vw, m_ILRGain, m_prevdLdW;
		realmtx_t m_optMtxA, m_optMtxB;//some optimizers require additional memory.

		real_t m_L2;//L2 (weight decay) regularizer coefficient
		real_t m_L1;//L1 regularizer coefficient

		real_t m_optBeta1t, m_optBeta2t;//storage for coefficients some optimizers (Adam, AdaMax) needed

		//////////////////////////////////////////////////////////////////////////
		// some static constants to make code consistent

		static constexpr bool defRegularizersIgnoresBiasWeights = true;
		static constexpr bool defNormIncludesBias = false;


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
				ar & NNTL_SERIALIZATION_NVP(m_optBeta1);
				ar & NNTL_SERIALIZATION_NVP(m_optBeta2);
				ar & NNTL_SERIALIZATION_NVP(m_numericStabilizerEps);
				ar & NNTL_SERIALIZATION_NVP(m_WeightVecNormSqared);
				ar & NNTL_SERIALIZATION_NVP(m_L2);
				ar & NNTL_SERIALIZATION_NVP(m_L1);
				ar & NNTL_SERIALIZATION_NVP(m_type);
				ar & NNTL_SERIALIZATION_NVP(m_flags);
				ar & NNTL_SERIALIZATION_NVP(m_optBeta1t);
				ar & NNTL_SERIALIZATION_NVP(m_optBeta2t);
			}

			if (utils::binary_option<true>(ar, serialization::serialize_grad_works_state)) {
				if (_optimizerRequiresMatrixA()) ar & NNTL_SERIALIZATION_NVP(m_optMtxA);
				if (_optimizerRequiresMatrixB()) ar & NNTL_SERIALIZATION_NVP(m_optMtxB);

				if (use_momentums()) ar & NNTL_SERIALIZATION_NVP(m_Vw);
				if (use_individual_learning_rates()) {
					ar & NNTL_SERIALIZATION_NVP(m_ILRGain);
					if (!use_momentums() | !m_flags[f_ApplyILRToMomentum]) ar & NNTL_SERIALIZATION_NVP(m_prevdLdW);
				}
			}
		}

	protected:
		void _flags_default()noexcept {
			m_flags.set(f_FirstRun).set(f_UseNesterovMomentum); // .reset(f_ApplyILRToMomentum);
		}

	public:
		~grad_works()noexcept {}

		//!! copy constructor not needed
		grad_works(const grad_works& other)noexcept = delete;
		//!!assignment is not needed
		grad_works& operator=(const grad_works& rhs) noexcept = delete;

		grad_works(const real_t lr) noexcept : m_momentum(real_t(0.0))
			, m_optBeta1(real_t(0.9)), m_optBeta2(real_t(0.999))
			, m_numericStabilizerEps(_impl::NUM_STAB_EPS<real_t>::value), m_WeightVecNormSqared(real_t(0.0))
			, m_L1(real_t(0.0)), m_L2(real_t(0.0)), m_actualL1(real_t(0.0)), m_actualL2(real_t(0.0))
			, m_optBeta1t(real_t(1.)), m_optBeta2t(real_t(1.))
			, m_type(ClassicalConstant)
		{
			learning_rate(lr);
			_flags_default();
		}

		//template<typename grad_init_t>
		bool init(const common_data_t& cd, const mtx_size_t& weightsSize)noexcept {
			//TODO: there must be some flag that prevents resetting of the data state between distinct calls to nnet.train()
			//(which causes init/deinit cycle)

			if (use_momentums()) {
				if (!m_Vw.resize(weightsSize)) return false;
				m_Vw.zeros();
			}

			if (_optimizerRequiresMatrixA()) {
				if (!m_optMtxA.resize(weightsSize))return false;
			}

			if (_optimizerRequiresMatrixB()) {
				if (!m_optMtxB.resize(weightsSize))return false;
			}

			if (use_individual_learning_rates()) {
				if (!m_ILRGain.resize(weightsSize))return false;
				if ((!use_momentums() | !m_flags[f_ApplyILRToMomentum]) && !m_prevdLdW.resize(weightsSize))return false;
				m_ILRGain.ones();
			}

			set_common_data(cd);
			m_flags.set(f_FirstRun);
			return true;
		}

		void deinit() noexcept {
			clean_common_data();
			
			//#TODO: current cleanup code is not compatible with sequential nnet::train() calls

			m_Vw.clear();
			m_optMtxA.clear();
			m_optMtxB.clear();
			m_ILRGain.clear();
			m_prevdLdW.clear();

			//m_ILR.clear();//we shouldn't clear this variable, as it contains only settings but not a run-time data
			//_flags_default();//same for flags
		}

		void pre_training_fprop(realmtxdef_t& weights) noexcept {
			if (use_nesterov_momentum()) {
				// (1)  vW`(t+1)= momentum*vW(t)
				// (2)  W`(t+1) = W(t) - momentum*vW(t)
				//				= W(t) - vW`(t+1)
				auto& iM = get_iMath();
				auto& iI = get_iInspect();
				iI.fprop_preNesterovMomentum(m_Vw, m_momentum, weights);
				iM.evMulC_ip_Sub_ip(m_Vw, m_momentum, weights);
				iI.fprop_postNesterovMomentum(m_Vw, weights);

				//this might seems unnecessary, because weights will be normalized during apply_grad(), however note this:
				//Nesterov momentum might change weights vectors significantly and if it'll make a weight vector norm significantly bigger
				// than a max_norm, it may seriously affect performance of an optimizer, that uses some function of the
				// weights (such as in Adam or RMSProp)
				// #todo should thoughtfully test this claim
				// though, it seems grounded, => leave it here until tests
				if (use_max_norm()) {
					iM.mCheck_normalize_rows(weights, m_WeightVecNormSqared, m_flags[f_NormIncludesBias]);
				}
			}
		}
		
		void apply_grad(realmtxdef_t& weights, realmtxdef_t& dLdW) noexcept {
			auto& iM = get_iMath();
			auto& iI = get_iInspect();

			iI.apply_grad_begin(weights, dLdW);

			NNTL_ASSERT(dLdW.size() == weights.size());

			const bool bFirstRun = m_flags[f_FirstRun];
			m_flags.reset(f_FirstRun);

			//applying L1-L2 penalties
			// BTW: because we're working with batches that averages dL/dW over many data samples and, possibly, over many 
			// individual dropout masks, it is highly unlikely that any element of dLdW could have a value of zero
			// (zero dL/dW means that we've found an optimal value of weight, that minimizes/maximizes the loss function, or a saddle point).
			// Therefore it is safe to assume, that weights tuning process is still incomplete here and we should apply other
			// weights changing mechanisms, such as L1 or L2 regularization (we shouldn't do it if dL/dW for a weight is zero,
			// because it'll drive the weight's value away from its optimal state).
			// 
			if (use_L1_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L1RegIgnoreBias];
				if (bIgnoreBiases) { dLdW.hide_last_col(); weights.hide_last_col(); }
				iM.evAddScaledSign_ip(dLdW, m_actualL1, weights);
				if (bIgnoreBiases) { dLdW.restore_last_col(); weights.restore_last_col(); }
			}

			if (use_L2_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L2RegIgnoreBias];
				if (bIgnoreBiases) { dLdW.hide_last_col(); weights.hide_last_col(); }
				iM.evAddScaled_ip(dLdW, m_actualL2, weights);
				if (bIgnoreBiases) { dLdW.restore_last_col(); weights.restore_last_col(); }
			}
			
			switch (m_type) {
			case ClassicalConstant:
				iM.evMulC_ip(dLdW, m_learningRate);
				break;

			case RMSProp_Hinton:
				if (bFirstRun) {
					iM.evSquare(m_optMtxA, dLdW);
					iM.evMulC_ip(dLdW, m_learningRate);
				} else iM.RMSProp_Hinton(dLdW, m_optMtxA, m_learningRate, m_optBeta1, m_numericStabilizerEps);
				break;

			case RMSProp_Graves:
				if (bFirstRun) {
					iM.evSquare(m_optMtxA, dLdW);
					dLdW.cloneTo(m_optMtxB);
					iM.evMulC_ip(dLdW, m_learningRate);
				} else iM.RMSProp_Graves(dLdW, m_optMtxA, m_optMtxB, m_learningRate, m_optBeta1, m_numericStabilizerEps);
				break;

			case RProp:
				iM.RProp(dLdW, m_learningRate);
				break;

			case ModProp:
				if (bFirstRun) {
					iM.evAbs(m_optMtxA, dLdW);
					iM.evMulC_ip(dLdW, m_learningRate);
				} else iM.ModProp(dLdW, m_optMtxA, m_learningRate, m_optBeta1, m_numericStabilizerEps);
				break;

			case Adam:
				if (bFirstRun) {
					m_optBeta1t = real_t(1.);
					m_optBeta2t = real_t(1.);
					m_optMtxA.zeros();
					m_optMtxB.zeros();
				};
				iM.Adam(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, m_optBeta2t, m_learningRate, m_optBeta1, m_optBeta2, m_numericStabilizerEps);
				break;

			case AdaMax:
				if (bFirstRun) {
					m_optBeta1t = real_t(1.);
					m_optMtxA.zeros();
					m_optMtxB.zeros();
				};
				iM.AdaMax(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, m_learningRate, m_optBeta1, m_optBeta2, m_numericStabilizerEps);
				break;

			default:
				NNTL_ASSERT(!"WTF??");
				STDCOUTL("*** " << NNTL_FUNCTION << ": Wrong type of optimizer specified!");
				abort();
			}
			iI.apply_grad_postOptimizer(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, m_optBeta2t);

			const auto bUseMomentums = use_momentums();
			if (use_individual_learning_rates()) {
				const auto bUseVelocity = bUseMomentums & m_flags[f_ApplyILRToMomentum];
				NNTL_ASSERT(bUseVelocity || m_prevdLdW.size() == dLdW.size());
				if (!bFirstRun) {
					const auto& prevdLdW = bUseVelocity ? m_Vw : m_prevdLdW;
					iI.apply_grad_preILR(dLdW, prevdLdW, m_ILRGain);
					iM.apply_ILR(dLdW, prevdLdW, m_ILRGain, m_ILR.mulDecr, m_ILR.mulIncr, m_ILR.capLow, m_ILR.capHigh);
					iI.apply_grad_postILR(dLdW, m_ILRGain);
				}
				if (!bUseVelocity) dLdW.cloneTo(m_prevdLdW);
			}

			bool bApplydLdW2Weights = true;
			if (bUseMomentums) {
				NNTL_ASSERT(m_Vw.size() == dLdW.size());
				if (m_flags[f_UseNesterovMomentum]) {
					// (3)  vW(t+1) = momentum*vW(t) + scaling*grad_Loss( W(t)-momentum*vW(t))
					//				= vW`(t+1) + scaling*grad_Loss( W`(t+1) )
					iI.apply_grad_preNesterovMomentum(m_Vw, dLdW);
					iM.evAdd_ip(m_Vw, dLdW);
					iI.apply_grad_postNesterovMomentum(m_Vw);
					// (4)  W(t+1)  = W(t) - vW(t+1) 
					//				= W(t) - vW`(t+1) - scaling*grad_Loss( W`(t+1) )
					//				= W`(t+1) - scaling * grad_Loss( W`(t+1) )
					//bApplydLdW2Weights=true;
				} else {
					//Vw = momentum.*Vw + dW
					iM.apply_momentum(m_Vw, m_momentum, dLdW);
					iI.apply_grad_update(weights, m_Vw);
					iM.evSub_ip(weights, m_Vw);
					bApplydLdW2Weights = false;
				}
			}

			if (bApplydLdW2Weights) {
				iI.apply_grad_update(weights, dLdW);
				iM.evSub_ip(weights, dLdW);
			}

			if (use_max_norm()) {
				iM.mCheck_normalize_rows(weights, m_WeightVecNormSqared, m_flags[f_NormIncludesBias]);
			}

			iI.apply_grad_end(weights);
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
				ret += m_actualL1 * get_iMath().vSumAbs(_W);
				if (bIgnoreBiases) _W.restore_last_col();
			}

			if (use_L2_regularization()) {
				const bool bIgnoreBiases = m_flags[f_L2RegIgnoreBias];
				if (bIgnoreBiases) _W.hide_last_col();
				ret += m_actualL2*real_t(.5) * get_iMath().vSumSquares(_W);
				if (bIgnoreBiases) _W.restore_last_col();
			}

			return ret;
		}
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return use_L1_regularization() | use_L2_regularization(); }

		//////////////////////////////////////////////////////////////////////////

		self_t& learning_rate(const real_t learningRate)noexcept {
			m_learningRate = learningRate;
			m_actualL1 = math::sign(m_learningRate)*m_L1;
			m_actualL2 = math::sign(m_learningRate)*m_L2;
			return *this;
		}
		const real_t learning_rate()const noexcept { return m_learningRate; }

		self_t& set_ILR(const real_t decr, const real_t incr, const real_t capLow, const real_t capHigh) noexcept {
			m_ILR.set(decr, incr, capLow, capHigh);
			m_flags[f_UseILR] = m_ILR.bUseMe();
			return *this;
		}
		self_t& clear_ILR() noexcept {
			m_ILR.clear();
			m_flags[f_UseILR] = false;
			return *this;
		}
		self_t& set_ILR(const GW::ILR_range<real_t>& ilr) noexcept {
			m_ILR = ilr;
			m_flags[f_UseILR] = m_ILR.bUseMe();
			return *this;
		}
		//static constexpr bool defApplyILR2MomentumVelocity = false;//true value for this setting may prevent setups with ILR,
		//dropout & momentum to learn correctly.
		self_t& applyILRToMomentum(const bool b)noexcept {
			m_flags[f_ApplyILRToMomentum] = b;
			if (!use_momentums() | !b) {
				if (m_ILRGain.rows() > 0 && m_ILRGain.cols() > 0 && !m_prevdLdW.resize(m_ILRGain)) {
					//#TODO: correct error return
					STDCOUTL("Failed to resize m_prevdLdW");
					abort();
				}
			}
			return *this;
		}

		self_t& momentum(const real_t m)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			m_flags[f_UseMomentum] = m_momentum > real_t(0.0);
			m_flags.reset(f_UseNesterovMomentum);
			return *this;
		}
		self_t& nesterov_momentum(const real_t m)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			m_flags[f_UseMomentum] = m_momentum > real_t(0.0);
			m_flags.set(f_UseNesterovMomentum);
			return *this;
		}

		self_t& beta1(const real_t c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_optBeta1 = c;
			return *this;
		}
		const real_t beta1()const noexcept { return m_optBeta1; }
		self_t& beta2(const real_t c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_optBeta2 = c;
			return *this;
		}
		const real_t beta2()const noexcept { return m_optBeta2; }

		self_t& numeric_stabilizer(const real_t n)noexcept {
			NNTL_ASSERT(n >= 0 && n < real_t(1.));
			m_numericStabilizerEps = n;
			return *this;
		}

		//Don't call this function during learning!!!
		self_t& set_type(const GradType gt)noexcept {
			m_type = gt;
			return *this;
		}

		//in general, for max_norm it is better to take biases into account when calculation the norm value - it doesn't
		//affect on the direction that the weight is point to but makes two weights with the same direction but different biases really different.
		// HOWEVER: if weights are getting small, but a bias has to be big, than there might be issues due to numeric problems.
		// In general: when there must be a big difference in weights (including bias) magnitude, it may make the things worse. When weights
		// and bias are similar in magnitude - it helps. To detect such condition try to learn with double precision type. Usually
		// double+max_norm(,false) works better or similar to float+max_norm(,true) - sign of numeric issues with MN
		
		//bNormIncludesBias parameter toggles whether the mn parameter describes the max norm of weight vector only (excluding bias weight - false)
		//or the max norm of a full weight vector (including bias - true). Anyway, full weight vector are scaled.
		// bNormIncludesBias==false might offer a bit more numeric stability due to overall magnitude similarity between weigths
		// (biases generally have different, mostly bigger, magnitudes)
		self_t& max_norm(const real_t L2normSquared, const bool bNormIncludesBias = defNormIncludesBias)noexcept {
			NNTL_ASSERT(L2normSquared >= real_t(0.0));
			m_WeightVecNormSqared = L2normSquared;
			m_flags[f_UseMaxNorm] = m_WeightVecNormSqared > real_t(0.0);
			m_flags[f_NormIncludesBias] = bNormIncludesBias;
			return *this;
		}
		const real_t max_norm()const noexcept { return use_max_norm() ? m_WeightVecNormSqared : real_t(.0); }

		//L1 is good for sparse signals
		self_t& L1(const real_t l1, const bool bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			NNTL_ASSERT(l1 >= real_t(0.0));
			m_L1 = l1;
			m_actualL1 = math::sign(m_learningRate)*m_L1;
			m_flags[f_UseL1] = m_L1 > real_t(0.0);
			m_flags[f_L1RegIgnoreBias] = bIgnoreBiasWeights;
			return *this;
		}
		const real_t L1()const noexcept { return m_L1; }
		//L2 is just good)
		self_t& L2(const real_t l2, const bool bIgnoreBiasWeights = defRegularizersIgnoresBiasWeights)noexcept {
			NNTL_ASSERT(l2 >= real_t(0.0));
			m_L2 = l2;
			m_actualL2 = math::sign(m_learningRate)*m_L2;
			m_flags[f_UseL2] = m_L2 > real_t(0.0);
			m_flags[f_L2RegIgnoreBias] = bIgnoreBiasWeights;
			return *this;
		}
		const real_t L2()const noexcept { return m_L2; }

		const bool use_momentums()const noexcept { return m_flags[f_UseMomentum]; } // m_momentum > real_t(0.0);
		const bool nesterov_momentum()const noexcept { return m_flags[f_UseNesterovMomentum]; }
		const bool use_nesterov_momentum()const noexcept { return m_flags[f_UseMomentum] & m_flags[f_UseNesterovMomentum]; }
		const bool use_classical_momentum()const noexcept { return m_flags[f_UseMomentum] & (!m_flags[f_UseNesterovMomentum]); }

		const bool use_individual_learning_rates()const noexcept { return m_flags[f_UseILR]; }  // m_ILR.bUseMe(); }
		const bool applyILRToMomentum()const noexcept { return m_flags[f_ApplyILRToMomentum]; }

		const bool use_max_norm()const noexcept { return m_flags[f_UseMaxNorm]; }
		const bool use_L1_regularization()const noexcept { return m_flags[f_UseL1]; }
		const bool use_L2_regularization()const noexcept { return m_flags[f_UseL2]; }
		const bool isFirstRun()const noexcept { return m_flags[f_FirstRun]; }

	protected:

	};

}