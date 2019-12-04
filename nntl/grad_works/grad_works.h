/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (aradvert@gmail.com; https://github.com/Arech)
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

//Current _grad_works concept is a mess...
//Still don't understand completely how it should look like, but going to change a lot here in a near future.

#include <bitset>
#include "../common_nn_data.h"
#include "ILR.h"
#include "loss_addendums.h"

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
			nntl_interface real_t learning_rate()const noexcept;

			//DON'T use these 2 functions for learning rate decay! They're intended to be used internally only!
			/* it seems that the whole idea was a mistake; pending for removal
			 *nntl_interface auto _learning_rate_scale(const real_t lrScale)noexcept;
			nntl_interface real_t _learning_rate_scale()const noexcept;*/

			nntl_interface void pre_training_fprop(realmtx_t& weights)noexcept;

			//dLdW can have any values on output (use it for temporary calculations if needed)
			nntl_interface void apply_grad(realmtxdef_t& weights, realmtxdef_t& dLdW)noexcept;

			//////////////////////////////////////////////////////////////////////////
			// The following function ALSO MUST BE IMPLEMENTED
			// They are commented out for the reason, - they are implemented as a mixin, and it's an issue to correctly specify
			// a "using" statement for them
			// 
			//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
			// l2Coefficient*Sum(weights.^2) )
			//nntl_interface real_t lossAddendum(const realmtxdef_t& weights)const noexcept;

			//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
			//nntl_interface bool hasLossAddendum()const noexcept;
		};
	}

	// on type of ILR (GW::ILR or GW::ILR_dummy) must be passed as MixinsT
	// #todo: split other _grad_works functionality into corresponding mixins
	template<typename FinalT, typename InterfacesT, template<typename, typename, size_t> class... MixinsT>
	class _grad_works
		: public _impl::_i_grad_works<typename InterfacesT::iMath_t::real_t>
		, public _impl::_common_data_consumer<InterfacesT>
		, public MixinsT< 
		FinalT, typename InterfacesT::iMath_t::real_t
		, utils::mixins::indexed::ref_index<MixinsT, sizeof...(MixinsT), sizeof...(MixinsT), MixinsT...>::value
		>...
	{
	protected:
		typedef FinalT self_t;
		NNTL_METHODS_SELF();

	public:
		typedef _impl::common_nn_data<interfaces_t> common_data_t;
		using _impl::_common_data_consumer<InterfacesT>::real_t;
		
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;

	protected:
		enum OptsList {
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
			// Steps (1)-(2) is done during pre_training_fprop(), steps (3)-(4) - during apply_grad()

			//if there's LRDropout and Nesterov Momentum to be applied, should we apply LRDropout to weight update during 
			//application of momentum velocity to the weights (steps (1)-(2) during fprop).
			// true by default
			f_apply_LRDropout_to_nesterov_momentum,

			f_UseMaxNorm,
			f_NormIncludesBias,//if true, the max-norm parameter describes full norm of weight vector
			
			opts_total
		};

	public:
		static constexpr size_t mixins_count = sizeof...(MixinsT);

		typedef utils::mixins::indexed::make_mixin_vec<FinalT, real_t, MixinsT...> mixins_tvec;
		//#TODO: opts_total must be extendable in derived classes
		typedef utils::mixins::make_mixin_options_count_vec_c<mixins_tvec, opts_total> mixin_opts_cnt;
		typedef utils::mixins::make_cumsum_vec_c<mixin_opts_cnt> mixin_opts_ofs;

		static constexpr size_t TotalOpts = utils::mixins::get_cumsum<mixin_opts_cnt>::value;

		//#TODO move it out of the class
		//this definition should be local to a _grad_works class definition. Other derivations of _impl::_i_grad_works could
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

			Nadam, //Timothy Dozat, ICLR 2016, "Incorporating Nesterov Momentum into Adam"
			Radam //https://github.com/tdozat/Optimization
		};

	protected:
		const bool _optimizerRequiresMatrixA()const noexcept {
			return m_type == RMSProp_Hinton || m_type == RMSProp_Graves || m_type == ModProp || m_type == Adam 
				|| m_type == AdaMax || m_type == Nadam || m_type == Radam;
		}
		const bool _optimizerRequiresMatrixB()const noexcept {
			return m_type == RMSProp_Graves || m_type == Adam || m_type == AdaMax || m_type == Nadam || m_type == Radam;
		}

	protected:
		NNTL_METHODS_MIXIN_ROOT_OPTIONS();

	public:
		//::std::bitset<opts_total> m_flags;
		//unfortunately, it must be left inside public scope at this moment
		// #TODO: should be private/protected
		utils::mixins::binary_options_storage<mixin_opts_ofs, TotalOpts> m_opts;//never use it directly! Only via get_opt()/set_opt()

	protected:		
		real_t m_learningRate, m_LRDropoutPercActive{real_t(1.)}; // , m_lrScale{ real_t(1.) };

		real_t m_momentum;
		real_t m_optBeta1, m_optBeta2, m_optGamma;
		real_t m_numericStabilizerEps;
		real_t m_WeightVecNormSqared;//coefficient of max-norm regularization ||W||2 <= c (see "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".2014)
		//hint: weights initialized from uniform distribution [-b,b]. It's second raw momentum is b^2/3, so the mean norm should
		//be about <row_vector_length>*b^2/3

		GradType m_type;

		realmtx_t m_Vw;
		realmtx_t m_optMtxA, m_optMtxB;//some optimizers require additional memory.

		real_t m_optBeta1t, m_optBeta2t;//storage for coefficients some optimizers (Adam, AdaMax) needed

	protected:
		//////////////////////////////////////////////////////////////////////////
		// some static constants to make code consistent
		//static constexpr bool defRegularizersIgnoresBiasWeights = true;
		static constexpr bool defNormIncludesBias = false;


		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.

			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_learningRate);
				ar & NNTL_SERIALIZATION_NVP(m_LRDropoutPercActive);
				//ar & NNTL_SERIALIZATION_NVP(m_lrScale);
				ar & NNTL_SERIALIZATION_NVP(m_momentum);
				ar & NNTL_SERIALIZATION_NVP(m_optBeta1);
				ar & NNTL_SERIALIZATION_NVP(m_optBeta2);
				ar & NNTL_SERIALIZATION_NVP(m_numericStabilizerEps);
				ar & NNTL_SERIALIZATION_NVP(m_WeightVecNormSqared);

				unsigned int typ = static_cast<unsigned int>(m_type);//#todo must support reading here!
				ar & NNTL_SERIALIZATION_NVP(typ);

				//ar & NNTL_SERIALIZATION_NVP(m_opts); //#todo: must save&restore this variable

				ar & NNTL_SERIALIZATION_NVP(m_optBeta1t);
				ar & NNTL_SERIALIZATION_NVP(m_optBeta2t);
			}

			if (utils::binary_option<true>(ar, serialization::serialize_grad_works_state)) {
				if (_optimizerRequiresMatrixA()) ar & NNTL_SERIALIZATION_NVP(m_optMtxA);
				if (_optimizerRequiresMatrixB()) ar & NNTL_SERIALIZATION_NVP(m_optMtxB);

				if (use_momentums()) ar & NNTL_SERIALIZATION_NVP(m_Vw);
			}

			ILR_serialize(ar, version);
		}

	protected:
		void _flags_default()noexcept {
			set_opt(f_FirstRun, true)
				.set_opt(f_UseNesterovMomentum, true)
				.set_opt(f_apply_LRDropout_to_nesterov_momentum, true); // .reset(f_ApplyILRToMomentum);
		}

		~_grad_works()noexcept {}

		//!! copy constructor not needed
		_grad_works(const _grad_works& other)noexcept = delete;
		//!!assignment is not needed
		_grad_works& operator=(const _grad_works& rhs) noexcept = delete;

		_grad_works(const real_t lr) noexcept : m_momentum(real_t(0.0))
			, m_optBeta1(real_t(0.9)), m_optBeta2(real_t(0.999)), m_optGamma(real_t(0.05))
			, m_numericStabilizerEps(_impl::NUM_STAB_EPS<real_t>::value), m_WeightVecNormSqared(real_t(0.0))
			, m_optBeta1t(real_t(1.)), m_optBeta2t(real_t(1.))
			, m_type(ClassicalConstant) //, m_lrScale(real_t(1.))
		{
			learning_rate(lr);
			_flags_default();

			_la_construct();
		}
		
	public:
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

			if (!ILR_init(weightsSize))return false;

			set_common_data(cd);

			//we would need twice weightsNumel to make LRDropout for NesterovMomentum if necessary
			get_iMath().preinit(math::smatrix_td::sNumel(weightsSize)*(1 + (
				use_nesterov_momentum() && bApplyLRDropoutToNesterovMomentum() && bLRDropout()
				)));

			if (!_la_init(weightsSize))return false;

			set_opt(f_FirstRun,true);
			return true;
		}

		void deinit() noexcept {
			clean_common_data();
			
			ILR_deinit();
			_la_deinit();

			//#TODO: current cleanup code is not compatible with sequential nnet::train() calls

			m_Vw.clear();
			m_optMtxA.clear();
			m_optMtxB.clear();

			//_flags_default();//we shouldn't clear this variable, as it contains only settings but not a run-time data
		}

		void pre_training_fprop(realmtxdef_t& weights) noexcept {
			if (!isLearningBlocked() && use_nesterov_momentum()) {
				// (1)  vW`(t+1)= momentum*vW(t)
				// (2)  W`(t+1) = W(t) - momentum*vW(t)
				//				= W(t) - vW`(t+1)
				auto& iM = get_iMath();
				auto& iI = get_iInspect();
				iI.fprop_preNesterovMomentum(m_Vw, m_momentum, weights);

				if (bLRDropout() && bApplyLRDropoutToNesterovMomentum()) {
					//we can't make everything here in a single step, so doing in a long way
					//step (1)
					iM.evMulC_ip(m_Vw, m_momentum);

					//storing m_Vw in temporary matrix to apply dropout further
					NNTL_ASSERT(!weights.emulatesBiases());
					const auto weightsNumel = weights.numel();
					auto pVwTmp = iM._istor_alloc(weightsNumel);
					auto pDoMask = iM._istor_alloc(weightsNumel);
					realmtx_t tmpVw(pVwTmp, weights), doMask(pDoMask, weights);
					m_Vw.copy_to(tmpVw);

					//creating and applying do mask
					get_iRng().gen_matrix_norm(doMask);

					iI.fprop_preLRDropout4NesterovMomentum(tmpVw, m_LRDropoutPercActive, doMask);
					iM.apply_dropout_mask(tmpVw, m_LRDropoutPercActive, doMask);
					iI.fprop_postLRDropout4NesterovMomentum(tmpVw);

					//applying weight updates
					iM.evSub_ip(weights, tmpVw);

					tmpVw.clear();
					doMask.clear();
					iM._istor_free(pDoMask, weightsNumel);
					iM._istor_free(pVwTmp, weightsNumel);
				} else {
					iM.evMulC_ip_Sub_ip(m_Vw, m_momentum, weights);
				}
				iI.fprop_postNesterovMomentum(m_Vw, weights);

				//this might seems unnecessary, because weights will be normalized during apply_grad(), however note this:
				//Nesterov momentum might change weights vectors significantly and if it'll make a weight vector norm significantly bigger
				// than a max_norm, it may seriously affect performance of an optimizer, that uses some function of the
				// weights (such as in Adam or RMSProp)
				// #todo should thoughtfully test this claim
				// though, it seems grounded, => leave it here until tests
				if (use_max_norm()) {
					iM.mCheck_normalize_rows(weights, m_WeightVecNormSqared, get_opt(f_NormIncludesBias));
				}
			}
		}
		
		//#todo this code should be refactored.
		void apply_grad(realmtxdef_t& weights, realmtxdef_t& dLdW) noexcept {
			NNTL_ASSERT(dLdW.size() == weights.size());
#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
			NNTL_ASSERT(weights.test_noNaNs());
			NNTL_ASSERT(dLdW.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK

			auto& iI = get_iInspect();

			iI.apply_grad_begin(weights, dLdW);

			//loss addendums MUST be applied before optimizers, because they reflect actual/desired error surface
			// and optimizers must take that into account
			_applyLossAddendums(weights, dLdW);

			if (isLearningBlocked()) {
				iI.apply_grad_end(weights);
				return; //do nothing, leave the state intact
			}

			//_applyLossAddendums(weights, dLdW);

			const bool bFirstRun = get_opt(f_FirstRun);
			set_opt(f_FirstRun,false);

			auto& iM = get_iMath();

			/*//changing nesterov momentum vars with fresh dL/dW (should do the same with classical momentum #todo)
			if (use_momentums() && get_opt(f_UseNesterovMomentum)) {
				NNTL_ASSERT(m_Vw.size() == dLdW.size());
				// (3)  vW(t+1) = momentum*vW(t) + scaling*grad_Loss( W(t)-momentum*vW(t))
				//				= vW`(t+1) + scaling*grad_Loss( W`(t+1) )
				iI.apply_grad_preNesterovMomentum(m_Vw, dLdW);
				//#todo: need a separate scaling coefficient here. Better leave m_learningRate to optimizer's only use.
				iM.evAddScaled_ip(m_Vw, m_learningRate, dLdW);
				iI.apply_grad_postNesterovMomentum(m_Vw);
				// (4)  W(t+1)  = W(t) - vW(t+1) 
				//				= W(t) - vW`(t+1) - scaling*grad_Loss( W`(t+1) )
				//				= W`(t+1) - scaling * grad_Loss( W`(t+1) )
			}*/

			const real_t curLr = m_learningRate;// *m_lrScale;


			switch (m_type) {
			case ClassicalConstant:
				iM.evMulC_ip(dLdW, curLr);
				break;

			case RMSProp_Hinton:
				if (bFirstRun) {
					iM.evSquare(m_optMtxA, dLdW);
					iM.evMulC_ip(dLdW, curLr);
				} else iM.RMSProp_Hinton(dLdW, m_optMtxA, curLr, m_optBeta1, m_numericStabilizerEps);
				break;

			case RMSProp_Graves:
				if (bFirstRun) {
					iM.evSquare(m_optMtxA, dLdW);
					dLdW.clone_to(m_optMtxB);
					iM.evMulC_ip(dLdW, curLr);
				} else iM.RMSProp_Graves(dLdW, m_optMtxA, m_optMtxB, curLr, m_optBeta1, m_numericStabilizerEps);
				break;

			case RProp:
				iM.RProp(dLdW, curLr);
				break;

			case ModProp:
				if (bFirstRun) {
					iM.evAbs(m_optMtxA, dLdW);
					iM.evMulC_ip(dLdW, curLr);
				} else iM.ModProp(dLdW, m_optMtxA, curLr, m_optBeta1, m_numericStabilizerEps);
				break;

			case Adam:
				if (bFirstRun) {
					m_optBeta1t = real_t(1.);
					m_optBeta2t = real_t(1.);
					m_optMtxA.zeros();
					m_optMtxB.zeros();
				};
				iM.Adam(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, m_optBeta2t, curLr, m_optBeta1, m_optBeta2, m_numericStabilizerEps);
				break;

			case AdaMax:
				if (bFirstRun) {
					m_optBeta1t = real_t(1.);
					m_optMtxA.zeros();
					m_optMtxB.zeros();
				};
				iM.AdaMax(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, curLr, m_optBeta1, m_optBeta2, m_numericStabilizerEps);
				break;

			case Nadam:
			case Radam:
				if (bFirstRun) {
					m_optBeta1t = real_t(1.);
					m_optBeta2t = real_t(1.);
					m_optMtxA.zeros();
					m_optMtxB.zeros();
					if (Nadam==m_type) {
						m_optGamma = real_t(0.);
					}
				};
				iM.RNadam(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, m_optBeta2t, curLr, m_optBeta1, m_optBeta2, m_optGamma, m_numericStabilizerEps);
				break;

			default:
				NNTL_ASSERT(!"WTF??");
				STDCOUTL("*** " << NNTL_FUNCTION << ": Wrong type of optimizer specified!");
				abort();
			}
			iI.apply_grad_postOptimizer(dLdW, m_optMtxA, m_optMtxB, m_optBeta1t, m_optBeta2t);

			ILR_apply(bFirstRun, dLdW, m_Vw);

			/*if (use_momentums() && !get_opt(f_UseNesterovMomentum)) {
				//Vw = momentum.*Vw + dW
				iM.apply_momentum(m_Vw, m_momentum, dLdW);
				iI.apply_grad_update(weights, m_Vw);
				iM.evSub_ip(weights, m_Vw);
			} else {
				iI.apply_grad_update(weights, dLdW);
				iM.evSub_ip(weights, dLdW);
			}*/

			bool bApplydLdW2Weights = true;
			if (use_momentums()) {
				NNTL_ASSERT(m_Vw.size() == dLdW.size());
				if (get_opt(f_UseNesterovMomentum)) {
					// (3)  vW(t+1) = momentum*vW(t) + scaling*grad_Loss( W(t)-momentum*vW(t))
					//				= vW`(t+1) + scaling*grad_Loss( W`(t+1) )
					iI.apply_grad_preNesterovMomentum(m_Vw, dLdW);
					iM.evAdd_ip(m_Vw, dLdW);
					iI.apply_grad_postNesterovMomentum(m_Vw);
					// (4)  W(t+1)  = W(t) - vW(t+1) 
					//				= W(t) - vW`(t+1) - scaling*grad_Loss( W`(t+1) )
					//				= W`(t+1) - scaling * grad_Loss( W`(t+1) )
				} else {
					//Vw = momentum.*Vw + dW
					iM.apply_momentum(m_Vw, m_momentum, dLdW);
					//note that we're going to apply m_Vw to the weights and we don't need dLdW anymore

					if (bLRDropout()) {
						//to apply lrdropout just store weight updates back to dLdW and proceed further
						m_Vw.copy_to(dLdW);
						//note that we cannot use m_Vw directly, because dropout will make useless some elements of it
					} else {
						//applying weight updates now
						iI.apply_grad_update(weights, m_Vw);
						iM.evSub_ip(weights, m_Vw);
						bApplydLdW2Weights = false;
					}
				}
			}

			if (bApplydLdW2Weights) {
				if (bLRDropout()) { //applying LR dropout, arxiv:1912.00144
					const auto storAllocSize = dLdW.numel();
					auto pStor = iM._istor_alloc(storAllocSize);

					realmtx_t doMask(pStor, dLdW);
					get_iRng().gen_matrix_norm(doMask);
					
					iI.apply_grad_preLRDropout(dLdW, m_LRDropoutPercActive, doMask);
					iM.apply_dropout_mask(dLdW, m_LRDropoutPercActive, doMask);
					iI.apply_grad_postLRDropout(dLdW);

					doMask.clear();
					iM._istor_free(pStor, storAllocSize);
				}

				iI.apply_grad_update(weights, dLdW);
				iM.evSub_ip(weights, dLdW);
			}

			if (use_max_norm()) {
				iM.mCheck_normalize_rows(weights, m_WeightVecNormSqared, get_opt(f_NormIncludesBias));
			}

#ifdef NNTL_AGGRESSIVE_NANS_DBG_CHECK
			NNTL_ASSERT(weights.test_noNaNs());
#endif // NNTL_AGGRESSIVE_NANS_DBG_CHECK

			iI.apply_grad_end(weights);
		}

		//////////////////////////////////////////////////////////////////////////

		self_ref_t learning_rate(const real_t learningRate)noexcept {
			m_learningRate = learningRate;
			return get_self();
		}
		real_t learning_rate()const noexcept { return m_learningRate; }

		//DON'T use these 2 functions for learning rate decay! They're intended to be used internally only!
		/*it seems that the whole idea was a mistake; pending for removal
		 *self_ref_t _learning_rate_scale(const real_t lrScale)noexcept {
			m_lrScale = lrScale;
			return get_self();
		}
		real_t _learning_rate_scale()const noexcept { return m_lrScale; }*/


		//Learning rate dropout, arxiv:1912.00144
		bool bLRDropout()const noexcept { return m_LRDropoutPercActive < real_t(1.); }

		real_t LRDropoutPercentActive()const noexcept { return m_LRDropoutPercActive; }

		self_ref_t LRDropoutPercentActive(const real_t dpa)noexcept {
			NNTL_ASSERT(real_t(0.) <= dpa && dpa <= real_t(1.));
			m_LRDropoutPercActive = (dpa <= real_t(+0.) || dpa > real_t(1.)) ? real_t(1.) : dpa;
			return get_self();
		}

		bool bApplyLRDropoutToNesterovMomentum()const noexcept {
			return get_opt(f_apply_LRDropout_to_nesterov_momentum);
		}
		self_ref_t setApplyLRDropoutToNesterovMomentum(const bool v)noexcept {
			set_opt(f_apply_LRDropout_to_nesterov_momentum, v);
			return get_self();
		}

		self_ref_t momentum(const real_t m)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			set_opt(f_UseMomentum, m_momentum > real_t(0.0));
			set_opt(f_UseNesterovMomentum, false);
			return get_self();
		}
		self_ref_t nesterov_momentum(const real_t m)noexcept {
			NNTL_ASSERT(m >= 0 && m < 1);
			m_momentum = m;
			set_opt(f_UseMomentum, m_momentum > real_t(0.0));
			set_opt(f_UseNesterovMomentum, true);
			return get_self();
		}

		self_ref_t beta1(const real_t c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_optBeta1 = c;
			return get_self();
		}
		real_t beta1()const noexcept { return m_optBeta1; }
		self_ref_t beta2(const real_t c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_optBeta2 = c;
			return get_self();
		}
		real_t beta2()const noexcept { return m_optBeta2; }

		self_ref_t gamma(const real_t c)noexcept {
			NNTL_ASSERT(c > 0 && c < 1);
			m_optGamma = c;
			return get_self();
		}
		real_t gamma()const noexcept { return m_optGamma; }


		self_ref_t numeric_stabilizer(const real_t n)noexcept {
			NNTL_ASSERT(n >= 0 && n < real_t(1.));
			m_numericStabilizerEps = n;
			return get_self();
		}

		//Don't call this function during learning!!!
		self_ref_t set_type(const GradType gt)noexcept {
			m_type = gt;
			return get_self();
		}



		//for max_norm it might be better to take biases into account during calculation of norm value - it doesn't
		//affect the direction that the weight is point to but makes two weights with the same direction but different biases really different.
		// HOWEVER: if weights are getting small, but a bias has to be big, than there might be issues due to numeric problems.
		// In general: when there must be a big difference in weights (including bias) magnitude, it may make the things worse. When weights
		// and bias are similar in magnitude - it helps. To detect such condition try to learn with double precision type. Usually
		// double+max_norm(,false) works better or similar to float+max_norm(,true) - sign of numeric issues with MN
		
		//bNormIncludesBias parameter toggles whether the mn parameter describes the max norm of weight vector only (excluding bias weight - false)
		//or the max norm of a full weight vector (including bias - true). Anyway, full weight vector are scaled.
		// bNormIncludesBias==false might offer a bit more numeric stability due to overall magnitude similarity between weigths
		// (biases generally have different, mostly bigger, magnitudes).
		// And one more thing to consider: weight vector magnitude (norm) defines the maximum "amplification" that previous layers
		// activations will get during scalar multiplication (pre-activation calculation step). When the prev.layer activations
		// is ranged in [-1,1] (tanh-function family) or [0,1] (sigm-function famility) it might be fairly safe to assume
		// (though not always) that bias weights will have the same magnitude as activation weights. However, for other activation
		// functions, especically for ReLU-style functions, this assumption is clearly very fragile and unsounded. Therefore,
		// generally it is better NOT to include bias weight in max-norm constraint and set bNormIncludesBias parameter to false.
		self_ref_t max_norm(const real_t L2normSquared, const bool bNormIncludesBias = defNormIncludesBias)noexcept {
			NNTL_ASSERT(L2normSquared >= real_t(0.0));
			m_WeightVecNormSqared = L2normSquared;
			set_opt(f_UseMaxNorm, m_WeightVecNormSqared > real_t(0.0));
			set_opt(f_NormIncludesBias, bNormIncludesBias);
			return get_self();
		}
		real_t max_norm()const noexcept { return use_max_norm() ? m_WeightVecNormSqared : real_t(.0); }

		bool use_momentums()const noexcept { return get_opt(f_UseMomentum); } // m_momentum > real_t(0.0);
		bool nesterov_momentum()const noexcept { return get_opt(f_UseNesterovMomentum); }
		bool use_nesterov_momentum()const noexcept { return get_opt(f_UseMomentum) & get_opt(f_UseNesterovMomentum); }
		bool use_classical_momentum()const noexcept { return get_opt(f_UseMomentum) & (!get_opt(f_UseNesterovMomentum)); }

		bool use_max_norm()const noexcept { return get_opt(f_UseMaxNorm); }
		bool isFirstRun()const noexcept { return get_opt(f_FirstRun); }
	};


	template<typename InterfacesT, template<typename, typename, size_t> class... MixinsT>
	class grad_works_f final : public _grad_works<grad_works_f<InterfacesT, MixinsT...>, InterfacesT, MixinsT...>{
	public:
		~grad_works_f()noexcept{}
		grad_works_f(const typename InterfacesT::iMath_t::real_t lr) noexcept 
			: _grad_works<grad_works_f<InterfacesT, MixinsT...>, InterfacesT, MixinsT...>(lr){}
	};

	template<typename InterfacesT>
	using grad_works = grad_works_f<
		InterfacesT
		, GW::ILR
		, GW::Loss_Addendums_builder< ::std::tuple<
		    loss_addendum::L1<typename InterfacesT::real_t>
		    , loss_addendum::L2<typename InterfacesT::real_t>
		>>::template type
	>;

}