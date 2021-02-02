/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
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

#include "fully_connected.h"

namespace nntl {

	// _layer_output supports #supportsBatchInRow only for previous layer activations. There's no particular issue to
	// support it for m_activations, just need to update corresponding iM functions
	template<typename FinalPolymorphChild, typename ActivFunc, typename GradWorks>
	class _layer_output 
		: public m_layer_output
		, public m_layer_learnable
		, public _LFC_FProp<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc, false>
	{
	private:
		typedef _LFC_FProp<FinalPolymorphChild, typename GradWorks::interfaces_t, ActivFunc, false> _base_class_t;

	public:
		static_assert(bActivationForOutput && bActivationForHidden, "ActivFunc template parameter must be derived from "
			"activation::_i_activation_loss<> and activation::_i_activation<>");

		typedef GradWorks grad_works_t;
		static_assert(::std::is_base_of<_impl::_i_grad_works<real_t>, grad_works_t>::value, "GradWorks template parameter should be derived from _i_grad_works");
		
		static constexpr const char _defName[] = "outp";

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		grad_works_t m_gradientWorks;

		realmtxdef_t m_dLdW;//doesn't guarantee to retain it's value between usage in different code flows;
		//may share memory with some other data structure. Must be deformable for grad_works_t
		
		//real_t m_nTiledTimes{ 0 };

		real_t m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd;
		bool m_bRestrictdLdZ;//restriction flag should be permanent for init/deinit calls and changed only by explicit calls to respective functions
						
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class ::boost::serialization::access;
		template<class Archive>
		void save(Archive & ar, const unsigned int version) const {
			NNTL_UNREF(version);
			ar & ::boost::serialization::base_object<_base_class_t>(*this);

			//NB: DONT touch ANY of .useExternalStorage() matrices here, because it's absolutely temporary meaningless data
			// and moreover, underlying storage may have already been freed.			
			if (utils::binary_option<true>(ar, serialization::serialize_grad_works)) ar & m_gradientWorks;//dont use nvp or struct here for simplicity
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_bRestrictdLdZ);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictLowerBnd);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictUpperBnd);
			}
		}
		template<class Archive>
		void load(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);
			ar & ::boost::serialization::base_object<_base_class_t>(*this);
			
			if (utils::binary_option<true>(ar, serialization::serialize_training_parameters)) {
				ar & NNTL_SERIALIZATION_NVP(m_bRestrictdLdZ);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictLowerBnd);
				ar & NNTL_SERIALIZATION_NVP(m_dLdZRestrictUpperBnd);
			}
			//#todo other vars!
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER();

		//////////////////////////////////////////////////////////////////////////
		//methods
	public:
		~_layer_output() noexcept {};

		_layer_output(const char* pCustomName, const neurons_count_t _neurons_cnt, real_t learningRate = real_t(.01)) noexcept
			: _base_class_t(pCustomName, _neurons_cnt), m_dLdW(), m_gradientWorks(learningRate)
			, m_bRestrictdLdZ(false), m_dLdZRestrictLowerBnd(.0), m_dLdZRestrictUpperBnd(.0)
		{
			NNTL_ASSERT(!m_activations.emulatesBiases());
		};
		_layer_output(const neurons_count_t _neurons_cnt, const real_t learningRate = real_t(.01), const char*const pCustomName = nullptr) noexcept
			: _layer_output(pCustomName, _neurons_cnt, learningRate)
		{}

		grad_works_t& get_gradWorks()noexcept { return m_gradientWorks; }
		const grad_works_t& get_gradWorks()const noexcept { return m_gradientWorks; }

		ErrorCode init(_layer_init_data_t& lid)noexcept {
			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});

			auto ec = _base_class_t::init(lid);
			if (ErrorCode::Success != ec) return ec;

			//we need this to be able to include layer_output into LPT (though, it won't work now, at least because LPT doesn't
			// propagate m_output_layer marker)
			NNTL_ASSERT(lid.nTiledTimes == 1 || !"If you've updated LPT to incapsulate this layer, uncomment every m_nTiledTimes usage if it is still necessary");
			//and note, that that nTiledTimes machinery is actually a hack to make LPT pass a grad check. So it's better to make grad checker
			//routine aware of layer differences.
			//m_nTiledTimes = real_t(lid.nTiledTimes);
			
			const numel_cnt_t prmsNumel = get_self().bUpdateWeights() ? m_weights.numel() : 0;
			lid.nParamsToLearn = prmsNumel;
			
// 			get_iMath().preinit(::std::max({
// 				realmtx_t::sNumel(get_common_data().biggest_batch_size(), get_self().get_incoming_neurons_cnt() + 1)//for prev activations.
// 			}));

			if (get_common_data().is_training_possible() && get_self().bUpdateWeights() /*&& get_self().bDoBProp()*/) {
				//There's no dLdA coming into the output layer, therefore leave max_dLdA_numel it zeroed
				//lid.max_dLdA_numel = 0;
				
				// we'll need 1 temporarily matrix for bprop(): dL/dW [m_neurons_cnt x get_incoming_neurons_cnt()+1]
				lid.maxMemTrainingRequire = prmsNumel;
			}

			if (!get_self().get_gradWorks().init(get_common_data(), m_weights))return ErrorCode::CantInitializeGradWorks;

			//#BUGBUG current implementation of GW::hasLossAddendum could return false because LAs are currently disabled,
			//however they could be enabled later. Seems like not a major bug, so I'll leave it to fix later.
			//#SeeAlso LFC
			lid.bLossAddendumDependsOnWeights = get_self().get_gradWorks().hasLossAddendum();

			bSuccessfullyInitialized = true;
			return ec;
		}

		void deinit()noexcept {
			get_gradWorks().deinit();
			m_dLdW.clear();
			//m_nTiledTimes = real_t(0);
			_base_class_t::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			NNTL_UNREF(cnt);
			if (get_common_data().is_training_possible() && get_self().bUpdateWeights()) {
				NNTL_ASSERT(ptr && cnt >= m_weights.numel());
				m_dLdW.useExternalStorage(ptr, m_weights);
				NNTL_ASSERT(!m_dLdW.emulatesBiases());
			} else NNTL_ASSERT(m_dLdW.empty());
		}

		void _on_fprop_in_training_mode()noexcept {
			get_self().get_gradWorks().pre_training_fprop(m_weights);
		}

	protected:

		void _cust_inspect(const realmtx_t&)const noexcept { }

		// #supportsBatchInRow for prevActivations, data_y and m_activations (iif the dLdZ, obtained from
		// activation_t::dLdZ() has bBatchInRow() layout -- need to update other iM func to make it work completely) 
		template<typename YT>
		unsigned _outp_bprop(const math::smatrix<YT>& data_y, const realmtx_t& prevActivations, const bool bPrevLayerWBprop, realmtx_t& dLdAPrev)noexcept {
			NNTL_ASSERT(prevActivations.test_biases_strict());
			//NNTL_ASSERT(real_t(1.) == get_gradWorks()._learning_rate_scale());//output layer mustn't have lr scaled.
			// Don't implement lrdecay using the _learning_rate_scale(), it's for internal use only!
			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

			NNTL_ASSERT_MTX_NO_NANS(prevActivations);
			NNTL_ASSERT_MTX_NO_NANS(data_y);

			auto& _iI = get_iInspect();
			_iI.bprop_begin(get_layer_idx(), data_y);
			//_iI.bprop_finaldLdA(dLdA); //--doesn't apply here actually

			data_y.assert_storage_does_not_intersect(dLdAPrev);
			NNTL_ASSERT(get_common_data().is_training_mode());
			//NNTL_ASSERT(m_activations.size() == data_y.size()); //too strong. We may need loss functions that may require different columns count
			NNTL_ASSERT(m_activations.batch_size() == data_y.batch_size());
			NNTL_ASSERT(m_activations.batch_size() == get_common_data().get_cur_batch_size());
			NNTL_ASSERT(get_common_data().get_cur_batch_size() == prevActivations.batch_size() 
				&& get_incoming_neurons_cnt() == prevActivations.sample_size());
			NNTL_ASSERT(!bPrevLayerWBprop || dLdAPrev.size() == prevActivations.size_no_bias());
			NNTL_ASSERT(!bPrevLayerWBprop || dLdAPrev.bBatchInRow() == prevActivations.bBatchInRow());

			auto& iM = get_iMath();

			_iI.bprop_predLdZOut(m_activations, data_y);
			//compute dL/dZ into m_activations. Note that there's no requirement to make layout of dLdZ the same as m_activations, therefore
			//remembering it to restore later (this is not necessary now, but will be in future)
			const auto bActBatchesInRows = m_activations.bBatchInRow();
			_activation_bprop_output(data_y, iM);
			//now dLdZ is calculated into m_activations

			//#todo: once upgrade finished, remove the following assert
			NNTL_ASSERT(m_activations.bBatchInColumn());

			realmtx_t & dLdZ = m_activations;
			_iI.bprop_dLdZ(dLdZ);

			if (m_bRestrictdLdZ) {
				iM.evClamp(dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);
				_iI.bprop_postClampdLdZ(dLdZ, m_dLdZRestrictLowerBnd, m_dLdZRestrictUpperBnd);
			}

			get_self()._cust_inspect(dLdZ);

			if (bPrevLayerWBprop) {
				NNTL_ASSERT(!m_weights.emulatesBiases());
				//compute dL/dAprev to use in lower layer. Before that make m_weights looks like there is no bias weights
				//m_weights.hide_last_col();
				//iM.mMulAB_C(dLdZ, m_weights, dLdAPrev);
				//m_weights.restore_last_col();//restore weights back
				// #TODO support bBatchInRow for dLdZ!!!
				iM.mMul_dLdZ_weights_2_dLdAPrev(dLdZ, m_weights, dLdAPrev);
			}

			if (get_self().bUpdateWeights()) {
				dLdAPrev.assert_storage_does_not_intersect(m_dLdW);
				NNTL_ASSERT(!m_dLdW.emulatesBiases() && m_dLdW.bBatchInColumn() && m_weights.bBatchInColumn());
				NNTL_ASSERT(m_dLdW.size() == m_weights.size());

				//NNTL_ASSERT(m_nTiledTimes > 0);
				//#hack to make gradient check of LPT happy, we would ignore the tiling count here entirely (because numerical error is
				// implicitly normalized to batch size, however, analytical - doesn't and it's normalized to the current batch size. Inner
				// layers of LPT have different batch sizes, so we just going to ignore that here. Probably we should:
				//#todo update grad checking algo to take tiling count into account.
				//However, the tiled layer in a real life would have a m_nTiledTimes bigger batch size, than the modeled layer should have,
				// therefore we'll upscale the gradient m_nTiledTimes times.
				//compute dL/dW = 1/batchsize * (dL/dZ)` * Aprev
				//iM.mScaledMulAtB_C(real_t(1.) / real_t(dLdZ.rows()), dLdZ, prevActivations, m_dLdW);
				// #TODO support bBatchInRow for dLdZ!!!
				iM.mMulScaled_dLdZ_prevAct_2_dLdW(real_t(1.) / real_t(dLdZ.rows()), dLdZ, prevActivations, m_dLdW);

				_iI.bprop_dLdW(dLdZ, prevActivations, m_dLdW);

				//now we can apply gradient to the weights
				get_gradWorks().apply_grad(m_weights, m_dLdW);
			}
			//restoring layout possibly changed by dLdZ computation
			m_activations.set_batchInRow(bActBatchesInRows);

			_iI.bprop_end(dLdAPrev);
			return 1;
		}
	public:
		
		template <typename YT, typename LowerLayer>
		unsigned bprop(const math::smatrix<YT>& data_y, const LowerLayer& lowerLayer, realmtx_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			//NNTL_ASSERT(get_self().bDoBProp());
			return get_self()._outp_bprop(data_y, lowerLayer.get_activations(), is_layer_with_bprop<LowerLayer>::value, dLdAPrev);
		}
		
		//////////////////////////////////////////////////////////////////////////

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		real_t lossAddendum()const noexcept { return get_gradWorks().lossAddendum(m_weights); }
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return get_gradWorks().hasLossAddendum(); }

		//////////////////////////////////////////////////////////////////////////

		//use this function to put a restriction on dL/dZ value - this may help in training large networks
		//(see Alex Graves's "Generating Sequences With Recurrent Neural Networks(2013)" )
		self_ref_t restrict_dL_dZ(real_t lowerBnd, real_t upperBnd)noexcept {
			if (upperBnd == 0 || lowerBnd==0) {
				NNTL_ASSERT(upperBnd == 0 && lowerBnd == 0);
				m_bRestrictdLdZ = false;
			} else {
				NNTL_ASSERT(lowerBnd < 0 && upperBnd>0);
				m_bRestrictdLdZ = true;
			}
			m_dLdZRestrictLowerBnd = lowerBnd;
			m_dLdZRestrictUpperBnd = upperBnd;
			return get_self();
		}
		self_ref_t drop_dL_dZ_restriction()noexcept { 
			m_bRestrictdLdZ = false; 
			return get_self();
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_output
	// If you need to derive a new class, derive it from _layer_output (to make static polymorphism work)
	template <typename ActivFunc = activation::sigm_quad_loss<d_interfaces::real_t>,
		typename GradWorks = grad_works<d_interfaces>
	> class layer_output final : public _layer_output<layer_output<ActivFunc, GradWorks>, ActivFunc, GradWorks>
	{
		typedef _layer_output<layer_output<ActivFunc, GradWorks>, ActivFunc, GradWorks> _base_class_t;
	public:
		template<typename...ArgsT>
		layer_output(ArgsT&&... ar)noexcept : _base_class_t(::std::forward<ArgsT>(ar)...) {}
	};
}

