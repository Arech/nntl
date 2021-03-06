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

//#todo now layers can reshape data and change batch size, so this class should be upgraged/optimized for the case of preconstructed input

// layer_pack_tile (LPT) is a similar layer type to layer_pack_horizontal in a sense that it breaks input neurons into
// K groups to be fed into K processing units (layers) and gathers output of that K processing units into a single activation
// matrix. The difference is in a fact that LPT uses only single one processing unit. LPT just tiles it K times to cover all
// inputs. It may be very helpful when one needs to process K groups of data, that were made by a single original data source
// and therefore should be processed by the same feature detector.
// One may think of LPT as a kind of 1D convolutional layer with a stride equal to a kernel size.
// Here is a picture
// 
//   |a1_1..a1_a. . . .ai_1..ai_a. . . .ak_1..ak_a|			- LPT activation matrix consists of k individual activation submatrices.
// |----------------layer_pack_tile-----------------|			It has k*a columns (and a bias column) and m rows.
// |	|	|			 |	 |			  |	  | 	|
// |  a1_1..a1_a   .   ai_1..ai_a   .   ak_1..ak_a  |
// | | L_inst_1 |  .  | L_inst_i |  .  | L_inst_k | |		- k "instances" of the same layer (actually it's the same layer application)
// |  x1_1..x1_n   .   xi_1..xi_n   .   xk_1..xk_n  |		- source data, k groups with n columns each. (actually, n+1 columns
// |   |   |   |		|   |   |		 |   |   |	|		-  - remember the bias column).
// |------------------------------------------------|
//   |x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n|			- source data concatentenated, k*n columns (k*n+1 with the bias column) and m rows.
//
// Now the tricky part: how to make it work? We must implement correct fprop() and bprop() routines, i.e. we should produce correct
// activations in fprop() and by processing dLdA in bprop() produce correct dLdAPrev and update layer`s weights.
// We may take the LPH<> approach for fprop() and apply the layer k times to k groups of data to produce activation values.
// However, we can't run bprop() after that, because by design any layer expects that bprop() is called for the same data that was
// passed to fprop(). For example, dropout relies on that property and expects it's internal variables to retain its values between
// corresponding fprop() and bprop() calls. That would not be the case, because when the bprop() will be called first time
// for the first layer "instance", its variables will be describing the dropout state for the last layer instance, not the first.
// All internal layer state that corresponds to all fprop() calls except for the last call will be lost.
// Moreover, because of weights update during each bprop() run, dLdAPrev calculated by the layer for all but the first processed
// instance will be corrupted.
// 
// Provided that the layer to be tiled could be in fact a compound layer itself, it looks impractical to try to save each fprop()
// state to be able to restore it later for corresponding bprop() session. Therefore, we have no real choice but to transform
// incoming data from [m, k*n+1] matrix to [k*m, n+1] matrix and feed it to the layer one time.
// Here's how the transformation should look like:
//																						|x1_1...x1_n 1|		:transformed data_x
//																						|........... 1|		:to be fed to the layer
// incoming to the LPT data_x=|x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n 1|	===>	|xi_1...xi_n 1|
//																						|........... 1|
//																						|xk_1...xk_n 1|
//																
// Upon layer.fprop() finishes, we'd need to transform [k*m,a+1] activation matrix back to normal(expected) size of [m,k*a+1].
// On bprop(), we'd need to transform activations back to size [k*m,a+1], and dLdA from the size [m,k*a] to [k*m,a].
// Similar to dLdAPrev, but only resize from [m,k*n] to [k*m,n] would be needed (no actual data moving required for dLdAPrev).
// Then we'd run layer.bprop() on 'em and after that we should transform dLdAPrev from [k*m,n] to [m,k*n].
// 
// Notice how the bias handling issue arises: source data_x.numel()==m*k*n+m, however transformed data_x would require more
// storage. It's numel()==m*k*n+m*k. The same applies to activation matrix. However, the dLdA/dLdAPrev transformation is
// free of this, there are no bias columns in these matrices and both versions of matrices occupy the same amount of bytes.
// Here's how we could deal with source data_x and activation matrix:
// data_x:
// 1. The most efficient way is to require data_x to have necessary (transformed to [k*m, n+1]) structure before
//		training process starts. It's a fairly natural way when the data_x is actually a part of an input layer and the LPT is
//		the only user of the data.
//		In that case we aren't required to do any data_x processing on fprop() and bprop(). It could be fed into tiled layer
//		directly.
// 
// 2. Or we could transform the data_x from [m, k*n+1] to [k*m, n+1]. It could be the case when LPT is placed on top of an output
//		of a set of feature detectors. It could be done in two ways:
//		1) We could require the original (incoming into LPT) matrix to have an additional special (placeholder) (k-1) columns.
//			In that case we could just transform data_x inplace and save additional memory. However, in general case we'd
//			have to transform the data_x back into original form before fprop() returns, because it may be used a bit
//			later during fprop() by another (parallel) layer's fprop(). The same procedure applies to bprop(). So, it's
//			more memory friendly, however very costly in terms of execution.
//		
//		2) or we may allocate additional new_data_x matrix of size [k*m, n+1] and transform original data_x into new_data_x.
//			In that case we don't modify the data_x and therefore shouldn't do anything else on fprop()/bprop() beginning and
//			ending. So it's a preferred solution, though it'd require more memory.
//

#include "_activation_storage.h"
#include "_pack_.h"
#include "../utils.h"

namespace nntl {
	

	//Special parameters are:
	// K_tiles - the number of tiles
	//
	// note, that by this time bExpectSpecialDataX is buggy and can not be used, so always pass false for this parameter
	// bExpectSpecialDataX - set it to true when LPT is placed on top of (probably, a part of) a layer_input and original
	//		training/testing data are specially prepared to be used by and only by the LPT. This parameter set corresponds
	//		to the first data_x scenario described earlier. It means, that incoming data despite being stored in [m, k*(n+1)] matrix,
	//		actually stored there as [k*m,n+1] matrix. And the last column has already have biases incorporated by a data source.
	//		If bExpectSpecialDataX is set to false, then incoming data is expected to be in a usual form of [m, k*n+1]
	//		matrix that must be transformed by the layer into separate matrix of [k*m,n+1], usable by the underlying layer.
	// note, that by this time bExpectSpecialDataX is buggy and can not be used, so always pass false for this parameter
	// moreover, deprecated bExpectSpecialDataX until clarification
	
	template<typename FinalPolymorphChild, typename LayerT /*, bool bExpectSpecialDataX*/>
	class _LPT : public _impl::_act_stor<FinalPolymorphChild, typename LayerT::interfaces_t>
		, public _impl::m_prop_stops_bprop_marker<LayerT>
	{
		static constexpr bool bExpectSpecialDataX = false;//but leave it here to be able to leave it's mentions in the code
		//static_assert(!bExpectSpecialDataX, "Sorry, bExpectSpecialDataX==true is buggy, set it to false");
		// Besides buggy implementation here in LPT, we have to solve the issue of passing (possibly) wrongly sized activation
		// matrix of previous (input) layer here in a way that only this layer could safely use it and also no
		// properly placed asserts in other layers (including compound layers that may contain this LPT) will be triggered.
		// Also note, that any other layer attempting to fetch such input_layer's activations, that are fed to LPT
		// with bExpectSpecialDataX==true, must fail in compile time.
		// Also note, that some compound layers, such as LPH, may allocate additional memory storage for its needs
		// that rely at least on current batch size (or may rely on previous layer's activations count)

	private:
		static_assert(!is_layer_input<LayerT>::value && !is_layer_output<LayerT>::value,
			"Tiled layer LayerT can't be an input or output layer!");
		static_assert(::std::is_base_of<_i_layer<real_t>, LayerT>::value,
			"LayerT parameter must implement _i_layer interface!");

		typedef _impl::_act_stor<FinalPolymorphChild, typename LayerT::interfaces_t> _base_class_t;

	public:
		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;
		typedef typename LayerT tiled_layer_t;

		static constexpr bool bAssumeFPropOnly = is_layer_stops_bprop<tiled_layer_t>::value;
		
	protected:
		tiled_layer_t& m_tiledLayer;

		realmtxdef_t m_innerLowerLayerActivations;
		// m_innerLowerLayerActivations is a transformed version of incoming data_x (or just a wrapper if the data_x is already transformed)
		
		//common_data_t m_innerCD;//struct to pass to m_tiledLayer, must be persistent to the m_tiledLayer init/deinit duration
		//It must be instantiated separately with the duration of m_tiledLayer because it'll contain DIFFERENT
		// values for maxInnerFPropRowsCount/maxInnerBPropRowsCount
		// no longer true
		
		//can't be const, or serialization will not work
		/*const*/ neurons_count_t m_tiles_count{ 0 };

		//////////////////////////////////////////////////////////////////////////
		//
	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(inc_neurons_cnt > 0);

			//for bExpectSpecialDataX inc_neurons_cnt must correspond to a number of columns of a matrix size [m, k*(n+1)],
			//		i.e. it is (k*(n+1)-1) - excluding added by the engine bias column
			//for !bExpectSpecialDataX, it must correspond to a matrix size [m, k*n+1], i.e. (k*n)
			NNTL_ASSERT((bExpectSpecialDataX && (inc_neurons_cnt % m_tiles_count == m_tiles_count-1))
				|| (!bExpectSpecialDataX && (inc_neurons_cnt % m_tiles_count == 0)));

			_base_class_t::_preinit_layer(ili, inc_neurons_cnt);

			//by c++ design, integer division is rounded towards zero, i.e. floor()ed
			_impl::_preinit_layers initializer(ili, inc_neurons_cnt / m_tiles_count);
			initializer(m_tiledLayer);
		}
		
	public:
		~_LPT()noexcept {}
		_LPT(const char* pCustomName, tiled_layer_t& tl, neurons_count_t KTiles)noexcept
			: _base_class_t(KTiles*tl.get_neurons_cnt(), pCustomName)
			, m_tiledLayer(tl) //, m_innerCD()//initialize m_innerCD by default
			, m_tiles_count(KTiles)
		{
			NNTL_ASSERT(KTiles > 0 || !"Hey, KTiles MUST be positive!");
			m_innerLowerLayerActivations.will_emulate_biases();
		}
		static constexpr const char _defName[] = "lpt";

		//////////////////////////////////////////////////////////////////////////
		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			call_F_for_each_layer(::std::forward<_Func>(f), m_tiledLayer);
		}
		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			call_F_for_each_layer_down(::std::forward<_Func>(f), m_tiledLayer);
		}
		template<typename _Func> void for_each_packed_layer(_Func&& f)const noexcept { ::std::forward<_Func>(f)(m_tiledLayer); }
		template<typename _Func> void for_each_packed_layer_down(_Func&& f)const noexcept { ::std::forward<_Func>(f)(m_tiledLayer); }

		//////////////////////////////////////////////////////////////////////////
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		bool hasLossAddendum()const noexcept { return m_tiledLayer.hasLossAddendum(); }

		//returns a loss function summand, that's caused by this layer
		real_t lossAddendum()const noexcept { 
			//return tiles_count*m_tiledLayer.lossAddendum();
			//m_tiledLayer works on transformed version of activations, not the partial version. Therefore, it computes the correct within itself
			return m_tiledLayer.lossAddendum();
		}

		//////////////////////////////////////////////////////////////////////////
		ErrorCode layer_init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			//we must first initialize tiled layer to obtain correct outgoing batch sizes
			_layer_init_data_t initD(lid.commonData, lid.incBS*m_tiles_count);
			initD.pass_to_upper_layer();
			auto ec = m_tiledLayer.layer_init(initD, nullptr);
			if (ErrorCode::Success != ec)return ec;
			//we'll perform aggregate_from() once finish updating lid with our own props

			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().layer_deinit();
			});

			NNTL_ASSERT(0 == initD.outgBS.maxBS % m_tiles_count);
			NNTL_ASSERT(0 == initD.outgBS.maxTrainBS % m_tiles_count);
			lid.outgBS = initD.outgBS / m_tiles_count;

			ec = _base_class_t::layer_init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			//allocating m_innerLowerLayerActivations matrix
			NNTL_ASSERT(m_innerLowerLayerActivations.emulatesBiases());
			if (!bExpectSpecialDataX) {
				//we'll use transformed matrix only if an incoming data is in the non-specialized format
				if (!m_innerLowerLayerActivations.resize(initD.outgBS.biggest(), m_tiledLayer.get_incoming_neurons_cnt()))
					return ErrorCode::CantAllocateMemoryForInnerLLActivations;
			}

			//intializing layer memory requirements
			NNTL_ASSERT(0 == lid.max_dLdA_numel && 0 == lid.maxMemFPropRequire && 0 == lid.maxMemTrainingRequire);
			//lid.max_dLdA_numel = realmtx_t::sNumel(max_fprop_batch_size(), get_neurons_cnt());

			if (get_common_data().is_training_possible() /*&& get_self().bDoBProp()*/) {
				//we're going to use dLdA & dLdAPrev variables to also hold dLdA & dLdAPrev for the tiled layer.
				//Therefore biggest dLdA could have a size of [max_BPropRowsCount, max(get_neurons_cnt(), m_tiledLayer.get_incoming_neurons_cnt())]
				// note that m_tiledLayer requirements will be taken in account with lid.aggregate_from(initD) later. Here we
				// describe only this layer's reqs, however, taking into account that it is this layer, who also "emulates" prev layer
				// for m_tiledLayer.
				lid.max_dLdA_numel = ::std::max(
					realmtx_t::sNumel(lid.outgBS.maxTrainBS, get_neurons_cnt()),//dLdA coming into this layer
					realmtx_t::sNumel(initD.incBS.maxTrainBS, m_tiledLayer.get_incoming_neurons_cnt()) //sizeof dLdAPrev for m_tiledLayer
				);
			}
			//finally integrating m_tiledLayer reqs into lid
			lid.aggregate_from(initD);

			bSuccessfullyInitialized = true;
			//#TODO need some way to return failedLayerIdx
			return ErrorCode::Success;
		}

		void layer_deinit() noexcept {
			m_tiledLayer.layer_deinit();
			m_innerLowerLayerActivations.clear();
			//m_innerCD.deinit();
			_base_class_t::layer_deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			m_tiledLayer.initMem(ptr, cnt);
		}

		vec_len_t on_batch_size_change(const vec_len_t incBatchSize, real_t*const pNewActivationStorage = nullptr)noexcept {
			//changing the mode of m_tiledLayer.
			//m_innerCD.set_mode_and_batch_size(get_common_data().is_training_mode(), tiledRowsCnt);
			const auto tiledIncBS = incBatchSize*m_tiles_count;
			const auto tiledOutgBS = m_tiledLayer.on_batch_size_change(tiledIncBS, nullptr);
			NNTL_ASSERT(tiledOutgBS > 0 && 0 == tiledOutgBS%m_tiles_count);

			const auto outBs = _base_class_t::on_batch_size_change(tiledOutgBS / m_tiles_count, pNewActivationStorage);

			//updating supplemental matrices
			if (!bExpectSpecialDataX)
				m_innerLowerLayerActivations.deform_batch_size_with_biases(tiledIncBS);
			return outBs;
		}

	protected:

		template<typename LLWrapT, bool _C = bExpectSpecialDataX>
		::std::enable_if_t<_C> _lpt_fprop_prepareInnerLLAct(const realmtx_t& prevAct)noexcept {
			static_assert(is_layer_input<LLWrapT>::value, "When bExpectSpecialDataX is set, the lowerLayer must be layer_input!");
			//and moreover, it must produce specially prepared data!
			NNTL_ASSERT(prevAct.bBatchInColumn() && m_innerLowerLayerActivations.bBatchInColumn());
// 			NNTL_ASSERT(prevAct.size()
// 				== realmtx_t::mtx_size_t(m_activations.rows(), m_tiles_count*(m_tiledLayer.get_incoming_neurons_cnt() + 1)));
			NNTL_ASSERT(prevAct.sample_size() == m_tiles_count*m_tiledLayer.get_incoming_neurons_cnt());
			NNTL_ASSERT(prevAct.test_biases_strict());

			// m_innerLowerLayerActivations are NOT expected to be changed anywhere later, therefore
			// the trick with the const_cast<> should do no harm.
			m_innerLowerLayerActivations.useExternalStorage(m_tiles_count*prevAct.batch_size()
				, m_tiledLayer.get_incoming_neurons_cnt() + 1, const_cast<real_t*>(prevAct.data()), true);
		}

		template<typename LLWrapT, bool _C = bExpectSpecialDataX>
		::std::enable_if_t<!_C> _lpt_fprop_prepareInnerLLAct(const realmtx_t& prevAct)noexcept {
			NNTL_ASSERT(prevAct.bBatchInColumn() && m_innerLowerLayerActivations.bBatchInColumn());
			NNTL_ASSERT(prevAct.sample_size() == m_tiles_count*m_tiledLayer.get_incoming_neurons_cnt());
			NNTL_ASSERT(prevAct.test_biases_strict());
			NNTL_ASSERT(m_innerLowerLayerActivations.emulatesBiases());
			NNTL_ASSERT(m_innerLowerLayerActivations.size()
				== realmtx_t::mtx_size_t(m_tiles_count*prevAct.rows(), m_tiledLayer.get_incoming_neurons_cnt() + 1));
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_strict());

			get_iMath().mTilingRoll(prevAct, m_innerLowerLayerActivations);
		}
		
		template<typename LLWrapT>
		void _lpt_fprop(const realmtx_t& prevAct)noexcept {
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());

			auto& iI = get_iInspect();
			iI.fprop_begin(get_layer_idx(), prevAct, get_common_data().is_training_mode());

			_lpt_fprop_prepareInnerLLAct<LLWrapT>(prevAct);
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_strict());

			m_tiledLayer.fprop(LLWrapT(m_innerLowerLayerActivations));

			get_iMath().mTilingUnroll(m_tiledLayer.get_activations(), m_activations);

			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());

			iI.fprop_activations(m_activations);
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		// in order to implement backprop for the m_tiledLayer, we must provide it with a correct dLdA and dLdAPrev
		template<typename LLWrapT>
		unsigned _lpt_bprop(realmtxdef_t& dLdA, realmtxdef_t& dLdAPrev, const realmtx_t& prevAct)noexcept {
			NNTL_ASSERT(is_activations_shared() || m_activations.test_biases_strict());
			NNTL_ASSERT(prevAct.test_biases_strict());

			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

			NNTL_ASSERT(get_common_data().is_training_mode());
			//we'd use m_innerLowerLayerActivations instead of lowerLayer.get_activations()
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_strict());
			NNTL_ASSERT(dLdA.size() == m_activations.size_no_bias());
			NNTL_ASSERT((!is_layer_with_bprop<LLWrapT>::value) || dLdAPrev.size() == prevAct.size_no_bias());

			auto& iI = get_iInspect();
			iI.bprop_begin(get_layer_idx(), dLdA);
			iI.bprop_finaldLdA(dLdA);

			// The only problem now is in size of dLdA and dLdAPrev. Therefore we'll change matrices size appropriately and
			// then transform dLdA into dLdAPrev storage. Then we'll use dLdAPrev as dLdA and vice versa. Once the bprop() finishes
			// we'd transform dLdAPrev into dLdA storage to propagate dLdA correctly down over the layers stack
			NNTL_ASSERT(!dLdA.emulatesBiases() && !dLdAPrev.emulatesBiases());
			dLdAPrev.deform_like_no_bias(m_tiledLayer.get_activations());

			auto& iM = get_iMath();
			iM.mTilingRoll(dLdA, dLdAPrev);

			constexpr bool bProducedLdAPrev = is_layer_with_bprop<LLWrapT>::value;
			//now the correct dLdA for the m_tiledLayer is actually in dLdAPrev. We're going to store dLdAPrev in dLdA
			//BTW: we've just switched the matrices, therefore at this moment we must return (1^switchMtxs) from bprop()
			if (bProducedLdAPrev) {
				dLdA.deform_like_no_bias(m_innerLowerLayerActivations);
			} else dLdA.deform(0, 0);

			const auto switchMtxs = m_tiledLayer.bprop(dLdAPrev, LLWrapT(m_innerLowerLayerActivations), dLdA);
			//here we must use a return value of (1^switchMtxs)

			if (bProducedLdAPrev) {
				//reassigning variables to forget about switching
				// if switchMtxs, then the true dLdA (for THIS layer) is still the first m_tiledLayer.bprop() argument,
				//		i.e. it's stored in dLdAPrev variable.
				// if switchMtxs, then the true dLdAPrev (for THIS layer) is still the third m_tiledLayer.bprop() argument,
				//		i.e. it's stored in dLdA variable.
				realmtxdef_t& _dLdA = switchMtxs ? dLdAPrev : dLdA;
				realmtxdef_t& _dLdAPrev = switchMtxs ? dLdA : dLdAPrev;
				NNTL_ASSERT(_dLdAPrev.size() == m_innerLowerLayerActivations.size_no_bias());

				//now we should unroll _dLdAPrev value into the correct rolled version
				_dLdA.deform_like_no_bias(prevAct);
				iM.mTilingUnroll(_dLdAPrev, _dLdA);
				//we've switched dLdA matrices again. This adds another 1 to the function return value.
				//We should return (1^1^switchMtxs)==(0^switchMtxs)==switchMtxs here
			}
			const unsigned ret = switchMtxs ^ static_cast<unsigned>(!bProducedLdAPrev);

			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_strict());
			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}

	public:
		////////////////////////////////////////////////////////////////////////////
		
		template <typename LowerLayer>
		void fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(::std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			get_self()._lpt_fprop<_impl::wrap_trainable_layer<LowerLayer>>(lowerLayer.get_activations());
		}

		template <typename LowerLayer>
		unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(::std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");
			static_assert(!bAssumeFPropOnly, "");
			return get_self()._lpt_bprop<_impl::wrap_trainable_layer<LowerLayer>>(dLdA, dLdAPrev, lowerLayer.get_activations());
		}

	private:
		//support for ::boost::serialization
		friend class ::boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			NNTL_UNREF(version);

			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
			if (utils::binary_option<true>(ar, serialization::serialize_data_x)) ar & NNTL_SERIALIZATION_NVP(m_innerLowerLayerActivations);
			
			ar & NNTL_SERIALIZATION_NVP(m_tiles_count);

			ar & serialization::make_named_struct(m_tiledLayer.get_layer_name_str().c_str(), m_tiledLayer);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _LPT
	// If you need to derive a new class, derive it from _LPT (to make static polymorphism work)

	//to shorten class name to get rid of C4503
	template <typename LayerT>
	class LPT final : public _LPT<LPT<LayerT>, LayerT> {
	public:
		~LPT() noexcept {};
		LPT(LayerT& tl, neurons_count_t KTiles, const char* pCustomName=nullptr) noexcept
			: _LPT<LPT<LayerT>, LayerT>(pCustomName, tl, KTiles)
		{};

		LPT(const char* pCustomName, LayerT& tl, neurons_count_t KTiles) noexcept
			: _LPT<LPT<LayerT>, LayerT>(pCustomName, tl, KTiles)
		{};
	};

	template <typename LayerT>
	using layer_pack_tile = typename LPT<LayerT>;

}
