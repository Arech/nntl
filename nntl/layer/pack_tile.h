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

// layer_pack_tile (LPT) is a similar layer type to layer_pack_horizontal in a sense that it breaks input neurons into
// K groups to be fed into K processing units (layers) and gathers output of that K processing units into a single activation
// matrix. The difference is in a fact that LPT uses only single one processing unit. LPT just tiles it K times to cover all
// inputs. It may be very helpful when one needs to process K groups of data, that were made by a single original data source
// and therefore should be processed by the same feature detector.
// Here is a picture
// 
//   |a1_1..a1_a. . . .ai_1..ai_a. . . .ak_1..ak_a|			- LPT activation matrix consists of k individual activation submatrices.
// |----------------layer_pack_tile-----------------|			It has k*a columns (k*a+1 for the bias column) and m rows.
// |	|	|			 |	 |			  |	  | 	|
// |  a1_1..a1_a   .   ai_1..ai_a   .   ak_1..ak_a  |
// | | L_inst_1 |  .  | L_inst_i |  .  | L_inst_k | |		- k "instances" of the same layer (actually it's the same layer application)
// |  x1_1..x1_n   .   xi_1..xi_n   .   xk_1..xk_n  |		- source data, k groups with n columns each. (actually, n+1 columns
// |   |   |   |		|   |   |		 |   |   |	|		-  - remember the bias column).
// |------------------------------------------------|
//   |x1_1..x1_n. . . .xi_1..xi_n. . . .xk_1..xk_n|			- source data concatentenated, k*n columns (k*n+1 for the bias column) and m rows.
//
// Now the tricky part: how to make it work? We must implement correct fprop() and bprop() routines, i.e. we should produce correct
// activations in fprop() and by processing dLdA in bprop() produce correct dLdAPrev and update layer`s weights.
// We can take layer_pack_horizontal approach for fprop() and apply the layer k times to k groups of data to produce activation values.
// However, we can't run bprop() after that, because by design any layer expects that bprop() is called for the same data that was
// passed to fprop(). For example, dropout relies on that property and expects it's dropoutMask variable to retain it's value between
// corresponding fprop() and bprop() calls. That would not be the case, because when the bprop() will be called first time
// for the first layer "instance", it's dropoutMask will be describing the dropout state for the last layer instance, not the first.
// All internal layer's state that corresponds to all fprop() calls except for the last call will be lost.
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
// free of this, there're no bias columns in these matrices and both versions of matrices occupy the same amount of bytes.
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
// Almost the same applies to the activation matrix. We could either
// 1. allocate activation matrix of size [k*m,a+1], fprop() data directly into it, and then transform it inplace into [m,k*a+1].
//		That would require us to make the inverse transformation before layer.bprop() call.
// 2. or allocate in fact two activation matrices. One of size [k*m,a+1] to be used inside layer.fprop(), and the other of
//		size [m,k*a+1] to be used as data source for upper layers. This approach doen't require additional inverse transformation
//		before layer.bprop() at the cost of additional (m*k*a+m)*sizeof(real_t) bytes



#include "_pack_.h"
#include "../utils.h"

namespace nntl {

	//Special parameters are:
	// K_tiles - the number of tiles
	// bExpectSpecialDataX - set it to true when LPT is placed on top of (probably, a part of) a layer_input and original
	//		training/testing data are specially prepared to be used in and only by the LPT. This parameter set corresponds
	//		to the first data_x scenario described earlier. It means, that incoming data despite being stored in [m, k*(n+1)] matrix,
	//		actually stored there as [k*m,n+1] matrix. And the last column already has biases incorporated by a data source.
	//		If bExpectSpecialDataX is set to false, then incoming data is expected to be in a usual form of [m, k*n+1]
	//		matrix that must be transformed by the layer into separate matrix of [k*m,n+1], usable by the underlying layer.
	//		
	// NOT SUPPORTED YET bool bSingleActivationMatrix - set this parameter to true to use a single output activation matrix to reduce memory usage at
	//		the cost of additional transformation step. Use with care because after bprop() step activation matrix will be
	//		transformed.
	// 
	template<typename FinalPolymorphChild, typename LayerT, neurons_count_t K_tiles, bool bExpectSpecialDataX>
	class _layer_pack_tile 
		: public _layer_base<FinalPolymorphChild, typename LayerT::interfaces_t>
	{
	private:
		static_assert(!std::is_base_of<m_layer_input, LayerT>::value && !std::is_base_of<m_layer_output, LayerT>::value,
			"Tiled layer LayerT can't be an input or output layer!");
		static_assert(std::is_base_of<_i_layer<real_t>, LayerT>::value,
			"LayerT parameter must implement _i_layer interface!");

		typedef _layer_base<FinalPolymorphChild, typename LayerT::interfaces_t> _base_class;

	public:
		//LayerPack_t is used to distinguish ordinary layers from layer packs (for example, to implement call_F_for_each_layer())
		typedef self_t LayerPack_t;
		typedef typename LayerT tiled_layer_t;
		
		static constexpr neurons_count_t tiles_count = K_tiles;
		static_assert(tiles_count > 1, "Tiles count must be greater than one!");

	protected:
		tiled_layer_t& m_tiledLayer;

		realmtxdef_t m_activations, m_innerActivations, m_innerLowerLayerActivations;
		//m_innerActivations is a matrix of size [k*m,a+1] to receive output from m_tiledLayer. It's content then transformed
		//		into [m,k*a+1] m_activations matrix.
		//		We're allocating additional matrix and will pass it into m_tiledLayer's init()/on_batch_size_change() instead of using
		//			m_tiledLayer's own activations to anticipate inplace activation matrix transformation (which is not
		//			implemented yet). This should make it clear that activation matrix could be transformed.
		//			WTFIT???
		//			
		// m_innerLowerLayerActivations is a transformed version of incoming data_x (or just a wrapper if the data_x is already
		//		transformed)

		//m_dropSamplesMask uses its chunk of memory passed to initMem() to evaluate non-trivial m_tiledLayer.drop_samples() control flow
		//Size [k*m,1].
		realmtxdef_t m_dropSamplesMask;
		
		common_data_t m_innerCD;//struct to pass to m_tiledLayer, must be persistent to the m_tiledLayer init/deinit duration
		//It must be instantiated separately with the duration of m_tiledLayer because it'll contain DIFFERENT
		// values for maxInnerFPropRowsCount/maxInnerBPropRowsCount
		
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
			NNTL_ASSERT((bExpectSpecialDataX && (inc_neurons_cnt % tiles_count == tiles_count-1))
				|| (!bExpectSpecialDataX && (inc_neurons_cnt % tiles_count == 0)));

			_base_class::_preinit_layer(ili, inc_neurons_cnt);

			//by c++ design, integer division is rounded towards zero, i.e. floor()ed
			_impl::_preinit_layers initializer(ili, inc_neurons_cnt / tiles_count);
			initializer(m_tiledLayer);
		}
		
	public:
		~_layer_pack_tile()noexcept {}
		_layer_pack_tile(const char* pCustomName, tiled_layer_t& tl)noexcept
			: _base_class(tiles_count*tl.get_neurons_cnt(), pCustomName)
			, m_tiledLayer(tl), m_innerCD()//initialize m_innerCD by default
		{
			m_activations.will_emulate_biases();
			m_innerActivations.will_emulate_biases();
			m_innerLowerLayerActivations.will_emulate_biases();
			m_dropSamplesMask.dont_emulate_biases();
		}
		static constexpr const char _defName[] = "lpt";

		const realmtxdef_t& get_activations()const noexcept { 
			NNTL_ASSERT(m_bActivationsValid);
			return m_activations;
		}
		const mtx_size_t get_activations_size()const noexcept { return m_activations.size(); }

		const bool is_activations_shared()const noexcept {
			const auto r = _base_class::is_activations_shared();
			NNTL_ASSERT(!m_tiledLayer.is_activations_shared());
			NNTL_ASSERT(!r || m_activations.bDontManageStorage());//shared activations can't manage their own storage
			return r;
		}

		//////////////////////////////////////////////////////////////////////////
		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)const noexcept {
			call_F_for_each_layer(std::forward<_Func>(f), m_tiledLayer);
		}
		template<typename _Func>
		void for_each_layer_down(_Func&& f)const noexcept {
			call_F_for_each_layer_down(std::forward<_Func>(f), m_tiledLayer);
		}
		template<typename _Func> void for_each_packed_layer(_Func&& f)const noexcept { std::forward<_Func>(f)(m_tiledLayer); }
		template<typename _Func> void for_each_packed_layer_down(_Func&& f)const noexcept { std::forward<_Func>(f)(m_tiledLayer); }

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
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			auto ec = _base_class::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;
			bool bSuccessfullyInitialized = false;
			utils::scope_exit onExit([&bSuccessfullyInitialized, this]() {
				if (!bSuccessfullyInitialized) get_self().deinit();
			});
			
			const auto maxInnerFPropRowsCount = get_self().get_common_data().max_fprop_batch_size()*tiles_count;
			const auto maxInnerBPropRowsCount = get_self().get_common_data().training_batch_size()*tiles_count;
			const auto biggestInnerRowsCount = std::max(maxInnerFPropRowsCount, maxInnerBPropRowsCount);
			const auto biggestBatchSize = get_self().get_common_data().biggest_batch_size();

			//initialize common data for the m_tiledLayer
			m_innerCD.setInterfacesFrom(get_self().get_common_data());
			m_innerCD.init(maxInnerFPropRowsCount, maxInnerBPropRowsCount);

			//allocating m_activations
			NNTL_ASSERT(m_activations.emulatesBiases());
			if (pNewActivationStorage) {
				m_activations.useExternalStorage(pNewActivationStorage, biggestBatchSize, get_self().get_neurons_cnt() + 1, true);
			} else {
				if (!m_activations.resize(biggestBatchSize, get_self().get_neurons_cnt()))
					return ErrorCode::CantAllocateMemoryForActivations;
			}

			//allocating innerActivations matrix
			NNTL_ASSERT(m_innerActivations.emulatesBiases());
			if (!m_innerActivations.resize(biggestInnerRowsCount, m_tiledLayer.get_neurons_cnt()))
				return ErrorCode::CantAllocateMemoryForInnerActivations;

			//allocating m_innerLowerLayerActivations matrix
			NNTL_ASSERT(m_innerLowerLayerActivations.emulatesBiases());
			if (!bExpectSpecialDataX) {
				//we'll use transformed matrix only if an incoming data is in the non-specialized format
				if (!m_innerLowerLayerActivations.resize(biggestInnerRowsCount, m_tiledLayer.get_incoming_neurons_cnt()))
					return ErrorCode::CantAllocateMemoryForInnerLLActivations;
			}

			//intializing layer memory requirements
			NNTL_ASSERT(0 == lid.max_dLdA_numel && 0 == lid.maxMemFPropRequire && 0 == lid.maxMemTrainingRequire);
			//lid.max_dLdA_numel = realmtx_t::sNumel(get_self().max_fprop_batch_size(), get_self().get_neurons_cnt());

			//we're going to use dLdA & dLdAPrev variables to also hold dLdA & dLdAPrev for the tiled layer.
			//Therefore biggest dLdA could have a size of [max_BPropRowsCount, max(get_neurons_cnt(), m_tiledLayer.get_incoming_neurons_cnt())]
			lid.max_dLdA_numel = std::max(
				realmtx_t::sNumel(get_self().get_common_data().training_batch_size(), get_self().get_neurons_cnt()),//dLdA coming into this layer
				realmtx_t::sNumel(maxInnerBPropRowsCount, m_tiledLayer.get_incoming_neurons_cnt()) //sizeof dLdAPrev for m_tiledLayer
			);

			//now we must initialize m_tiledLayer. BTW, it's safe to pass dLdA & dLdAPrev to the layer, provided
			// that we'd return correct (aggregated) value for the max_dLdA_numel
			_layer_init_data_t initD(m_innerCD);
			initD.clean_using(lid);
			initD.bActivationsShareSpace = false;
			initD.nTiledTimes *= tiles_count;
			ec = m_tiledLayer.init(initD, m_innerActivations.data());
			if (ErrorCode::Success != ec)return ec;
			lid.update(initD);

			//requesting memory to store drop_samples() mask
			if (get_self().is_drop_samples_mbc()) {
				lid.maxMemFPropRequire += realmtx_t::sNumel(maxInnerFPropRowsCount, 1);
				lid.maxMemTrainingRequire += realmtx_t::sNumel(maxInnerBPropRowsCount, 1);
			}

			bSuccessfullyInitialized = true;
			//#TODO need some way to return failedLayerIdx
			return ec;
		}

		void deinit() noexcept {
			m_tiledLayer.deinit();
			m_activations.clear();
			m_innerActivations.clear();
			m_innerLowerLayerActivations.clear();
			m_dropSamplesMask.clear();
			m_innerCD.deinit();
			_base_class::deinit();
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			if (get_self().is_drop_samples_mbc()) {
				const auto maxInnerFPropRowsCount = get_self().get_common_data().max_fprop_batch_size()*tiles_count;
				const auto maxInnerBPropRowsCount = get_self().get_common_data().training_batch_size()*tiles_count;
				const auto biggestInnerRowsCount = std::max(maxInnerFPropRowsCount, maxInnerBPropRowsCount);
				m_dropSamplesMask.useExternalStorage(ptr, biggestInnerRowsCount, 1, false);
				const auto ne = m_dropSamplesMask.numel();
				ptr += ne;
				NNTL_ASSERT(cnt >= ne);
				cnt -= ne;
			}
			m_tiledLayer.initMem(ptr, cnt);
		}

		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			const vec_len_t batchSize = get_self().get_common_data().get_cur_batch_size();
			NNTL_ASSERT(batchSize > 0);
			NNTL_ASSERT(m_activations.emulatesBiases());
			// now we must resize m_activations and update activations of inner layers with on_batch_size_change variation
			m_bActivationsValid = false;
			const auto _biggest_batch_size = get_self().get_common_data().biggest_batch_size();
			NNTL_ASSERT(batchSize <= _biggest_batch_size);

			if (pNewActivationStorage) {
				NNTL_ASSERT(m_activations.bDontManageStorage());
				//m_neurons_cnt + 1 for biases
				m_activations.useExternalStorage(pNewActivationStorage, batchSize, get_self().get_neurons_cnt() + 1, true);
				//should not restore biases here, because for compound layers its a job for their fprop() implementation
			} else {
				NNTL_ASSERT(!m_activations.bDontManageStorage());
				m_activations.deform_rows(batchSize);
				if (batchSize != _biggest_batch_size) m_activations.set_biases();
				NNTL_ASSERT(m_activations.test_biases_ok());
			}

			//updating supplemental matrices
			NNTL_ASSERT(m_innerActivations.emulatesBiases());
			const auto tiledRowsCnt = batchSize*tiles_count;
			const auto tiled_biggest_batch_size = _biggest_batch_size*tiles_count;
			m_innerActivations.deform_rows(tiledRowsCnt);
			if (tiledRowsCnt != tiled_biggest_batch_size) m_innerActivations.set_biases();
			NNTL_ASSERT(m_innerActivations.test_biases_ok());

			if (!bExpectSpecialDataX) {
				m_innerLowerLayerActivations.deform_rows(tiledRowsCnt);
				if (tiledRowsCnt != tiled_biggest_batch_size) m_innerLowerLayerActivations.set_biases();
				NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());
			}

			if (get_self().is_drop_samples_mbc()) {
				NNTL_ASSERT(!m_dropSamplesMask.emulatesBiases() && !m_dropSamplesMask.empty() && 1 == m_dropSamplesMask.cols());
				m_dropSamplesMask.deform_rows(tiledRowsCnt);
			}

			//changing the mode of m_tiledLayer.
			//m_innerCD.set_training_mode(get_self().is_training_mode());
			m_innerCD.set_mode_and_batch_size(get_self().get_common_data().is_training_mode(), tiledRowsCnt);
			m_tiledLayer.on_batch_size_change(m_innerActivations.data());
		}

		////////////////////////////////////////////////////////////////////////////
		// FProp for the incoming data that was properly transformed for use in a tiled layer
		template <typename LowerLayer, bool _C = bExpectSpecialDataX>
		std::enable_if_t<_C> fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			static_assert(std::is_base_of<m_layer_input, LowerLayer>::value, "When bExpectSpecialDataX is set the lowerLayer must be layer_input!");
			//and moreover, it must produce specially prepared data!
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().get_common_data().is_training_mode());

			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());
			//restoring biases, should they were altered in drop_samples()
			if (m_activations.isHoleyBiases() && !get_self().is_activations_shared()) {
				m_activations.set_biases();
			}

			auto& llAct = lowerLayer.get_activations();
			NNTL_ASSERT(llAct.test_biases_ok());
			NNTL_ASSERT(llAct.size() == realmtx_t::mtx_size_t(m_activations.rows()
				, tiles_count*(m_tiledLayer.get_incoming_neurons_cnt() + 1)));

			// m_innerLowerLayerActivations are NOT expected to be changed anywhere later, therefore the trick with the const_cast<> should do no harm.
			m_innerLowerLayerActivations.useExternalStorage(const_cast<real_t*>(llAct.data())
				, tiles_count*m_activations.rows(), m_tiledLayer.get_incoming_neurons_cnt() + 1, true);
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());
			
			m_tiledLayer.fprop(_impl::trainable_layer_wrapper<LowerLayer>(m_innerLowerLayerActivations));

			get_self().get_iMath().mTilingUnroll(m_innerActivations, m_activations);

			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}
		// FProp for the incoming data that wasn't transformed for use in a tiled layer
		template <typename LowerLayer, bool _C = bExpectSpecialDataX>
		std::enable_if_t<!_C> fprop(const LowerLayer& lowerLayer)noexcept {
			static_assert(std::is_base_of<_i_layer_fprop, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_fprop");
			auto& iI = get_self().get_iInspect();
			iI.fprop_begin(get_self().get_layer_idx(), lowerLayer.get_activations(), get_self().get_common_data().is_training_mode());

			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());
			auto& llAct = lowerLayer.get_activations();
			NNTL_ASSERT(llAct.test_biases_ok());
			NNTL_ASSERT(llAct.size() == realmtx_t::mtx_size_t(m_activations.rows()
				, tiles_count*m_tiledLayer.get_incoming_neurons_cnt() + 1));
			NNTL_ASSERT(m_innerLowerLayerActivations.emulatesBiases());
			NNTL_ASSERT(m_innerLowerLayerActivations.size() == realmtx_t::mtx_size_t(tiles_count*m_activations.rows()
				, m_tiledLayer.get_incoming_neurons_cnt() + 1));
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());

			auto& iM = get_self().get_iMath();
			iM.mTilingRoll(llAct, m_innerLowerLayerActivations);

			m_tiledLayer.fprop(_impl::trainable_layer_wrapper<LowerLayer>(m_innerLowerLayerActivations));

			iM.mTilingUnroll(m_innerActivations, m_activations);

			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());
			iI.fprop_end(m_activations);
			m_bActivationsValid = true;
		}

		// in order to implement backprop for the m_tiledLayer, we must provide it with a correct dLdA and dLdAPrev
		template <typename LowerLayer>
		const unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept {
			static_assert(std::is_base_of<_i_layer_trainable, LowerLayer>::value, "Template parameter LowerLayer must implement _i_layer_trainable");

			NNTL_ASSERT(m_bActivationsValid);
			m_bActivationsValid = false;

			auto& iI = get_self().get_iInspect();
			iI.bprop_begin(get_self().get_layer_idx(), dLdA);

			NNTL_ASSERT(m_activations.rows() == get_self().get_common_data().get_cur_batch_size());
			NNTL_ASSERT(get_self().get_common_data().is_training_mode());
			//we'd use m_innerLowerLayerActivations instead of lowerLayer.get_activations()
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());
			NNTL_ASSERT(m_innerActivations.test_biases_ok());

			NNTL_ASSERT(dLdA.size() == m_activations.size_no_bias());

			NNTL_ASSERT((std::is_base_of<m_layer_input, LowerLayer>::value) || dLdAPrev.size() == lowerLayer.get_activations().size_no_bias());

			// The only problem now is in size of dLdA and dLdAPrev. Therefore we'll change matrices size appropriately and
			// then transform dLdA into dLdAPrev storage. Then we'll use dLdAPrev as dLdA and vice versa. Once the bprop() finishes
			// we'd transform dLdAPrev into dLdA storage to propagate dLdA correctly down over the layers stack
			NNTL_ASSERT(!dLdA.emulatesBiases() && !dLdAPrev.emulatesBiases());
			dLdAPrev.deform_like_no_bias(m_innerActivations);

			auto& iM = get_self().get_iMath();
			iM.mTilingRoll(dLdA, dLdAPrev);

			constexpr bool bProducedLdAPrev = !std::is_base_of<m_layer_input, LowerLayer>::value;
			//now the correct dLdA for the m_tiledLayer is actually in dLdAPrev. We're going to store dLdAPrev in dLdA
			//BTW: we've just switched the matrices, therefore at this moment we must return (1^switchMtxs) from bprop()
			if (bProducedLdAPrev) {
				dLdA.deform_like_no_bias(m_innerLowerLayerActivations);
			}else dLdA.deform(0, 0);

			const auto switchMtxs = m_tiledLayer.bprop(dLdAPrev, _impl::trainable_layer_wrapper<LowerLayer>(m_innerLowerLayerActivations), dLdA);
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
				_dLdA.deform_like_no_bias(lowerLayer.get_activations());
				iM.mTilingUnroll(_dLdAPrev, _dLdA);
				//we've switched dLdA matrices again. This adds another 1 to the function return value.
				//We should return (1^1^switchMtxs)==(0^switchMtxs)==switchMtxs here
			}
			const unsigned ret = switchMtxs ^ static_cast<unsigned>(!bProducedLdAPrev);
			NNTL_ASSERT(m_innerActivations.test_biases_ok());
			NNTL_ASSERT(m_innerLowerLayerActivations.test_biases_ok());
			iI.bprop_end(ret ? dLdAPrev : dLdA);
			return ret;
		}
		//////////////////////////////////////////////////////////////////////////

		//#TODO we should adopt last_layer().drop_activations_is_trivial() function signature here, but it look like non-trivial
		//to detect if the constexpr attribute was used.
		//If the m_tiledLayer.drop_activations_is_trivial() then ours drop_activations() is trivial too
		const bool is_trivial_drop_samples() const noexcept { return m_tiledLayer.is_trivial_drop_samples(); }

		void drop_samples(const realmtx_t& mask, const bool bBiasesToo)noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT(get_self().is_drop_samples_mbc());
			NNTL_ASSERT(!get_self().is_activations_shared() || !bBiasesToo);
			NNTL_ASSERT(!mask.emulatesBiases() && 1 == mask.cols() && m_activations.rows() == mask.rows() && mask.isBinary());
			NNTL_ASSERT(m_activations.emulatesBiases());

			m_activations.hide_last_col();
			get_self().get_iMath().mrwMulByVec(m_activations, mask.data());
			m_activations.restore_last_col();

			if (bBiasesToo) {
				m_activations.copy_biases_from(mask.data());
			}

			if (m_tiledLayer.is_trivial_drop_samples()) {
				//just skipping m_tiledLayer.drop_samples() completely. BProp will be fine due to a correct (holey) dLdA passed
			} else {
				//we should preallocate  memory for the rolled mask. However, we can't use iMath's internal storage
				//because there're no guarantees it won't be used during m_tiledLayer.drop_activations().
				// Also, we shouldn't make this matrix/memory for it private for this object only - it's too expensive
				if (!get_self().is_drop_samples_mbc()) {
					STDCOUTL("Unexpected call to .drop_samples(). _layer_init_data_t::bDropSamplesMightBeCalled must be set during init!");
					abort();
				}

				NNTL_ASSERT(!m_dropSamplesMask.empty() && mask.rows()*tiles_count == m_dropSamplesMask.rows() && 1 == m_dropSamplesMask.cols());

				m_dropSamplesMask.deform(mask.rows(), tiles_count);
				get_self().get_iMath().mCloneCol(mask, m_dropSamplesMask);
				m_dropSamplesMask.deform(mask.rows()* tiles_count, 1);

				m_tiledLayer.drop_samples(m_dropSamplesMask);
			}
		}


	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive & ar, const unsigned int version) {
			if (utils::binary_option<true>(ar, serialization::serialize_activations)) ar & NNTL_SERIALIZATION_NVP(m_activations);
			if (utils::binary_option<true>(ar, serialization::serialize_data_x)) ar & NNTL_SERIALIZATION_NVP(m_innerLowerLayerActivations);
			
			ar & NNTL_SERIALIZATION_NVP(tiles_count);

			ar & serialization::make_named_struct(m_tiledLayer.get_layer_name_str().c_str(), m_tiledLayer);
		}
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_pack_tile
	// If you need to derive a new class, derive it from _layer_pack_tile (to make static polymorphism work)

	//to shorten class name to get rid of C4503
	template <typename LayerT, neurons_count_t K_tiles, bool bExpectSpecialDataX>
	class LPT final
		: public _layer_pack_tile<LPT<LayerT, K_tiles, bExpectSpecialDataX>, LayerT, K_tiles, bExpectSpecialDataX>
	{
	public:
		~LPT() noexcept {};
		LPT(LayerT& tl, const char* pCustomName=nullptr) noexcept
			: _layer_pack_tile<LPT<LayerT, K_tiles, bExpectSpecialDataX>, LayerT, K_tiles, bExpectSpecialDataX>(pCustomName, tl)
		{};

		LPT(const char* pCustomName, LayerT& tl) noexcept
			: _layer_pack_tile<LPT<LayerT, K_tiles, bExpectSpecialDataX>, LayerT, K_tiles, bExpectSpecialDataX>(pCustomName, tl)
		{};
	};

	template <typename LayerT, neurons_count_t K_tiles, bool bExpectSpecialDataX>
	using layer_pack_tile = typename LPT<LayerT, K_tiles, bExpectSpecialDataX>;

	template <neurons_count_t K_tiles, bool bExpectSpecialDataX, typename LayerT> inline constexpr
	LPT <LayerT, K_tiles, bExpectSpecialDataX> make_layer_pack_tile(LayerT& tl, const char* pCustomName = nullptr) noexcept
	{
		return LPT<LayerT, K_tiles, bExpectSpecialDataX>(tl, pCustomName);
	}
}
