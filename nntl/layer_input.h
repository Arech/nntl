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

#include "_layer_base.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	// class to derive from when making final input layer. Need it to propagate correct FinalPolymorphChild to
	// static polymorphism implementation here and in layer__base
	//template<typename FinalPolymorphChild, typename MathInterface = nnet_def_interfaces::Math, typename RngInterface = nnet_def_interfaces::Rng>
// 	class _layer_input : public m_layer_input, public _layer_base<FinalPolymorphChild, MathInterface, RngInterface> {
// 	private:
// 		typedef _layer_base<FinalPolymorphChild, MathInterface, RngInterface> _base_class;
	template<typename FinalPolymorphChild>
	class _layer_input : public m_layer_input, public _layer_base<FinalPolymorphChild> {
	private:
		typedef _layer_base<FinalPolymorphChild> _base_class;

	public:

		_layer_input(const neurons_count_t _neurons_cnt)noexcept :
			_base_class(_neurons_cnt), m_pActivations(nullptr) {};
		~_layer_input() noexcept {};

		const floatmtx_t& get_activations()const noexcept {
			NNTL_ASSERT(nullptr != m_pActivations);
			return *m_pActivations;
		}

		constexpr bool is_input_layer()const noexcept { return true; }

		//template <typename i_math_t = nnet_def_interfaces::Math, typename i_rng_t = nnet_def_interfaces::Rng>
		//ErrorCode init(vec_len_t batchSize, numel_cnt_t& minMemFPropRequire, numel_cnt_t& minMemBPropRequire, i_math_t& iMath, i_rng_t& iRng)noexcept {
		template<typename _layer_init_data_t>
		ErrorCode init(_layer_init_data_t& lid)noexcept{
			static_assert(std::is_base_of<math::_i_math, _layer_init_data_t::i_math_t>::value, "i_math_t type should be derived from _i_math");
			static_assert(std::is_base_of<rng::_i_rng, _layer_init_data_t::i_rng_t>::value, "i_rng_t type should be derived from _i_rng");
			m_pActivations = nullptr;
			return ErrorCode::Success;
		}
		void deinit()noexcept {
			m_pActivations = nullptr;
		}


		void initMem(float_t_* ptr, numel_cnt_t cnt)noexcept {}
		void set_mode(vec_len_t batchSize)noexcept {}

		//template <typename i_math_t = nnet_def_interfaces::Math, typename i_rng_t = nnet_def_interfaces::Rng>
		//void fprop(const floatmtx_t& data_x, i_math_t& iMath, i_rng_t& iRng, const bool bInTraining)noexcept{
		//template <typename LowerLayer>
		void fprop(const floatmtx_t& data_x)noexcept{
			//NNTL_ASSERT(data_x.emulatesBiases());
			//cant check it here because in mini-batch version data_x will be a slice (including bias) from original train_x. Therefore
			// just leave the check on nnet class

			m_pActivations = &data_x;
		}

		//template <typename i_math_t = nnet_def_interfaces::Math>
		//void bprop(const floatmtx_t& dLdA, const floatmtx_t& prevActivations, floatmtx_t& dLdAPrev, i_math_t& iMath, const bool bPrevLayerIsInput)noexcept {
		template <typename LowerLayer>
		void bprop(floatmtx_t& dLdA, const LowerLayer& lowerLayer, floatmtx_t& dLdAPrev)noexcept{
			//static_assert(false, "There is no bprop() for input_layer!");
			// will be used in invariant backprop algo
			std::cout << "***** bprop in input layer " << (int)get_layer_idx() << std::endl;
		}

	protected:
		friend class _preinit_layers;
		void _preinit_layer(const layer_index_t idx, const neurons_count_t inc_neurons_cnt)noexcept {
			NNTL_ASSERT(0 == idx);
			NNTL_ASSERT(0 == inc_neurons_cnt);
			_base_class::_preinit_layer(idx, inc_neurons_cnt);

			//don't allocate activation vector here, cause it'll be received from fprop().
			//m_activations.resize(m_neurons_cnt);//there is no need to initialize allocated memory
		}

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		const floatmtx_t* m_pActivations;
	};


	//////////////////////////////////////////////////////////////////////////
	// final implementation of layer with all functionality of _layer_input
	// If you need to derive a new class, derive it from _layer_input (to make static polymorphism work)
	class layer_input final : public _layer_input<layer_input> {
	public:
		layer_input(const neurons_count_t _neurons_cnt) noexcept :
			_layer_input<layer_input>(_neurons_cnt) {};
		~layer_input() noexcept {};
	};

}