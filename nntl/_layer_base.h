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

//#include "layers_pack.h"
#include "_nnet_errs.h"
#include "nnet_def_interfaces.h"
#include "grad_works.h"

namespace nntl {

	namespace _impl {

		template<typename i_math_t_, typename i_rng_t_>
		struct _layer_init_data {
			typedef i_math_t_ i_math_t;
			typedef i_rng_t_ i_rng_t;

			static_assert(std::is_base_of<math::_i_math, i_math_t>::value, "i_math_t type should be derived from _i_math");
			static_assert(std::is_base_of<rng::_i_rng, i_rng_t>::value, "i_rng_t type should be derived from _i_rng");

			typedef math_types::floatmtx_ty floatmtx_t;
			typedef floatmtx_t::vec_len_t vec_len_t;
			typedef floatmtx_t::numel_cnt_t numel_cnt_t;


			IN i_math_t& iMath;
			IN i_rng_t& iRng;
			//fprop and bprop may use different batch sizes during single training session (for example, fprop()/bprop() uses small batch size
			// during learning process, but whole data_x.rows() during fprop() for loss function computation. Therefore to reduce memory
			// consumption during learning, we will demark fprop() memory requirements and bprop() mem reqs.
			OUT numel_cnt_t maxMemFPropRequire;//this value should be set by layer.init()
			OUT numel_cnt_t maxMemBPropRequire;//this value should be set by layer.init()
			OUT numel_cnt_t max_dLdA_numel;//this value should be set by layer.init()
			OUT numel_cnt_t nParamsToLearn;//total number of parameters, that layer has to learn during training
			IN const vec_len_t max_fprop_batch_size;//usually this is data_x.rows()
			IN const vec_len_t training_batch_size;//usually this is batchSize

			_layer_init_data(i_math_t& im, i_rng_t& ir, vec_len_t fbs, vec_len_t bbs) noexcept
				: iMath(im), iRng(ir), max_fprop_batch_size(fbs), training_batch_size(bbs)
			{
				NNTL_ASSERT(max_fprop_batch_size >= training_batch_size);
			}
		};

	}

	//////////////////////////////////////////////////////////////////////////
	// layer interface definition
	class _i_layer {
	protected:
		_i_layer()noexcept {};
		~_i_layer()noexcept {};

		//!! copy constructor not needed
		_i_layer(const _i_layer& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		_i_layer& operator=(const _i_layer& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		typedef _nnet_errs::ErrorCode ErrorCode;
		typedef math_types::floatmtx_ty floatmtx_t;
		typedef floatmtx_t::vec_len_t vec_len_t;
		typedef floatmtx_t::numel_cnt_t numel_cnt_t;
		typedef floatmtx_t::value_type float_t_;
		typedef floatmtx_t::mtx_size_t mtx_size_t;

		//////////////////////////////////////////////////////////////////////////
		// base interface
		// each call to own functions should go through get_self() to make polymorphyc function work
		nntl_interface auto get_self() const noexcept;
		nntl_interface const layer_index_t get_layer_idx() const noexcept;
		nntl_interface const neurons_count_t get_incoming_neurons_cnt()const noexcept;
		nntl_interface const bool is_input_layer()const noexcept;
		nntl_interface const bool is_output_layer()const noexcept;

		nntl_interface const float_t_ learning_rate()const noexcept;
		nntl_interface auto learning_rate(float_t_ lr)noexcept;

		//it turns out that this function looks quite contradictory... Leave it until find out what's better to do..
		nntl_interface const floatmtx_t& get_activations()const noexcept;

		//batchSize==0 puts layer into training mode with batchSize predefined by init()::lid.training_batch_size
		// any batchSize>0 puts layer into evaluation/testing mode with that batchSize. bs must be <= init()::lid.max_fprop_batch_size
		nntl_interface void set_mode(vec_len_t batchSize)noexcept;

		//minMemFPropRequire and minMemBPropRequire - vars to be set by init() implementation. Provides a way to reserve necessary
		// memory to hold temporary data during fprop() and bprop(). At least this amount (minMemFPropRequire+minMemBPropRequire) of memory
		// will be passed to subsequent call to initMem().
		//template <typename i_math_t = nnet_def_interfaces::Math, typename i_rng_t = nnet_def_interfaces::Rng>
		//nntl_interface ErrorCode init(vec_len_t batchSize, numel_cnt_t& minMemFPropRequire, numel_cnt_t& minMemBPropRequire, i_math_t& iMath, i_rng_t& iRng)noexcept;
		template<typename _layer_init_data_t>
		nntl_interface ErrorCode init(_layer_init_data_t& lid)noexcept;

		//frees any temporary resources that doesn't contain layer-specific information (i.e. layer weights shouldn't be freed).
		//In other words, this routine burns some unnecessary fat layer gained during training, but don't touch any data necessary
		// for new subsequent call to nnet.train()
		nntl_interface void deinit() noexcept;

		//provides a temporary storage for a layer. It is guaranteed, that during fprop() or bprop() the storage can be modified only by the layer
		// (it's a shared memory and it can be modified elsewhere between calls to fprop()/bprop())
		//function is guaranteed to be called if (minMemFPropRequire+minMemBPropRequire) set to >0 during init()
		// cnt is guaranteed to be at least as big as (minMemFPropRequire+minMemBPropRequire)
		nntl_interface void initMem(float_t_* ptr, numel_cnt_t cnt)noexcept;

		//input layer should use slightly different specialization: void fprop(const floatmtx_t& data_x)noexcept
		template <typename LowerLayer>
		nntl_interface void fprop(const LowerLayer& lowerLayer)noexcept;

		//dLdA is derivative of loss function wrt this layer neuron activations. Size [batchSize x (layer_neuron_cnt+1)]
		//dLdAPrev is derivative of loss function wrt to previous (lower) layer activations to compute by bprop(). Size [batchSize x (prev_layer_neuron_cnt+1)]
		// Also during bprop() after computation of dLdAPrev layer must compute dL/dW and adjust its weights accordingly.
		//flag bPrevLayerIsInput is set when previous layer is input layer and if not IBP, dont calc dLdAPrev
		template <typename LowerLayer>
		nntl_interface void bprop(floatmtx_t& dLdA, const LowerLayer& lowerLayer, floatmtx_t& dLdAPrev)noexcept;
		//output layer should use bprop(const floatmtx_t& data_y, ...)
		//need non-const for dLdA to make dropout work
	};

	//special marks for type checking of input and output layers
	struct m_layer_input {};
	struct m_layer_output {};


	//////////////////////////////////////////////////////////////////////////
	// base class for all layers. Implements compile time polymorphism to get rid of virtual functions
	template<typename FinalPolymorphChild>
	class _layer_base : public _i_layer{
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;
		
		//////////////////////////////////////////////////////////////////////////
		//constructors-destructor
		~_layer_base()noexcept {};
		_layer_base(const neurons_count_t _neurons_cnt)
			noexcept : m_layerIdx(0), m_neurons_cnt(_neurons_cnt), m_incoming_neurons_cnt(0), m_bTraining(false)
		{
			NNTL_ASSERT(m_neurons_cnt > 0);
			//m_activations.will_emulate_biases();
		};
		

		//////////////////////////////////////////////////////////////////////////
		//nntl_interface overridings
		self_ref_t get_self() noexcept {
			static_assert(std::is_base_of<_layer_base<FinalPolymorphChild>, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_base<FinalPolymorphChild>");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<_layer_base<FinalPolymorphChild>, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_base<FinalPolymorphChild>");
			return static_cast<self_cref_t>(*this);
		}

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { return m_incoming_neurons_cnt; }
		//const floatmtx_t& get_activations()const noexcept { return m_activations; }

		constexpr bool is_input_layer()const noexcept { return false; }
		constexpr bool is_output_layer()const noexcept { return false; }

		const float_t_ learning_rate()const noexcept { return float_t_(0.0); }
		self_ref_t learning_rate(float_t_ lr)noexcept { return get_self(); }
	
		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		//this is how we going to initialize layer indexes.
		//template <typename LCur, typename LPrev> friend void _init_layers::operator()(LCur&& lc, LPrev&& lp, bool bFirst)noexcept;
		friend class _preinit_layers;
		void _preinit_layer(const layer_index_t idx, const neurons_count_t inc_neurons_cnt)noexcept{
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx);
			NNTL_ASSERT(!m_incoming_neurons_cnt);

			if (m_layerIdx || m_incoming_neurons_cnt) abort();
			m_layerIdx = idx;
			if (idx) m_incoming_neurons_cnt = inc_neurons_cnt;

			//here is probably the best place to allocate memory, but this might be changed in future when
			// will tune for processor cache efficiency
			// upd: this is a work for derived classes. For example, input_layer doesn't need it at all, cause it will receive it
			// from nnet during fprop() phase
			//m_activations.resize(m_neurons_cnt);//there is no need to initialize allocated memory
		}

	//////////////////////////////////////////////////////////////////////////
	//members section (in "biggest first" order)
	protected:
		// matrix of layer neurons activations: <batch_size rows> x <m_neurons_cnt+1(bias) cols> for fully connected layer
		//floatmtx_t m_activations;

	public:
		const neurons_count_t m_neurons_cnt;

	private:
		neurons_count_t m_incoming_neurons_cnt;
		layer_index_t m_layerIdx;		

	protected:
		bool m_bTraining;

	public:

	};

}