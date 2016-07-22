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

#include "../_nnet_errs.h"
#include "../nnet_def_interfaces.h"
#include "../serialization/serialization.h"

#include "../grad_works.h"
#include "_init_layers.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	//each layer_pack_* layer is expected to have a special typedef self_t LayerPack_t

	//recognizer of layer_pack_* classes
	// primary template handles types that have no nested ::LayerPack_t member:
	template< class, class = std::void_t<> >
	struct is_layer_pack : std::false_type { };
	// specialization recognizes types that do have a nested ::LayerPack_t member:
	template< class T >
	struct is_layer_pack<T, std::void_t<typename T::LayerPack_t>> : std::true_type {};

	//helper function to call internal _for_each_layer(f) for layer_pack_* classes
	template<typename Func, typename LayerT> inline
		std::enable_if_t<is_layer_pack<LayerT>::value> call_F_for_each_layer(Func& F, LayerT& l)noexcept
	{
		l.for_each_layer(F);
	}
	template<typename Func, typename LayerT> inline
		std::enable_if_t<!is_layer_pack<LayerT>::value> call_F_for_each_layer(Func& F, LayerT& l)noexcept
	{
		F(l);
	}


	//////////////////////////////////////////////////////////////////////////
	// interface that must be implemented by a layer in order to make fprop() function work (it PrevLayer_type, so this will help us to isolate
	// which API in particular previous layer must implement)
	template <typename RealT>
	class _i_layer_fprop {
	protected:
		_i_layer_fprop()noexcept {};
		~_i_layer_fprop()noexcept {};

		//!! copy constructor not needed
		_i_layer_fprop(const _i_layer_fprop& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		_i_layer_fprop& operator=(const _i_layer_fprop& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		typedef RealT real_t;
		typedef math::simple_matrix<real_t> realmtx_t;
		typedef math::simple_matrix_deformable<real_t> realmtxdef_t;
		static_assert(std::is_base_of<realmtx_t, realmtxdef_t>::value, "simple_matrix_deformable must be derived from simple_matrix!");

	public:
		nntl_interface const realmtx_t& get_activations()const noexcept;
	};

	// and the same for bprop(). Derives from _i_layer_fprop because it generally need it API
	template <typename RealT>
	class _i_layer_trainable : public _i_layer_fprop<RealT>{
	protected:
		_i_layer_trainable()noexcept {};
		~_i_layer_trainable()noexcept {};

		//!! copy constructor not needed
		_i_layer_trainable(const _i_layer_trainable& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
															 //!!assignment is not needed
		_i_layer_trainable& operator=(const _i_layer_trainable& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		//nntl_interface const bool is_input_layer()const noexcept;
	};

	//////////////////////////////////////////////////////////////////////////
	// layer interface definition
	template<typename RealT>
	class _i_layer : public _i_layer_trainable<RealT>, public math::simple_matrix_typedefs {
	protected:
		_i_layer()noexcept {};
		~_i_layer()noexcept {};

		//!! copy constructor not needed
		_i_layer(const _i_layer& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		_i_layer& operator=(const _i_layer& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		typedef _nnet_errs::ErrorCode ErrorCode;

		// Each layer must define an alias for type of math interface and rng interface (and they must be the same for all layers)
		//typedef .... iMath_t
		//typedef .... iRng_t

		//////////////////////////////////////////////////////////////////////////
		// base interface
		// each call to own functions should go through get_self() to make polymorphyc function work
		nntl_interface auto get_self() const noexcept;
		nntl_interface const layer_index_t get_layer_idx() const noexcept;
		nntl_interface const neurons_count_t get_neurons_cnt() const noexcept;
		nntl_interface const neurons_count_t get_incoming_neurons_cnt()const noexcept;
		
		//must obey to matlab variables naming convention
		nntl_interface void get_layer_name(char* pName, const size_t cnt)const noexcept;
		nntl_interface std::string get_layer_name_str()const noexcept;

		// batchSize==0 puts layer into training mode with batchSize predefined by init()::lid.training_batch_size
		// any batchSize>0 puts layer into evaluation/testing mode with that batchSize. bs must be <= init()::lid.max_fprop_batch_size
		// pNewActivationStorage is used in conjunction with compound layers, such as layer_pack_horizontal, that use
		// their inner layers activations as own activation. If set, the layer must use this pointer as destination for activation units value
		// (do something like m_activations.useExternalStorage(pNewActivationStorage) ). Resetting of biases is not required at this case, however.
		// Layers, that should never be a part of other compound layers, should totally omit this parameter.
		nntl_interface void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept;

		// ATTN: more specific and non-templated version available for this function, see _layer_base for an example
		// pNewActivationStorage - see comments to set_mode(). Layers, that should never be a part of other compound layers, should totally omit this parameter.
		template<typename _layer_init_data_t>
		nntl_interface ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept;

		//frees any temporary resources that doesn't contain layer-specific information (i.e. layer weights shouldn't be freed).
		//In other words, this routine burns some unnecessary fat layer gained during training, but don't touch any data necessary
		// for new subsequent call to nnet.train()
		nntl_interface void deinit() noexcept;

		//provides a temporary storage for a layer. It is guaranteed, that during fprop() or bprop() the storage can be modified only by the layer
		// (it's a shared memory and it can be modified elsewhere between calls to fprop()/bprop())
		//function is guaranteed to be called if (minMemFPropRequire+minMemBPropRequire) set to >0 during init()
		// cnt is guaranteed to be at least as big as (minMemFPropRequire+minMemBPropRequire)
		nntl_interface void initMem(real_t* ptr, numel_cnt_t cnt)noexcept;

		//input layer should use slightly different specialization: void fprop(const realmtx_t& data_x)noexcept
		template <typename LowerLayer>
		nntl_interface void fprop(const LowerLayer& lowerLayer)noexcept;

		// dLdA is derivative of loss function wrt this layer neuron activations.
		// Size [batchSize x layer_neuron_cnt] (bias units ignored - they're updated during dLdW application)
		// 
		// dLdAPrev is derivative of loss function wrt to previous (lower) layer activations to compute by bprop().
		// Size [batchSize x prev_layer_neuron_cnt] (bias units ignored)
		// 
		// A layer must compute dL/dW (derivative of loss function wrt layer parameters (weights)) and adjust
		// its parameters accordingly after a computation of dLdAPrev during bprop() function.
		//  
		// realmtxdef_t type is used in pack_* layers. Non-compound layers should use realmtxt_t type instead.
		// Function is allowed to use dLdA once it's not needed anymore as it wants (resizing operation included,
		// provided that it won't resize it greater than max size. BTW, beware! The run-time check works only
		// in DEBUG builds!). Same for dLdAPrev, but on exit from bprop() it must have a proper size and content.
		template <typename LowerLayer>
		nntl_interface unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept;
		//output layer must use form void bprop(const realmtx_t& data_y, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev);
		//On return value: in short, simple/single layers should return 1.
		// In long: during init() phase, each layer returns the size of its dLdA matrix in _layer_init_data_t::max_dLdA_numel.
		// This values from every layer are aggregated by max() into greatest possible dLdA size for whole NNet.
		// Then two matrices of this (greatest) size are allocated and passed to layers::bprop() function. One of these
		// matrices will be used as dLdA and the other as dLdAPrev during each layer::bprop() call.
		// What does the return value from a bprop() is it governs whether the caller must alternate these matrices
		// on a call to lower layer bprop() (i.e. whether dLdAPrev is actually stored in dLdAPrev variable (return 1) or
		// dLdAPrev is really stored in (aprropriately resized) dLdA variable -return 0).
		// So, simple/single layers, that don't switch these matrices, should always return 1. However, compound layers
		// (such as layer_pack_vertical), that consists of other layers (and must call bprop() on them), may reuse
		// dLdA/dLdAPrev in order to eliminate the neccessity of additional temporary dLdA matrices and data coping,
		// just by switching between dLdA and dLdAPrev between calls to inner bprop()'s. So, if there was
		// even number of calls to inner layers bprop() occured, then the actual dLdAPrev of the whole compound
		// layer will be inside of dLdA and a caller of compound layer's bprop() should NOT switch matrices on
		// subsequent call to lower layer bprop(). Therefore, compound layer's bprop() must return 0 in that case.
		

		
		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		// At this moment, the code of layers::calcLossAddendum() depends on a (possibly non-stable) fact, that a loss function
		// addendum to be calculated doesn't depend on data_x or data_y (it depends on only internal nn properties, such as weights).
		// This might not be the case in a future, - update layers::calcLossAddendum() definition then.
		nntl_interface real_t lossAddendum()const noexcept;
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		nntl_interface bool hasLossAddendum()const noexcept;

	private:
		//support for boost::serialization
		friend class boost::serialization::access;
		template<class Archive> nntl_interface void serialize(Archive & ar, const unsigned int version) {}
	};

	
	//////////////////////////////////////////////////////////////////////////
	// base class for all layers. Implements compile time polymorphism to get rid of virtual functions
	template<typename Interfaces, typename FinalPolymorphChild>
	class _layer_base : public _i_layer<typename Interfaces::iMath_t::real_t>{
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs
		typedef FinalPolymorphChild self_t;
		typedef FinalPolymorphChild& self_ref_t;
		typedef const FinalPolymorphChild& self_cref_t;
		typedef FinalPolymorphChild* self_ptr_t;

		typedef typename Interfaces::iMath_t iMath_t;
		static_assert(std::is_base_of<math::_i_math<real_t>, iMath_t>::value, "Interfaces::iMath type should be derived from _i_math");

		typedef typename Interfaces::iRng_t iRng_t;
		static_assert(std::is_base_of<rng::_i_rng, iRng_t>::value, "Interfaces::iRng type should be derived from _i_rng");

		typedef _impl::_layer_init_data<iMath_t, iRng_t> _layer_init_data_t;

		//////////////////////////////////////////////////////////////////////////
		//members section (in "biggest first" order)
	protected:
		const neurons_count_t m_neurons_cnt;

	private:
		neurons_count_t m_incoming_neurons_cnt;
		layer_index_t m_layerIdx;

	protected:
		bool m_bTraining;

	public:		
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
			static_assert(std::is_base_of<_layer_base, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_base<FinalPolymorphChild>");
			return static_cast<self_ref_t>(*this);
		}
		self_cref_t get_self() const noexcept {
			static_assert(std::is_base_of<_layer_base, FinalPolymorphChild>::value
				, "FinalPolymorphChild must derive from _layer_base<FinalPolymorphChild>");
			return static_cast<self_cref_t>(*this);
		}

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { 
			NNTL_ASSERT(m_neurons_cnt);
			return m_neurons_cnt;
		}
		const neurons_count_t get_incoming_neurons_cnt()const noexcept { 
			NNTL_ASSERT(!m_layerIdx || m_incoming_neurons_cnt);
			return m_incoming_neurons_cnt;
		}

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, "unk%d", static_cast<unsigned>(get_layer_idx()));
		}
		std::string get_layer_name_str()const noexcept {
			constexpr size_t ml = 16;
			char n[ml];
			get_self().get_layer_name(n, ml);
			return std::string(n);
		}

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		constexpr const real_t lossAddendum()const noexcept { return real_t(0.0); }
			
		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		//this is how we going to initialize layer indexes.
		//template <typename LCur, typename LPrev> friend void _init_layers::operator()(LCur&& lc, LPrev&& lp, bool bFirst)noexcept;
		friend class _impl::_preinit_layers;
		//idx is passed by reference. On function enter it contains the lowest free layer index withing a NN.
		// On function exit after this (and possibly encapsulated into this) layer preinitialization it 
		// must contain next lowest free index.
		void _preinit_layer(layer_index_t& idx, const neurons_count_t inc_neurons_cnt)noexcept{
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx);
			NNTL_ASSERT(!m_incoming_neurons_cnt);

			if (m_layerIdx || m_incoming_neurons_cnt) abort();
			m_layerIdx = idx;
			if (idx++) {
				NNTL_ASSERT(inc_neurons_cnt);
				m_incoming_neurons_cnt = inc_neurons_cnt;
			}
		}

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class boost::serialization::access;
		//nothing to do here at this moment, also leave nntl_interface marker to prevent calls.
		template<class Archive> nntl_interface void serialize(Archive & ar, const unsigned int version);
	};


}