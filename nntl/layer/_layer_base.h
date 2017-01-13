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
#include "../interfaces.h"
#include "../serialization/serialization.h"

#include "../grad_works.h"
#include "_init_layers.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//each layer_pack_* layer is expected to have a special typedef self_t LayerPack_t
	// and it must implement for_each_layer() function

	//recognizer of layer_pack_* classes
	// primary template handles types that have no nested ::LayerPack_t member:
	template< class, class = std::void_t<> >
	struct is_layer_pack : std::false_type { };
	// specialization recognizes types that do have a nested ::LayerPack_t member:
	template< class T >
	struct is_layer_pack<T, std::void_t<typename T::LayerPack_t>> : std::true_type {};

	//helper function to call internal _for_each_layer(f) for layer_pack_* classes
	template<typename Func, typename LayerT> inline
		std::enable_if_t<is_layer_pack<LayerT>::value> call_F_for_each_layer(Func&& F, LayerT& l)noexcept
	{
		l.for_each_layer(std::forward<Func>(F));
	}
	template<typename Func, typename LayerT> inline
		std::enable_if_t<!is_layer_pack<LayerT>::value> call_F_for_each_layer(Func&& F, LayerT& l)noexcept
	{
		std::forward<Func>(F)(l);
	}

	template<typename Func, typename LayerT> inline
		std::enable_if_t<is_layer_pack<LayerT>::value> call_F_for_each_layer_down(Func&& F, LayerT& l)noexcept
	{
		l.for_each_layer_down(std::forward<Func>(F));
	}
	template<typename Func, typename LayerT> inline
		std::enable_if_t<!is_layer_pack<LayerT>::value> call_F_for_each_layer_down(Func&& F, LayerT& l)noexcept
	{
		std::forward<Func>(F)(l);
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// layer with ::grad_works_t type defined is expected to have m_gradientWorks member
	// (nonstandartized at this moment)

	template< class, class = std::void_t<> >
	struct layer_has_gradworks : std::false_type { };
	// specialization recognizes types that do have a nested ::grad_works_t member:
	template< class T >
	struct layer_has_gradworks<T, std::void_t<typename T::grad_works_t>> : std::true_type {};


	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template <typename RealT>
	struct _i_layer_td : public math::smatrix_td {
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;
		static_assert(std::is_base_of<realmtx_t, realmtxdef_t>::value, "smatrix_deform must be derived from smatrix!");
	};

	////////////////////////////////////////////////////////////////////////// 
	// interface that must be implemented by a layer in order to make fprop() function work
	// Layer, passed to fprop as the PrevLayer parameter must obey this interface.
	// (#TODO is it necessary? Can we just drop it?)
	template <typename RealT>
	class _i_layer_fprop : public _i_layer_td<RealT> {
	protected:
		_i_layer_fprop()noexcept {};
		~_i_layer_fprop()noexcept {};

		//!! copy constructor not needed
		_i_layer_fprop(const _i_layer_fprop& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		_i_layer_fprop& operator=(const _i_layer_fprop& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		//get_activations() is allowed to call after fprop() only. bprop() invalidates activation values!
		nntl_interface const realmtxdef_t& get_activations()const noexcept;
		nntl_interface const mtx_size_t get_activations_size()const noexcept;
	};

	template <typename RealT>
	class _i_layer_gate : private _i_layer_td<RealT> {
	protected:
		_i_layer_gate()noexcept {};
		~_i_layer_gate()noexcept {};

		//!! copy constructor not needed
		_i_layer_gate(const _i_layer_gate& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
																	 //!!assignment is not needed
		_i_layer_gate& operator=(const _i_layer_gate& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		nntl_interface const realmtx_t& get_gate()const noexcept;
		nntl_interface const vec_len_t get_gate_width()const noexcept;
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
	class _i_layer : public _i_layer_trainable<RealT> {
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

		//DON'T call this function unless you know what you're doing
		nntl_interface void _set_neurons_cnt(const neurons_count_t nc)noexcept;

		nntl_interface const neurons_count_t get_incoming_neurons_cnt()const noexcept;
		
		//must obey to matlab variables naming convention
		nntl_interface auto set_custom_name(const char* pCustName)noexcept;
		nntl_interface const char* get_custom_name()const noexcept;
		nntl_interface void get_layer_name(char* pName, const size_t cnt)const noexcept;
		nntl_interface std::string get_layer_name_str()const noexcept;

	private:
		//redefine in derived class in public scope. Array-style definition MUST be preserved.
		//the _defName must be unique for each final layer class and mustn't be longer than sizeof(layer_type_id_t) (it's also used as a layer typeId)
		static constexpr const char _defName[] = "_i_layer";
	public:
		//returns layer type id based on layer's _defName
		nntl_interface static constexpr layer_type_id_t get_layer_type_id()noexcept;

		// batchSize==0 puts layer into training mode with batchSize predefined by init()::lid.training_batch_size
		// any batchSize>0 puts layer into evaluation/testing mode with that batchSize. bs must be <= init()::lid.max_fprop_batch_size
		// pNewActivationStorage is used in conjunction with compound layers, such as layer_pack_horizontal, that 
		// provide their internal activation storage for embedded layers (to reduce data copying)
		// If pNewActivationStorage is set, the layer must store its activations under this pointer
		// (by doing something like m_activations.useExternalStorage(pNewActivationStorage) ).
		// Resetting of biases is not required at this case, however.
		// Layers, that should never be a part of other compound layers, should totally omit this parameter
		// from function signature (not recommeded)
		nntl_interface void set_mode(vec_len_t batchSize, real_t* pNewActivationStorage = nullptr)noexcept;

		// ATTN: more specific and non-templated version available for this function, see _layer_base for an example
		// On the pNewActivationStorage see comments to the set_mode(). Layers that should never be on the top of a stack of layers
		// inside of compound layers, should totally omit this parameter to break compilation.
		// For the _layer_init_data_t parameter see the _impl::_layer_init_data<>.
		template<typename _layer_init_data_t>
		nntl_interface ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept;
		//If a layer is given a pNewActivationStorage, then it MUST NOT touch a bit in the bias column of the activation storage.

		//frees any temporary resources that doesn't contain layer-specific information (i.e. layer weights shouldn't be freed).
		//In other words, this routine burns some unnecessary fat layer gained during training, but don't touch any data necessary
		// for new subsequent call to nnet.train()
		nntl_interface void deinit() noexcept;

		//provides a temporary storage for a layer. It is guaranteed, that during fprop() or bprop() the storage
		// can be modified only by the layer. However, it's not a persistent storage and layer mustn't rely on it
		// to retain it's content between calls to fprop()/bprop().
		// Compound layers (that call other's layers fprop()/bprop()) should use this storage with a great care.
		// Function is guaranteed to be called if
		// max(_layer_init_data::minMemFPropRequire,_layer_init_data::minMemBPropRequire) set to be >0 during init()
		// cnt is guaranteed to be at least as big as max(minMemFPropRequire,minMemBPropRequire)
		nntl_interface void initMem(real_t* ptr, numel_cnt_t cnt)noexcept;

		//input layer should use slightly different specialization: void fprop(const realmtx_t& data_x)noexcept
		template <typename LowerLayer>
		nntl_interface void fprop(const LowerLayer& lowerLayer)noexcept;
		//If a layer is given a pNewActivationStorage, then it MUST NOT touch a bit in the bias column of the activation storage.

		// dLdA is derivative of loss function wrt this layer neuron activations.
		// Size [batchSize x layer_neuron_cnt] (bias units ignored - their weights actually belongs to upper layer
		// and therefore are updated during that layer's bprop() phase and dLdW application)
		// 
		// dLdAPrev is derivative of loss function wrt to previous (lower) layer activations to compute by bprop().
		// Size [batchSize x prev_layer_neuron_cnt] (bias units ignored)
		// 
		// A layer must compute dL/dW (derivative of loss function wrt layer parameters (weights)) and adjust
		// its parameters accordingly after a computation of dLdAPrev during bprop() function.
		//  
		// realmtxdef_t type is used in pack_* layers. Non-compound layers should use realmtxt_t type instead.
		// Function is allowed to use dLdA once it's not needed anymore as it wants (resizing operation included,
		// provided that it won't resize it greater than max size. BTW, beware! The run-time check of maximum matrix size works only
		// in DEBUG builds!). Same for dLdAPrev, but on exit from bprop() it must have a proper size and content.
		template <typename LowerLayer>
		nntl_interface unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept;
		//If a layer is given a pNewActivationStorage, then it MUST NOT touch a bit in the bias column of the activation storage.
		//output layer must use form void bprop(const realmtx_t& data_y, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev);
		//On return value: in a short, simple/single layers should return 1.
		// In a long: during the init() phase, each layer returns the size of its dLdA matrix in _layer_init_data_t::max_dLdA_numel.
		// This values gathered from every layer in a layers stack are aggregated by max() into the biggest possible dLdA size for whole NNet.
		// Then two matrices of this (biggest) size are allocated and passed to layers::bprop() function. One of these
		// matrices will be used as dLdA and the other as dLdAPrev during each call to a layer::bprop().
		// What does the return value from a bprop() do is it governs whether the caller must alternate these matrices
		// on a call to lower layer bprop() (i.e. whether a real dLdAPrev is actually stored in dLdAPrev variable (return 1) or
		// the dLdAPrev is really stored in the (appropriately resized) dLdA variable - return 0).
		// So, simple/single layers, that don't switch/reuse these matrices, should always return 1. However, compound layers
		// (such as layer_pack_vertical), that consists of other layers (and must call bprop() on them), may reuse
		// dLdA&dLdAPrev variable in order to eliminate the necessity of additional temporary dLdA matrices and corresponding data coping,
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
	// poly base class, Implements compile time polymorphism (to get rid of virtual functions)
	// and default _layer_name_ machinery
	template<typename FinalPolymorphChild, typename RealT>
	class _cpolym_layer_base : public _i_layer<RealT> {
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs
		typedef FinalPolymorphChild self_t;
		NNTL_METHODS_SELF_CHECKED((std::is_base_of<_cpolym_layer_base, FinalPolymorphChild>::value)
			, "FinalPolymorphChild must derive from _cpolym_layer_base<FinalPolymorphChild>");

		//layer name could be used for example to name Matlab's variables,
		//so there must be some reasonable limit. Don't overcome this limit!
		static constexpr size_t layerNameMaxChars = 50;
		//limit for custom name length
		static constexpr size_t customNameMaxChars = layerNameMaxChars - 10;
	private:
		//redefine in derived class in public scope. Array-style definition MUST be preserved.
		//the _defName must be unique for each final layer class (it's also served as a layer typeId)
		static constexpr const char _defName[] = "_cpoly";

	protected:
		//just a pointer as passed, because don't want to care about memory allocation and leave a footprint as small as possible,
		//because it's just a matter of convenience.
		const char* m_customName;

	protected:
		bool m_bTraining;
		bool m_bActivationsValid;

	protected:
		void init()noexcept { m_bActivationsValid = false; }
		void deinit()noexcept { m_bActivationsValid = false; }

	public:
		//////////////////////////////////////////////////////////////////////////
		~_cpolym_layer_base()noexcept {}
		_cpolym_layer_base(const char* pCustName=nullptr)noexcept : m_bTraining(false), m_bActivationsValid(false) {
			set_custom_name(pCustName);
		}

		static constexpr const char* get_default_name()noexcept { return self_t::_defName; }
		self_ref_t set_custom_name(const char* pCustName)noexcept {
			NNTL_ASSERT(!pCustName || strlen(pCustName) < customNameMaxChars);
			m_customName = pCustName;
			return get_self();
		}
		const char* get_custom_name()const noexcept { return m_customName ? m_customName : get_self().get_default_name(); }

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			sprintf_s(pName, cnt, "%s_%d", get_self().get_custom_name(),static_cast<unsigned>(get_self().get_layer_idx()));
		}
		std::string get_layer_name_str()const noexcept {
			constexpr size_t ml = layerNameMaxChars;
			char n[ml];
			get_self().get_layer_name(n, ml);
			return std::string(n);
		}

	private:
		template<unsigned LEN>
		static constexpr layer_type_id_t _get_layer_type_id(const char(&pStr)[LEN], const unsigned pos = 0)noexcept {
			return pos < LEN ? (layer_type_id_t(pStr[pos]) | (_get_layer_type_id(pStr, pos + 1) << 8)) : 0;
		}
	public:
		static constexpr layer_type_id_t get_layer_type_id()noexcept {
			static_assert(sizeof(self_t::_defName) <= sizeof(layer_type_id_t), "Too long default layer name has been used. Can't use it to derive layer_type_id");
			return _get_layer_type_id(self_t::_defName);
		}

	};
	
	//////////////////////////////////////////////////////////////////////////
	// base class for most of layers.
	// Implements compile time polymorphism (to get rid of virtual functions),
	// default _layer_name_ machinery, some default basic typedefs and basic support machinery
	// (init() function with common_data_t, layer index number, neurons count)
	template<typename InterfacesT, typename FinalPolymorphChild>
	class _layer_base 
		: public _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::iMath_t::real_t>
		, public _impl::_common_data_consumer<InterfacesT>
	{
	private:
		typedef _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::iMath_t::real_t> _base_class;
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs		
		typedef _impl::_layer_init_data<common_data_t> _layer_init_data_t;

		using _base_class::real_t;

		//////////////////////////////////////////////////////////////////////////
		//members section (in "biggest first" order)

	private:
		neurons_count_t m_neurons_cnt, m_incoming_neurons_cnt;
		layer_index_t m_layerIdx;

		static constexpr const char _defName[] = "_base";

	public:		
		//////////////////////////////////////////////////////////////////////////
		//constructors-destructor
		~_layer_base()noexcept {};
		_layer_base(const neurons_count_t _neurons_cnt, const char* pCustomName=nullptr) noexcept 
			: _base_class(pCustomName)
			, m_layerIdx(0), m_neurons_cnt(_neurons_cnt), m_incoming_neurons_cnt(0)
		{};
		
		//////////////////////////////////////////////////////////////////////////
		//nntl_interface overridings
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			_base_class::init();
			set_common_data(lid.commonData);

			get_self().get_iInspect().init_layer(get_self().get_layer_idx(), get_self().get_layer_name_str(), get_self().get_layer_type_id());

			return ErrorCode::Success;
		}
		void deinit() noexcept { 
			clean_common_data();
			_base_class::deinit();
		}

		const layer_index_t get_layer_idx() const noexcept { return m_layerIdx; }
		const neurons_count_t get_neurons_cnt() const noexcept { 
			NNTL_ASSERT(m_neurons_cnt);
			return m_neurons_cnt;
		}
		//for layers that need to calculate their neurons count in run-time (layer_pack_horizontal)
		void _set_neurons_cnt(const neurons_count_t nc)noexcept {
			NNTL_ASSERT(nc);
			//NNTL_ASSERT(!m_neurons_cnt || nc==m_neurons_cnt);//to prevent double calls
			NNTL_ASSERT(!m_neurons_cnt);//to prevent double calls
			m_neurons_cnt = nc;
		}

		const neurons_count_t get_incoming_neurons_cnt()const noexcept { 
			NNTL_ASSERT(!m_layerIdx || m_incoming_neurons_cnt);//m_incoming_neurons_cnt will be zero in input layer (it has m_layerIdx==0)
			return m_incoming_neurons_cnt;
		}		

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		constexpr const real_t lossAddendum()const noexcept { return real_t(0.0); }
			
		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept{
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(!m_layerIdx);
			NNTL_ASSERT(!m_incoming_neurons_cnt);

			if (m_layerIdx || m_incoming_neurons_cnt) abort();
			m_layerIdx = ili.newIndex();
			if (m_layerIdx) {//special check for the first (input) layer that doesn't have any incoming neurons
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