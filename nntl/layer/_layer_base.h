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

#include "../_nnet_errs.h"
#include "../interfaces.h"
#include "../serialization/serialization.h"

#include "../grad_works/grad_works.h"
#include "_init_layers.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//each layer_pack_* layer is expected to have a special typedef self_t LayerPack_t
	// and it must implement for_each_layer() and for_each_packed_layer() function families
	
	//recognizer of layer_pack_* classes
	// primary template handles types that have no nested ::LayerPack_t member:
	template< class, class = ::std::void_t<> >
	struct is_layer_pack : ::std::false_type { };
	// specialization recognizes types that do have a nested ::LayerPack_t member:
	template< class T >
	struct is_layer_pack<T, ::std::void_t<typename T::LayerPack_t>> : ::std::true_type {};

	//helper function to call internal _for_each_layer(f) for layer_pack_* classes
	//it iterates through the layers from the lowmost (input) to the highmost (output).
	// layer_pack's are also passed to F!
	// Therefore the .for_each_layer() is the main mean to apply F to every layer in a network/pack
	template<typename Func, typename LayerT> inline
		::std::enable_if_t<is_layer_pack<LayerT>::value> call_F_for_each_layer(Func&& F, LayerT& l)noexcept
	{
		l.for_each_layer(F); //mustn't forward, because we'll be using F later!
		
		//must also call for the layer itself
		::std::forward<Func>(F)(l);//it's OK to cast to rvalue here if suitable, as we don't care what will happens with F after that.
	}
	template<typename Func, typename LayerT> inline
		::std::enable_if_t<!is_layer_pack<LayerT>::value> call_F_for_each_layer(Func&& F, LayerT& l)noexcept
	{
		::std::forward<Func>(F)(l);//OK to forward if suitable
	}

	//probably we don't need it, but let it be
	template<typename Func, typename LayerT> inline
		::std::enable_if_t<is_layer_pack<LayerT>::value> call_F_for_each_layer_down(Func&& F, LayerT& l)noexcept
	{
		F(l);//mustn't forward, we'll use it later
		l.for_each_layer_down(::std::forward<Func>(F));//OK, last use
	}
	template<typename Func, typename LayerT> inline
		::std::enable_if_t<!is_layer_pack<LayerT>::value> call_F_for_each_layer_down(Func&& F, LayerT& l)noexcept
	{
		::std::forward<Func>(F)(l);//OK to forward
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// layer with ::grad_works_t type defined is expected to have m_gradientWorks member
	// (nonstandartized at this moment)

	template< class, class = ::std::void_t<> >
	struct layer_has_gradworks : ::std::false_type { };
	// specialization recognizes types that do have a nested ::grad_works_t member:
	template< class T >
	struct layer_has_gradworks<T, ::std::void_t<typename T::grad_works_t>> : ::std::true_type {};


	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template <typename RealT>
	struct _i_layer_td : public math::smatrix_td {
		typedef RealT real_t;
		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;
		static_assert(::std::is_base_of<realmtx_t, realmtxdef_t>::value, "smatrix_deform must be derived from smatrix!");
	};

	////////////////////////////////////////////////////////////////////////// 
	// interface that must be implemented by a layer in order to make fprop() function work
	// Every layer passed to fprop() function as PrevLayer parameter must obey this interface.
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
		//It is allowed to call get_activations() after fprop() and before bprop() only. bprop() invalidates activation values!
		//Furthermore, DON'T ever change the activation matrix values from the outside of the layer!
		// Layer object expects it to be unchanged between fprop()/bprop() calls to make a proper gradient.
		nntl_interface const realmtxdef_t& get_activations()const noexcept;
		nntl_interface mtx_size_t get_activations_size()const noexcept;
		
		//essentially the same as get_activations(), however, it is allowed to call this function anytime to obtain the pointer
		//However, if you are going to dereference the pointer, the same restrictions as for get_activations() applies.
		//NOTE: It won't trigger assert if it's dereferenced in a wrong moment, therefore you'll get invalid values,
		// so use it wisely only when you absolutely can't use the get_activations()
		nntl_interface const realmtxdef_t* get_activations_storage()const noexcept;
	};

/*
//deprecated
	template <typename RealT>
	class _i_layer_gate : private _i_layer_td<RealT>/ *, public m_layer_gate* / {
	protected:
		_i_layer_gate()noexcept {};
		~_i_layer_gate()noexcept {};

		//!! copy constructor not needed
		_i_layer_gate(const _i_layer_gate& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
																	 //!!assignment is not needed
		_i_layer_gate& operator=(const _i_layer_gate& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

	public:
		//NB: gate is BINARY matrix
		nntl_interface const realmtx_t& get_gate()const noexcept;
		nntl_interface const realmtx_t* get_gate_storage()const noexcept;
		//nntl_interface const real_t* get_bias_gate()const noexcept;
		nntl_interface const vec_len_t get_gate_width()const noexcept;
	};
	
	template<typename LayerT>
	struct is_layer_gate : public ::std::is_base_of<_i_layer_gate<typename LayerT::real_t>, LayerT> {};

	*/

	//////////////////////////////////////////////////////////////////////////

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

		//////////////////////////////////////////////////////////////////////////
		// base interface
		
		// class constructor MUST have const char* pCustomName as the first parameter.
		
		//shared activations implies that the bias column may hold not the biases, but activations of some another layer
		//Therefore use such activation matrix only in proper context (biases of shared activations are valid only in
		// contexts of fprop()/bprop() of upper layers).
		//If the layer was given a pNewActivationStorage parameter in init/on_batch_size_change, it implies
		//that the activations are shared.
		nntl_interface bool is_activations_shared()const noexcept;

		//use it only if you really know what you're are doing and it won't hurt derivatives calculation
		//SOME LAYERS may NOT implement this function!
		nntl_interface realmtxdef_t& _get_activations_mutable()const noexcept;

		//////////////////////////////////////////////////////////////////////////
		// Almost every call to layer's own functions should go through get_self() to make redefined in derived classes functions work.
		nntl_interface auto get_self() const noexcept;
		nntl_interface layer_index_t get_layer_idx() const noexcept;
		nntl_interface neurons_count_t get_neurons_cnt() const noexcept;

		//For internal use only. DON'T call this function unless you know very well what you're doing
		nntl_interface void _set_neurons_cnt(const neurons_count_t nc)noexcept;

		nntl_interface neurons_count_t get_incoming_neurons_cnt()const noexcept;
		
		//must obey to matlab variables naming convention
		nntl_interface auto set_custom_name(const char* pCustName)noexcept;
		nntl_interface const char* get_custom_name()const noexcept;
		nntl_interface void get_layer_name(char* pName, const size_t cnt)const noexcept;
		nntl_interface ::std::string get_layer_name_str()const noexcept;

	private:
		//redefine in derived class in public scope. Array-style definition MUST be preserved.
		//the _defName must be unique for each final layer class and mustn't be longer than sizeof(layer_type_id_t) (it's also used as a layer typeId)
		static constexpr const char _defName[] = "_i_layer";
		//#todo: this method of defining layer's name is ill. Should change it to something more workable.

	public:
		//returns layer type id based on layer's _defName
		nntl_interface static constexpr layer_type_id_t get_layer_type_id()noexcept;

		// ATTN: more specific and non-templated version available for this function, see _layer_base for an example
		// On the pNewActivationStorage see comments to the on_batch_size_change(). Layers that should never be on the top of a stack of layers
		// inside of compound layers, should totally omit this parameter to break compilation.
		// For the _layer_init_data_t parameter see the _impl::_layer_init_data<>.
		template<typename _layer_init_data_t>
		nntl_interface ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage /*= nullptr*/)noexcept;
		//If the layer was given a pNewActivationStorage (during the init() or on_batch_size_change()), then it MUST NOT touch a bit
		// in the bias column of the activation storage during bprop()/fprop() and everywhere else.
		// In general - if the layer allocates activation matrix by itself (when the pNewActivationStorage==nullptr),
		//					then it also allocates, sets up and manages biases keeping them in coherent state.
		//					That means, that:
		//					a) during on_batch_size_change() the layer has to restore its bias column, had the activation
		//					matrix been resized/deformed and the layer is not under a gate
		//					b) if the layer is under a gate, then it must copy gating mask to its biases during fprop()
		//					c) it's a very good practice to place asserts that checks the biases of previous layer activation matrix
		//					in fprop/bprop and layer's activation matrix on fprop() finish.
		//			  - if the layer is given pNewActivationStorage, then it uses it supposing there's a space for a bias column,
		//					however, layer must never touch data in that column.

		// Sets a batch size of the layer. The actual mode (evaluation or training) now should be set via common_data::set_training_mode()/isTraining()
		// 
		// pNewActivationStorage is used in conjunction with compound layers, such as layer_pack_horizontal, that 
		// provide their internal activation storage for embedded layers (to reduce data copying)
		// If pNewActivationStorage is set, the layer must store its activations under this pointer
		// (by doing something like m_activations.useExternalStorage(pNewActivationStorage) ).
		// Resetting of biases is not required at this case, however.
		// Layers, that should never be a part of other compound layers, should totally omit this parameter
		// from function signature (not recommended use-case, however)
		// 
		// #deprecated learningRateScale - is a scaling coefficient that must be applied to a learningRate value to account for some
		//	specific layer usage (inside LPT or LPHO, for example). //it seems that the whole idea was a mistake; pending for removal
		nntl_interface void on_batch_size_change(/*const real_t learningRateScale,*/ real_t*const pNewActivationStorage /*= nullptr*/)noexcept;
		//If the layer was given a pNewActivationStorage (during the init() or on_batch_size_change()), then it MUST NOT touch a bit
		// in the bias column of the activation storage during fprop() and everywhere else.
		// For more information about how memory storage is organized, see discussion in _init_layers.h::_layer_init_data{}

		//frees any temporary resources that doesn't contain layer-specific information (i.e. layer weights shouldn't be freed).
		//In other words, this routine burns some unnecessary fat layer gained during training, but don't touch any data necessary
		// for new subsequent call to nnet.train()
		nntl_interface void deinit() noexcept;

		//provides a temporary storage for a layer. It is guaranteed, that during fprop() or bprop() the storage
		// can be modified only by the layer. However, it's not a persistent storage and layer mustn't rely on it
		// to retain it's content between calls to fprop()/bprop().
		// Compound layers (that call other's layers fprop()/bprop()) should use this storage with a great care and make sure
		// they offsets the memory pointer they pass to inner layers.
		// Function is guaranteed to be called if any of {_layer_init_data::minMemFPropRequire,_layer_init_data::minMemBPropRequire}
		// were set to non zero during init()
		// cnt is guaranteed to be at least as big as max(minMemFPropRequire,minMemBPropRequire)
		nntl_interface void initMem(real_t* ptr, numel_cnt_t cnt)noexcept;

		// Forward propagation
		template <typename LowerLayer>
		nntl_interface void fprop(const LowerLayer& lowerLayer)noexcept;
		// Layer MUST NOT touch a bit in the bias column of the activation storage during fprop()/bprop() if the
		// activation storage are shared.
		//
		// input layer should use a different variant: void fprop(const realmtx_t& data_x)noexcept


		// dLdA is the derivative of the loss function wrt this layer neurons activations.
		// Size is [batchSize x layer_neuron_cnt] (bias units are ignored - their weights actually belongs to an upper layer
		// and therefore are updated during that layer's bprop() phase and dLdW application)
		// 
		// dLdAPrev is the derivative of the loss function wrt to a previous (lower) layer activations to be computed by the bprop().
		// Size is [batchSize x prev_layer_neuron_cnt] (bias units are also ignored)
		// 
		// The layer must compute a dL/dW (the derivative of the loss function wrt the layer parameters (weights) if any) and adjust
		// its parameters accordingly after a computation of dLdAPrev during the bprop() function.
		//  
		// realmtxdef_t type is used in pack_* layers. Non-compound layers should use less generic realmtxt_t type instead.
		// The function is allowed to use the dLdA parameter once it's not needed anymore as it wants (including resizing operation,
		// provided that it won't resize it greater than a max size previuosly set by the layer during init() phase.
		// BTW, beware! The run-time check of maximum matrix size works only
		// in DEBUG builds!).
		// Same for the dLdAPrev, - layer is free to use it as it wants during bprop(), - but on exit from the bprop()
		// it must have a proper size and expected content.
		// Note: if a layer has bDoBProp()==false, its bprop() function MUST NOT be called. --- not available at this moment
		template <typename LowerLayer>
		nntl_interface unsigned bprop(realmtxdef_t& dLdA, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev)noexcept;
		// Layer MUST NOT touch a bit in the bias column of the activation storage during fprop()/bprop()
		// 
		//Output layer must use different signature:
		//    void bprop(const realmtx_t& data_y, const LowerLayer& lowerLayer, realmtxdef_t& dLdAPrev);
		// 
		//On the function return value: in a short, simple/single layers should return 1.
		// In a long: during the init() phase, each layer returns the total (maximum) size of its dLdA matrix in _layer_init_data_t::max_dLdA_numel.
		// This value, gathered from every layer in a layers stack, are gets aggregated by the max()
		// into the biggest possible dLdA size for a whole NNet.
		// Then two matrices of this (biggest) size are allocated and passed to a layers::bprop() function during backpropagation. One of these
		// matrices will be used as a dLdA and the other as a dLdAPrev during an each call to every layer::bprop() in the layers stack.
		// What does the return value from a bprop() do is it governs whether the caller must alternate these matrices
		// on a call to a lower layer bprop() (i.e. whether a real dLdAPrev is actually stored in dLdAPrev variable (return 1) or
		// the dLdAPrev is really stored in the (appropriately resized by the layers bprop()) dLdA variable - return 0).
		// So, simple/single layers, that don't switch/reuse these matrices should return 1. However, compound layers
		// (such as the layer_pack_vertical), that consists of other layers (and must call bprop() on them), may reuse
		// dLdA and dLdAPrev variable in order to eliminate the necessity of additional temporary dLdA matrices and corresponding data coping,
		// just by switching between dLdA and dLdAPrev between calls to inner layer's bprop()s.
		// So (continuing layer_pack_vertical example) if there was
		// an even number of calls to inner layers bprop(), then the actual dLdAPrev of the whole compound
		// layer will be inside of dLdA variable and a caller of compound layer's bprop() should NOT switch matrices on
		// subsequent call to lower layer bprop(). Therefore, the compound layer's bprop() must return 0 in that case.
		
		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer applied to weights adds a term
		// l2Coefficient*Sum(weights.^2) )
		// At this moment, code of layers::calcLossAddendum() depends on a (possibly non-stable) fact, that a loss function
		// addendum to be calculated doesn't depend on data_x or data_y (it depends on only internal nn properties, such as weights).
		// This might not be the case in a future, - update layers::calcLossAddendum() definition then.
		nntl_interface real_t lossAddendum()const noexcept;
		//should return true, if the layer has a value to add to Loss function value (there's some regularizer attached)
		nntl_interface bool hasLossAddendum()const noexcept;

		// apply() function is used to pass real layer object to a parameter-function.
		// Useful for unwrapping possible wrappers around layer objects (see hlpr_array_of_layers_same_phl) wrapper
		template<typename F> nntl_interface void apply(F&& f)noexcept;
		template<typename F> nntl_interface void apply(F&& f)const noexcept;

	private:
		//support for ::boost::serialization
		friend class ::boost::serialization::access;
		template<class Archive> nntl_interface void serialize(Archive & ar, const unsigned int version) {}
	};

	//////////////////////////////////////////////////////////////////////////
	// the outermost common base layer type for (almost) every layer
	template<typename RealT>
	class _layer_core : public _i_layer<RealT> {
	protected:		
		layer_index_t m_layerIdx = invalid_layer_index;//must be reachable from derived classes, because it's set not here

		//////////////////////////////////////////////////////////////////////////
		//for a data packing reasons we have a plenty of space here to fit some flags
		// 
		//run-time flags (reset during init/deinit())
		bool m_bActivationsValid = false;

		//////////////////////////////////////////////////////////////////////////
		// persistent flags (generally unaffected by init/deinit(), but actually depends on derived class intent)

		//#todo this flag is probably worst possible solution, however we may need some flag to switch off nonlinearity in a run-time.
		//Is there a better (non-branching when it's not necessary) solution available?
		// Might be unused in some derived class (until conditional member variables are allowed). Lives here for packing reasons.
		bool m_bIgnoreActivation = false;

	private:
		// if true layer still MUST calculate correct dL/dAprev but MUST NOT update own weights
		// NEVER change after init()!
		bool m_bDoNotUpdateWeights = false;
		
		//if true layer MUST skip backpropagation phase completely. Note that every layer located below current MUST also have
		//this flag turned on or it will process invalid dLdA.
		// NEVER change after init()!
		//bool m_bDropBProp = false;//too much burden with proper implementation. Will do sometime in future

		NNTL_DEBUG_DECLARE(bool dbgm_bInitialized = false);

	protected:
		~_layer_core()noexcept{}
		_layer_core()noexcept{}

		void init()noexcept {
			m_bActivationsValid = false;
			NNTL_DEBUG_DECLARE(dbgm_bInitialized = true);
			//persistent flags are left untouched
		}
		void deinit()noexcept {
			m_bActivationsValid = false;
			NNTL_DEBUG_DECLARE(dbgm_bInitialized = false);
			//persistent flags are left untouched
		}

	public:
		//in most cases will be overriden in derived class to check whether 0 is a valid value
		layer_index_t get_layer_idx() const noexcept {
			NNTL_ASSERT(m_layerIdx != invalid_layer_index || !"Layer was not initialized!");
			return m_layerIdx;
		}

		bool bUpdateWeights()const noexcept { return !m_bDoNotUpdateWeights; }
		//bool bDoBProp()const noexcept { return !m_bDropBProp; }

		//Note, that meddling with _setUpdateWeights/_setDoBProp after init() phase may lead to memory corruption
		//because these variables define how much memory should be allocated
		void _setUpdateWeights(const bool b)noexcept {
			NNTL_ASSERT(!dbgm_bInitialized || !"Hey! Don't call that function after init() was run!");
			m_bDoNotUpdateWeights = !b;
		}
		//Note, that meddling with _setUpdateWeights/_setDoBProp after init() phase may lead to memory corruption
		//because these variables define how much memory should be allocated
		// use hlpr_setDoBProp_for_layerId_range
		//Also note that disabling bprop is actually a kind of a hack and there's no check of correct usage. So it's yours
		// responsibility to make sure that every layer under the layer with disabled bprop also have it disabled!
		/*void _setDoBProp(const bool b)noexcept {
			NNTL_ASSERT(!dbgm_bInitialized || !"Hey! Don't call that function after init() was run!");
			m_bDropBProp = !b;
		}*/
	};


	//////////////////////////////////////////////////////////////////////////
	// poly base class, Implements compile time polymorphism (to get rid of virtual functions)
	// and default _layer_name_ machinery
	// Every derived class MUST have typename FinalPolymorphChild as the first template parameter!
	template<typename FinalPolymorphChild, typename RealT>
	class _cpolym_layer_base : public _layer_core<RealT> {
		typedef _layer_core<RealT> _base_class_t;
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs
		typedef FinalPolymorphChild self_t;
		NNTL_METHODS_SELF_CHECKED((::std::is_base_of<_cpolym_layer_base, FinalPolymorphChild>::value)
			, "FinalPolymorphChild must derive from _cpolym_layer_base<FinalPolymorphChild>");

		//can't work here
		//static constexpr bool bOutputLayer = is_layer_output<self_t>::value;

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
		~_cpolym_layer_base()noexcept {}
		_cpolym_layer_base(const char* pCustName = nullptr)noexcept : _base_class_t(){
			set_custom_name(pCustName);
		}

	public:
		template<typename F> void apply(F&& f) noexcept { f(get_self()); }
		template<typename F> void apply(F&& f)const noexcept { f(get_self()); }

		static constexpr const char* get_default_name()noexcept { return self_t::_defName; }
		self_ref_t set_custom_name(const char* pCustName)noexcept {
			NNTL_ASSERT(!pCustName || strlen(pCustName) < customNameMaxChars);
			m_customName = pCustName;
			return get_self();
		}
		const char* get_custom_name()const noexcept { return m_customName ? m_customName : get_self().get_default_name(); }

		void get_layer_name(char* pName, const size_t cnt)const noexcept {
			::sprintf_s(pName, cnt, "%s_%d", get_self().get_custom_name(),static_cast<unsigned>(get_self().get_layer_idx()));
		}
		::std::string get_layer_name_str()const noexcept {
			constexpr size_t ml = layerNameMaxChars;
			char n[ml];
			get_self().get_layer_name(n, ml);
			return ::std::string(n);
		}

		//#todo: need a way to define layer type based on something more versatile than a self_t::_defName
		//probably based on https://akrzemi1.wordpress.com/2017/06/28/compile-time-string-concatenation/
		//or https://crazycpp.wordpress.com/2014/10/17/compile-time-strings-with-constexpr/
		// better use boost::hana:string, but need to switch compiler first.
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
	template<typename FinalPolymorphChild, typename InterfacesT>
	class _layer_base 
		: public _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::iMath_t::real_t>
		, public _impl::_common_data_consumer<InterfacesT>
	{
	private:
		typedef _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::iMath_t::real_t> _base_class_t;
	public:
		//////////////////////////////////////////////////////////////////////////
		//typedefs		
		typedef _impl::_layer_init_data<common_data_t> _layer_init_data_t;

		using _base_class_t::real_t;


	private:
		neurons_count_t m_neurons_cnt, m_incoming_neurons_cnt;

		//hiding from derived classes completely. Available methods to meddle are provided
		using _base_class_t::m_layerIdx;
		using _base_class_t::m_bIgnoreActivation;
		
	private:
		static constexpr const char _defName[] = "_base";

	protected:
		//////////////////////////////////////////////////////////////////////////
		//constructors-destructor
		~_layer_base()noexcept {};
		_layer_base(const neurons_count_t _neurons_cnt, const char* pCustomName=nullptr) noexcept 
			: _base_class_t(pCustomName)
			, m_neurons_cnt(_neurons_cnt), m_incoming_neurons_cnt(0)
		{
			//we can't dismiss a case when m_neurons_cnt == 0 here, because some layers can get valid neuron count a bit later.
			//So check it in constructor of derived class where applicable
			NNTL_ASSERT(m_neurons_cnt >= 0);
		};
	
	public:
		//////////////////////////////////////////////////////////////////////////
		//nntl_interface overridings
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage = nullptr)noexcept {
			NNTL_UNREF(pNewActivationStorage);
			_base_class_t::init();

			set_common_data(lid.commonData);

			get_iInspect().init_layer(get_layer_idx(), get_self().get_layer_name_str(), get_self().get_layer_type_id());

			return ErrorCode::Success;
		}
		void deinit() noexcept {
			clean_common_data();
			_base_class_t::deinit();
		}

		void initMem(real_t* , numel_cnt_t )noexcept {}

	public:
		layer_index_t get_layer_idx() const noexcept { 
			NNTL_ASSERT(is_layer_input<self_t>::value || m_layerIdx);
			return _base_class_t::get_layer_idx();
		}
		neurons_count_t get_neurons_cnt() const noexcept { 
			NNTL_ASSERT(m_neurons_cnt > 0);
			return m_neurons_cnt;
		}
		//for layers that need to calculate their neurons count in run-time (layer_pack_horizontal)
		void _set_neurons_cnt(const neurons_count_t nc)noexcept {
			NNTL_ASSERT(nc > 0);
			NNTL_ASSERT(m_neurons_cnt==0 || !"m_neurons_cnt has already been set!");//shouldn't be set multiple times
			m_neurons_cnt = nc;
		}

		neurons_count_t get_incoming_neurons_cnt()const noexcept { 
			NNTL_ASSERT((0 == _base_class_t::get_layer_idx() && is_layer_input<self_t>::value)
				|| m_incoming_neurons_cnt);//m_incoming_neurons_cnt will be zero in input layer (it has m_layerIdx==0)

			return m_incoming_neurons_cnt;
		}		

		//returns a loss function summand, that's caused by this layer (for example, L2 regularizer adds term
		// l2Coefficient*Sum(weights.^2) )
		constexpr real_t lossAddendum()const noexcept { return real_t(0.0); }
		
		//////////////////////////////////////////////////////////////////////////

		template<bool c = is_layer_learnable<self_t>::value >
		::std::enable_if_t<c, bool> bIgnoreActivation()const noexcept { return m_bIgnoreActivation; }

		//for a layer that is not learnable we should return false to make activation function work
		template<bool c = is_layer_learnable<self_t>::value >
		::std::enable_if_t<!c, bool> constexpr bIgnoreActivation()const noexcept { return false; }

		template<bool c = is_layer_learnable<self_t>::value >
		::std::enable_if_t<c> setIgnoreActivation(const bool b)noexcept { m_bIgnoreActivation = b; }

		//////////////////////////////////////////////////////////////////////////
		// other funcs
	protected:
		//this is how we going to initialize layer indexes.
		friend class _impl::_preinit_layers;
		void _preinit_layer(_impl::init_layer_index& ili, const neurons_count_t inc_neurons_cnt)noexcept{
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(m_layerIdx == invalid_layer_index);
			NNTL_ASSERT(!m_incoming_neurons_cnt);

			if (m_layerIdx != invalid_layer_index || m_incoming_neurons_cnt) ::abort();
			m_layerIdx = ili.newIndex();
			if (m_layerIdx) {//special check for the first (input) layer that doesn't have any incoming neurons
				NNTL_ASSERT(inc_neurons_cnt);
				m_incoming_neurons_cnt = inc_neurons_cnt;
			}
		}

	private:
		//////////////////////////////////////////////////////////////////////////
		//Serialization support
		friend class ::boost::serialization::access;
		//nothing to do here at this moment, also leave nntl_interface marker to prevent calls.
		//#TODO serialization function must be provided
		template<class Archive> nntl_interface void serialize(Archive & ar, const unsigned int version);
	};

	//////////////////////////////////////////////////////////////////////////
	// "light"-version of _layer_base that forwards its functions to some other layer, that is acceptable by get_self()._forwarder_layer()

	template<typename FinalPolymorphChild, typename InterfacesT>
	class _layer_base_forwarder 
		: public _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::real_t>
		, public interfaces_td<InterfacesT> 
	{
		typedef _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::real_t> _base_class_t;
	public:
		typedef typename InterfacesT::real_t real_t;

		static constexpr bool bAllowToBlockLearning = inspector::is_gradcheck_inspector<iInspect_t>::value;

	private:
		using _base_class_t::m_layerIdx;//hiding from derived classes completely

	protected:
		~_layer_base_forwarder()noexcept{}
		_layer_base_forwarder(const char* pCustName = nullptr)noexcept 
			: _cpolym_layer_base<FinalPolymorphChild, typename InterfacesT::real_t>(pCustName)
		{}

		//specialized _preinit_layer() to be called by derived class only
		void _preinit_layer(_impl::init_layer_index& ili)noexcept {
			//there should better be an exception, but we don't want exceptions at all.
			//anyway, there is nothing to help to those who'll try to abuse this API...
			NNTL_ASSERT(m_layerIdx == invalid_layer_index);
			if (m_layerIdx != invalid_layer_index) ::abort();
			m_layerIdx = ili.newIndex();
		}

	public:
		layer_index_t get_layer_idx() const noexcept {
			NNTL_ASSERT(is_layer_input<self_t>::value || m_layerIdx);
			return _base_class_t::get_layer_idx();
		}

		//////////////////////////////////////////////////////////////////////////
		// helpers to access common data 
		// #todo this implies, that the following functions are described in _i_layer interface. It's not the case at this moment.
		bool has_common_data()const noexcept { return get_self()._forwarder_layer().has_common_data(); }
		const auto& get_common_data()const noexcept { return get_self()._forwarder_layer().get_common_data(); }
		iMath_t& get_iMath()const noexcept { return get_self()._forwarder_layer().get_iMath(); }
		iRng_t& get_iRng()const noexcept { return get_self()._forwarder_layer().get_iRng(); }
		iInspect_t& get_iInspect()const noexcept { return get_self()._forwarder_layer().get_iInspect(); }

		template<bool B = bAllowToBlockLearning>
		::std::enable_if_t<B, const bool> isLearningBlocked()const noexcept {
			return get_self()._forwarder_layer().isLearningBlocked();
		}
		template<bool B = bAllowToBlockLearning>
		constexpr ::std::enable_if_t<!B, bool> isLearningBlocked() const noexcept { return false; }

		neurons_count_t get_neurons_cnt() const noexcept { return get_self()._forwarder_layer().get_neurons_cnt(); }
		neurons_count_t get_incoming_neurons_cnt()const noexcept { return  get_self()._forwarder_layer().get_incoming_neurons_cnt(); }

		const realmtxdef_t& get_activations()const noexcept { return get_self()._forwarder_layer().get_activations(); }
		const realmtxdef_t* get_activations_storage()const noexcept { return get_self()._forwarder_layer().get_activations_storage(); }
		realmtxdef_t& _get_activations_mutable()const noexcept { return get_self()._forwarder_layer()._get_activations_mutable(); }
		mtx_size_t get_activations_size()const noexcept { return get_self()._forwarder_layer().get_activations_size(); }
		bool is_activations_shared()const noexcept { return get_self()._forwarder_layer().is_activations_shared(); }

	};

}