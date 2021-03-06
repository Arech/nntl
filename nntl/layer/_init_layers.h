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

#include <type_traits>
#include <algorithm>
#include "_pack_traits.h"
#include "../common_nn_data.h"

namespace nntl {

	//special marks for type checking of input and output layers
	struct m_layer_input {};
	struct m_layer_output {};
	struct m_layer_gate {};
	//struct m_layer_tolerates_no_biases {};//marks that a layer doesn't require a bias column to perform fprop/bprop correctly. (LI)

	//mark layer as the layer to cancel chain of backpropagation, which means that this layer's bprop() (as well as every layer 
	// below the layer in stack, if any) must not be called
	struct m_layer_stops_bprop {};

	//this class marks a layer as learnable. It must provide get_weights/set_weights/reinit_weights/drop_weights/isWeightsSuitable()
	// , bIgnoreActivation()/setIgnoreActivation()
	// and get_gradWorks() functions,
	// as well as compute dL/dW during bprop() (if prev layer doesn't have a m_layer_stops_bprop marker)
	struct m_layer_learnable {};

	//when a layer is derived from this class, it is expected to be used inside of some layer_pack_* objects and it
	// doesn't have a neurons count specified in constructor. Instead, compound layer (or it's support objects) specifies
	// the number of neurons during their construction via _set_neurons_cnt().
	// See LI for an example
	struct m_layer_autoneurons_cnt {};

	//////////////////////////////////////////////////////////////////////////

	template<typename LayerT>
	using is_layer_learnable = ::std::is_base_of<m_layer_learnable, LayerT>;

	template<typename LayerT>
	using is_layer_output = ::std::is_base_of<m_layer_output, LayerT>;

	template<typename LayerT>
	using is_layer_input = ::std::is_base_of<m_layer_input, LayerT>;

	template<typename LayerT>
	using is_layer_gate = ::std::is_base_of<m_layer_gate, LayerT>;

	template<typename LayerT>
	using is_layer_stops_bprop = ::std::is_base_of<m_layer_stops_bprop, LayerT>;
	template<typename LayerT>
	using is_layer_with_bprop = ::std::negation<::std::is_base_of<m_layer_stops_bprop, LayerT>>;

	//////////////////////////////////////////////////////////////////////////
	template <typename T, typename = int>
	struct Has_bLayerToleratesNoBiases : std::false_type { };

	template <typename T>
	struct Has_bLayerToleratesNoBiases <T, decltype((void)T::bLayerToleratesNoBiases, 0)> : std::true_type { };


	template< class, class = ::std::void_t<> >
	struct layer_tolerates_no_biases : ::std::false_type { };
	template< class T >
	struct layer_tolerates_no_biases<T, ::std::void_t<decltype((void)T::bLayerToleratesNoBiases, 0)>>
		: ::std::integral_constant<bool, T::bLayerToleratesNoBiases>
	{
		//static_assert(::std::is_same<bool, T::bLayerToleratesNoBiases>::value, "");
	};

	template< class, class = ::std::void_t<> >
	struct layer_has_trivial_bprop : ::std::false_type { };
	template< class T >
	struct layer_has_trivial_bprop<T, ::std::void_t<decltype((void)T::bLayerHasTrivialBProp, 0)>>
		: ::std::integral_constant<bool, T::bLayerHasTrivialBProp>
	{
		//static_assert(::std::is_same<bool, T::bLayerHasTrivialBProp>::value, "");
	};

	//////////////////////////////////////////////////////////////////////////

	namespace _impl {

		struct _m_dummy {};//when condition forbids derivation from main mark

		template<bool b>
		using conditional_layer_output = ::std::conditional_t<b, m_layer_output, _m_dummy>;

		template<bool b>
		using conditional_layer_stops_bprop = ::std::conditional_t<b, m_layer_stops_bprop, _m_dummy>;

		//////////////////////////////////////////////////////////////////////////
		//this definition of m_prop_input_marker<> strips away information about LT
		template<typename RealT> struct _not_from_IL {
			typedef RealT real_t;
		};
		template<typename RealT> struct _from_IL : m_layer_input {
			typedef RealT real_t;
		};
		template<typename LT>
		using m_prop_input_marker = ::std::conditional_t<is_layer_input<LT>::value
			, _from_IL<typename LT::real_t>
			, _not_from_IL<typename LT::real_t>
		>;

		template<typename T>
		using is_m_prop_input_marker = ::std::disjunction<
			::std::is_base_of<_not_from_IL<typename T::real_t>, T>
			, ::std::is_base_of<_from_IL<typename T::real_t>, T>
		>;

		//////////////////////////////////////////////////////////////////////////
		struct _l_stops_bprop : public m_layer_stops_bprop {};
		struct _l_has_bprop {};

		template<typename LT>
		using m_prop_stops_bprop_marker = ::std::conditional_t<is_layer_stops_bprop<LT>::value, _l_stops_bprop, _l_has_bprop>;

		//////////////////////////////////////////////////////////////////////////

		template<typename LT>
		struct m_propagate_markers : public m_prop_input_marker<LT>, public m_prop_stops_bprop_marker<LT> {};

		//////////////////////////////////////////////////////////////////////////
		class init_layer_index {
		protected:
			layer_index_t& _idx;

		public:
			~init_layer_index()noexcept {}
			init_layer_index(layer_index_t& src)noexcept : _idx(src) {
				NNTL_ASSERT(0 == src);
			}
			init_layer_index(init_layer_index& other)noexcept : _idx(other._idx) {}

			layer_index_t newIndex()noexcept {
				NNTL_ASSERT(_idx != invalid_layer_index || !"WTF?! Too huge nnet having >65k layers?");
				return _idx++;
			}
		};

		//////////////////////////////////////////////////////////////////////////
		//class to help propagate NN structure through layer stack during that stack creation
		class _preinit_layers {
		public:
			char* m_pPHLCheckStorage;//variable to hold a pointer to an array, that's used to check whether inner layers
							// of _LPH cover all activation units of the range of underlying layer
							// Initialized to nullptr by default in constructors.

			init_layer_index m_ILI;

			neurons_count_t _incNeurons;
			//layer_index_t _idx;			


			~_preinit_layers()noexcept {
				if (m_pPHLCheckStorage) delete[] m_pPHLCheckStorage;
			}
			_preinit_layers(layer_index_t& src) noexcept : m_ILI(src), _incNeurons(0), m_pPHLCheckStorage(nullptr) {}
			_preinit_layers(init_layer_index& ili, const neurons_count_t n) noexcept
				: m_ILI(ili), _incNeurons(n), m_pPHLCheckStorage(nullptr)
			{
				NNTL_ASSERT(n);
			}

			//////////////////////////////////////////////////////////////////////////
			// variation to comply tuple_utils::for_eachwp_up() callback
			// To use with class layers{}
			template <typename LCur, typename LPrev>
			void operator()(LCur& lcur, LPrev& lprev, const bool bFirst)noexcept {
				static_assert(::std::is_base_of<_i_layer<typename ::std::remove_reference<LCur>::type::real_t >
					, ::std::remove_reference<LCur>::type>::value, "Each layer must derive from i_layer");
				static_assert(::std::is_base_of<_i_layer<typename ::std::remove_reference<LCur>::type::real_t >
					, ::std::remove_reference<LPrev>::type>::value, "Each layer must derive from i_layer");
				static_assert(::std::is_same<LCur::interfaces_t, LPrev::interfaces_t>::value, "interfaces_t must be the same for all layers!");

				if (bFirst) lprev._preinit_layer(m_ILI, _incNeurons);
				lcur._preinit_layer(m_ILI, lprev.get_neurons_cnt());
			}

			//////////////////////////////////////////////////////////////////////////
			// some machinery necessary for the layer_pack_horizontal class
			bool preparePHLCheck()noexcept {
				NNTL_ASSERT(_incNeurons && !m_pPHLCheckStorage);
				m_pPHLCheckStorage = new(::std::nothrow) char[_incNeurons];
				if (m_pPHLCheckStorage) memset(m_pPHLCheckStorage, 0, _incNeurons);
				return nullptr != m_pPHLCheckStorage;
			}

			// variation to comply with tuple_utils::for_each_up() callback. For use with PHL structures in _LPH
			template<typename PHLT>
			::std::enable_if_t<is_PHL<PHLT>::value> operator()(PHLT& phl)noexcept {
				static_assert(::std::is_base_of<_i_layer<PHLT::phl_original_t::real_t>, PHLT::phl_original_t>::value, "Each layer must derive from i_layer");
				static_assert(!is_layer_input<PHLT::phl_original_t>::value && !is_layer_output<PHLT::phl_original_t>::value,
					"No input/output layers is allowed within _LPH");

				NNTL_ASSERT(m_pPHLCheckStorage && phl.coord.m_count && phl.coord.m_offset < _incNeurons && (phl.coord.m_offset + phl.coord.m_count) <= _incNeurons);
				const auto pBeg = m_pPHLCheckStorage + phl.coord.m_offset;
				::std::fill(pBeg, pBeg + phl.coord.m_count, char(1));

				phl.l._preinit_layer(m_ILI, phl.coord.m_count);
			}

			bool PHLCheck()noexcept {
				NNTL_ASSERT(m_pPHLCheckStorage);
				const bool r = ::std::all_of(m_pPHLCheckStorage, m_pPHLCheckStorage + _incNeurons, [](const char c)->bool {
					return c == char(1);
				});
				delete[] m_pPHLCheckStorage;
				m_pPHLCheckStorage = nullptr;
				return r;
			}

			//////////////////////////////////////////////////////////////////////////
			// variation to use in other case just to preinit single layer
			template<typename Layr>
			::std::enable_if_t<!is_PHL<Layr>::value> operator()(Layr& layr)noexcept {
				layr._preinit_layer(m_ILI, _incNeurons);
			}
		};
	
		//////////////////////////////////////////////////////////////////////////
		// This structure is passed to a _i_layer::init() during initialization phase.
		// OUT marks variables that should be filled/returned by a layer if applicable. Most of this variables are used to
		// find out how many shared memory real_t's should be allocated by a nnet object during initialization phase
		// and passed to the layer.initMem().
		template<typename CommonNnData>
		struct _layer_init_data : public math::smatrix_td {
			typedef CommonNnData common_data_t;
			typedef typename CommonNnData::real_t real_t;

			// "IN" marks variables that are passed to init() function, "OUT" marks output from init()

			IN const common_data_t& commonData;

			// Here are some insights about how the memory handling is organized in NNTL in order to achieve minimum memory requirements.
			// There are three "types" of memory:
			// 1. There are memory regions that must be independent of everything else and store values as long as they are needed.
			//		For example, activation values should be stored almost all the time layer object exists. This type of memory can't be
			//		optimized and shared between layer objects. Therefore, it's "private" memory.
			// 2. However, some memory are needed only to hold temporary computations during some functions execution. Be it a kind
			//		of mathematical function (like a softmax, that requires some storage to hold rowwise max values 
			//		of pre-activations) or a fprop() or a bprop() functions that may require some temporary data like a computed dL/dW matrix
			//		for their work.
			//		In this case it makes sense:
			//		a) to allow iMath's object to have it's own temporary storage (see _i_math::preinit() and _i_math::init() functions and their
			//			implementations), and
			//		b) to share temporary memory storage between different layers. The only guarantee this shared memory could issue is that
			//			the memory will be left intact only during fprop() or bprop() execution.
			//
			// We'll discuss next how 2.b type of memory is organized.
			// 
			// fprop() and bprop() can use different batch sizes during a single training session. For example, during
			// the learning process fprop()/bprop() could use a small batch size, however a whole data_x.rows() of a source dataset 
			// could be necessary for fprop() for a loss function value computation. Therefore to reduce memory consumption
			// during nnet evaluating and learning, it makes sense to distinguish how much memory is required to evaluate nnet, from how much
			// memory is required for the training.
			// During the init() phase, layer gets a reference to common_data_t structure via _layer_init_data::commonData. This structure
			// contains description of maximum number of data rows for evaluating nnet and for training nnet. Using this information, the layer
			// should calculate how much shared memory (in number of real_t elements) it would require in both cases. The layer returns this
			// numbers in _layer_init_data::maxMemFPropRequire and _layer_init_data::maxMemTrainingRequire variables.
			// Then the .initMem() function gets called with a pointer to shared memory and a **maximum** total available elements in it
			// (max of _layer_init_data::maxMemFPropRequire and _layer_init_data::maxMemTrainingRequire). The layer then should "save"
			// this pointer, for example, by redistributing it among necessary matrices via realmtx_t::useExternalStorage()
			// After that, the nnet object sets the new batch size in the shared common_data_t structure and calls layer's
			// .on_batch_size_change() function to allow layer update sizes of it's internal variables.
			// Only then the .fprop()/.bprop() functions could be called (with the same batch_size, that was found during
			// last call to .on_batch_size_change()). When a new batch_size will be necessary, the nnet object will set it in common_data_t
			// and call .on_batch_size_change() once more.
			
			OUT numel_cnt_t maxMemFPropRequire;// total size of <real_t> array to be passed to layer.initMem() in order to compute fprop()
			OUT numel_cnt_t maxMemTrainingRequire;// same for bprop - that's how much <real_t>s must be addressed by pointer passed to .initMem() in order to bprop() to work
			OUT numel_cnt_t max_dLdA_numel;//Biggest size of dLdA matrix (numel) that could be passed into a layer for bprop()
			// We can always calculate this variable by knowing the batchSize and the layer's neurons count, aren't we?
			// NOOO!!! It's not the case for compound layers, especially such as layer_pack_tile, that hides it's m_tiledLayer
			// completely from the stack !!! We need this variable!
			OUT numel_cnt_t nParamsToLearn;//total number of parameters, that layer has to learn during training

			IN BatchSizes incBS;
			OUT BatchSizes outgBS;

			OUT bool bOutputDifferentDuringTraining;//set by layer.init() to note that the layer output (fprop results)
			// differs in the training and testing mode for the same data_x and parameters (for example, due to a dropout)

			OUT bool bLossAddendumDependsOnWeights;//layer.init() sets this option when there's at least one loss addendum that
			//depends on layer's weights values (necessary for correct loss addendum caching handling for training/testing sets).
			OUT bool bLossAddendumDependsOnActivations; //layer.init() sets this option when there's at least one loss addendum that
			//depends on activations values (necessary for correct loss addendum caching handling for training/testing sets).

			// public constructors expects the .pass_to_upper_layer() be called before use!
			_layer_init_data(const common_data_t& cd, const vec_len_t mbs, const vec_len_t tbs) noexcept 
				: commonData(cd), outgBS(mbs, tbs)//we always call .pass_to_upper_layer() first!
			{
				//clean(); //not necessary here because the struct is almost always reused and cleaned before each use.
				NNTL_ASSERT(outgBS.isValid());
			}

			_layer_init_data(const common_data_t& cd, const BatchSizes& bs) noexcept
				: commonData(cd), outgBS(bs.maxBS, bs.maxTrainBS)//not using copy constructor to trigger assertions if necessary
				//setting to the same value as inc_* because we always call .pass_to_upper_layer() first!
			{	//clean(); //not necessary here because the struct is almost always reused and cleaned before each use.
				NNTL_ASSERT(outgBS.isValid());
			}

		protected:
			_layer_init_data(const BatchSizes& inBs, const common_data_t& cd) noexcept : commonData(cd), incBS(inBs) {
				NNTL_ASSERT(incBS.isValid());
				_clean_exc_incBS();
			}
			void _clean_exc_incBS()noexcept {
				maxMemFPropRequire = maxMemTrainingRequire = max_dLdA_numel = nParamsToLearn = 0;
				outgBS.clear();
				bOutputDifferentDuringTraining = bLossAddendumDependsOnWeights = bLossAddendumDependsOnActivations = false;
			}

			void _clean(const BatchSizes& _bs)noexcept {				
				incBS.set_from(_bs);
				_clean_exc_incBS();
			}

		public:
			vec_len_t biggest_incoming_batch_size()const noexcept { return incBS.biggest(); }
			vec_len_t biggest_outgoing_batch_size()const noexcept { return outgBS.biggest(); }
		
			void layer_doesnt_change_batchsize()noexcept { outgBS.set_from(incBS); }

			//this function must be called on the object before it is passed to layer.init()
			void pass_to_upper_layer()noexcept {
				_clean(outgBS);
			}

			//used by compound layers to gather data from layers encapsulated into them.
			void aggregate_from(const _layer_init_data& o)noexcept {
				maxMemFPropRequire = ::std::max(maxMemFPropRequire, o.maxMemFPropRequire);
				maxMemTrainingRequire = ::std::max(maxMemTrainingRequire, o.maxMemTrainingRequire);
				max_dLdA_numel = ::std::max(max_dLdA_numel, o.max_dLdA_numel);
				nParamsToLearn += o.nParamsToLearn;
				bLossAddendumDependsOnWeights |= o.bLossAddendumDependsOnWeights;
				bLossAddendumDependsOnActivations |= o.bLossAddendumDependsOnActivations;
				bOutputDifferentDuringTraining |= o.bOutputDifferentDuringTraining;
			}

			//dupe() just like public constructors expects the .pass_to_upper_layer() to be called first
			_layer_init_data dupe()const noexcept {
				return _layer_init_data(commonData, incBS);
			}

			//produces proper duplicate with same commonData and incBS.
			//note that this object must already pass through .pass_to_upper_layer();
			_layer_init_data exact_dupe()const noexcept {
				NNTL_ASSERT(incBS.isValid());
				return _layer_init_data(incBS, commonData);
			}
		};

		//////////////////////////////////////////////////////////////////////////
		// structure to be filled during layers.init() to return necessary data back to nnet object
		struct layers_mem_requirements : public math::smatrix_td {
			numel_cnt_t maxMemLayerTrainingRequire,//for nnet.train()
				maxMemLayersFPropRequire,//#todo for nnet.eval()
				maxSingledLdANumel,//the biggest dLdA matrix required for bprop()
				totalParamsToLearn;//The total parameters count the model has

			bool bLossAddendumDependsOnWeights;
			bool bLossAddendumDependsOnActivations;
			bool bOutputDifferentDuringTraining;

			layers_mem_requirements() noexcept{
				zeros();
			}

			void zeros()noexcept {
				maxMemLayerTrainingRequire = 0;
				maxMemLayersFPropRequire = 0;
				maxSingledLdANumel = 0;//single! The biggest matrix.numel() to be used in a bprop()
				totalParamsToLearn = 0;
				bLossAddendumDependsOnWeights = false;
				bLossAddendumDependsOnActivations = false;
				bOutputDifferentDuringTraining = false;
			}

			void updateLayerReq(const numel_cnt_t& mmlF, const numel_cnt_t& mmlB
				, const numel_cnt_t& maxdLdA, const numel_cnt_t& nLP
				, const bool _bLossAddendumDependsOnWeights, const bool _bLossAddendumDependsOnActivations
				, const bool _OutputDifferentDuringTraining)noexcept
			{
				maxMemLayerTrainingRequire = ::std::max({ maxMemLayerTrainingRequire, mmlF, mmlB });
				maxMemLayersFPropRequire = ::std::max(maxMemLayersFPropRequire, mmlF);
				maxSingledLdANumel = ::std::max(maxSingledLdANumel, maxdLdA);
				totalParamsToLearn += nLP;
				bLossAddendumDependsOnWeights |= _bLossAddendumDependsOnWeights;
				bLossAddendumDependsOnActivations |= _bLossAddendumDependsOnActivations;
				bOutputDifferentDuringTraining |= _OutputDifferentDuringTraining;
			}

			template<typename _layer_init_data_t>
			void updateLayerReq(const _layer_init_data_t& lid)noexcept {
				return updateLayerReq(lid.maxMemFPropRequire, lid.maxMemTrainingRequire
					, lid.max_dLdA_numel, lid.nParamsToLearn
					, lid.bLossAddendumDependsOnWeights, lid.bLossAddendumDependsOnActivations
					, lid.bOutputDifferentDuringTraining);
			}
		};
	}

}