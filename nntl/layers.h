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

//provides the means to store and manage nn layers

#include <tuple>
#include <algorithm>
#include <array>

#include "layer/_layer_base.h"
#include "utils.h"

#include "_nnet_errs.h"

namespace nntl {

	// layers class is a special container for all layers used in nnet.
	// each layer is stored in layers object by its reference, therefore layer object has to be instantiated
	// somewhere by a caller. This is not so good, because semantically one thing - neural network object -
	// happens to be spread over a set of objects (individual layer objects, plus layers, plus nnet object).
	// It's possible to instantiate layer objects and layers completely withing nnet class though, but
	// in this case then I don't see a universal method to use non-default constructors of layers. And the absence of
	// non-default constructors looks worse at this moment because it can prevent some powerful optimizations to occur
	// (for example, when using constexpr's that has to be initialized from such constructors)
	// So, let's leave the possibility of using non-def constructors of layers at cost of some semantic fuzziness
	// (anyway, those who wants to have all correct can make nnet superclass by them self)
	// May be will provide a move semantic later that will allow to instantiate layers in one place, then move them into layers
	// and then move layers into nnet. But now there's no real need in this (I think)
	template <typename ...Layrs>
	class layers 
		: public math::smatrix_td
		, public interfaces_td<typename std::remove_reference<typename std::tuple_element<0, std::tuple<Layrs&...>>::type>::type::interfaces_t> {
	public:
		typedef const std::tuple<Layrs&...> _layers;

		static constexpr size_t layers_count = sizeof...(Layrs);
		static_assert(layers_count > 1, "Hey, what kind of NN with 1 layer you are gonna use?");
		typedef typename std::remove_reference<typename std::tuple_element<0, _layers>::type>::type input_layer_t;
		typedef typename std::remove_reference<typename std::tuple_element<layers_count - 1, _layers>::type>::type output_layer_t;
		typedef typename std::remove_reference<typename std::tuple_element<layers_count - 2, _layers>::type>::type preoutput_layer_t;

		//matrix type to feed into forward propagation

		typedef typename output_layer_t::common_data_t common_data_t;
		typedef typename output_layer_t::_layer_init_data_t _layer_init_data_t;

		typedef typename iMath_t::realmtx_t realmtx_t;
		typedef typename iMath_t::realmtxdef_t realmtxdef_t;

		typedef _nnet_errs::ErrorCode ErrorCode;
		typedef std::pair<ErrorCode, layer_index_t> layer_error_t;

		//we need 2 matrices for bprop()
		typedef std::array<realmtxdef_t, 2> realmtxdef_array_t;

		//test whether the first layer is m_layer_input and the last is m_layer_output derived
		static_assert(std::is_base_of<m_layer_input, input_layer_t>::value, "First/input layer must derive from m_layer_input!");
		static_assert(std::is_base_of<m_layer_output, output_layer_t>::value, "Last/output layer must derive from m_layer_output!");

		//////////////////////////////////////////////////////////////////////////
	protected:
		_layers m_layers;
		real_t m_lossAddendum;//cached value of a part of loss function addendum, that's based on regularizers properties,
		//such as L2 or L1 regularization. It doesn't depend on data_y or nnet activation and calculated during 
		// layer.lossAddendum() calls.
		//TODO: is it possible for lossAddendum() to depend on data_y or nnet activation??? Do we have the right to
		// cache this value in general case?
	private:
		layer_index_t m_totalLayersCount;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			for_each_packed_layer([&ar](auto& l) {
				ar & serialization::make_named_struct(l.get_layer_name_str().c_str(), l);
			});
		}

		//////////////////////////////////////////////////////////////////////////
	public:
		~layers()noexcept {}
		layers(Layrs&... layrs) noexcept : m_layers(layrs...), m_lossAddendum(0.0), m_totalLayersCount(0){
			//iterate over layers and check whether they i_layer derived and set their indexes
			_impl::_preinit_layers pil(m_totalLayersCount);
			utils::for_eachwp_up(m_layers, pil);
			//STDCOUTL("There are " << m_totalLayersCount << " layers at total");
		}

		//!! copy constructor not needed
		layers(const layers& other)noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is
		//!!assignment is not needed
		layers& operator=(const layers& rhs) noexcept; // = delete; //-it should be `delete`d, but factory function won't work if it is

		//better don't play with _layers directly
		_layers& get_layers()noexcept { return m_layers; }
		
		const layer_index_t total_layers()const noexcept {
			NNTL_ASSERT(m_totalLayersCount);
			return m_totalLayersCount;
		}

		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here
		template<typename _Func>
		void for_each_layer(_Func&& f)noexcept {
			//utils::for_each_up(m_layers, std::move(f));
			utils::for_each_up(m_layers, [&func{ std::forward<_Func>(f) }](auto& l) {
				call_F_for_each_layer(std::forward<_Func>(func), l);
			});
		}
		//This will apply f to every layer, packed in tuple no matter whether it is a _pack_* kind of layer or no
		template<typename _Func>
		void for_each_packed_layer(_Func&& f)noexcept {
			utils::for_each_up(m_layers, std::forward<_Func>(f));
		}

		//and apply function _Func(auto& layer) to each underlying (non-pack) layer here excluding the first
		template<typename _Func>
		void for_each_layer_exc_input(_Func&& f)noexcept {
			//utils::for_each_exc_first_up(m_layers, std::move(f));
			utils::for_each_exc_first_up(m_layers, [&func{ std::forward<_Func>(f) }](auto& l) {
				call_F_for_each_layer(std::forward<_Func>(func), l);
			});
		}
		//This will apply f to every layer excluding the first, packed in tuple no matter whether it is a _pack_* kind of layer or no
		template<typename _Func>
		void for_each_packed_layer_exc_input(_Func&& f)noexcept {
			utils::for_each_exc_first_up(m_layers, std::forward<_Func>(f));
		}

		input_layer_t& input_layer()const noexcept { return std::get<0>(m_layers); }
		output_layer_t& output_layer()const noexcept { return std::get<layers_count-1>(m_layers); }
		preoutput_layer_t& preoutput_layer()const noexcept { return std::get<layers_count - 2>(m_layers); }

		//perform layers initialization before training begins.
		layer_error_t init(const common_data_t& cd, _impl::layers_mem_requirements& LMR)const noexcept
		{
			ErrorCode ec = ErrorCode::Success;
			layer_index_t failedLayerIdx = 0;

			_layer_init_data_t lid(cd);

			utils::for_each_up(m_layers, [&](auto& lyr)noexcept {
				if (ErrorCode::Success == ec) {
					lid.clean();
					ec = lyr.init(lid);
					if (ErrorCode::Success == ec) {
						LMR.updateLayerReq(lid);
					} else {
						failedLayerIdx = lyr.get_layer_idx();
					}
				}
			});

			return layer_error_t(ec, failedLayerIdx);
		}

		void deinit()const noexcept {
			utils::for_each_up(m_layers, [](auto& lyr)noexcept {
				lyr.deinit();
			});
		}

		void initMem(real_t* ptr, numel_cnt_t cnt)noexcept {
			utils::for_each_up(m_layers, [=](auto& lyr)noexcept {
				lyr.initMem(ptr,cnt);
			});
		}

		// loss addendum value (at least at this moment with current loss functions) doesn't depend on data_x or data_y. It's purely
		// a function of weights. Therefore to prevent it double calculation during training/testing phases, we'll cache it upon first evaluation
		// and return cached value until prepToCalcLossAddendum() is called.
		// #TODO Should'n we introduce some flag to allow or disallow this kind of optimization? There's a risk to forget about it once
		// the loss function calculation will change.
		void prepToCalcLossAddendum()noexcept {
			m_lossAddendum = real_t(0.);
		}
		real_t calcLossAddendum()noexcept {
			if (m_lossAddendum==real_t(0.0)) {
				real_t ret(0.0);
				utils::for_each_up(m_layers, [&](auto& lyr)noexcept {
					const auto v = lyr.lossAddendum();
					//may assert here when learningRate<0. Should find a better way to test loss addendum correctness
					NNTL_ASSERT(v >= real_t(0.0));
					ret += v;
				});
				m_lossAddendum = ret;
			}
			
			return m_lossAddendum;
		}

		// batchSize==0 puts all layers into training mode with batchSize predefined by init()::lid.training_batch_size
		// any bs>0 puts layers into evaluation/testing mode with that batchSize. bs must be <= init()::lid.max_fprop_batch_size
		void set_mode(vec_len_t batchSize = 0)noexcept {
			utils::for_each_up(m_layers, [=](auto& lyr)noexcept { lyr.set_mode(batchSize); });
		}

		void fprop(const realmtx_t& data_x)const noexcept {
			NNTL_ASSERT(data_x.test_biases_ok());

			input_layer().fprop(data_x);

			utils::for_eachwp_up(m_layers, [](auto& lcur, auto& lprev, const bool)noexcept {
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
				lcur.fprop(lprev);
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
			});
		}

		void bprop(const realmtx_t& data_y, realmtxdef_array_t& a_dLdA)const noexcept {
			NNTL_ASSERT(a_dLdA.size() == 2);
			
			if (2 == layers_count) {
				a_dLdA[0].deform(0,0);
			} else a_dLdA[0].deform_like_no_bias(preoutput_layer().get_activations());

			output_layer().bprop(data_y, preoutput_layer(), a_dLdA[0]);
			unsigned mtxIdx = 0;

			utils::for_eachwn_downbp(m_layers, [&mtxIdx, &a_dLdA](auto& lcur, auto& lprev, const bool bPrevIsFirstLayer)noexcept {
				const unsigned nextMtxIdx = mtxIdx ^ 1;
				if (bPrevIsFirstLayer) {
					//TODO: for IBP we'd need a normal matrix
					a_dLdA[nextMtxIdx].deform(0, 0);
				} else {
					a_dLdA[nextMtxIdx].deform_like_no_bias(lprev.get_activations());
				}
				
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());
				NNTL_ASSERT(a_dLdA[mtxIdx].size() == lcur.get_activations().size_no_bias());
				const unsigned bAlternate = lcur.bprop(a_dLdA[mtxIdx], lprev, a_dLdA[nextMtxIdx]);
				NNTL_ASSERT(1 == bAlternate || 0 == bAlternate);
				NNTL_ASSERT(lprev.get_activations().test_biases_ok());

				mtxIdx ^= bAlternate;
			});
		}
	};

	template <typename ...Layrs> inline constexpr
	layers<Layrs...> make_layers(Layrs&... layrs) noexcept {
		return layers<Layrs...>(layrs...);
	}


	//////////////////////////////////////////////////////////////////////////
	// helpers to change various layer properties that may or may not exist
	struct hlpr_layer_set_learning_rate {
		template<typename _L> std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l, typename _L::real_t lr)noexcept {
			l.m_gradientWorks.learning_rate(lr);
		}
		template<typename _L> std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L& l, typename _L::real_t lr)noexcept {}
	};	
	struct hlpr_layer_learning_rate_decay {
		template<typename _L> std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l, typename _L::real_t decayCoeff)noexcept {
			l.m_gradientWorks.learning_rate(l.m_gradientWorks.learning_rate()*decayCoeff);
		}
		template<typename _L> std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L& l, typename _L::real_t decayCoeff)noexcept {}
	};

	struct hlpr_layer_set_nesterov_momentum {
		template<typename _L> std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l, typename _L::real_t nm)noexcept {
			l.m_gradientWorks.nesterov_momentum(nm);
		}
		template<typename _L> std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L& l, typename _L::real_t nm)noexcept {}
	};

	struct hlpr_layer_apply_func2gradworks_layer {
		template<typename _L, typename F> std::enable_if_t<nntl::layer_has_gradworks<_L>::value> operator()(_L& l, F&& f)noexcept {
			(std::forward<F>(f))(l);
		}
		template<typename _L, typename F> std::enable_if_t<!nntl::layer_has_gradworks<_L>::value> operator()(_L& l, F&& f)noexcept {}
	};
};
