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

#include "_layer_base.h"


namespace nntl {
namespace _impl {

	//_act_stor is a base class that handles m_activations matrix
	// If not mentioned explicitly in a function comment, any member function of the class #supportsBatchesInRows (at least it should)
	template<typename FinalPolymorphChild, typename InterfacesT>
	class _act_stor : public _layer_base<FinalPolymorphChild, InterfacesT> {
	private:
		typedef _layer_base<FinalPolymorphChild, InterfacesT> _base_class_t;

	protected:
		// matrix of layer neurons activations: <batch_size rows> x <m_neurons_cnt+1(bias) cols> for fully connected layer
		// Class assumes, that it's content on the beginning of the bprop() is the same as it was on exit from fprop().
		// Don't change it outside of this class or derived classes.
		realmtxdef_t m_activations;

	protected:
		~_act_stor()noexcept {}
		_act_stor(const neurons_count_t _neurons_cnt, const char* pCustomName = nullptr)noexcept
			: _base_class_t(_neurons_cnt, pCustomName)
		{
			m_activations.emulate_biases(!is_layer_output<self_t>::value);
		}

	public:
		// Class assumes that the content of the m_activations matrix on the beginning of the bprop() is the same as it was on exit from fprop().
		// Don't change it outside of the class!
		const realmtxdef_t& get_activations()const noexcept {
			NNTL_ASSERT(m_bActivationsValid);
			NNTL_ASSERT_MTX_NO_NANS(m_activations);
			return m_activations;
		}
		const realmtxdef_t* get_activations_storage()const noexcept { return &m_activations; }
		realmtxdef_t* get_activations_storage_mutable() noexcept { return &m_activations; }

		realmtxdef_t& _get_activations_mutable() noexcept {
			return const_cast<realmtxdef_t&>(get_activations());
		}

		mtx_size_t get_activations_size()const noexcept { return m_activations.size(); }

		bool is_activations_valid()const noexcept { return m_bActivationsValid; }

		bool is_activations_shared()const noexcept { return m_activations.bDontManageStorage(); }

		//////////////////////////////////////////////////////////////////////////
		ErrorCode init(_layer_init_data_t& lid, real_t* pNewActivationStorage)noexcept {
			const auto ec = _base_class_t::init(lid, pNewActivationStorage);
			if (ErrorCode::Success != ec) return ec;

			static constexpr bool bActivationMustHaveBiases = !is_layer_output<self_t>::value;

			//non output layers must have biases while output layers must not
			NNTL_ASSERT(!(m_activations.emulatesBiases() ^ bActivationMustHaveBiases));

			const auto neurons_cnt = get_neurons_cnt();
			NNTL_ASSERT(neurons_cnt);
			const auto biggestBatchSize = get_common_data().biggest_batch_size();
			if (pNewActivationStorage) {
				NNTL_ASSERT(!is_layer_output<self_t>::value || !"WTF? pNewActivationStorage can't be set for output layer");
// 				m_activations.useExternalStorage(pNewActivationStorage
// 					, biggestBatchSize, neurons_cnt + bActivationMustHaveBiases, bActivationMustHaveBiases); //+1 for biases
				m_activations.useExternalStorage(biggestBatchSize, neurons_cnt + bActivationMustHaveBiases//+1 for biases
					, pNewActivationStorage, bActivationMustHaveBiases);
			} else {
				if (!m_activations.resize_as_dataset(biggestBatchSize, neurons_cnt))
					return ErrorCode::CantAllocateMemoryForActivations;
			}
			return ErrorCode::Success;
		}

		void deinit() noexcept {
			static constexpr bool bActivationMustHaveBiases = !is_layer_output<self_t>::value;
			NNTL_ASSERT(!(m_activations.emulatesBiases() ^ bActivationMustHaveBiases));
			m_activations.clear();
			_base_class_t::deinit();
		}

		void on_batch_size_change(real_t*const pNewActivationStorage = nullptr)noexcept {
			constexpr bool bOutputLayer = is_layer_output<self_t>::value;

			NNTL_ASSERT(get_neurons_cnt());
			NNTL_ASSERT(m_activations.emulatesBiases() ^ bOutputLayer);
			m_bActivationsValid = false;

			const auto& CD = get_common_data();
			const vec_len_t batchSize = CD.get_cur_batch_size();
			NNTL_ASSERT(batchSize > 0 && batchSize <= CD.biggest_batch_size());

		#pragma warning(push)
		#pragma warning(disable : 4127)
			NNTL_ASSERT(!bOutputLayer || !pNewActivationStorage);
			if (!bOutputLayer && pNewActivationStorage) {
				NNTL_ASSERT(m_activations.bDontManageStorage());
				//m_neurons_cnt + 1 for biases
				//m_activations.useExternalStorage(pNewActivationStorage, batchSize, get_neurons_cnt() + 1, true);
				m_activations.useExternalStorage(batchSize, get_neurons_cnt() + 1, pNewActivationStorage, true);
				//should not restore biases here, because for compound layers its a job for their fprop() implementation
			} else {
				NNTL_ASSERT(!pNewActivationStorage);
				NNTL_ASSERT(m_activations.bOwnStorage());

				const auto oldBatchSize = m_activations.batch_size();
				m_activations.deform_batch_size(batchSize);
				//we must restore biases if the batch size has been changed
				if (!bOutputLayer && oldBatchSize != batchSize && batchSize != CD.biggest_batch_size()) {
					m_activations.set_biases();
				}
				NNTL_ASSERT(bOutputLayer || m_activations.test_biases_strict());
			}
		#pragma warning(pop)
		}
	};

}
}
