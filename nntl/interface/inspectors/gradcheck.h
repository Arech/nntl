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

#include "../_i_inspector.h"
#include "../../utils/gradcheck.h"
//#include "../../_supp/io/matfile.h"


//intended to be used for a numeric gradient check
namespace nntl {
namespace inspector {

	template<typename RealT, typename BaseInspT = _impl::_base<RealT>, size_t maxNnetDepth = 32>
	class GradCheck  : public BaseInspT {
		static_assert(std::is_base_of<_i_inspector<RealT>, BaseInspT>::value, "BaseInspT must derive from _i_inspector<>!");
	private:
		typedef BaseInspT _base_class_t;
	protected:
		typedef utils::layer_idx_keeper<layer_index_t, BaseInspT::_NoLayerIdxSpecified, maxNnetDepth> keeper_t;

	public:
		typedef GradCheck gradcheck_inspector_t;

		//we have to bring this types to the scope or MSVC will emit errors.
		using mtx_coords_t = typename _base_class_t::mtx_coords_t;
		using real_t = typename _base_class_t::real_t;
		using realmtx_t = typename _base_class_t::realmtx_t;
		using realmtxdef_t = typename _base_class_t::realmtxdef_t;

	protected:
		keeper_t m_curLayer;

		layer_index_t m_layerIdxToCheck;

		nntl::_impl::gradcheck_paramsGroup m_checkParamsGroup;
		nntl::_impl::gradcheck_phase m_checkPhase;

		mtx_coords_t m_coord;

		real_t m_stepSize;
		real_t m_analiticalValue;
		static_assert(std::numeric_limits<real_t>::has_quiet_NaN, "real_t MUST have quiet NaN available!");

		real_t* m_pChangedEl;
		real_t m_origElVal;
		
	public:
		~GradCheck() noexcept {}
		GradCheck() noexcept : m_layerIdxToCheck(0), m_pChangedEl(nullptr) {}

		void gc_reset()noexcept {
			m_layerIdxToCheck = 0;
			m_pChangedEl = nullptr;
		}

		void gc_init(const real_t& ss)noexcept {
			m_stepSize = ss;
			m_pChangedEl = nullptr;
		}
		void gc_deinit()noexcept{ m_pChangedEl = nullptr; }

		const auto gc_getCurParamsGroup()const noexcept {
			return m_checkParamsGroup;
		}

		void gc_prep_check_layer(const layer_index_t lidx, const nntl::_impl::gradcheck_paramsGroup gcpg, const mtx_coords_t& coord)noexcept {
			m_layerIdxToCheck = lidx;
			m_checkParamsGroup = gcpg;
			m_coord = coord;
			//return *this;
		}
		const real_t& get_analitical_value()const noexcept {
			NNTL_ASSERT(!std::isnan(m_analiticalValue));
			NNTL_ASSERT(nntl::_impl::gradcheck_phase::df_analitical == m_checkPhase);
			return m_analiticalValue;
		}

		void gc_set_phase(nntl::_impl::gradcheck_phase ph)noexcept {
			m_checkPhase = ph;
			m_analiticalValue = std::numeric_limits<real_t>::quiet_NaN();
		}

		//////////////////////////////////////////////////////////////////////////
		void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) noexcept {
			if (m_layerIdxToCheck) m_curLayer.push(lIdx);
			_base_class_t::fprop_begin(lIdx, prevAct, bTrainingMode);
		}

		void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)noexcept {
			if (m_layerIdxToCheck) {
				if (m_layerIdxToCheck == m_curLayer
					&& nntl::_impl::gradcheck_paramsGroup::dLdW == m_checkParamsGroup
					&& nntl::_impl::gradcheck_phase::df_analitical != m_checkPhase)
				{
					NNTL_ASSERT(!m_pChangedEl);
					m_pChangedEl = &const_cast<realmtx_t&>(W).get(m_coord);
					m_origElVal = *m_pChangedEl;
					*m_pChangedEl += nntl::_impl::gradcheck_phase::df_numeric_plus == m_checkPhase ? m_stepSize : -m_stepSize;
				}
			}
			_base_class_t::fprop_makePreActivations(W, prevAct);
		}

		void fprop_preactivations(const realmtx_t& Z) noexcept {
			if (m_layerIdxToCheck) {
				if (m_layerIdxToCheck == m_curLayer
					&& nntl::_impl::gradcheck_paramsGroup::dLdW == m_checkParamsGroup
					&& nntl::_impl::gradcheck_phase::df_analitical != m_checkPhase
					&& m_pChangedEl)
				{
					*m_pChangedEl = m_origElVal;
					m_pChangedEl = nullptr;
				}
			}
			_base_class_t::fprop_preactivations(Z);
		}

		void fprop_end(const realmtx_t& Act) noexcept {
			if (m_layerIdxToCheck){
				if (m_layerIdxToCheck == m_curLayer 
					&& nntl::_impl::gradcheck_paramsGroup::dLdA == m_checkParamsGroup
					&& nntl::_impl::gradcheck_phase::df_analitical != m_checkPhase
					&& m_curLayer.bUpperLayerDifferent())
				{
					const_cast<realmtx_t&>(Act).get(m_coord) += nntl::_impl::gradcheck_phase::df_numeric_plus == m_checkPhase 
						? m_stepSize : -m_stepSize;
				}
				m_curLayer.pop();
			}
			_base_class_t::fprop_end(Act);
		}

		void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) noexcept {
			if (m_layerIdxToCheck) {
				m_curLayer.push(lIdx);

				if (m_layerIdxToCheck==lIdx
					&& nntl::_impl::gradcheck_paramsGroup::dLdA == m_checkParamsGroup
					&& nntl::_impl::gradcheck_phase::df_analitical == m_checkPhase
					&& m_curLayer.bUpperLayerDifferent())
				{
					NNTL_ASSERT(std::isnan(m_analiticalValue));
					m_analiticalValue = dLdA.get(m_coord);
				}
			}
			_base_class_t::bprop_begin(lIdx,dLdA);
		}

		void bprop_dLdW(const realmtx_t& dLdZ, const realmtx_t& prevAct, const realmtx_t& dLdW) noexcept {
			if (m_layerIdxToCheck) {
				if (m_layerIdxToCheck == m_curLayer
					&& nntl::_impl::gradcheck_paramsGroup::dLdW == m_checkParamsGroup
					&& nntl::_impl::gradcheck_phase::df_analitical == m_checkPhase
					&& m_curLayer.bUpperLayerDifferent())
				{
					NNTL_ASSERT(std::isnan(m_analiticalValue));
					m_analiticalValue = dLdW.get(m_coord);
				}
			}
			_base_class_t::bprop_dLdW(dLdZ, prevAct, dLdW);
		}

		void bprop_end(const realmtx_t& dLdAPrev) noexcept {
			if (m_layerIdxToCheck) m_curLayer.pop();
			_base_class_t::bprop_end(dLdAPrev);
		}
	};

	template< class, class = std::void_t<> >
	struct is_gradcheck_inspector : std::false_type { };
	template< class T >
	struct is_gradcheck_inspector<T, std::void_t<typename T::gradcheck_inspector_t>> : std::true_type {};

}
}