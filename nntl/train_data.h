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

#include "serialization/serialization.h"

namespace nntl {


	//dummy struct to handle training data
	template<typename BaseT>
	class train_data {
	public:
		typedef BaseT value_type;
		typedef math::smatrix<value_type> mtx_t;
		typedef math::smatrix_deform<value_type> mtxdef_t;

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		mtxdef_t m_train_x, m_train_y, m_test_x, m_test_y;

		//////////////////////////////////////////////////////////////////////////
		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & serialization::make_nvp("train_x", m_train_x);
			ar & serialization::make_nvp("train_y", m_train_y);
			ar & serialization::make_nvp("test_x", m_test_x);
			ar & serialization::make_nvp("test_y", m_test_y);
			NNTL_ASSERT(absorbsion_will_succeed(m_train_x, m_train_y, m_test_x, m_test_y));
// 			STDCOUTL("serialize_training_parameters is " << std::boolalpha
// 				<< utils::binary_option(ar, serialization::serialize_training_parameters) << std::noboolalpha);
		}

	public:
		train_data()noexcept {}

		//!! copy constructor not needed
		train_data(const train_data& other)noexcept = delete;
		//!!assignment is not needed
		train_data& operator=(const train_data& rhs) noexcept = delete;

		//////////////////////////////////////////////////////////////////////////

		const bool operator==(const train_data& rhs)const noexcept {
			return m_train_x == rhs.m_train_x && m_train_y == rhs.m_train_y && m_test_x == rhs.m_test_x && m_test_y == rhs.m_test_y;
		}

		const mtxdef_t& train_x()const noexcept { return m_train_x; }
		const mtxdef_t& train_y()const noexcept { return m_train_y; }
		const mtxdef_t& test_x()const noexcept { return m_test_x; }
		const mtxdef_t& test_y()const noexcept { return m_test_y; }

		mtxdef_t& train_x()noexcept { return m_train_x; }
		mtxdef_t& train_y()noexcept { return m_train_y; }
		mtxdef_t& test_x()noexcept { return m_test_x; }
		mtxdef_t& test_y()noexcept { return m_test_y; }

		mtxdef_t& train_x_mutable() noexcept { return m_train_x; }
		mtxdef_t& train_y_mutable() noexcept { return m_train_y; }

		const bool empty()const noexcept {
			return m_train_x.empty() || m_train_y.empty() || m_test_x.empty() || m_test_y.empty();
		}

		const bool absorb(mtx_t&& _train_x, mtx_t&& _train_y, mtx_t&& _test_x, mtx_t&& _test_y)noexcept{
			//, const bool noBiasEmulationNecessary=false)noexcept {
			
			if (!absorbsion_will_succeed(_train_x, _train_y,_test_x,_test_y))  return false;
			NNTL_ASSERT(_train_x.test_biases_ok());
			NNTL_ASSERT(_test_x.test_biases_ok());

			m_train_x = std::move(_train_x);
			m_train_y = std::move(_train_y);
			m_test_x = std::move(_test_x);
			m_test_y = std::move(_test_y);
			return true;
		}

		static const bool absorbsion_will_succeed(const mtx_t& _train_x, const mtx_t& _train_y
			, const mtx_t& _test_x, const mtx_t& _test_y)noexcept //, const bool noBiasEmulationNecessary) noexcept
		{
			return !_train_x.empty() && !_train_y.empty() && _train_x.rows() == _train_y.rows()
				&& !_test_x.empty() && !_test_y.empty() && _test_x.rows() == _test_y.rows()
				&& _train_y.cols() == _test_y.cols()
				&& _train_x.cols() == _test_x.cols()
				&& !_train_y.emulatesBiases() && !_test_y.emulatesBiases()
				&& _train_x.emulatesBiases() && _test_x.emulatesBiases()
				&& !_train_x.bDontManageStorage() && !_test_x.bDontManageStorage()
				&& !_train_y.bDontManageStorage() && !_test_y.bDontManageStorage()
				;
				//&& (noBiasEmulationNecessary ^ _train_x.emulatesBiases()) && (noBiasEmulationNecessary ^ _test_x.emulatesBiases());
		}

		const bool replace_Y_will_succeed(const mtx_t& _train_y, const mtx_t& _test_y)noexcept
		{
			return !_train_y.empty() && _train_y.rows() == m_train_y.rows()
				&& !_test_y.empty() && _test_y.rows() == m_test_y.rows()
				&& _train_y.cols() == _test_y.cols()
				&& !_train_y.emulatesBiases() && !_test_y.emulatesBiases()
				&& !_train_y.bDontManageStorage() && !_test_y.bDontManageStorage()
				;
		}

		const bool replace_Y(mtx_t&& _train_y, mtx_t&& _test_y)noexcept {
			if (!replace_Y_will_succeed(_train_y, _test_y)) return false;

			m_train_y = std::move(_train_y);
			m_test_y = std::move(_test_y);
			return true;
		}
	};
}
