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

namespace nntl {


	//dummy struct to handle training data
	class train_data {
	public:
		//typedef math_types::realmtx_ty realmtx_t;
		typedef math_types::real_ty real_t;
		typedef math::simple_matrix<real_t> realmtx_t;

		train_data()noexcept {}
		~train_data() noexcept {};

		//!! copy constructor not needed
		train_data(const train_data& other)noexcept = delete;
		//!!assignment is not needed
		train_data& operator=(const train_data& rhs) noexcept = delete;

		//////////////////////////////////////////////////////////////////////////

		const realmtx_t& train_x()const noexcept { return m_train_x; }
		const realmtx_t& train_y()const noexcept { return m_train_y; }
		const realmtx_t& test_x()const noexcept { return m_test_x; }
		const realmtx_t& test_y()const noexcept { return m_test_y; }

		realmtx_t& train_x_mutable() noexcept { return m_train_x; }
		realmtx_t& train_y_mutable() noexcept { return m_train_y; }

		const bool empty()const noexcept {
			return m_train_x.empty() || m_train_y.empty() || m_test_x.empty() || m_test_y.empty();
		}

		const bool absorb(realmtx_t&& _train_x, realmtx_t&& _train_y, realmtx_t&& _test_x, realmtx_t&& _test_y)noexcept{
			//, const bool noBiasEmulationNecessary=false)noexcept {
			
			if (!absorbsion_will_succeed(_train_x, _train_y,_test_x,_test_y))  return false;
			m_train_x = std::move(_train_x);
			m_train_y = std::move(_train_y);
			m_test_x = std::move(_test_x);
			m_test_y = std::move(_test_y);
			return true;
		}

		static const bool absorbsion_will_succeed(
			realmtx_t& _train_x, realmtx_t& _train_y, realmtx_t& _test_x, realmtx_t& _test_y)noexcept //, const bool noBiasEmulationNecessary) noexcept
		{
			return !_train_x.empty() && !_train_y.empty() && _train_x.rows() == _train_y.rows()
				&& !_test_x.empty() && !_test_y.empty() && _test_x.rows() == _test_y.rows()
				&& _train_y.cols() == _test_y.cols()
				&& _train_x.emulatesBiases() && _test_x.emulatesBiases();
				//&& (noBiasEmulationNecessary ^ _train_x.emulatesBiases()) && (noBiasEmulationNecessary ^ _test_x.emulatesBiases());
		}

		//////////////////////////////////////////////////////////////////////////
		//members
	protected:
		realmtx_t m_train_x, m_train_y, m_test_x, m_test_y;

	};
}
