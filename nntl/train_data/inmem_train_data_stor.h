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

#include "../serialization/serialization.h"

#include "_i_train_data.h"

namespace nntl {
	//namespace _impl {

		//dummy struct to handle training data
		template<typename XT, typename YT = XT>
		class inmem_train_data_stor : public virtual DataSetsId {
		public:
			typedef XT x_t;
			typedef YT y_t;
			// 		typedef BaseT value_type;
			// 		typedef BaseT real_t;
			// 		typedef math::smatrix<value_type> realmtx_t;
			// 		typedef math::smatrix_deform<real_t> realmtxdef_t;

			typedef math::smatrix<x_t> x_mtx_t;
			typedef math::smatrix<y_t> y_mtx_t;
			typedef math::smatrix_deform<x_t> x_mtxdef_t;
			typedef math::smatrix_deform<y_t> y_mtxdef_t;

		protected:
			static_assert(train_set_id == 0 && test_set_id == 1, "");
			typedef ::std::array<x_mtxdef_t, 2> x_mtx_array_t;
			typedef ::std::array<y_mtxdef_t, 2> y_mtx_array_t;

		protected:
			x_mtx_array_t m_x;
			y_mtx_array_t m_y;

		public:
			// #supportsBatchInRow
			const x_mtxdef_t& train_x()const noexcept { return m_x[train_set_id]; }
			const y_mtxdef_t& train_y()const noexcept { return m_y[train_set_id]; }
			const x_mtxdef_t& test_x()const noexcept { return m_x[test_set_id]; }
			const y_mtxdef_t& test_y()const noexcept { return m_y[test_set_id]; }
			// #supportsBatchInRow
			const x_mtxdef_t& X(data_set_id_t dataSetId)const noexcept { NNTL_ASSERT(dataSetId >= 0 && dataSetId <= 1); return m_x[dataSetId]; }
			const y_mtxdef_t& Y(data_set_id_t dataSetId)const noexcept { NNTL_ASSERT(dataSetId >= 0 && dataSetId <= 1); return m_y[dataSetId]; }
			// #supportsBatchInRow
			x_mtxdef_t& train_x_mutable() noexcept { return m_x[train_set_id]; }
			y_mtxdef_t& train_y_mutable() noexcept { return m_y[train_set_id]; }
			x_mtxdef_t& test_x_mutable() noexcept { return m_x[test_set_id]; }
			y_mtxdef_t& test_y_mutable() noexcept { return m_y[test_set_id]; }
			// #supportsBatchInRow
			x_mtxdef_t& X_mutable(data_set_id_t dataSetId) noexcept { NNTL_ASSERT(dataSetId >= 0 && dataSetId <= 1); return m_x[dataSetId]; }
			y_mtxdef_t& Y_mutable(data_set_id_t dataSetId) noexcept { NNTL_ASSERT(dataSetId >= 0 && dataSetId <= 1); return m_y[dataSetId]; }

			//////////////////////////////////////////////////////////////////////////
			//Serialization support
		private:
			friend class ::boost::serialization::access;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int) {
				ar & serialization::make_nvp("train_x", train_x_mutable());
				ar & serialization::make_nvp("train_y", train_y_mutable());
				ar & serialization::make_nvp("test_x", test_x_mutable());
				ar & serialization::make_nvp("test_y", test_y_mutable());
				NNTL_ASSERT(absorbsion_will_succeed(train_x_mutable(), train_y_mutable(), test_x_mutable(), test_y_mutable()));
				// 			STDCOUTL("serialize_training_parameters is " << ::std::boolalpha
				// 				<< utils::binary_option(ar, serialization::serialize_training_parameters) << ::std::noboolalpha);
			}

		public:
			inmem_train_data_stor()noexcept {}

			//!! copy constructor not needed
			inmem_train_data_stor(const inmem_train_data_stor& other)noexcept = delete;
			//!!assignment is not needed
			inmem_train_data_stor& operator=(const inmem_train_data_stor& rhs) noexcept = delete;
			
			// #supportsBatchInRow
			inmem_train_data_stor& operator=(inmem_train_data_stor&& rhs) noexcept {
				if (this != &rhs) {
					if (rhs.empty()) {
						clear();
					} else {
						train_x_mutable() = ::std::move(rhs.train_x_mutable());
						train_y_mutable() = ::std::move(rhs.train_y_mutable());
						test_x_mutable() = ::std::move(rhs.test_x_mutable());
						test_y_mutable() = ::std::move(rhs.test_y_mutable());
					}
				}
				return *this;
			}

			// #supportsBatchInRow
			void absorb(inmem_train_data_stor&& rhs)noexcept {
				*this = ::std::move(rhs);
			}

			//////////////////////////////////////////////////////////////////////////

			//using designated function instead of operator= to prevent accidental use
			// #supportsBatchInRow
			bool dupe(inmem_train_data_stor& td)const noexcept {
				x_mtxdef_t trx, tx;
				y_mtxdef_t trY, ty;

				if (!trx.cloneFrom(train_x())) return false;
				if (!trY.cloneFrom(train_y())) return false;
				if (!tx.cloneFrom(test_x())) return false;
				if (!ty.cloneFrom(test_y())) return false;

				return td.absorb(::std::move(trx), ::std::move(trY), ::std::move(tx), ::std::move(ty));
			}

			//////////////////////////////////////////////////////////////////////////
			// #supportsBatchInRow
			bool operator==(const inmem_train_data_stor& rhs)const noexcept {
				return train_x() == rhs.train_x() && train_y() == rhs.train_y() && test_x() == rhs.test_x() && test_y() == rhs.test_y();
			}

			// #supportsBatchInRow
			bool empty()const noexcept {
				return train_x().empty() || train_y().empty() || test_x().empty() || test_y().empty();
			}

			// #supportsBatchInRow
			void clear()noexcept {
				for (data_set_id_t i = 0; i < 2; ++i) {
					X_mutable(i).clear();
					Y_mutable(i).clear();
				}
			}

			// #supportsBatchInRow
			bool absorb(x_mtxdef_t&& _train_x, y_mtxdef_t&& _train_y, x_mtxdef_t&& _test_x, y_mtxdef_t&& _test_y)noexcept {
				if (!absorbsion_will_succeed(_train_x, _train_y, _test_x, _test_y))  return false;
				NNTL_ASSERT(_train_x.test_biases_strict());
				NNTL_ASSERT(_test_x.test_biases_strict());

				train_x_mutable() = ::std::move(_train_x);
				train_y_mutable() = ::std::move(_train_y);
				test_x_mutable() = ::std::move(_test_x);
				test_y_mutable() = ::std::move(_test_y);
				return true;
			}

			// #supportsBatchInRow
			static bool absorbsion_will_succeed(const x_mtxdef_t& _train_x, const y_mtxdef_t& _train_y
				, const x_mtxdef_t& _test_x, const y_mtxdef_t& _test_y)noexcept //, const bool noBiasEmulationNecessary) noexcept
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

			// #supportsBatchInRow
			bool replace_Y_will_succeed(const y_mtxdef_t& _train_y, const y_mtxdef_t& _test_y)const noexcept
			{
				return !_train_y.empty() && _train_y.rows() == train_y().rows()
					&& !_test_y.empty() && _test_y.rows() == test_y().rows()
					&& _train_y.cols() == _test_y.cols()
					&& !_train_y.emulatesBiases() && !_test_y.emulatesBiases()
					&& !_train_y.bDontManageStorage() && !_test_y.bDontManageStorage()
					;
			}

			// #supportsBatchInRow
			bool replace_Y(y_mtxdef_t&& _train_y, y_mtxdef_t&& _test_y)noexcept {
				if (!replace_Y_will_succeed(_train_y, _test_y)) return false;

				train_y_mutable() = ::std::move(_train_y);
				test_y_mutable() = ::std::move(_test_y);
				return true;
			}

			//opposite of absorb()
			// #supportsBatchInRow
			void extract(x_mtxdef_t& _train_x, y_mtxdef_t& _train_y, x_mtxdef_t& _test_x, y_mtxdef_t& _test_y)noexcept {
				NNTL_ASSERT(!empty());

				_train_x = ::std::move(train_x_mutable());
				_train_y = ::std::move(train_y_mutable());
				_test_x = ::std::move(test_x_mutable());
				_test_y = ::std::move(test_y_mutable());

				NNTL_ASSERT(empty());
			}

			// #supportsBatchInRow
			bool samplesStorageCoherent()const noexcept {
				const auto bb = train_x().bBatchInRow();
				return bb == train_y().bBatchInRow() && bb == test_x().bBatchInRow() && bb == test_y().bBatchInRow();
			}
			// #supportsBatchInRow
			bool samplesXStorageCoherent()const noexcept {
				return train_x().bBatchInRow() == test_x().bBatchInRow();
			}
			bool samplesYStorageCoherent()const noexcept {
				return train_y().bBatchInRow() == test_y().bBatchInRow();
			}

			//////////////////////////////////////////////////////////////////////////
			// Note that you must know what you're doing if you are to use this function
			// Stick to #supportsBatchInRow tagged API only then when will use transposed data
			// Note that if some matrix's value_type is not floating_point, there may be blood (due to hack to use
			// of BLAS routine for transposition), or may be not (shouldn't, I guess, for common x86/x64)
			template<typename iMathT>
			void transpose()noexcept {
				NNTL_ASSERT(!empty());
				iMathT::s_mTranspose_ip(train_x_mutable());
				iMathT::s_mTranspose_ip(train_y_mutable());
				iMathT::s_mTranspose_ip(test_x_mutable());
				iMathT::s_mTranspose_ip(test_y_mutable());
			}
			template<typename iMathT>
			void transposeX()noexcept {
				NNTL_ASSERT(!empty());
				iMathT::s_mTranspose_ip(train_x_mutable());
				iMathT::s_mTranspose_ip(test_x_mutable());
			}
			template<typename iMathT>
			void transposeY()noexcept {
				NNTL_ASSERT(!empty());
				iMathT::s_mTranspose_ip(train_y_mutable());
				iMathT::s_mTranspose_ip(test_y_mutable());
			}
		};

	//}
}

