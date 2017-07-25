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

#include <array>
#include <numeric>
#include "../interface/math/smatrix.h"

namespace nntl {
	namespace utils {

		//derive from this struct your own evaluators
		template<typename RealT>
		class ConfusionMtx : public math::smatrix_td {
		public:
			typedef RealT real_t;

		protected:
			enum {
				_iTrueNegCnt, //!bGroundTrue && !bModelPredictsTrue
				_iFalseNegCnt,// bGroundTrue && !bModelPredictsTrue
				_iFalsePosCnt,//!bGroundTrue &&  bModelPredictsTrue
				_iTruePosCnt, // bGroundTrue &&  bModelPredictsTrue

				_iCntTotal
			};
			::std::array<vec_len_t, _iCntTotal> m_asArray;

		public:
			void update(const bool& bGroundTrue, const bool& bModelPredictsTrue)noexcept {
				++m_asArray[bModelPredictsTrue * 2 + bGroundTrue];
			}

			vec_len_t& truePosCnt()noexcept { return m_asArray[_iTruePosCnt]; }
			vec_len_t& falsePosCnt()noexcept { return m_asArray[_iFalsePosCnt]; }
			vec_len_t& trueNegCnt()noexcept { return m_asArray[_iTrueNegCnt]; }
			vec_len_t& falseNegCnt()noexcept { return m_asArray[_iFalseNegCnt]; }

			const vec_len_t& truePosCnt()const noexcept { return m_asArray[_iTruePosCnt]; }
			const vec_len_t& falsePosCnt()const noexcept { return m_asArray[_iFalsePosCnt]; }
			const vec_len_t& trueNegCnt()const noexcept { return m_asArray[_iTrueNegCnt]; }
			const vec_len_t& falseNegCnt()const noexcept { return m_asArray[_iFalseNegCnt]; }

/*
			vec_len_t& TP()noexcept { return truePosCnt(); }
			vec_len_t& FP()noexcept { return falsePosCnt(); }
			vec_len_t& TN()noexcept { return trueNegCnt(); }
			vec_len_t& FN()noexcept { return falseNegCnt(); }*/

			const vec_len_t& TP()const noexcept { return truePosCnt(); }
			const vec_len_t& FP()const noexcept { return falsePosCnt(); }
			const vec_len_t& TN()const noexcept { return trueNegCnt(); }
			const vec_len_t& FN()const noexcept { return falseNegCnt(); }

			vec_len_t TotalPosCnt()const noexcept { return TP() + FP(); }
			vec_len_t TotalNegCnt()const noexcept { return TN() + FN(); }

			vec_len_t DataPosCnt()const noexcept { return TP() + FN(); }
			vec_len_t DataNegCnt()const noexcept { return TN() + FP(); }

			vec_len_t AllCnt()const noexcept { return ::std::accumulate(m_asArray.begin(), m_asArray.end(), 0); }

			
			real_t Sensitivity()const noexcept { return static_cast<real_t>(TP()) / DataPosCnt(); }
			real_t Recall()const noexcept { return Sensitivity(); }
			real_t TPR()const noexcept { return Sensitivity(); }

			real_t Specificity()const noexcept { return static_cast<real_t>(TN()) / DataNegCnt(); }
			real_t TNR()const noexcept { return Specificity(); }

			real_t Precision()const noexcept { return static_cast<real_t>(TP()) / TotalPosCnt(); }
			real_t PPV()const noexcept { return Precision(); }

			real_t NPV()const noexcept { return static_cast<real_t>(TN()) / TotalNegCnt(); }

			real_t Accuracy()const noexcept { return static_cast<real_t>(TP() + TN()) / AllCnt(); }

			real_t Informedness()const noexcept { return TPR() + TNR() - real_t(1.); }
			real_t Markedness()const noexcept { return PPV() + NPV() - real_t(1.); }
			
			real_t MCC()const noexcept {
				const auto den = (TP() + FP())*(TP() + FN())*(TN() + FP())*(TN() + FN());
				return den ? static_cast<real_t>(TP()*TN() - FP()*FN()) / ::std::sqrt(static_cast<real_t>(den))  : real_t(0.);
			}

			real_t F1Score()const noexcept {
				const auto tp2 = 2 * TP();
				return static_cast<real_t>(tp2) / (tp2 + FP() + FN());
			}
			//beta>1 weighs recall higher than precision
			//beta<1 weighs recall lower than precision
			real_t FBetaScore(const real_t& beta)const noexcept {
				const auto b2 = beta*beta;
				const auto tp2 = (real_t(1.) + b2) * TP();
				return static_cast<real_t>(tp2) / (tp2 + FP() + b2*FN());
			}


		public:
			//////////////////////////////////////////////////////////////////////////
			ConfusionMtx()noexcept {}
			~ConfusionMtx()noexcept {}

			void clear()noexcept {
				::std::fill(m_asArray.begin(), m_asArray.end(), 0);
			}
			bool operator==(const ConfusionMtx& rhs)const noexcept {
				return m_asArray == rhs.m_asArray;
			}

			void operator=(const ConfusionMtx& rhs)noexcept {
				m_asArray = rhs.m_asArray;
			}
			void operator+=(const ConfusionMtx& rhs)noexcept {
				for (unsigned i = 0; i < m_asArray.size(); ++i) {
					m_asArray[i] += rhs.m_asArray[i];
				}
			}
			void reduce(const ConfusionMtx*const pCM, const unsigned& _cnt)noexcept {
				NNTL_ASSERT(_cnt > 0 && pCM);
				*this = *pCM;
				for (unsigned i = 1; i < _cnt; ++i) *this += pCM[i];
			}

		};

	}
}