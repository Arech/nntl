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

#include <array>

namespace nntl {

	template< class, class = ::std::void_t<> >
	struct has_real_t : public ::std::false_type { };
	// specialization recognizes types that do have a nested ::real_t member:
	template< class T >
	struct has_real_t<T, ::std::void_t<typename T::real_t>> : public ::std::true_type {};

	namespace math {

		//better DON'T use d_real_t
		typedef NNTL_CFG_DEFAULT_TYPE d_real_t;
		//static constexpr char* d_real_t_name = NNTL_STRINGIZE(NNTL_CFG_DEFAULT_TYPE); //obsolete?

		//////////////////////////////////////////////////////////////////////////
		//computes ln(x+1)
		template <typename T> T log1p(T v)noexcept {
			NNTL_ASSERT(v >= T(-1.));
#if NNTL_CFG_CAREFULL_LOG_EXP
			return ::std::log1p(v);
#else
			return ::std::log(T(1.0) + v);
#endif
		}
		//computes exp(x)-1
		template <typename T> T expm1(T v)noexcept {
#if NNTL_CFG_CAREFULL_LOG_EXP
			return ::std::expm1(v);
#else
			return ::std::exp(v) - T(1.0);
#endif
		}

#pragma float_control(push)
#pragma float_control(precise, on)

		template <typename T> T log_eps(T v)noexcept {
			NNTL_ASSERT(v >= T(0.));
			return ::std::log(v + ::std::numeric_limits<T>::min());
		}

		template <typename T> T log1p_eps(T v)noexcept {
			NNTL_ASSERT(v >= T(-1.));
			//#if NNTL_CFG_CAREFULL_LOG_EXP
			//		return ::std::log1p(v + ::std::numeric_limits<T>::epsilon());
			//we can't use is, because we have to add epsilon() to move v out of -1 (worst case scenario), however the error near v==0
			//becomes too big;
			//#else
#if NNTL_CFG_CAREFULL_LOG_EXP
#pragma message("Warning! nntl::math::log1p_eps() will NOT use ::std::log1p(), error is too large")
#endif
			return ::std::log((T(1.0) + v) + ::std::numeric_limits<T>::min());
			//#endif
		}

#pragma float_control(pop)



		//////////////////////////////////////////////////////////////////////////
		//thanks to http://stackoverflow.com/a/4609795
		template <typename T> constexpr int sign(const T val) {
			return (T(+0.0) < val) - (val < T(-0.0));
		}

		template <typename _T> struct real_t_limits {};
		template <> struct real_t_limits<double> {
			//natural log of closest to zero but non zero (realmin) value
			static constexpr double log_almost_zero = double(-708.3964185322642);

			//returns minimum value greater than zero, that can be added to v and the result be represented by double
			static double eps_greater(double v)noexcept {
				return ::std::nextafter(v, ::std::numeric_limits<double>::infinity()) - v;
			}
			static double eps_greater_n(double v, double n)noexcept {
				return n*eps_greater(v);
			}

			//returns minimum value greater than zero, that can be subtracted from v and the result be represented by double
			static double eps_lower(double v)noexcept {
				return v - ::std::nextafter(v, -::std::numeric_limits<double>::infinity());
			}
			static double eps_lower_n(double v, double n)noexcept {
				return n*eps_lower(v);
			}

			typedef ::std::uint64_t similar_FWI_t;// probably it's a "similar fixed width integer"
			static_assert(sizeof(double) == sizeof(similar_FWI_t), "Wrong type sizes!");
			static_assert(::std::is_unsigned<similar_FWI_t>::value, "");

			static constexpr double _rone = double(1.);
			static constexpr double _rposz = double(+0.);
			static constexpr double _rnegz = double(-0.);

			static constexpr double _rNaN = ::std::numeric_limits<double>::quiet_NaN();
		};
		template <> struct real_t_limits<float> {
			//natural log of closest to zero but non zero (realmin) value
			static constexpr float log_almost_zero = float(-87.336544750402);

			//returns minimum value greater than zero, that can be added to v and the result be represented by float
			static float eps_greater(float v)noexcept {
				return ::std::nextafter(v, ::std::numeric_limits<float>::infinity()) - v;
			}
			static float eps_greater_n(float v, float n)noexcept {
				return n*eps_greater(v);
			}

			//returns minimum value greater than zero, that can be subtracted from v and the result be represented by float
			static float eps_lower(float v)noexcept {
				return v - ::std::nextafter(v, -::std::numeric_limits<float>::infinity());
			}
			static float eps_lower_n(float v, float n)noexcept {
				return n*eps_lower(v);
			}

			typedef ::std::uint32_t similar_FWI_t;
			static_assert(sizeof(float) == sizeof(similar_FWI_t), "Wrong type sizes!");
			static_assert(::std::is_unsigned<similar_FWI_t>::value, "");

			static constexpr float _rone = float(1.);
			static constexpr float _rposz = float(+0.);
			static constexpr float _rnegz = float(-0.);

			static constexpr float _rNaN = ::std::numeric_limits<float>::quiet_NaN();
		};

		//////////////////////////////////////////////////////////////////////////
		//#TODO WARNING!!! ���� �� �����, ������������� ����������, ���������� ��������������� ������ ���� � ������
		//�������� ������������� ���������� �� ��������� � ��, ��� ��� ��� �������� - ������ �������� �����������,
		// ������� ����� � ����� ������ ���������.
		// ��������� ������ ����� memcpy
		//////////////////////////////////////////////////////////////////////////
		template<typename RealT>
		static inline typename real_t_limits<RealT>::similar_FWI_t similar_FWI_one() noexcept {
			typename real_t_limits<RealT>::similar_FWI_t r;
			memcpy(&r, &real_t_limits<RealT>::_rone, sizeof(typename real_t_limits<RealT>::similar_FWI_t));
			return r;
			//return *reinterpret_cast<const typename real_t_limits<RealT>::similar_FWI_t*>(&real_t_limits<RealT>::_rone);
		}
		template<typename RealT>
		static inline typename real_t_limits<RealT>::similar_FWI_t similar_FWI_pos_zero() noexcept {
			typename real_t_limits<RealT>::similar_FWI_t r;
			memcpy(&r, &real_t_limits<RealT>::_rposz, sizeof(typename real_t_limits<RealT>::similar_FWI_t));
			return r;
			//return *reinterpret_cast<const typename real_t_limits<RealT>::similar_FWI_t*>(&real_t_limits<RealT>::_rposz);
		}
		template<typename RealT>
		static inline typename real_t_limits<RealT>::similar_FWI_t similar_FWI_neg_zero() noexcept {
			typename real_t_limits<RealT>::similar_FWI_t r;
			memcpy(&r, &real_t_limits<RealT>::_rnegz, sizeof(typename real_t_limits<RealT>::similar_FWI_t));
			return r;
			//return *reinterpret_cast<const typename real_t_limits<RealT>::similar_FWI_t*>(&real_t_limits<RealT>::_rnegz);
		}

		//////////////////////////////////////////////////////////////////////////
		// to return back to proper real_t from similar_FWI_t
		template <typename T> struct int_t_limits {};

		template<> struct int_t_limits<::std::int32_t> {
			typedef float similar_FP_t;
		};
		template<> struct int_t_limits<::std::uint32_t> {
			typedef float similar_FP_t;
		};

		template<> struct int_t_limits<::std::int64_t> {
			typedef double similar_FP_t;
		};
		template<> struct int_t_limits<::std::uint64_t> {
			typedef double similar_FP_t;
		};

	}
}
