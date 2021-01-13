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

#include "../_defs.h"
#include "../common.h"

// loss_addendums are small classes that implements some kind of (surprise!) addendum to a loss function.
// Typically it's kind of a penalty like the weight-decay.
// 

namespace nntl {
namespace loss_addendum {
	
	//////////////////////////////////////////////////////////////////////////
	// Each _i_loss_addendum derived class must also be:
	// - default constructible (it must be instantiate-able as other class's member)
	// - inactive by default
	template<typename RealT>
	class _i_loss_addendum : public math::smatrix_td {
	public:
		typedef RealT real_t;

		typedef math::smatrix<real_t> realmtx_t;
		typedef math::smatrix_deform<real_t> realmtxdef_t;
		//typedef init_struct init_struct_t;

		//static constexpr const char _defName[] = "_LA_name_not_set";
		nntl_interface const char* getName()const noexcept;// { return "_LA_name_not_set"; }

		//computes loss addendum for a given matrix of values (for weight-decay Vals parameter is a weight matrix)
		template <typename CommonDataT>
		nntl_interface real_t lossAdd(const realmtx_t& Vals, const CommonDataT& CD) noexcept;

		//must provide a static constexpr bool calcOnFprop 
		//static constexpr bool calcOnFprop = false/true;
		template <typename CommonDataT, bool c = calcOnFprop>
		nntl_interface ::std::enable_if_t<c> on_fprop(const realmtx_t& Vals, const CommonDataT& CD) noexcept;

		template <typename CommonDataT>
		nntl_interface void dLossAdd(const realmtx_t& Vals, realmtx_t& dLossdVals, const CommonDataT& CD) noexcept;

		//#todo: computation of some loss functions may be optimized, if some data is cached between corresponding calls to lossAdd/dLossAdd.
		//However it's really useful only for a fullbatch learning with a full error calculation, which is a quite rare thing.
		// (usually, there's no special need to compute a full error (take loss addendums into account); it's usually enough
		// to compute error from the basic loss function (quadratic or cross-entropy) only).
		// Therefore until I'd really need it
		// I don't want to bother about caching and a huge set of accompanying cache-freshness related conditions...

		nntl_interface const bool bEnabled()const noexcept;

		// Performs initialization.
		//At mininum, it must call iMath.preinit()
		//override in derived class to suit needs
		template <typename CommonDataT>
		static bool init(const mtx_size_t biggestMtx, const CommonDataT& CD) noexcept {
			CD.iMath().preinit( realmtx_t::sNumel(biggestMtx) );
			return true;
		}
		static void deinit()noexcept{}
	};

	template<typename LaT>
	struct is_loss_addendum_impl : public ::std::is_base_of<_i_loss_addendum<typename LaT::real_t>, LaT>{};
	
	template<typename AnyT>
	struct is_loss_addendum : public ::std::conditional_t<has_real_t<AnyT>::value
		, is_loss_addendum_impl<AnyT>
		, ::std::false_type>
	{};

	template<typename LaT>
	using works_on_fprop = ::std::integral_constant<bool, LaT::calcOnFprop>;

	template<typename LaT>
	using depends_on_many = ::std::integral_constant<bool, LaT::dependsOnManyElements>;

	template<typename LaT>
	using depends_on_many_and_on_fprop = ::std::integral_constant<bool, LaT::dependsOnManyElements && LaT::calcOnFprop>;


	template<typename LaT>
	struct comparatorTpl {
		template<typename LaT2>
		using cmp_tpl = ::std::is_same<LaT, LaT2>;
	};

// 	enum {
// 		la_CalcOnBprop = 0, //- calculate loss derivatives using values in bprop stage (prone to dropout'ed neurons if LA used on activations)
// 		la_CalcOnFprop = 1, //- same, but for fprop stage (occurs before dropout, but requires additional memory to store gradient)
// 		la_MASK_onFpropType = 1,
// 
// 		la_appendToAnyGradient = 0, //append loss addendum ignoring current gradient value
// 		la_appendToNonZeroGradient = 2, //append loss addendum only if current gradient value is non zero
// 		la_MASK_appendGradientType = 2
// 	};
// 
// 	static inline constexpr bool is_la_mode(const unsigned int val, const unsigned int mode, const unsigned int mask)noexcept {
// 		return mode==(val & mask);
// 	}

	namespace _impl {
		//if bCalcOnFProp set to true, then it computes the necessary derivate during fprop step and stores it internally
		//until bprop() phase. Be sure to undertand the whole computation sequence when change this parameter (especially,
		// pay attention to the dropout and LPHG(usage of mask from drop_samples() call is not supported with bCalcOnFprop==true at this moment!))
		// bDependsOnManyElements flag to define if the loss is calculated elementwise (such as L2 or L1), or it 
		//		requires many elements (such as DeCov). I.e., whether the dLossAddendum/dA depends on a single element or many elements.
		template<typename RealT, bool bCalcOnFprop, bool bAppendToNZGrad, bool bDependsOnManyElements>
		class scaled_addendum : public _i_loss_addendum<RealT> {
		public:
			static constexpr bool calcOnFprop = bCalcOnFprop;
			static constexpr bool appendToNZGrad = bAppendToNZGrad;
			static constexpr bool dependsOnManyElements = bDependsOnManyElements;

		protected:
			real_t m_scale;

		public:
			scaled_addendum()noexcept : m_scale(real_t(0.)) {}

			void scale(const real_t s)noexcept {
				//NNTL_ASSERT(s >= real_t(0.));
				m_scale = s;
			}
			real_t scale()const noexcept { return m_scale; }

			const bool bEnabled()const noexcept { return real_t(0.) != m_scale; }

		protected:
			template<typename iMathT, bool c = appendToNZGrad>
			::std::enable_if_t<c> _appendGradient(iMathT& iM, realmtx_t& dLossdVals, const realmtx_t& newGrad)const noexcept {
				iM.evNZAddScaled_ip(dLossdVals, m_scale, newGrad);
			}
			template<typename iMathT, bool c = appendToNZGrad>
			::std::enable_if_t<!c> _appendGradient(iMathT& iM, realmtx_t& dLossdVals, const realmtx_t& newGrad)const noexcept {
				iM.evAddScaled_ip(dLossdVals, m_scale, newGrad);
			}
		};

		template<typename RealT, bool bCalcOnFprop, bool bAppendToNZGrad, bool bDependsOnManyElements>
		struct scaled_addendum_with_mtx4fprop : public scaled_addendum<RealT, bCalcOnFprop, bAppendToNZGrad, bDependsOnManyElements> {};

		template<typename RealT, bool bAppendToNZGrad, bool bDependsOnManyElements>
		struct scaled_addendum_with_mtx4fprop<RealT, true, bAppendToNZGrad, bDependsOnManyElements>
			: public scaled_addendum<RealT, true, bAppendToNZGrad, bDependsOnManyElements>
		{
		private:
			typedef scaled_addendum<RealT, true, bAppendToNZGrad, bDependsOnManyElements> _base_class_t;
		protected:
			math::smatrix_deform<RealT> m_Mtx;

		public:
			template <typename CommonDataT>
			bool init(const mtx_size_t biggestMtx, const CommonDataT& CD) noexcept {
				if (!_base_class_t::init(biggestMtx, CD))return false;
				return m_Mtx.resize(biggestMtx);
			}

			void deinit()noexcept {
				_base_class_t::deinit();
				m_Mtx.clear();
			}
		};
	}

}
}