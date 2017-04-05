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

//we need boost::mpl to handle variadic parameter pack of mixins
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/size_t.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>
//#include <boost/mpl/transform.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/arithmetic.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/push_back.hpp>
//#include <boost/mpl/pop_front.hpp>


// NEVER!!! call get_opt()/set_opt() in mixin constructors (before the root (which instantiates and constucts m_opts) constructor has been run)
#define NNTL_METHODS_MIXIN_OPTIONS(mixinIdx) \
static_assert(mixinIdx > 0, "MixinIdx must be positive! Zero is reserved for a root class"); \
private: \
const bool get_opt(const size_t& oIdx)const { return get_self().m_opts.get<mixinIdx>(oIdx); 	} \
auto& set_opt(const size_t& oIdx, const bool& b) { get_self().m_opts.set<mixinIdx>(oIdx, b); return *this; }

//set_opt() (for mixins and for the root) MUST return *this, and not a get_self()!

#define NNTL_METHODS_MIXIN_ROOT_OPTIONS() protected: \
const bool get_opt(const size_t& oIdx)const { return get_self().m_opts.get<0>(oIdx); } \
auto& set_opt(const size_t& oIdx, const bool& b) { get_self().m_opts.set<0>(oIdx, b); return *this; } \
private:

//////////////////////////////////////////////////////////////////////////

namespace nntl {
namespace utils {

namespace mixins {
	// we need a parametrized mixin classes that share a storage for some common data (we going to handle that data by the 
	// root class)

	//namespace to have the data types related to indexed mixins (i.e. that is also parametrized by their index in mixins stack).
	//Mixin must be indexed in order to address its range of storage space within a root class.
	// I.e. for our purpose it must have the following signature:
	// template<typename FinalT, typename CfgT, std::size_t MixinIdx> class Mixin;
	namespace indexed {

		template <template<typename, typename, std::size_t> class T1, template<typename, typename, std::size_t> class T2>
		struct is_same : public std::false_type {};
		template<template<typename, typename, std::size_t> class T>
		struct is_same<T, T> : public std::true_type {};

		//////////////////////////////////////////////////////////////////////////
		// machinery necessary for passing a mixin index to the mixin during a root class declaration phase
		template <template<typename, typename, std::size_t> typename CheckM, std::size_t N, std::size_t I
			, template<typename, typename, std::size_t> typename... MList>
		struct ref_index;

		template <template<typename, typename, std::size_t> typename CheckM, std::size_t N, std::size_t I
			, template<typename, typename, std::size_t> typename Head, template<typename, typename, std::size_t> typename... Tail>
		struct mixin_ref_index_imp : std::conditional<
			is_same<CheckM, Head>::value
			, std::integral_constant<std::size_t, N - I + 1>
			, ref_index<CheckM, N, I - 1, Tail...>
		>::type {};

		template <template<typename, typename, std::size_t> typename CheckM, std::size_t N, template<typename, typename, std::size_t> typename... Head>
		struct mixin_ref_index_imp0 : std::integral_constant<std::size_t, N> {};

		//based on http://stackoverflow.com/questions/33999868/stdtuple-get-item-by-inherited-type
		template <template<typename, typename, std::size_t> typename CheckM, std::size_t N, std::size_t I
			, template<typename, typename, std::size_t> typename... MList>
		struct ref_index : std::conditional< 1 == I
			, mixin_ref_index_imp0<CheckM, N, MList...>
			, mixin_ref_index_imp<CheckM, N, I, MList...>
		>::type {};

		//////////////////////////////////////////////////////////////////////////
		// support for a root class
			
		//creating vector of types of mixins
		template <class FinalT, class CfgT, template<typename, typename, std::size_t> typename... MList>
		using make_mixin_vec = boost::mpl::vector<
			MList< FinalT, CfgT, ref_index<MList, sizeof...(MList), sizeof...(MList), MList...>::value >...
		>;
	}
		

	// If a mixin class provides a plain enum that quantifies options that is stored by the root class, the enum must start from 0
	// up to opts_total (which is a placeholder for the total options count)
	template <class M>
	struct options_count : std::integral_constant<std::size_t, M::opts_total> {};

	struct lf_options_count2seq {
		template <class St, class M>
		struct apply {
			typedef typename boost::mpl::push_back<St, boost::mpl::size_t< options_count<M>::value > >::type type;
		};
	};

	template<typename Mixin_Seq, std::size_t RootTotalOpts>
	using make_mixin_options_count_vec_c = typename boost::mpl::fold<
		Mixin_Seq
		, boost::mpl::vector_c<std::size_t, RootTotalOpts>
		, lf_options_count2seq
	>::type;


	template<typename Seq>
	using get_cumsum = typename boost::mpl::fold<
		Seq
		, boost::mpl::size_t<0>
		, boost::mpl::lambda< boost::mpl::plus< boost::mpl::_1, boost::mpl::_2 > >::type
	>::type;


	struct lf_cumsum_seq {
		template <class St, class C>
		struct apply {
			typedef typename boost::mpl::back<St>::type prevValT;
			typedef typename boost::mpl::push_back<St, boost::mpl::size_t< prevValT::value + C::value > >::type type;
		};
	};

	template<typename Seq>
	using make_cumsum_vec_c = typename boost::mpl::fold<
		Seq
		, boost::mpl::vector_c<std::size_t, 0>
		, lf_cumsum_seq
	>::type;


	template<typename Opts_Ofs, std::size_t N>
	class binary_options_storage : protected std::bitset<N>{
	protected:
		typedef std::bitset<N> base_class_t;

	public:
		template<size_t mixinIdx>
		const bool get(const size_t& oIdx)const {//we could do this func constexpr, but there's no point of that
			return (*this)[oIdx + boost::mpl::at_c<Opts_Ofs, mixinIdx>::type::value];
		}

		template<size_t mixinIdx>
		void set(const size_t& oIdx, const bool b) {
			(*this)[oIdx + boost::mpl::at_c<Opts_Ofs, mixinIdx>::type::value] = b;
		}

		//Serialization support
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & *(static_cast<base_class_t*>(this));
		}
	};

}
}
}