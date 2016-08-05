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

//this header contains necessary includes of boost::serialization headers and additional
//definitions of types that are helpful in nntl objects serialization.

#include <type_traits>

#define BOOST_SERIALIZATION_NO_LIB

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/string.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>

#include "options.h"

namespace nntl {
namespace serialization {

	//////////////////////////////////////////////////////////////////////////
	// just an alias for nvp class
	template<typename T>
	using nvp = ::boost::serialization::nvp<T>;

	template<class T>
	inline nvp< T > make_nvp(const char * name, T & t) {
		return nvp< T >(name, t);
	}
#define NNTL_SERIALIZATION_NVP(v) ::nntl::serialization::make_nvp( NNTL_STRINGIZE(v), v )

	//////////////////////////////////////////////////////////////////////////
	// special helper type to structurize data in archives
	template<typename T>
	struct named_struct : public nvp<T> {
		~named_struct(){};
		named_struct(const named_struct & rhs) : nvp(static_cast<const nvp&>(rhs)) {}
		named_struct(const nvp & rhs) : nvp(rhs) {}
		explicit named_struct(const char * name_, T & t) : nvp(name_, t) {};
	};

	template<class T>
	inline named_struct< T > make_named_struct(const char * name, T & t) {
		return named_struct< T >(name, t);
	}
#define NNTL_SERIALIZATION_STRUCT(v) ::nntl::serialization::make_named_struct( NNTL_STRINGIZE(v), v )

	//////////////////////////////////////////////////////////////////////////
	// alias to call base's class serialize()
// 	template<class Base, class Derived>
// 	inline typename ::boost::serialization::detail::base_cast<Base, Derived>::type & serialize_base_class(Derived &d) {
// 		return ::boost::serialization::base_object(d);
// 	}

	//using serialize_base_class = ::boost::serialization::base_object<Base, Derived>(Derived &d);
	//using serialize_base_class = ::boost::serialization::template base_object<Base, Derived>; //??

	//////////////////////////////////////////////////////////////////////////
	// Saving Archive concept base class
	template<typename FinalChildT, bool bSavingArchive>
	class simple_archive {
	public:
		typedef FinalChildT self_t;
		typedef self_t& self_ref_t;

		self_ref_t get_self()noexcept { return static_cast<self_ref_t>(*this); }
		const self_ref_t get_self()const noexcept { return static_cast<const self_ref_t>(*this); }

		~simple_archive()noexcept {}
		simple_archive() noexcept {}
		
	public:
		///////////////////////////////////////////////////
		// Implement requirements for archive concept

		typedef ::boost::mpl::bool_<!bSavingArchive> is_loading;
		typedef ::boost::mpl::bool_<bSavingArchive> is_saving;

		// this can be a no-op since we ignore pointer polymorphism
		template<class T> void register_type(const T * = NULL) {}
		nntl_interface unsigned int get_library_version() { return 0; }
		nntl_interface void save_binary(const void *address, std::size_t count) { static_assert(!"save_binary"); }
		nntl_interface void load_binary(void *address, std::size_t count) { static_assert(!"load_binary"); }

		template<class T>
		self_ref_t operator<<(T const & t) {
			boost::serialization::serialize_adl(get_self(), const_cast<T &>(t), ::boost::serialization::version< T >::value);
			return get_self();
		}
		template<class T>
		self_ref_t operator>>(T& t) {
			boost::serialization::serialize_adl(get_self(), t, ::boost::serialization::version< T >::value);
			return get_self();
		}

		// the & operator 
		template<class T, bool b = bSavingArchive>
		std::enable_if_t<b, self_ref_t> operator&(const T & t) {
			return get_self() << t;
		}
		template<class T, bool b = bSavingArchive>
		std::enable_if_t<!b, self_ref_t> operator&(T & t) {
			return get_self() >> t;
		}
	};
}
}
