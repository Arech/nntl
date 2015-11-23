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

//This code has been taken from boost/algorithm/clamp.hpp
//I've just copy-pasted and simplified it here to make NNTL work without requiring boost

/*
Copyright (c) Marshall Clow 2008-2012.

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Revision history:
27 June 2009 mtc First version
23 Oct  2010 mtc Added predicate version

*/

namespace nntl {
namespace utils {
namespace boost {
namespace algorithm {

	/// \fn clamp ( T const& val, 
	///               typename boost::mpl::identity<T>::type const & lo, 
	///               typename boost::mpl::identity<T>::type const & hi, Pred p )
	/// \return the value "val" brought into the range [ lo, hi ]
	///     using the comparison predicate p.
	///     If p ( val, lo ) return lo.
	///     If p ( hi, val ) return hi.
	///     Otherwise, return the original value.
	/// 
	/// \param val   The value to be clamped
	/// \param lo    The lower bound of the range to be clamped to
	/// \param hi    The upper bound of the range to be clamped to
	/// \param p     A predicate to use to compare the values.
	///                 p ( a, b ) returns a boolean.
	///
	template<typename T, typename Pred>
	T const & clamp(T const& val,
		T const & lo,
		T const & hi, Pred p)
	{
		return p(val, lo) ? lo : p(hi, val) ? hi : val;
	}


	/// \fn clamp ( T const& val, 
	///               typename boost::mpl::identity<T>::type const & lo, 
	///               typename boost::mpl::identity<T>::type const & hi )
	/// \return the value "val" brought into the range [ lo, hi ].
	///     If the value is less than lo, return lo.
	///     If the value is greater than "hi", return hi.
	///     Otherwise, return the original value.
	///
	/// \param val   The value to be clamped
	/// \param lo    The lower bound of the range to be clamped to
	/// \param hi    The upper bound of the range to be clamped to
	///
	template<typename T>
	T const& clamp(const T& val,
		T const & lo,
		T const & hi)
	{
		return (clamp)(val, lo, hi, std::less<T>());
	}

	/// \fn clamp_range ( InputIterator first, InputIterator last, OutputIterator out, 
	///       std::iterator_traits<InputIterator>::value_type const & lo, 
	///       std::iterator_traits<InputIterator>::value_type const & hi )
	/// \return clamp the sequence of values [first, last) into [ lo, hi ]
	/// 
	/// \param first The start of the range of values
	/// \param last  One past the end of the range of input values
	/// \param out   An output iterator to write the clamped values into
	/// \param lo    The lower bound of the range to be clamped to
	/// \param hi    The upper bound of the range to be clamped to
	///
	template<typename InputIterator, typename OutputIterator>
	OutputIterator clamp_range(InputIterator first, InputIterator last, OutputIterator out,
		typename std::iterator_traits<InputIterator>::value_type const & lo,
		typename std::iterator_traits<InputIterator>::value_type const & hi)
	{
		// this could also be written with bind and std::transform
		while (first != last)
			*out++ = clamp(*first++, lo, hi);
		return out;
	}

	/// \fn clamp_range ( InputIterator first, InputIterator last, OutputIterator out, 
	///       std::iterator_traits<InputIterator>::value_type const & lo, 
	///       std::iterator_traits<InputIterator>::value_type const & hi, Pred p )
	/// \return clamp the sequence of values [first, last) into [ lo, hi ]
	///     using the comparison predicate p.
	/// 
	/// \param first The start of the range of values
	/// \param last  One past the end of the range of input values
	/// \param out   An output iterator to write the clamped values into
	/// \param lo    The lower bound of the range to be clamped to
	/// \param hi    The upper bound of the range to be clamped to
	/// \param p     A predicate to use to compare the values.
	///                 p ( a, b ) returns a boolean.

	///
	template<typename InputIterator, typename OutputIterator, typename Pred>
	OutputIterator clamp_range(InputIterator first, InputIterator last, OutputIterator out,
		typename std::iterator_traits<InputIterator>::value_type const & lo,
		typename std::iterator_traits<InputIterator>::value_type const & hi, Pred p)
	{
		// this could also be written with bind and std::transform
		while (first != last)
			*out++ = clamp(*first++, lo, hi, p);
		return out;
	}
}
}
}
}