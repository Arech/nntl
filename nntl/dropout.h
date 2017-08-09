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

//This file defines wrappers around different dropout implementations
//Layer classes will be derived from this classes using public inheritance, so code wisely

#include "dropout/dropout.h"
#include "dropout/alpha_dropout.h"

namespace nntl {

	//namespace _impl {
	//helper to instantiate AlphaDropout class with correct template parameters when applicable
	template< class, class = ::std::void_t<> >
	struct AlphaDropoutT_proxy { };

	template< class _AF >
	struct AlphaDropoutT_proxy<_AF, ::std::void_t<decltype(_AF::_TP_alpha), decltype(_AF::_TP_lambda), decltype(_AF::_TP_fpMean)
		, decltype(_AF::_TP_fpVar), decltype(_AF::_TP_corrType)>>
		: AlphaDropout<typename _AF::real_t, _AF::_TP_alpha, _AF::_TP_lambda, _AF::_TP_fpMean, _AF::_TP_fpVar, _AF::_TP_corrType>
	{};

	//}


	template<typename ActFT>
	using default_dropout_for = ::std::conditional_t <
		activation::is_type_of<activation::type_selu, ActFT>::value
		, AlphaDropoutT_proxy<ActFT>
		, Dropout<typename ActFT::real_t>
	>;

}