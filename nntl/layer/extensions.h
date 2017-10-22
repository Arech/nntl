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

#include "extender.h"

#include "pack_horizontal.h"

namespace nntl {

	//////////////////////////////////////////////////////////////////////////
	//template to extend an LFC with penalized activations

	template <typename ActivFunc, typename GradWorks, typename LossAddsTupleTOrVoid, typename DropoutT = NoDropout<typename ActivFunc::real_t>>
	struct _LFC_PA_DO {
		template <typename FC>
		using _tpl_LFC = _LFC<FC, ActivFunc, GradWorks>;

		typedef LPAt_DO<_tpl_LFC, DropoutT, LossAddsTupleTOrVoid> type;
	};

	template <typename LossAddsTupleT
		, typename ActivFunc = activation::sigm<d_interfaces::real_t>
		, typename GradWorks = grad_works<d_interfaces>
	> using LFC_PA = typename _LFC_PA_DO<ActivFunc, GradWorks, LossAddsTupleT>::type;

	template <typename ActivFunc = activation::sigm<d_interfaces::real_t>
		,typename GradWorks = grad_works<d_interfaces>, typename DropoutT = default_dropout_for<ActivFunc>
	> using LFC_DO = typename _LFC_PA_DO<ActivFunc, GradWorks, void, DropoutT>::type;


	template <typename LossAddsTupleT, typename ActivFunc = activation::sigm<d_interfaces::real_t>
		, typename GradWorks = grad_works<d_interfaces>, typename DropoutT = default_dropout_for<ActivFunc>
	> using LFC_PA_DO = typename _LFC_PA_DO<ActivFunc, GradWorks, LossAddsTupleT, DropoutT>::type;

	//////////////////////////////////////////////////////////////////////////
	//template to extend an LPH with penalized activations

	template <typename LossAddsTupleT, typename DropoutT, typename PHLsTupleT>
	struct _LPH_PA_DO {
		template <typename FC>
		using _tpl_LPH = _LPH<FC, PHLsTupleT>;

		typedef LPAt_DO<_tpl_LPH, DropoutT, LossAddsTupleT> type;
	};

	template <typename LossAddsTupleT, typename ...PHLsT>
	using LPH_PA = typename _LPH_PA_DO<LossAddsTupleT, void, ::std::tuple<PHLsT...>>::type;

	template <typename LossAddsTupleT,  typename PHLsTupleT>
	using LPHt_PA = typename _LPH_PA_DO<LossAddsTupleT, void, PHLsTupleT>::type;

	template <typename DropoutT, typename ...PHLsT>
	using LPH_DO = typename _LPH_PA_DO<void, DropoutT, ::std::tuple<PHLsT...>>::type;

	template <typename DropoutT, typename PHLsTupleT>
	using LPHt_DO = typename _LPH_PA_DO<void, DropoutT, PHLsTupleT>::type;

	template <typename LossAddsTupleT, typename DropoutT, typename PHLsTupleT>
	using LPHt_PA_DO = typename _LPH_PA_DO<LossAddsTupleT, DropoutT, PHLsTupleT>::type;

	//////////////////////////////////////////////////////////////////////////
	//template to extend an LPHG with penalized activations
	/*
	template <typename LossAddsTupleT, typename DropoutT, int iBinarize1e6, bool bDoBinarizeGate, typename PHLsTupleT>
	struct _LPHG_PA_DO {
		template <typename FC>
		using _tpl_LPHG = _LPHG<FC, iBinarize1e6, bDoBinarizeGate, PHLsTupleT>;

		typedef LPAt_DO<_tpl_LPHG, DropoutT, LossAddsTupleT> type;
	};

	template <typename LossAddsTupleT, int iBinarize1e6, typename ...PHLsT>
	using LPHG_PA = typename _LPHG_PA_DO<LossAddsTupleT, void, iBinarize1e6, true, ::std::tuple<PHLsT...>>::type;

	template <typename LossAddsTupleT, int iBinarize1e6, typename PHLsTupleT>
	using LPHGt_PA = typename _LPHG_PA_DO<LossAddsTupleT, void, iBinarize1e6, true, PHLsTupleT>::type;

	template <typename LossAddsTupleT, typename ...PHLsT>
	using LPHGFI_PA = typename _LPHG_PA_DO<LossAddsTupleT,void, 0, false, ::std::tuple<PHLsT...>>::type;

	template <typename LossAddsTupleT, typename PHLsTupleT>
	using LPHGFIt_PA = typename _LPHG_PA_DO<LossAddsTupleT,void, 0, false, PHLsTupleT>::type;

	template <typename DropoutT, typename ...PHLsT>
	using LPHGFI_DO = typename _LPHG_PA_DO<void, DropoutT, 0, false, ::std::tuple<PHLsT...>>::type;
	
	template <typename LossAddsTupleT, typename DropoutT, typename ...PHLsT>
	using LPHGFI_PA_DO = typename _LPHG_PA_DO<LossAddsTupleT, DropoutT, 0, false, ::std::tuple<PHLsT...>>::type;

	template <typename LossAddsTupleT, typename DropoutT, typename PHLsTupleT>
	using LPHGFIt_PA_DO = typename _LPHG_PA_DO<LossAddsTupleT, DropoutT, 0, false, PHLsTupleT>::type;
	*/

}
