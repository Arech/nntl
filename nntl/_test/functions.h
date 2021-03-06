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

//#TODO: there's no reason to make this file a part of NNTL. Move it to the tests project. Actually, it applies to the whole containing folder

namespace nntl {
//namespace tests {

	template<bool bActNorm = false, typename FET, typename FST, typename FMT, typename FB>
	void test_dLdZ_corr(FET&& fet, FST&& fst, FMT&& fmt, FB&& fb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, descr);
		constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

		//no biases here by intent, because dLdZ works with output layer
		realmtx_t A(rowsCnt, colsCnt), ASrc(rowsCnt, colsCnt), dLdZ_ET(rowsCnt, colsCnt), Y(rowsCnt, colsCnt);
		ASSERT_TRUE(!A.isAllocationFailed() && !ASrc.isAllocationFailed() && !dLdZ_ET.isAllocationFailed() && !Y.isAllocationFailed());

		iM.preinit(A.numel());
		ASSERT_TRUE(iM.init());
		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			if (bActNorm) {
				rg.gen_matrix_norm(ASrc);
				rg.gen_matrix_norm(Y);
			} else {
				rg.gen_matrix(ASrc, real_t(5.));
				rg.gen_matrix(Y, real_t(5.));
			}

			ASrc.clone_to(dLdZ_ET);
			//(::std::forward<FET>(fet))(Y, dLdZ_ET);
			fet(Y, dLdZ_ET);

			ASrc.clone_to(A);
			//(::std::forward<FST>(fst))(Y, A);
			fst(Y, A);
			ASSERT_MTX_EQ(dLdZ_ET, A, "_st");

			ASrc.clone_to(A);
			//(::std::forward<FMT>(fmt))(Y, A);
			fmt(Y, A);
			ASSERT_MTX_EQ(dLdZ_ET, A, "_mt");

			ASrc.clone_to(A);
			//(::std::forward<FB>(fb))(Y, A);
			fb(Y, A);
			ASSERT_MTX_EQ(dLdZ_ET, A, "()");
		}
	}

	template<bool XHasBiases, bool bXNorm = false, typename FET, typename FST, typename FMT, typename FB>
	void test_f_x_corr(FET&& fet, FST&& fst, FMT&& fmt, FB&& fb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, descr);
		constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

		//no biases here by intent, because dLdZ works with output layer
		realmtx_t X(rowsCnt, colsCnt, XHasBiases), XSrc(rowsCnt, colsCnt, XHasBiases), X_ET(rowsCnt, colsCnt, XHasBiases);
		ASSERT_TRUE(!X.isAllocationFailed() && !XSrc.isAllocationFailed() && !X_ET.isAllocationFailed());

		iM.preinit(X.numel());
		ASSERT_TRUE(iM.init());
		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			if (bXNorm) {
				if (XHasBiases) {
					rg.gen_matrix_no_bias_norm(XSrc);
				} else rg.gen_matrix_norm(XSrc);
			} else {
				if (XHasBiases) {
					rg.gen_matrix_no_bias(XSrc, real_t(5.));
				} else rg.gen_matrix(XSrc, real_t(5.));
			}
			ASSERT_TRUE(!XHasBiases || XSrc.test_biases_strict());

			XSrc.clone_to(X_ET);
			//(::std::forward<FET>(fet))(X_ET);
			fet(X_ET);
			ASSERT_TRUE(!XHasBiases || X_ET.test_biases_strict());

			XSrc.clone_to(X);
			//(::std::forward<FST>(fst))(X);
			fst(X);
			ASSERT_TRUE(!XHasBiases || X.test_biases_strict());
			ASSERT_MTX_EQ(X_ET, X, "_st");

			XSrc.clone_to(X);
			//(::std::forward<FMT>(fmt))(X);
			fmt(X);
			ASSERT_TRUE(!XHasBiases || X.test_biases_strict());
			ASSERT_MTX_EQ(X_ET, X, "_mt");

			XSrc.clone_to(X);
			//(::std::forward<FB>(fb))(X);
			fb(X);
			ASSERT_TRUE(!XHasBiases || X.test_biases_strict());
			ASSERT_MTX_EQ(X_ET, X, "()");
		}
	}

	template<typename EpsT,bool XHasBiases, bool bXNorm = false, typename FET, typename FST, typename FMT, typename FB>
	void test_f_x_corr_eps(FET&& fet, FST&& fst, FMT&& fmt, FB&& fb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, descr);
		constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

		//no biases here by intent, because dLdZ works with output layer
		realmtx_t X(rowsCnt, colsCnt, XHasBiases), XSrc(rowsCnt, colsCnt, XHasBiases), X_ET(rowsCnt, colsCnt, XHasBiases);
		ASSERT_TRUE(!X.isAllocationFailed() && !XSrc.isAllocationFailed() && !X_ET.isAllocationFailed());

		iM.preinit(X.numel());
		ASSERT_TRUE(iM.init());
		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			if (bXNorm) {
				if (XHasBiases) {
					rg.gen_matrix_no_bias_norm(XSrc);
				} else rg.gen_matrix_norm(XSrc);
			} else {
				if (XHasBiases) {
					rg.gen_matrix_no_bias(XSrc, real_t(5.));
				} else rg.gen_matrix(XSrc, real_t(5.));
			}
			ASSERT_TRUE(!XHasBiases || XSrc.test_biases_strict());

			XSrc.clone_to(X_ET);
			//(::std::forward<FET>(fet))(X_ET);
			fet(X_ET);
			ASSERT_TRUE(!XHasBiases || X_ET.test_biases_strict());

			XSrc.clone_to(X);
			//(::std::forward<FST>(fst))(X);
			fst(X);
			ASSERT_TRUE(!XHasBiases || X.test_biases_strict());
			ASSERT_REALMTX_NEAR(X_ET, X, "_st", EpsT::eps);

			XSrc.clone_to(X);
			//(::std::forward<FMT>(fmt))(X);
			fmt(X);
			ASSERT_TRUE(!XHasBiases || X.test_biases_strict());
			ASSERT_REALMTX_NEAR(X_ET, X, "_mt", EpsT::eps);

			XSrc.clone_to(X);
			//(::std::forward<FB>(fb))(X);
			fb(X);
			ASSERT_TRUE(!XHasBiases || X.test_biases_strict());
			ASSERT_REALMTX_NEAR(X_ET, X, "()", EpsT::eps);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<bool XHasBiases, bool bXNorm = false, typename EPST, typename FET, typename FST, typename FMT, typename FB>
	void test_f_x_xbasedET_corr(FET&& fet, FST&& fst, FMT&& fmt, FB&& fb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, descr);
		constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

		//no biases here by intent, because dLdZ works with output layer
		realmtx_t X(rowsCnt, colsCnt, XHasBiases), F(rowsCnt, colsCnt, XHasBiases), F_ET(rowsCnt, colsCnt, XHasBiases);
		ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !F_ET.isAllocationFailed());

		iM.preinit(X.numel());
		ASSERT_TRUE(iM.init());
		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			if (bXNorm) {
				if (XHasBiases) {
					rg.gen_matrix_no_bias_norm(X);
				} else rg.gen_matrix_norm(X);
			} else {
				if (XHasBiases) {
					rg.gen_matrix_no_bias(X, real_t(5.));
				} else rg.gen_matrix(X, real_t(5.));
			}
			ASSERT_TRUE(!XHasBiases || F.test_biases_strict());

			//(::std::forward<FET>(fet))(X, F_ET);
			fet(X, F_ET);
			ASSERT_TRUE(!XHasBiases || F_ET.test_biases_strict());

			X.clone_to(F);
			//(::std::forward<FST>(fst))(F);
			fst(F);
			ASSERT_TRUE(!XHasBiases || F.test_biases_strict());
			ASSERT_REALMTX_NEAR(F_ET, F, "_st", EPST::eps);

			X.clone_to(F);
			//(::std::forward<FMT>(fmt))(F);
			fmt(F);
			ASSERT_TRUE(!XHasBiases || F.test_biases_strict());
			ASSERT_REALMTX_NEAR(F_ET, F, "_mt", EPST::eps);

			X.clone_to(F);
			//(::std::forward<FB>(fb))(F);
			fb(F);
			ASSERT_TRUE(!XHasBiases || F.test_biases_strict());
			ASSERT_REALMTX_NEAR(F_ET, F, "()", EPST::eps);
		}
	}

	template<typename EPST, typename FET, typename DFET, typename DFST, typename DFMT, typename DFB>
	void test_df_x_xbasedET_corr(FET&& fet, DFET&& dfet, DFST&& dfst, DFMT&& dfmt, DFB&& dfb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		MTXSIZE_SCOPED_TRACE(rowsCnt, colsCnt, descr);
		constexpr unsigned testCorrRepCnt = TEST_CORRECTN_REPEATS_COUNT;

		//no biases here by intent, because dLdZ works with output layer
		realmtx_t X(rowsCnt, colsCnt, true), F(rowsCnt, colsCnt, true), DF(rowsCnt, colsCnt, false)
			, df_ET(rowsCnt, colsCnt, false);
		ASSERT_TRUE(!X.isAllocationFailed() && !F.isAllocationFailed() && !DF.isAllocationFailed() && !df_ET.isAllocationFailed());

		iM.preinit(X.numel());
		ASSERT_TRUE(iM.init());
		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());

		for (unsigned r = 0; r < testCorrRepCnt; ++r) {
			rg.gen_matrix_no_bias(X, real_t(5.));
			ASSERT_TRUE(X.test_biases_strict());

			//(::std::forward<DFET>(dfet))(X, df_ET);
			dfet(X, df_ET);
			//(::std::forward<FET>(fet))(X, F);
			fet(X, F);
			ASSERT_TRUE(F.test_biases_strict());

			F.clone_to_no_bias(DF);
			//(::std::forward<DFST>(dfst))(DF);
			dfst(DF);
			ASSERT_REALMTX_NEAR(df_ET, DF, "_st", EPST::eps);

			F.clone_to_no_bias(DF);
			//(::std::forward<DFMT>(dfmt))(DF);
			dfmt(DF);
			ASSERT_REALMTX_NEAR(df_ET, DF, "_mt", EPST::eps);

			F.clone_to_no_bias(DF);
			//(::std::forward<DFB>(dfb))(DF);
			dfb(DF);
			ASSERT_REALMTX_NEAR(df_ET, DF, "()", EPST::eps);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////// 

	//if XValuesSpan1e3==0, then gen_matrix_norm() is used
	template<unsigned XValuesSpan1e3 = 10000, typename FST, typename FMT, typename FB>
	void test_f_x_perf(FST&& fst, FMT&& fmt, FB&& fb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
		STDCOUTL("**** testing " << descr << "() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

		constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
		realmtx_t X(rowsCnt, colsCnt);// , XSrc(rowsCnt, colsCnt);
		ASSERT_TRUE(!X.isAllocationFailed());

		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());
		utils::tictoc tSt, tMt, tB, tSt2, tMt2;
		real_t s = real_t(0);
		threads::prioritize_workers<threads::PriorityClass::PerfTesting, imath_basic_t::iThreads_t> pw(iM.ithreads());
		for (unsigned r = 0; r < maxReps; ++r) {
#pragma warning(disable:4127)
			if (XValuesSpan1e3) {
				rg.gen_matrix(X, real_t(XValuesSpan1e3) / real_t(1e3));
			} else rg.gen_matrix_norm(X);
#pragma warning(default:4127)
			tSt.tic();
			fst(X);
			tSt.toc();
			for (auto& e : X) s += e;			s = ::std::log10(::std::abs(s));

#pragma warning(disable:4127)
			if (XValuesSpan1e3) {
				rg.gen_matrix(X, real_t(XValuesSpan1e3) / real_t(1e3));
			} else rg.gen_matrix_norm(X);
#pragma warning(default:4127)
			tMt.tic();
			fmt(X);
			tMt.toc();
			for (auto& e : X) s += e;			s = ::std::log10(::std::abs(s));

#pragma warning(disable:4127)
			if (XValuesSpan1e3) {
				rg.gen_matrix(X, real_t(XValuesSpan1e3) / real_t(1e3));
			} else rg.gen_matrix_norm(X);
#pragma warning(default:4127)
			tSt2.tic();
			fst(X);
			tSt2.toc();
			for (auto& e : X) s += e;			s = ::std::log10(::std::abs(s));

#pragma warning(disable:4127)
			if (XValuesSpan1e3) {
				rg.gen_matrix(X, real_t(XValuesSpan1e3) / real_t(1e3));
			} else rg.gen_matrix_norm(X);
#pragma warning(default:4127)
			tMt2.tic();
			fmt(X);
			tMt2.toc();
			for (auto& e : X) s += e;			s = ::std::log10(::std::abs(s));

#pragma warning(disable:4127)
			if (XValuesSpan1e3) {
				rg.gen_matrix(X, real_t(XValuesSpan1e3) / real_t(1e3));
			} else rg.gen_matrix_norm(X);
#pragma warning(default:4127)
			tB.tic();
			fb(X);
			tB.toc();
			for (auto& e : X) s += e;			s = ::std::log10(::std::abs(s));
		}
		tSt.say("st");
		tSt2.say("st2");
		tMt.say("mt");
		tMt2.say("mt2");

		tB.say("best");
		STDCOUTL(s);
	}

	//////////////////////////////////////////////////////////////////////////

	template<bool bActNorm = false, typename FST, typename FMT, typename FB>
	void test_dLdZ_perf(FST&& fst, FMT&& fmt, FB&& fb, const char* descr, vec_len_t rowsCnt, vec_len_t colsCnt = 10) {
		const auto dataSize = realmtx_t::sNumel(rowsCnt, colsCnt);
		STDCOUTL("**** testing " << descr << "() over " << rowsCnt << "x" << colsCnt << " matrix (" << dataSize << " elements) ****");

		constexpr unsigned maxReps = TEST_PERF_REPEATS_COUNT;
		realmtx_t A(rowsCnt, colsCnt), Y(rowsCnt, colsCnt);
		ASSERT_TRUE(!A.isAllocationFailed() && !Y.isAllocationFailed());

		d_interfaces::iRng_t rg;
		rg.init_ithreads(iM.ithreads());
		utils::tictoc tSt, tMt, tB, tSt2, tMt2;
		real_t s = real_t(0);
		threads::prioritize_workers<threads::PriorityClass::PerfTesting, decltype(iM)::iThreads_t> pw(iM.ithreads());
		for (unsigned r = 0; r < maxReps; ++r) {
			if (bActNorm) {
				rg.gen_matrix_norm(Y);  rg.gen_matrix_norm(A);
			} else {
				rg.gen_matrix(Y, real_t(5.));  rg.gen_matrix(A, real_t(5.));
			}
			tSt.tic();
			fst(Y, A);
			tSt.toc();
			for (auto& e : A) s += e;			s = ::std::log10(::std::abs(s));

			if (bActNorm) {
				rg.gen_matrix_norm(Y);  rg.gen_matrix_norm(A);
			} else {
				rg.gen_matrix(Y, real_t(5.));  rg.gen_matrix(A, real_t(5.));
			}
			tMt.tic();
			fmt(Y, A);
			tMt.toc();
			for (auto& e : A) s += e;
			s = ::std::log10(::std::abs(s));

			if (bActNorm) {
				rg.gen_matrix_norm(Y);  rg.gen_matrix_norm(A);
			} else {
				rg.gen_matrix(Y, real_t(5.));  rg.gen_matrix(A, real_t(5.));
			}
			tSt2.tic();
			fst(Y, A);
			tSt2.toc();
			for (auto& e : A) s += e;
			s = ::std::log10(::std::abs(s));

			if (bActNorm) {
				rg.gen_matrix_norm(Y);  rg.gen_matrix_norm(A);
			} else {
				rg.gen_matrix(Y, real_t(5.));  rg.gen_matrix(A, real_t(5.));
			}
			tMt2.tic();
			fmt(Y, A);
			tMt2.toc();
			for (auto& e : A) s += e;
			s = ::std::log10(::std::abs(s));

			if (bActNorm) {
				rg.gen_matrix_norm(Y);  rg.gen_matrix_norm(A);
			} else {
				rg.gen_matrix(Y, real_t(5.));  rg.gen_matrix(A, real_t(5.));
			}
			tB.tic();
			fb(Y, A);
			tB.toc();
			for (auto& e : A) s += e;
			s = ::std::log10(::std::abs(s));
		}
		tSt.say("st");
		tSt2.say("st2");
		tMt.say("mt");
		tMt2.say("mt2");
		tB.say("best");
		STDCOUTL(s);
	}

//};
};