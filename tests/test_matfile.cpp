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
#include "stdafx.h"

#if NNTL_MATLAB_AVAILABLE

#include "../nntl/math.h"
#include "../nntl/common.h"

#include "../nntl/_supp/io/binfile.h"
#include "../nntl/_supp/io/matfile.h"

#include "../nntl/nntl.h"

#include "asserts.h"

using namespace nntl;
using namespace nntl_supp;

typedef d_interfaces::real_t real_t;

#define MNIST_FILE_DEBUG "../data/mnist200_100.bin"
#define MNIST_SMALL_FILE MNIST_FILE_DEBUG


template<typename T_>
bool get_td(train_data<T_>& td)noexcept {
	binfile reader;
	auto rec = reader.read("./test_data/td.bin", td);
	EXPECT_EQ(binfile::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	return binfile::ErrorCode::Success == rec;
}

TEST(TestMatfile, ReadWriteMat) {
	math::smatrix<float> float_mtx(3, 2), float_mtx2;
	for (unsigned i = 0; i < float_mtx.numel(); ++i) float_mtx.data()[i] = float(1.1) * i;
	math::smatrix<double> double_mtx(3, 2), double_mtx2;
	for (unsigned i = 0; i < double_mtx.numel(); ++i) double_mtx.data()[i] = double(2.1) * i;

	const double dET = 3.9;
	const float fET = 2.6f;

	train_data<real_t> td_ET,td;
	ASSERT_TRUE(get_td(td_ET));

	const char *const pFileName = "./test_data/test.mat";

	{
		SCOPED_TRACE("Testing omatfile");
		omatfile<> mf;

		ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName));
		
		mf << serialization::make_nvp("d", dET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

		mf << serialization::make_nvp("f", fET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

		mf << NNTL_SERIALIZATION_NVP(float_mtx);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		
		mf << NNTL_SERIALIZATION_NVP(double_mtx);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

		mf & serialization::make_named_struct("td", td_ET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
	}
	{
		double d;
		float f;

		SCOPED_TRACE("Testing imatfile");
		imatfile<> mf;

		ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName));

		mf & NNTL_SERIALIZATION_NVP(d);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_DOUBLE_EQ(dET, d) << "double differs";

		mf & NNTL_SERIALIZATION_NVP(f);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_FLOAT_EQ(fET, f) << "float differs";

		mf >> serialization::make_nvp("float_mtx", float_mtx2);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_MTX_EQ(float_mtx, float_mtx2, "Saved and loaded float matrices differ!");

		mf >> serialization::make_nvp("double_mtx", double_mtx2);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_MTX_EQ(double_mtx, double_mtx2, "Saved and loaded double matrices differ!");

		mf & NNTL_SERIALIZATION_STRUCT(td);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(td_ET, td) << "train_data comparison failed";
	}
}

#ifdef NNTL_DEBUG
#define TESTS_MATFILE_ALLOW_NNTL_ASSERTION_FAILURE 0
#else
#define TESTS_MATFILE_ALLOW_NNTL_ASSERTION_FAILURE 1
#endif

TEST(TestMatfile, ManualStructsWriting) {
	const double dET = 3.9, d2ET = 412.54;
	const float fET = 2.6f, f2ET = 4.23f;
	const char *const pFileName = "./test_data/test_structs.mat";

	struct Struct1T {
		double VarD;
		//we can't use templates here, therefore had to make 2 function definitions.
		void serialize(omatfileEx<> & ar, const unsigned int version) {
			ar & NNTL_SERIALIZATION_NVP(VarD);
			ASSERT_EQ(ar.ErrorCode::Success, ar.get_last_error()) << ar.get_last_error_str();
		}
		void serialize(imatfile<> & ar, const unsigned int version) {
			ar & NNTL_SERIALIZATION_NVP(VarD);
			ASSERT_EQ(ar.ErrorCode::Success, ar.get_last_error()) << ar.get_last_error_str();
		}
		const bool operator==(const Struct1T& rhs)const noexcept { return VarD == rhs.VarD; }
	};
	const Struct1T s1ET = { dET }, s1ET2 = { d2ET };

	struct Struct2T {
		float VarF;
		void serialize(imatfile<> & ar, const unsigned int version) {
			ar & NNTL_SERIALIZATION_NVP(VarF);
			ASSERT_EQ(ar.ErrorCode::Success, ar.get_last_error()) << ar.get_last_error_str();
		}
		const bool operator==(const Struct2T& rhs)const noexcept { return VarF == rhs.VarF; }
	};
	const Struct2T s2ET = { fET }, s2ET2 = { f2ET };

	struct StructOverwriteT {
		float VarF;
		void serialize(imatfile<> & ar, const unsigned int version) { 
			ar & NNTL_SERIALIZATION_NVP(VarF);
			ASSERT_EQ(ar.ErrorCode::Success, ar.get_last_error()) << ar.get_last_error_str();
#if TESTS_MATFILE_ALLOW_NNTL_ASSERTION_FAILURE
			//The following code would emit assertion inside of matfile during DEBUG build
			double VarD = -1.;
			ar & NNTL_SERIALIZATION_NVP(VarD);
			ASSERT_EQ(-1., VarD) << "VarD mustn't change";
			ASSERT_EQ(ar.ErrorCode::FailedToTransferVariable, ar.get_last_error()) << "Read a field that must not be existed";
			ar._drop_last_error();
#endif
		}
		const bool operator==(const StructOverwriteT& rhs)const noexcept { return VarF == rhs.VarF; }
	};
	const StructOverwriteT soET = { fET };

	struct StructUpdateT {
		double VarD;
		float VarF;
		void serialize(imatfile<> & ar, const unsigned int version) {
			ar & NNTL_SERIALIZATION_NVP(VarF);
			ASSERT_EQ(ar.ErrorCode::Success, ar.get_last_error()) << ar.get_last_error_str();
			ar & NNTL_SERIALIZATION_NVP(VarD);
			ASSERT_EQ(ar.ErrorCode::Success, ar.get_last_error()) << ar.get_last_error_str();			
		}
		const bool operator==(const StructUpdateT& rhs)const noexcept { return VarF == rhs.VarF && VarD==rhs.VarD; }
	};
	const StructUpdateT suET = { dET, fET };

	{
		SCOPED_TRACE("Testing omatfileEx with manual struct writing mode");
		omatfileEx<> mf;
		const char* pNS_normal = "Struct1", *pNS_overwrite = "StructOverwrite", *pNS_update = "StructUpdate";

		ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName, mf.FileOpenMode::UpdateDelete));

		mf << serialization::make_nvp("d", dET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

		//////////////////////////////////////////////////////////////////////////
		// testing simple struct saving
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string(pNS_normal), false, false)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarD", dET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();

		//////////////////////////////////////////////////////////////////////////
		// testing overwriting
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string(pNS_overwrite), false, false)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarF", f2ET);
		mf << serialization::make_nvp("VarD", dET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();

		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string(pNS_overwrite), false, false)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarF", fET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();

		//////////////////////////////////////////////////////////////////////////
		// testing updating
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string(pNS_update), false, false)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarF", f2ET);
		mf << serialization::make_nvp("VarD", dET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();

		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string(pNS_update), true, false)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarF", fET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();

		//////////////////////////////////////////////////////////////////////////
		// testing nested calls
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string("Sf_1"), false, false)) << mf.get_last_error_str();
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string("Sd_1"), false, true)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarD", dET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarF", fET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();

		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string("Sf_2"), false, false)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarF", f2ET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_begin(::std::string("Sd_2"), false, true)) << mf.get_last_error_str();
		mf << serialization::make_nvp("VarD", d2ET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();
		ASSERT_EQ(mf.ErrorCode::Success, mf.save_struct_end()) << mf.get_last_error_str();


		// finally writing another variable and a struct to test all is fine
		Struct1T S1dupe = { dET };
		mf << NNTL_SERIALIZATION_STRUCT(S1dupe);

		mf << serialization::make_nvp("f", fET);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();

	}
	{
		SCOPED_TRACE("Testing imatfile for manual struct writing mode");
		imatfile<> mf;
		double d;
		float f;

		ASSERT_EQ(mf.ErrorCode::Success, mf.open(pFileName));

		mf & NNTL_SERIALIZATION_NVP(d);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_DOUBLE_EQ(dET, d) << "double differs";

		mf & NNTL_SERIALIZATION_NVP(f);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_FLOAT_EQ(fET, f) << "float differs";

		Struct1T S1dupe;
		mf & NNTL_SERIALIZATION_STRUCT(S1dupe);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(s1ET, S1dupe) << "Struct1T S1dupe differs";

		//////////////////////////////////////////////////////////////////////////
		//reading structs
		Struct1T Struct1;
		mf & NNTL_SERIALIZATION_STRUCT(Struct1);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(s1ET, Struct1) << "Struct1T differs";

		StructOverwriteT StructOverwrite;
		mf & NNTL_SERIALIZATION_STRUCT(StructOverwrite);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(soET, StructOverwrite) << "StructOverwrite differs";

		StructUpdateT StructUpdate;
		mf & NNTL_SERIALIZATION_STRUCT(StructUpdate);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(suET, StructUpdate) << "StructUpdate differs";

		Struct1T Sd_1, Sd_2;
		mf & NNTL_SERIALIZATION_STRUCT(Sd_1);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(s1ET, Sd_1) << "Sd_1 differs";
		mf & NNTL_SERIALIZATION_STRUCT(Sd_2);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(s1ET2, Sd_2) << "Sd_2 differs";

		Struct2T Sf_1, Sf_2;
		mf & NNTL_SERIALIZATION_STRUCT(Sf_1);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(s2ET, Sf_1) << "Sf_1 differs";
		mf & NNTL_SERIALIZATION_STRUCT(Sf_2);
		ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
		ASSERT_EQ(s2ET2, Sf_2) << "Sf_2 differs";
	}
}


TEST(TestMatfile, DumpNnet) {
	train_data<real_t> td;
	binfile reader;

	binfile::ErrorCode rec = reader.read(NNTL_STRING(MNIST_SMALL_FILE), td);
	ASSERT_EQ(binfile::ErrorCode::Success, rec) << "Error code description: " << reader.get_last_error_str();
	ASSERT_TRUE(td.train_x().emulatesBiases());
	ASSERT_TRUE(td.test_x().emulatesBiases());

	size_t epochs = 5;
	const real_t learningRate = real_t(.002);

	layer_input<> inp(td.train_x().cols_no_bias());
	layer_fully_connected<> fcl(60, learningRate);
	layer_fully_connected<> fcl2(50, learningRate);
	layer_output<> outp(td.train_y().cols(), learningRate);

	auto lp = make_layers(inp, fcl, fcl2, outp);

	nnet_train_opts<> opts(epochs);

	opts.batchSize(100).ImmediatelyDeinit(false);
	auto nn = make_nnet(lp);
	auto ec = nn.train(td, opts);
	ASSERT_EQ(decltype(nn)::ErrorCode::Success, ec) << "Error code description: " << nn.get_last_error_string();

	//doing actual work
	omatfile<> mf;
	mf.turn_on_all_options();
	ASSERT_EQ(mf.ErrorCode::Success, mf.open("./test_data/nn_dump.mat"));

	mf & nn;
	ASSERT_EQ(mf.ErrorCode::Success, mf.get_last_error()) << mf.get_last_error_str();
}

#else

TEST(TestMatfile, Warning) {
	STDCOUT("# Test set has been skipped. Define NNTL_MATLAB_AVAILABLE to nonzero value to enable the set.");
}

#endif