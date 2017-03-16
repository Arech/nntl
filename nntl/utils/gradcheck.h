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

namespace nntl {

	namespace _impl {
		enum class gradcheck_paramsGroup {
			dLdA,
			dLdW
		};

		enum class gradcheck_mode {
			batch,//use batch mode to check parameters that affects a whole batch, e.g. single neuron weight gradient (dL/dW)
			online//online (single sample) mode should be used to check parameters, that affects a loss function value
				  // on only a single sample, e.g. activation value gradient (dL/dA)
		};

		enum class gradcheck_phase {
			df_analitical,
			df_numeric_plus,
			df_numeric_minus
		};
	}

	enum class gradcheck_settingsCountMode {
		total_min,//min(<paramsToCheck_Setting>, total_params_to_check) parameters
		percent_or_min// max(<NPerc_Setting>% of total_params_to_check, min(<paramsToCheck_Setting>,total_params_to_check))"
	};

	struct gradcheck_groupSetts {
	protected:
		gradcheck_settingsCountMode countMode;
		union{
			neurons_count_t paramsToCheck;

			struct {
				neurons_count_t paramsToCheck;
				unsigned NPercents;
			} perc;
		} u;

	public:
		~gradcheck_groupSetts()noexcept{}
		gradcheck_groupSetts()noexcept { set(20, 10); }
		gradcheck_groupSetts(neurons_count_t ptc)noexcept { set(ptc); }
		gradcheck_groupSetts(neurons_count_t ptc, unsigned percCnt)noexcept { set(ptc, percCnt); }

		void set(neurons_count_t ptc)noexcept {
			NNTL_ASSERT(ptc > 0);
			countMode = gradcheck_settingsCountMode::total_min;
			u.paramsToCheck = ptc > 0 ? ptc : 1;
		}

		void set(neurons_count_t ptc, unsigned percCnt)noexcept {
			NNTL_ASSERT(percCnt <= 100 && percCnt > 0 && ptc > 0);
			countMode = gradcheck_settingsCountMode::percent_or_min;
			u.perc.paramsToCheck = ptc > 0 ? ptc : 1;
			u.perc.NPercents = percCnt > 100 ? 100 : percCnt;
		}

		neurons_count_t countToCheck(neurons_count_t maxParams)const noexcept {
			neurons_count_t r = 0;
			switch (countMode) {
			case nntl::gradcheck_settingsCountMode::total_min:
				r = std::min(maxParams, u.paramsToCheck);
				break;

			case nntl::gradcheck_settingsCountMode::percent_or_min:
				r = std::max(
					(maxParams * u.perc.NPercents) / 100
					, std::min(maxParams, u.perc.paramsToCheck)
				);
				break;

			default:
				NNTL_ASSERT(!"WTF???");
				break;
			}
			NNTL_ASSERT(r > 0 && r <= maxParams);
			return r;
		}
	};

	namespace _impl {
		template <typename _T> struct gradcheck_def_relErrs_dLdA {};
		template <> struct gradcheck_def_relErrs_dLdA<double> { 
			static constexpr double warn = double(1e-6);
			static constexpr double fail = double(1e-4);
		};
		template <> struct gradcheck_def_relErrs_dLdA<float> { 
			static constexpr float warn = float(1e-5);
			static constexpr float fail = float(1e-3);
		};

		template <typename _T> struct gradcheck_def_relErrs_dLdW {};
		template <> struct gradcheck_def_relErrs_dLdW<double> {
			static constexpr double warn = double(1e-5);
			static constexpr double fail = double(1e-4);
		};
		template <> struct gradcheck_def_relErrs_dLdW<float> {
			static constexpr float warn = float(1e-4);
			static constexpr float fail = float(1e-3);
		};
	}

	template<typename RealT>
	struct gradcheck_evalSetts_group {
		typedef RealT real_t;

		real_t relErrWarnThrsh;
		real_t relErrFailThrsh;

		unsigned percOfZeros;//max percent of dL/dX==0 from the total number of dL/dX to check. If there's more zeroed 
							 //dL/dX than this threshold, error will be emitted
							 //in order to reduce number of zeroed entries one may try to increase batch_size, however, this still may not work
							 //as some neurons still might be left unused.

		gradcheck_evalSetts_group(const real_t& warn, const real_t& fail)noexcept
			:percOfZeros(0), relErrWarnThrsh(warn) , relErrFailThrsh(fail)
		{}
	};


	template<typename RealT>
	struct gradcheck_evalSetts {
		typedef RealT real_t;

		gradcheck_evalSetts_group<real_t> dLdA_setts;
		gradcheck_evalSetts_group<real_t> dLdW_setts;

		bool bIgnoreZerodLdWInUndelyingLayer;

		gradcheck_evalSetts()noexcept
			: dLdA_setts(_impl::gradcheck_def_relErrs_dLdA<real_t>::warn, _impl::gradcheck_def_relErrs_dLdA<real_t>::fail)
			, dLdW_setts(_impl::gradcheck_def_relErrs_dLdW<real_t>::warn, _impl::gradcheck_def_relErrs_dLdW<real_t>::fail)
			, bIgnoreZerodLdWInUndelyingLayer(false)
		{}
	};


	namespace _impl {
		template <typename _T> struct gradcheck_def_stepSize {};
		template <> struct gradcheck_def_stepSize<double> { static constexpr double value = double(1e-5); };
		template <> struct gradcheck_def_stepSize<float> { static constexpr float value = float(1e-4); };
	}

	// numeric gradient check settings 
	template<typename RealT>
	struct gradcheck_settings {
		typedef RealT real_t;

		// For the sake of gradcheck coverage we use separate thresholds for a number of parameters to check within a group - groupSetts
		// (e.g. neurons count for dL/dA or dL/dW check) and a number of parameters to check within each subgroup - subgroupSetts(e.g.
		// particular weight of a neuron selected by groupSetts)
		gradcheck_groupSetts groupSetts;
		gradcheck_groupSetts subgroupSetts;

		gradcheck_evalSetts<real_t> evalSetts;

		const real_t stepSize;
		const bool bVerbose;

		//////////////////////////////////////////////////////////////////////////
		gradcheck_settings()noexcept : stepSize(_impl::gradcheck_def_stepSize<real_t>::value), bVerbose(true) {}

		gradcheck_settings(bool vb, real_t ss = _impl::gradcheck_def_stepSize<real_t>::value)noexcept 
			: stepSize(ss), bVerbose(vb)
		{}
	};

	namespace _impl {

		template<typename RealT>
		class gradcheck_dataHolder : public nntl::math::smatrix_td {
		public:
			typedef RealT real_t;
			typedef nntl::math::smatrix<real_t> realmtx_t;
			typedef nntl::math::smatrix_deform<real_t> realmtxdef_t;

		protected:
			const realmtx_t* m_pDataX;
			const realmtx_t* m_pDataY;

			realmtxdef_t m_batchX, m_batchY;

			std::vector<vec_len_t> m_rowsIdxs, m_curBatchIdxs;

			vec_len_t m_lastUsedRow;

		public:
			~gradcheck_dataHolder()noexcept {
				deinit();
			}
			gradcheck_dataHolder()noexcept : m_pDataX(nullptr), m_pDataY(nullptr) {
				m_batchX.will_emulate_biases();
				m_batchY.dont_emulate_biases();
			}

			const std::vector<vec_len_t>& curBatchIdxs()const noexcept { 
				NNTL_ASSERT(m_curBatchIdxs.size() > 0);
				return m_curBatchIdxs;
			}

			const realmtx_t& batchX()const noexcept {
				NNTL_ASSERT(!m_batchX.empty() && m_batchX.numel() > 0);
				return m_batchX;
			}
			const realmtx_t& batchY()const noexcept {
				NNTL_ASSERT(!m_batchY.empty() && m_batchY.numel() > 0);
				return m_batchY;
			}

			void deinit()noexcept {
				m_batchX.clear();
				m_batchY.clear();
				m_rowsIdxs.clear();
				m_curBatchIdxs.clear();
				m_pDataX = nullptr;
				m_pDataY = nullptr;
			}
			void init(const vec_len_t maxBatchSize, const realmtx_t& data_x, const realmtx_t& data_y)noexcept {
				NNTL_ASSERT(maxBatchSize > 0 && maxBatchSize <= data_x.rows() && data_x.rows() == data_y.rows());
				NNTL_ASSERT(data_x.emulatesBiases() && !data_y.emulatesBiases());
				NNTL_ASSERT(m_batchX.empty() && m_batchY.empty());
				NNTL_ASSERT(m_batchX.emulatesBiases() && !m_batchY.emulatesBiases());

				m_pDataX = &data_x;
				m_pDataY = &data_y;
				m_batchX.resize(maxBatchSize, data_x.cols_no_bias());
				m_batchY.resize(maxBatchSize, data_y.cols());

				m_rowsIdxs.resize(data_x.rows());
				std::iota(m_rowsIdxs.begin(), m_rowsIdxs.end(), 0);
				m_lastUsedRow = data_x.rows();

				m_curBatchIdxs.reserve(maxBatchSize);
			}

			void prepateToBatchSize(const vec_len_t batchSize)noexcept {
				m_batchX.deform_rows(batchSize);
				m_batchY.deform_rows(batchSize);
				m_curBatchIdxs.resize(batchSize);
			}
			const vec_len_t curBatchSize()const noexcept { return m_batchX.rows(); }

			template<typename iRngT, typename iMathT>
			void nextBatch(iRngT& iR, iMathT& iM)noexcept {
				if (m_lastUsedRow + curBatchSize() >= m_pDataX->rows()) {
					m_lastUsedRow = 0;
					std::random_shuffle(m_rowsIdxs.begin(), m_rowsIdxs.end(), iR);
				}

				vec_len_t* pSrc = &m_rowsIdxs[m_lastUsedRow];
				std::copy(pSrc, pSrc + curBatchSize(), m_curBatchIdxs.begin());

				m_lastUsedRow += curBatchSize();

				iM.mExtractRows(*m_pDataX, m_curBatchIdxs.begin(), m_batchX);
				iM.mExtractRows(*m_pDataY, m_curBatchIdxs.begin(), m_batchY);
			}
		};

	}
}