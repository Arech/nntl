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

#include "../_i_inspector.h"

namespace nntl {
	namespace inspector {

		namespace conds {
			struct EpochNum : public math::smatrix_td {
				typedef std::vector<size_t> epochs_to_dump_t;

				vector_conditions m_dumpEpochCond;
				epochs_to_dump_t m_epochsToDump;

				void on_init_nnet(const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
					const auto lastEpochIdx = totalEpochs - 1;
					if (!m_epochsToDump.size()) m_epochsToDump.push_back(lastEpochIdx);

					m_dumpEpochCond.clear().resize(totalEpochs, false);
					for (const auto etd : m_epochsToDump) {
						if (etd < totalEpochs) {
							m_dumpEpochCond.verbose(etd);
						} else m_dumpEpochCond.verbose(lastEpochIdx);
					}
				}

				const bool on_train_epochBegin(const size_t epochIdx)const noexcept {
					return m_dumpEpochCond(epochIdx);
				}
				static constexpr bool on_train_batchBegin(const bool _bDoDump, const vec_len_t batchIdx, const size_t epochIdx) noexcept {
					return _bDoDump && 0 == batchIdx;
				}
			};

			//useful to debug initial training steps
			struct FirstBatches : public math::smatrix_td {
				size_t batchesRun, maxBatches, stride;
				

				FirstBatches()noexcept:batchesRun(0), maxBatches(0), stride(0){}

				void on_init_nnet(const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
					batchesRun = 0;
					if (!maxBatches) maxBatches = totalEpochs*totalBatches;
					if (!stride) stride = 1;
				}

				static constexpr bool on_train_epochBegin(const size_t epochIdx)noexcept { return true; }
				const bool on_train_batchBegin(const bool _bDoDump, const vec_len_t batchIdx, const size_t epochIdx) noexcept {
					return batchesRun < maxBatches && !(batchesRun++ % stride);
				}
			};

		}

		


		template<typename FinalChildT, typename RealT, typename ArchiveT, typename CondDumpT, size_t maxNnetDepth = 10>
		class _dumper_base : public _impl::_bwlist<FinalChildT, RealT>
		{
		public:
			typedef ArchiveT archive_t;
			typedef typename archive_t::ErrorCode ArchiveError_t;

			typedef CondDumpT cond_dump_t;

			typedef std::vector<std::string> layer_names_t;

		protected:
			typedef _impl::layer_idx_keeper<layer_index_t, _NoLayerIdxSpecified, maxNnetDepth> keeper_t;

			//#todo: this settings should be given be a special ParamsT class passed here as a template parameter
			static constexpr bool bVerbose = false;
			//dumps significantly (x10+) times faster when set to true
			static constexpr bool bSplitFiles = true;

			static constexpr bool bIgnoreLayersNesting = true;

		protected:
			cond_dump_t m_condDump;
			
			layer_names_t m_layerNames;
			size_t m_layersCount, m_epochIdx;
			vec_len_t m_batchIdx;

			keeper_t m_curLayer;

			const char* m_pDirToDump;

			static constexpr size_t maxFileNameLength = MAX_PATH;
			static constexpr size_t maxDirNameLength = maxFileNameLength - 10;

		private:
			archive_t* m_pArch;
			const bool m_bOwnArch;		

		protected:
			

			template<bool b=bVerbose>
			std::enable_if_t<!b> _verbalize(const char*)const noexcept {}
			template<bool b = bVerbose>
			std::enable_if_t<b> _verbalize(const char* s)const noexcept {
				STDCOUTL(_layer_name() << " - " << s);
			}

		private:
			void _make_archive(std::nullptr_t pA)noexcept{
				NNTL_ASSERT(m_bOwnArch);
				m_pArch = new (std::nothrow) archive_t;
			}
			void _make_archive(archive_t* pA)noexcept {
				NNTL_ASSERT(pA);
				NNTL_ASSERT(!m_bOwnArch);
				m_pArch = pA;
			}

			void _ctor()noexcept {
				m_epochIdx = -1;
				m_batchIdx = -1;
				m_layersCount = 0;
				m_pDirToDump = nullptr;
			}

		protected:
			~_dumper_base()noexcept {
				if (m_bOwnArch) delete m_pArch;
				m_pArch = nullptr;
			}
			_dumper_base()noexcept : m_bOwnArch(true){
				_ctor();
				_make_archive(nullptr);
				NNTL_ASSERT(m_pArch);
			}

			template<typename PArchT = std::nullptr_t>
			_dumper_base(const char* pDirToDump, PArchT pA=nullptr)noexcept 
				: m_bOwnArch(!pA)
			{
				_ctor();
				if (pDirToDump) set_dir_to_dump(pDirToDump);
				_make_archive(pA);
				NNTL_ASSERT(m_pArch);
			}

			static constexpr char* _noLayerName = "_NoName";
			const char*const _layer_name()const noexcept {
				NNTL_ASSERT(m_curLayer < m_layersCount && !m_layerNames[m_curLayer].empty());
				return m_curLayer < m_layersCount ? m_layerNames[m_curLayer].c_str() : _noLayerName;
			}

			static void _epic_fail(const char* r1, const char* r2=nullptr, const char* r3=nullptr) noexcept {
				STDCOUT(r1);
				if (r2) STDCOUT(r2);
				if (r3) STDCOUT(r3);
				abort();
			}

			void _check_err(const ArchiveError_t ec, const char* descr)const noexcept {
				NNTL_ASSERT(ArchiveError_t::Success == ec);
				if (ArchiveError_t::Success != ec) {
					_epic_fail(descr, " failed. Reason: ", get_self().getArchive().get_error_str(ec));
				}
			}

			template<bool b = bSplitFiles>
			std::enable_if_t<b> _make_file_name(char* n, const size_t ml, const bool bFprop)const noexcept {
				NNTL_ASSERT(m_pDirToDump);
				if (!m_pDirToDump) _epic_fail("m_pDirToDump is not set!");
				sprintf_s(n, ml, bFprop ? "%s/epoch%zd_%df.mat" : "%s/epoch%zd_%db.mat", m_pDirToDump, m_epochIdx + 1, m_batchIdx);
			}
			template<bool b = bSplitFiles>
			std::enable_if_t<!b> _make_file_name(char* n, const size_t ml, const bool bFprop)const noexcept {
				NNTL_ASSERT(m_pDirToDump);
				if (!m_pDirToDump) _epic_fail("m_pDirToDump is not set!");
				sprintf_s(n, ml, "%s/epoch%zd_%d.mat", m_pDirToDump, m_epochIdx + 1, m_batchIdx);
			}

			self_ref_t _open_archive(const bool bFprop=false)noexcept {
				char n[maxFileNameLength];
				_make_file_name(n, maxFileNameLength, bFprop);
				_check_err(get_self().getArchive().open(n, archive_t::FileOpenMode::UpdateDelete), "Opening file for updating");				
				return get_self();
			}
			self_ref_t _close_archive()noexcept {
				_check_err(get_self().getArchive().close(), "Closing file");
				return get_self();
			}
			//default implementations
			void on_train_batchBegin(const vec_len_t batchIdx) const noexcept {}
			void on_train_batchEnd()const noexcept {}
			void on_train_preFprop(const realmtx_t& data_x)const noexcept {}
			void on_train_preBprop(const realmtx_t& data_y)const noexcept {}
			void on_fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept {}
			void on_fprop_end(const realmtx_t& act) const noexcept {}
			void on_bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) const noexcept {}
			void on_bprop_end(const realmtx_t& dLdAPrev) const noexcept {}

		public:
			const bool bDoDump()const noexcept { return m_bDoDump; }
			const bool bDoDump(const layer_index_t& lidx)const noexcept { 
				return m_bDoDump && _bwlist_layerAllowed(lidx);
			}

			self_ref_t set_dir_to_dump(const char* pDirName)noexcept {
				NNTL_ASSERT(pDirName);
				m_pDirToDump = pDirName;
				if (strlen(m_pDirToDump) > maxDirNameLength) _epic_fail(NNTL_FUNCTION, ": Too long directory name");
				return get_self();
			}

			archive_t& getArchive()const noexcept {
				NNTL_ASSERT(m_pArch);
				return *m_pArch;
			}

			cond_dump_t& getCondDump()noexcept { return m_condDump; }
			
			//////////////////////////////////////////////////////////////////////////

			//to notify about total layer, epoch and batches count
			void init_nnet(const size_t totalLayers, const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
				NNTL_ASSERT(m_pDirToDump);
				NNTL_ASSERT(totalLayers && totalEpochs && totalBatches);
				m_layersCount = totalLayers;
				m_bDoDump = false;

				//#exceptions STL
				m_layerNames.clear();
				m_layerNames.resize(m_layersCount);
				m_layerNames.shrink_to_fit();

				_bwlist_init(totalLayers);

				m_condDump.on_init_nnet(totalEpochs, totalBatches);
				//if (!m_pDirToDump) STDCOUTL("*beware, dumping has been disabled due to unset directory to dump");
			}

			template<typename StrT>
			void init_layer(const layer_index_t lIdx, StrT&& LayerName, const layer_type_id_t layerTypeId)noexcept {
				NNTL_ASSERT(lIdx < m_layersCount);
				//#exceptions STL
				m_layerNames[lIdx].assign(std::forward<StrT>(LayerName));
				_bwlist_updateLayer(lIdx, layerTypeId);
			};

			void train_epochBegin(const size_t epochIdx)noexcept {
				m_epochIdx = epochIdx;
				m_bDoDump = m_condDump.on_train_epochBegin(epochIdx);
			}
			void train_batchBegin(const vec_len_t batchIdx) noexcept {
				m_batchIdx = batchIdx;
				m_bDoDump = m_condDump.on_train_batchBegin(m_bDoDump, batchIdx, m_epochIdx);
				if (m_bDoDump) {
					STDCOUTL("Going to dump the dataflow...");
					get_self()._open_archive(true)
						.on_train_batchBegin(batchIdx);
				}
			}

			void train_preBprop(const realmtx_t& data_y) noexcept {
				if (m_bDoDump) {
					if (bSplitFiles) get_self()._close_archive()
						._open_archive(false);

					get_self().on_train_preBprop(data_y);
				}
			}

			void train_batchEnd() noexcept {
				if (m_bDoDump) {
					get_self().on_train_batchEnd();
					get_self()._close_archive();
				}
				m_bDoDump = false;
			}

			//////////////////////////////////////////////////////////////////////////
			// dumping functions implies that m_bDoDump is changed only during train_*() calls
			void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) noexcept {
				//m_bDoDump = m_bDoDump && bTrainingMode;
				//we shouldn't update m_bDoDump here, because it's already set by train_batchBegin(). bTrainingMode
				// is guaranteed to be true when m_bDoDump==true
				NNTL_ASSERT(!m_bDoDump || bTrainingMode);
				m_curLayer.push(lIdx);
				if (bDoDump(lIdx)) get_self().on_fprop_begin(lIdx, prevAct, bTrainingMode);
			}
			void fprop_end(const realmtx_t& act) noexcept {
				if (bDoDump(m_curLayer)) get_self().on_fprop_end(act);
				m_curLayer.pop();
			}
			void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) noexcept {
				m_curLayer.push(lIdx);
				if (bDoDump(lIdx)) get_self().on_bprop_begin(lIdx, dLdA);
			}
			void bprop_end(const realmtx_t& dLdAPrev) noexcept {
				if (bDoDump(m_curLayer)) get_self().on_bprop_end(dLdAPrev);
				m_curLayer.pop();
			}
		};

		template<typename FinalChildT, typename RealT, typename ArchiveT, typename CondDumpT, size_t maxNnetDepth = 10>
		class _dumper : public _dumper_base<FinalChildT, RealT, ArchiveT, CondDumpT, maxNnetDepth> {
		private:
			typedef _dumper_base<FinalChildT, RealT, ArchiveT, CondDumpT, maxNnetDepth> _base_class;

		protected:
			~_dumper()noexcept {}
			_dumper()noexcept {}
			_dumper(const char* pDirToDump)noexcept : _base_class(pDirToDump, nullptr){}
			_dumper(const char* pDirToDump, archive_t* pA)noexcept : _base_class(pDirToDump, pA) {}

		public:
			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////
			//NB: every _i_inspector's function implementation (they aren't staring with on_) should check if bDoDump() first
			
			void on_fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept {
				NNTL_ASSERT(bTrainingMode);
				_verbalize("on_fprop_begin");
				auto& ar = getArchive();
				_check_err(ar.save_struct_begin(get_self()._layer_name(), false, bIgnoreLayersNesting), "on_fprop_begin: save_struct_begin");
			}

			/*void fprop_preNesterovMomentum(const realmtx_t& vW, const real_t momentum, const realmtx_t& W)const noexcept {
				if (bDoDump()) {
					_verbalize("fprop_preNesterovMomentum");
					auto& ar = getArchive();
					ar & serialization::make_nvp("f_preNM_vW",vW);
					_check_err(ar.get_last_error(), "fprop_preNesterovMomentum: saving vW");
					ar & serialization::make_nvp("f_preNM_momentum", momentum);
					_check_err(ar.get_last_error(), "fprop_preNesterovMomentum: saving momentum");
				}
			}*/

			void on_fprop_end(const realmtx_t& A)const noexcept {
				_verbalize("on_fprop_end");
				auto& ar = getArchive();				
				ar & serialization::make_nvp("A",A);
				_check_err(ar.get_last_error(), "on_fprop_end: saving activations");

				_check_err(ar.save_struct_end(), "on_fprop_end: save_struct_end");
			}

			//////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////

			void on_bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA)const noexcept {
				_verbalize("on_bprop_begin");
				auto& ar = getArchive();
				_check_err (ar.save_struct_begin(get_self()._layer_name(), bSplitFiles ? false : true, bIgnoreLayersNesting),"on_bprop_begin: save_struct_begin");
				ar & NNTL_SERIALIZATION_NVP(dLdA);
				_check_err(ar.get_last_error(),"on_bprop_begin: saving dLdA");
			}

			/*void bprop_dLdZ(const realmtx_t& dLdZ) const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("bprop_dLdZ");
					auto& ar = getArchive();
					ar & NNTL_SERIALIZATION_NVP(dLdZ);
					_check_err(ar.get_last_error(), "bprop_dLdZ: saving dLdZ");
				}
			}*/

			void apply_grad_begin(const realmtx_t& W, const realmtx_t& dLdW)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_begin");
					auto& ar = getArchive();
					ar & NNTL_SERIALIZATION_NVP(dLdW);
					_check_err(ar.get_last_error(), "apply_grad_begin: saving dLdW");
				}
			}
			void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_update");
					auto& ar = getArchive();
					ar & serialization::make_nvp("a_preUpd_W",W);
					_check_err(ar.get_last_error(), "apply_grad_update: saving weights");

					ar & serialization::make_nvp("a_preUpd_WUpd",WUpd);
					_check_err(ar.get_last_error(), "apply_grad_update: saving weight updates");
				}
			}

			void on_bprop_end(const realmtx_t& dLdAPrev) const noexcept {
				_verbalize("on_bprop_end");
				_check_err(getArchive().save_struct_end(), "on_bprop_end: save_struct_end");
			}
		};

		template<typename RealT, typename ArchiveT, typename CondDumpT = conds::EpochNum, size_t maxNnetDepth = 10>
		class dumper final : public _dumper< dumper<RealT, ArchiveT, CondDumpT, maxNnetDepth>, RealT, ArchiveT, CondDumpT, maxNnetDepth> {
		private:
			typedef _dumper< dumper<RealT, ArchiveT, CondDumpT, maxNnetDepth>, RealT, ArchiveT, CondDumpT, maxNnetDepth> _base_class;

		public:
			~dumper()noexcept {}
			dumper()noexcept {}
			dumper(const char* pDirToDump)noexcept : _base_class(pDirToDump) {}
			dumper(const char* pDirToDump, archive_t* pA)noexcept : _base_class(pDirToDump, pA) {}
		};
	}
}