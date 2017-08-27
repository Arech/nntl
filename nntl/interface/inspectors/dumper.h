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
#include "../../utils/bwlist.h"

namespace nntl {
	namespace inspector {

		namespace conds {

			//dumps first batch of an epoch
			struct EpochNum : public math::smatrix_td {
				typedef ::std::vector<size_t> epochs_to_dump_t;
				//static constexpr bool bNewBatch2NewFile = true;

				vector_conditions m_dumpEpochCond;
				epochs_to_dump_t m_epochsToDump;

				void on_init_nnet(const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
					NNTL_UNREF(totalBatches);
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
#pragma warning(disable : 4100)
				static constexpr bool on_train_batchBegin(const bool _bDoDump, const vec_len_t batchIdx, const size_t epochIdx) noexcept {
					return _bDoDump && 0 == batchIdx;
				}
				static constexpr bool on_train_calcErr(const size_t epochIdx, const bool bOnTrainSet) noexcept {return false;};
#pragma warning(default : 4100)
			};

			//This function will dump first batch of specified epochs AND the training error calculation
			struct CalcErrByEpochNum : public EpochNum {
				typedef EpochNum parent_t;
				bool bDumpInitialErr;

				void on_init_nnet(const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
					parent_t::on_init_nnet(totalEpochs, totalBatches);
					bDumpInitialErr = ::std::any_of(m_epochsToDump.begin(), m_epochsToDump.end(), [](const auto& V)->bool {
						return V == static_cast<size_t>(-1);
					});
				}

				const bool on_train_calcErr(const size_t epochIdx, const bool bOnTrainSet)const noexcept {
					return bOnTrainSet && (epochIdx == static_cast<size_t>(-1) ? bDumpInitialErr : m_dumpEpochCond(epochIdx));
				}
			};

			//This function will dump only the training error calculation for a specified epoch
			struct CalcErrOnlyByEpochNum : public CalcErrByEpochNum {
#pragma warning(disable : 4100)
				static constexpr bool on_train_epochBegin(const size_t epochIdx) noexcept { return false; }
				static constexpr bool on_train_batchBegin(const bool _bDoDump, const vec_len_t batchIdx, const size_t epochIdx) noexcept {
					return false;
				}
#pragma warning(default : 4100)
			};

			/*struct FullEpochNum : public EpochNum {
				static constexpr bool bNewBatch2NewFile = false;
				static constexpr bool on_train_batchBegin(const bool _bDoDump, const vec_len_t batchIdx, const size_t epochIdx) noexcept {
					return _bDoDump;
				}
			};*/

			//useful to debug initial training steps
			struct FirstBatches : public math::smatrix_td {
				//static constexpr bool bNewBatch2NewFile = true;

				size_t batchesRun, maxBatches, stride;
				

				FirstBatches()noexcept:batchesRun(0), maxBatches(0), stride(0){}

				void on_init_nnet(const size_t totalEpochs, const vec_len_t totalBatches)noexcept {
					batchesRun = 0;
					if (!maxBatches) maxBatches = totalEpochs*totalBatches;
					if (!stride) stride = 1;
				}

#pragma warning(disable : 4100)
				static constexpr bool on_train_epochBegin(const size_t epochIdx)noexcept { return true; }
				const bool on_train_batchBegin(const bool _bDoDump, const vec_len_t batchIdx, const size_t epochIdx) noexcept {
					return batchesRun < maxBatches && !(batchesRun++ % stride);
				}
				static constexpr bool on_train_calcErr(const size_t epochIdx, const bool bOnTrainSet) noexcept { return false; };
#pragma warning(default : 4100)
			};
		}


		template<typename FinalChildT, typename RealT, typename ArchiveT, typename CondDumpT, size_t maxNnetDepth = 32>
		class _dumper_base 
			: public utils::_bwlist<RealT>
			, public _impl::_base<RealT>
		{
		public:
			typedef FinalChildT self_t;
			NNTL_METHODS_SELF_CHECKED((::std::is_base_of<_dumper_base, FinalChildT>::value), "FinalChildT must derive from _dumper_base");

			typedef ArchiveT archive_t;
			typedef typename archive_t::ErrorCode ArchiveError_t;

			typedef CondDumpT cond_dump_t;

			typedef ::std::vector<::std::string> layer_names_t;

		protected:
			typedef utils::layer_idx_keeper<layer_index_t, _NoLayerIdxSpecified, maxNnetDepth> keeper_t;

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

			::std::string m_DirToDump;

			bool m_bDoDump;//it's 'protected' for some rare special unforeseen cases. Use the bDoDump() or the CondDumpT

			static constexpr size_t maxFileNameLength = MAX_PATH;
			static constexpr size_t maxDirNameLength = maxFileNameLength - 10;

		private:
			archive_t* m_pArch;
			const bool m_bOwnArch;		

		protected:
			

			template<bool b=bVerbose>
			::std::enable_if_t<!b> _verbalize(const char*)const noexcept {}
			template<bool b = bVerbose>
			::std::enable_if_t<b> _verbalize(const char* s)const noexcept {
				STDCOUTL(_layer_name() << " - " << s);
			}

		private:
			void _make_archive(::std::nullptr_t pA)noexcept{
				NNTL_ASSERT(m_bOwnArch);
				m_pArch = new (::std::nothrow) archive_t;
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
				m_bDoDump = false;
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

			template<typename PArchT = ::std::nullptr_t>
			_dumper_base(const char* pDirToDump, PArchT pA=nullptr)noexcept  : m_bOwnArch(!pA) {
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

			/*template<bool b = cond_dump_t::bNewBatch2NewFile>
			::std::enable_if_t<b, const char*> _make_var_name(const char* szBaseName, char* dest, const size_t ds)const noexcept {
				return szBaseName;
			}
			template<bool b = cond_dump_t::bNewBatch2NewFile>
			::std::enable_if_t<!b, const char*> _make_var_name(const char* szBaseName, char* dest, const size_t ds)const noexcept {
				sprintf_s(dest, ds, "%s_%d", szBaseName, m_batchIdx);
				return dest;
			}*/

			enum class _ToDump {
				FProp=0,
				BProp,
				CalcErrTrainset,
				CalcErrTestset
			};

			template<bool b = bSplitFiles>
			::std::enable_if_t<b> _make_file_name(char* n, const size_t ml, const _ToDump omode)const noexcept {
				NNTL_ASSERT(!m_DirToDump.empty());
				if (m_DirToDump.empty()) _epic_fail("m_DirToDump is not set!");
				if (omode == _ToDump::FProp || omode == _ToDump::BProp) {
					//if (cond_dump_t::bNewBatch2NewFile) {
						sprintf_s(n, ml, omode == _ToDump::FProp ? "%s/ep%03zd_%df.mat" : "%s/ep%03zd_%db.mat"
							, m_DirToDump.c_str(), m_epochIdx + 1, m_batchIdx);
// 					} else {
// 						sprintf_s(n, ml, omode == _ToDump::FProp ? "%s/epoch%zdf.mat" : "%s/epoch%zdb.mat"
// 							, m_pDirToDump, m_epochIdx + 1);
// 					}
				} else {
					sprintf_s(n, ml, omode == _ToDump::CalcErrTrainset ? "%s/ep%03zd_train.mat" : "%s/ep%03zd_test.mat"
						, m_DirToDump.c_str(), m_epochIdx + 1);
				}
			}
			template<bool b = bSplitFiles>
			::std::enable_if_t<!b> _make_file_name(char* n, const size_t ml, const _ToDump omode)const noexcept {
				NNTL_ASSERT(!m_DirToDump.empty());
				if (m_DirToDump.empty()) _epic_fail("m_DirToDump is not set!");
				if (omode == _ToDump::FProp || omode == _ToDump::BProp) {
					//if (cond_dump_t::bNewBatch2NewFile) {
						sprintf_s(n, ml, "%s/ep%03zd_%d.mat", m_DirToDump.c_str(), m_epochIdx + 1, m_batchIdx);
// 					} else {
// 						sprintf_s(n, ml, "%s/epoch%zd.mat", m_pDirToDump, m_epochIdx + 1);
// 					}
				} else {
					sprintf_s(n, ml, omode == _ToDump::CalcErrTrainset ? "%s/ep%03zd_train.mat" : "%s/ep%03zd_test.mat"
						, m_DirToDump.c_str(), m_epochIdx + 1);
				}
			}

			self_ref_t _open_archive(const _ToDump omode)noexcept {
				char n[maxFileNameLength];
				_make_file_name(n, maxFileNameLength, omode);
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
			void on_train_preCalcError(const bool bOnTrainSet)const noexcept {}
			void on_train_postCalcError()const noexcept {}
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
				m_DirToDump = pDirName;
				if (m_DirToDump.length() > maxDirNameLength) _epic_fail(NNTL_FUNCTION, ": Too long directory name");
				return get_self();
			}

			template<typename S>
			self_ref_t set_dir_to_dump(S&& str)noexcept {
				NNTL_ASSERT(!str.empty());
				m_DirToDump = ::std::forward<S>(str);
				if (m_DirToDump.length() > maxDirNameLength) _epic_fail(NNTL_FUNCTION, ": Too long directory name");
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
				NNTL_ASSERT(!m_DirToDump.empty());
				NNTL_ASSERT(totalLayers && totalEpochs && totalBatches);
				m_layersCount = totalLayers;
				m_bDoDump = false;
				m_epochIdx = -1;
				m_batchIdx = -1;

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
				m_layerNames[lIdx].assign(::std::forward<StrT>(LayerName));
				_bwlist_updateLayer(lIdx, layerTypeId);
			};

			void train_epochBegin(const size_t epochIdx)noexcept {
				m_epochIdx = epochIdx;
				m_bDoDump = m_condDump.on_train_epochBegin(epochIdx);
			}
			void train_batchBegin(const vec_len_t batchIdx) noexcept {
				m_bDoDump = m_condDump.on_train_batchBegin(m_bDoDump, batchIdx, m_epochIdx);
				if (m_bDoDump) {
					m_batchIdx = batchIdx;
					STDCOUTL("Going to dump the training dataflow...");
					get_self()._open_archive(_ToDump::FProp)
						.on_train_batchBegin(batchIdx);
				}
			}

			void train_preBprop(const realmtx_t& data_y) noexcept {
				if (m_bDoDump) {
					if (bSplitFiles) get_self()._close_archive()
						._open_archive(_ToDump::BProp);

					get_self().on_train_preBprop(data_y);
				}
			}

			void train_batchEnd() noexcept {
				if (m_bDoDump) {
					get_self().on_train_batchEnd();
					get_self()._close_archive();
					m_bDoDump = false;
				}
			}

			void train_preCalcError(const bool bOnTrainSet) noexcept {
				m_bDoDump = m_condDump.on_train_calcErr(m_epochIdx, bOnTrainSet);
				if (m_bDoDump) {
					STDCOUTL("Going to dump the calc error dataflow...");
					get_self()._open_archive(bOnTrainSet ? _ToDump::CalcErrTrainset : _ToDump::CalcErrTestset);
					get_self().on_train_preCalcError(bOnTrainSet);
				}
			};
			void train_postCalcError()noexcept {
				if (m_bDoDump) {
					get_self().on_train_postCalcError();
					get_self()._close_archive();
					m_bDoDump = false;
				}
			};

			//////////////////////////////////////////////////////////////////////////
			// dumping functions implies that m_bDoDump is changed only during train_*() calls
			void fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) noexcept {
				//m_bDoDump = m_bDoDump && bTrainingMode;
				//we shouldn't update m_bDoDump here, because it's already set by train_batchBegin(). bTrainingMode
				// is guaranteed to be true when m_bDoDump==true
				//NNTL_ASSERT(!m_bDoDump || bTrainingMode);
				m_curLayer.push(lIdx);
				if (bDoDump(lIdx) && m_curLayer.bUpperLayerDifferent()) get_self().on_fprop_begin(lIdx, prevAct, bTrainingMode);
			}
			void fprop_end(const realmtx_t& act) noexcept {
				if (bDoDump(m_curLayer) && m_curLayer.bUpperLayerDifferent()) get_self().on_fprop_end(act);
				m_curLayer.pop();
			}
			void bprop_begin(const layer_index_t lIdx, const realmtx_t& dLdA) noexcept {
				m_curLayer.push(lIdx);
				if (bDoDump(lIdx) && m_curLayer.bUpperLayerDifferent()) get_self().on_bprop_begin(lIdx, dLdA);
			}
			void bprop_end(const realmtx_t& dLdAPrev) noexcept {
				if (bDoDump(m_curLayer) && m_curLayer.bUpperLayerDifferent()) get_self().on_bprop_end(dLdAPrev);
				m_curLayer.pop();
			}
		};

		//////////////////////////////////////////////////////////////////////////
		// _dumper class is more an *example* of how to use the _dumper_base<> class. Derive your own code from _dumper_base<>.
		// To make it universally useful, it should be somehow parametrized with an info which data it should dump and which one shouldn't
		// Without this parametrization one just have to comment/uncomment the necessary code...
		template<typename FinalChildT, typename RealT, typename ArchiveT, typename CondDumpT, size_t maxNnetDepth = 32>
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
			// NB: every _i_inspector's function implementation (they aren't staring with on_) should check if bDoDump(m_curLayer) first
			// NB2: there are some functions commented out below - their code is perfectly fine, just surplus to some task.
			// Feel free to restore it, or better derive your own class from _dumper_base<> with necessary functions to be independent of _dumper
			
			void on_fprop_begin(const layer_index_t lIdx, const realmtx_t& prevAct, const bool bTrainingMode) const noexcept {
				//NNTL_ASSERT(bTrainingMode);
				_verbalize("on_fprop_begin");
				auto& ar = getArchive();
				_check_err(ar.save_struct_begin(get_self()._layer_name(), false, bIgnoreLayersNesting), "on_fprop_begin: save_struct_begin");
			}
			void on_fprop_end(const realmtx_t& A)const noexcept {
				_verbalize("on_fprop_end");
				auto& ar = getArchive();
				ar & serialization::make_nvp("A",A);
				_check_err(ar.get_last_error(), "on_fprop_end: saving activations");

				_check_err(ar.save_struct_end(), "on_fprop_end: save_struct_end");
			}

			void fprop_preNesterovMomentum(const realmtx_t& vW, const real_t momentum, const realmtx_t& W)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("fprop_preNesterovMomentum");
					auto& ar = getArchive();
					ar & serialization::make_nvp("f_preNM_vW",vW);
					_check_err(ar.get_last_error(), "fprop_preNesterovMomentum: saving vW");
					ar & serialization::make_nvp("f_preNM_momentum", momentum);
					_check_err(ar.get_last_error(), "fprop_preNesterovMomentum: saving momentum");
				}
			}

			void fprop_makePreActivations(const realmtx_t& W, const realmtx_t& prevAct)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("fprop_makePreActivations");
					auto& ar = getArchive();
					ar & serialization::make_nvp("W", W);
					_check_err(ar.get_last_error(), "fprop_makePreActivations: saving W");
				}
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
			void on_bprop_end(const realmtx_t& dLdAPrev) const noexcept {
				_verbalize("on_bprop_end");
				_check_err(getArchive().save_struct_end(), "on_bprop_end: save_struct_end");
			}

			void bprop_dLdZ(const realmtx_t& dLdZ) const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("bprop_dLdZ");
					auto& ar = getArchive();
					ar & NNTL_SERIALIZATION_NVP(dLdZ);
					_check_err(ar.get_last_error(), "bprop_dLdZ: saving dLdZ");
				}
			}

			void apply_grad_begin(const realmtx_t& W, const realmtx_t& dLdW)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_begin");
					auto& ar = getArchive();
					ar & NNTL_SERIALIZATION_NVP(W);
					_check_err(ar.get_last_error(), "apply_grad_begin: saving W");

					ar & NNTL_SERIALIZATION_NVP(dLdW);
					_check_err(ar.get_last_error(), "apply_grad_begin: saving dLdW");
				}
			}

			void apply_grad_update(const realmtx_t& W, const realmtx_t& WUpd)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_update");
					auto& ar = getArchive();
					ar & serialization::make_nvp("a_WUpd_W",W);
					_check_err(ar.get_last_error(), "apply_grad_update: saving weights");

					ar & serialization::make_nvp("a_WUpd",WUpd);
					_check_err(ar.get_last_error(), "apply_grad_update: saving weight updates");
				}
			}

			void apply_grad_end(const realmtx_t& W)const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_end");
					auto& ar = getArchive();
					ar & serialization::make_nvp("a_W", W);
					_check_err(ar.get_last_error(), "apply_grad_end: saving weights");
				}
			}
			
			void apply_grad_preILR(const realmtx_t& dLdW, const realmtx_t& prevdLdW, const realmtx_t& Gain) const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_preILR");
					auto& ar = getArchive();
					/*ar & serialization::make_nvp("dLdW_preILR", dLdW);
					_check_err(ar.get_last_error(), "apply_grad_preILR: saving dLdW");*/
					ar & serialization::make_nvp("Gain_preILR", Gain);
					_check_err(ar.get_last_error(), "apply_grad_preILR: saving Gain");
				}
			}

			void apply_grad_postILR(const realmtx_t& dLdW, const realmtx_t& Gain) const noexcept {
				if (bDoDump(m_curLayer)) {
					_verbalize("apply_grad_postILR");
					auto& ar = getArchive();
// 					ar & serialization::make_nvp("dLdW_postILR", dLdW);
// 					_check_err(ar.get_last_error(), "apply_grad_postILR: saving dLdW");
					ar & serialization::make_nvp("Gain_postILR", Gain);
					_check_err(ar.get_last_error(), "apply_grad_postILR: saving Gain");
				}
			}
		};

		template<typename RealT, typename ArchiveT, typename CondDumpT = conds::EpochNum, size_t maxNnetDepth = 32>
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