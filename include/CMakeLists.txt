# TODO: get rid of matlab dependency
option_list("NNTL_MATLAB_AVAILABLE" OFF "Is Matlab binaries installed" ON)

project("nntl" DESCRIPTION "Neural Network Templates Library" VERSION "1.0.0")

add_library("${PROJECT_NAME}" INTERFACE)

target_compile_definitions("${PROJECT_NAME}" INTERFACE
	NNTL_MATLAB_AVAILABLE=$<BOOL:${NNTL_MATLAB_AVAILABLE}>)

target_include_directories("${PROJECT_NAME}" INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries("${PROJECT_NAME}" INTERFACE OpenBLAS::OpenBLAS)

# find ./nntl/ -maxdepth 10 -type f -not -path '*/*\.txt' | sed 's/^\.\///'
target_sources("${PROJECT_NAME}" INTERFACE FILE_SET HEADERS BASE_DIRS nntl FILES 
	nntl/activation.h
	nntl/activations/elogu.h
	nntl/activations/elu.h
	nntl/activations/identity.h
	nntl/activations/loglogu.h
	nntl/activations/logu.h
	nntl/activations/relu_family.h
	nntl/activations/selu.h
	nntl/activations/sigm.h
	nntl/activations/softmax.h
	nntl/activations/softsigm.h
	nntl/activations/softsign.h
	nntl/activations/step.h
	nntl/activations/_i_activation.h
	nntl/activations/_loss_parts.h
	nntl/common.h
	nntl/common_nn_data.h
	nntl/compiler.h
	nntl/dropout/alpha_dropout.h
	nntl/dropout/dropout.h
	nntl/dropout/_base.h
	nntl/dropout.h
	nntl/errors.h
	nntl/grad_works/grad_works.h
	nntl/grad_works/ILR.h
	nntl/grad_works/ILR_Props.h
	nntl/grad_works/loss_addendums.h
	# nntl/interface/imemmgr/imemmgr.h  #pending for remove
	nntl/interface/inspectors/dummy.h
	nntl/interface/inspectors/dumper.h
	nntl/interface/inspectors/gradcheck.h
	nntl/interface/inspectors/stdcout.h
	# nntl/interface/math/bindings/b_acml.h  #pending for remove
	nntl/interface/math/bindings/b_open_blas.h
	nntl/interface/math/mathn.h
	nntl/interface/math/mathn_mt.h
	nntl/interface/math/mathn_thr.h
	nntl/interface/math/smath.h
	nntl/interface/math/smath_thr.h
	nntl/interface/math/smatrix.h
	nntl/interface/math/_base.h
	nntl/interface/math/_mcwFindKOrdered_hlpr.h
	# nntl/interface/mt_dispatcher/mt_dispatcher.h   # not finished
	# nntl/interface/mt_dispatcher/profiler.h        # not finished
	nntl/interface/rng/afrand.h
	nntl/interface/rng/afrand_as.h
	nntl/interface/rng/afrand_mt.h
	nntl/interface/rng/afrand_mt_thr.h
	nntl/interface/rng/cstd.h
	nntl/interface/rng/distr_normal_naive.h
	nntl/interface/threads/bgworkers.h
	nntl/interface/threads/parallel_range.h
	nntl/interface/threads/prioritize_workers.h
	nntl/interface/threads/workers.h
	nntl/interface/threads/_sync_primitives.h
	nntl/interface/_i_bgworkers.h
	nntl/interface/_i_imemmgr.h
	nntl/interface/_i_inspector.h
	nntl/interface/_i_math.h
	nntl/interface/_i_rng.h
	nntl/interface/_i_threads.h
	nntl/interfaces.h
	nntl/layer/extender.h
	nntl/layer/extensions.h
	nntl/layer/fully_connected.h
	nntl/layer/identity.h
	nntl/layer/input.h
	nntl/layer/output.h
	nntl/layer/pack_horizontal.h
	nntl/layer/pack_horizontal_optional.h
	nntl/layer/pack_tile.h
	nntl/layer/pack_vertical.h
	nntl/layer/_activation_storage.h
	nntl/layer/_activation_wrapper.h
	nntl/layer/_init_layers.h
	nntl/layer/_layer_base.h
	nntl/layer/_lph_base.h
	nntl/layer/_pack_.h
	nntl/layer/_pack_traits.h
	nntl/layer/_penalized_activations_base.h
	nntl/layer/_tuple_utils.h
	nntl/layers.h
	nntl/loss_addendum/DeCov.h
	nntl/loss_addendum/L1.h
	nntl/loss_addendum/L2.h
	nntl/loss_addendum/_i_loss_addendum.h
	nntl/math_details.h
	nntl/nnet.h
	nntl/nnet_evaluator.h
	nntl/nnet_train_opts.h
	nntl/nntl.h
	nntl/serialization/options.h
	nntl/serialization/serialization.h
	nntl/training_observer.h
	nntl/train_data/inmem_train_data.h
	nntl/train_data/inmem_train_data_stor.h
	nntl/train_data/seq_data.h
	nntl/train_data/transf_train_data.h
	nntl/train_data/_i_train_data.h
	nntl/train_data/_td_base.h
	nntl/train_data/_td_norm.h
	nntl/train_data/_train_data_simple.h
	nntl/train_data.h
	nntl/utils/bwlist.h
	nntl/utils/call_wrappers.h
	nntl/utils/chrono.h
	nntl/utils/confusion_mtx.h
	nntl/utils/dataHolder.h
	nntl/utils/data_buffer.h
	nntl/utils/denormal_floats.h
	nntl/utils/epochs_props.h
	nntl/utils/forwarder.h
	nntl/utils/gradcheck.h
	nntl/utils/lambdas.h
	nntl/utils/layers_settings.h
	nntl/utils/layer_idx_keeper.h
	nntl/utils/matlab.h
	nntl/utils/mixins.h
	nntl/utils/mtx2Normal.h
	nntl/utils/options.h
	nntl/utils/own_or_use_ptr.h
	nntl/utils/scope_exit.h
	nntl/utils/tictoc.h
	nntl/utils/tuple_utils.h
	nntl/utils/variant.h
	nntl/utils/vector_conditions.h
	nntl/utils.h
	nntl/weights_init/LsuvExt.h
	nntl/weights_init/static.h
	nntl/weights_init.h
	nntl/_defs.h
	nntl/_nnet_errs.h
	nntl/_SNN_common.h
	nntl/_supp/io/binfile.h
	nntl/_supp/io/jsonreader.h
	nntl/_supp/io/matfile.h
	nntl/_test/functions.h
	nntl/_test/test.h
	nntl/_test/test_weights_init.h
)

install(TARGETS "${PROJECT_NAME}" EXPORT "${PROJECT_NAME}Export"
  FILE_SET HEADERS DESTINATION include/nntl)

install(EXPORT "${PROJECT_NAME}Export" NAMESPACE nntl::
  DESTINATION lib/cmake/nntl)
