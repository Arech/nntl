include(GoogleTest)
include("${REPO_ROOT}/maintenance/cmake/test_utils.cmake")

enable_testing()

set(TESTS_NAME "nntl_tests")

# find ./src/ -maxdepth 10 -type f | sed 's/^\.\///'
add_executable("${TESTS_NAME}"
	#src/tests.cpp
	src/test_activations.cpp
	#src/test_binfile.cpp
	#src/test_b_open_blas.cpp
	#src/test_imath_basic.cpp
	#src/test_imath_basic_thr.cpp
	#src/test_inspectors.cpp
	#src/test_jsonreader.cpp
	#src/test_layer_pack_horizontal.cpp
	#src/test_layer_pack_horizontal_optional.cpp
	#src/test_layer_pack_tile.cpp
	#src/test_layer_pack_vertical.cpp
	#src/test_layer_penalized_activations.cpp
	#src/test_matfile.cpp
	#src/test_nnet.cpp
	#src/test_perf_decisions.cpp
	#src/test_rng.cpp
	#src/test_selu.cpp
	#src/test_simple_math.cpp
	#src/test_simple_math_thr.cpp
	#src/test_simple_matrix.cpp
	#src/test_threading.cpp
	#src/test_util.cpp
)

set_common_test_target_properties("${TESTS_NAME}")

target_include_directories("${TESTS_NAME}" PRIVATE include)

target_link_libraries("${TESTS_NAME}" PRIVATE "nntl")



gtest_discover_tests("${TESTS_NAME}")
