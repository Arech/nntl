include(GoogleTest)
include("${REPO_ROOT}/build-support/cmake/test_utils.cmake")

enable_testing()

set(TESTS_NAME "nntl_tests")

add_executable("${TESTS_NAME}")

set_common_test_target_properties("${TESTS_NAME}")

target_include_directories("${TESTS_NAME}" PRIVATE include)

target_link_libraries("${TESTS_NAME}" PRIVATE "nntl")



gtest_discover_tests("${TESTS_NAME}")
