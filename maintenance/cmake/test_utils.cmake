# Sets various properties common to test targets.
#

# this function is for internal use, one probably never need to call it themself
function(set_common_google_project_props target)
  # GTest makes use of variadic macros called with empty argument lists, which
  # triggers the `-Wgnu-zero-variadic-macro-arguments` warning when building
  # with Clang as it's technically not supported by the C/C++ standard (but is
  # supported by most compilers), so we have to disable the warning when
  # building with Clang.
  if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    target_compile_options("${target}" PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
  endif()

  if(ENABLE_CLANG_TIDY)
    # - Code expanded from GoogleTest and gMock macros trigger a number of
    #   clang-tidy checks, so disable those checks for the test target.
    # - Disable magic number lint checks to make writing test code easier.
    set_target_properties("${target}"
      PROPERTIES
        CXX_CLANG_TIDY "clang-tidy-12;-extra-arg=-std=c++${CMAKE_CXX_STANDARD};--checks=-cppcoreguidelines-owning-memory,-cppcoreguidelines-special-member-functions,-hicpp-special-member-functions,-*-non-private-member-variables-in-classes,-*-magic-numbers,-*-avoid-goto,-cppcoreguidelines-avoid-non-const-global-variables"
    )
  endif()

  if(ENABLE_CPPCHECK)
    # cppcheck 1.90 emits a generic "syntax error" warning on GoogleTest macros
    set_target_properties("${target}" PROPERTIES CXX_CPPCHECK "")
  endif()
endfunction()

# ! Note ! GTest could be used in multiple contexts, including those that shouldn't be executed in
# the context of ctest.
# Don't add into `set_common_test_target_properties()` and `target_link_gmock()` functions any
# additional functionality that implies something beyond building a binary, that uses GTest.
#
# Params:
# - target  Name of the test target to modify.
function(set_common_test_target_properties target)
  find_package(GTest REQUIRED)
  find_package(Threads REQUIRED)

  # Add common GTest library targets.
  #
  # GTest 1.11.0 introduced namespaced library targets, e.g., `GTest::gtest`.
  # Older non-namespaced targets are still available, but vcpkg does not export
  # these, so we have to explicitly support either possibility.
  if(TARGET GTest::gtest)
    target_link_libraries("${target}" PRIVATE GTest::gtest_main GTest::gtest)
  else()
    target_link_libraries("${target}" PRIVATE gtest_main gtest)
  endif()

  target_link_libraries("${target}" PRIVATE Threads::Threads)

  set_common_google_project_props("${target}")
endfunction()

# Params:
# - target  Name of the benchmark target to modify.
function(set_common_benchmark_target_properties target)
  find_package(benchmark REQUIRED)

  # Newer benchmark lib introduced namespaced library targets, e.g., 
  # `benchmark::benchmark`.
  if(TARGET benchmark::benchmark)
    target_link_libraries("${target}" PRIVATE benchmark::benchmark_main benchmark::benchmark)
  else()
    target_link_libraries("${target}" PRIVATE benchmark_main benchmark)
  endif()

  set_common_google_project_props("${target}")
endfunction()


# Adds gMock to the link libraries for a test target.
#
# ! Note ! see the note to set_common_test_target_properties() function
#
# Params:
# - target  Name of the test target to modify.
function(target_link_gmock target)
  if(TARGET GTest::gmock)
    target_link_libraries("${target}" PRIVATE GTest::gmock_main GTest::gmock)
  else()
    target_link_libraries("${target}" PRIVATE gmock_main gmock)
  endif()
endfunction()
