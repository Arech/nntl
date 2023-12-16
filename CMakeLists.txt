cmake_minimum_required(VERSION 3.25.1)

include(maintenance/cmake/basics.cmake)

option_list(NNTL_BUILD_TESTS ON "Whether to build tests." OFF)
#option_list(NNTL_BUILD_UNITBENCHS ON "Whether to build unit benchmarks." OFF)
#option_list(NNTL_BUILD_EXAMPLES ON "Whether to build examples." OFF)
option_list(NNTL_BUILD_UNITBENCHS OFF "Whether to build unit benchmarks." ON)
option_list(NNTL_BUILD_EXAMPLES OFF "Whether to build examples." ON)

# vcpkg features should be selected before the first project() call
if(NOT DEFINED VCPKG_MANIFEST_FEATURES)
    set(VCPKG_MANIFEST_FEATURES "")
endif()

if(NNTL_BUILD_TESTS)
    list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()
if(NNTL_BUILD_UNITBENCHS)
    list(APPEND VCPKG_MANIFEST_FEATURES "benchmarks")
endif()

message(STATUS "Selected vcpkg features: ${VCPKG_MANIFEST_FEATURES}")

##################################
project("nntl-dev" DESCRIPTION "NNTL dev framework" VERSION "1.0.0")

generic_compiler_setup()

include(GNUInstallDirs) # it's also mandatory for proper vcpkg porting!
include(CMakePackageConfigHelpers)

#project wide packages
find_package(OpenBLAS CONFIG REQUIRED)
#find_package(Microsoft.GSL CONFIG REQUIRED)

# Project-wide definitions
#add_compile_definitions()

add_subdirectory(include)

if(NNTL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NNTL_BUILD_UNITBENCHS)
    add_subdirectory(unit-benchmarks)
endif()

if(NNTL_BUILD_EXAMPLES)
    message(FATAL_ERROR "building examples isn't supported yet")
    add_subdirectory(examples)
endif()
