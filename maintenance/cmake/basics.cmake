# a collection of CMake instructions to make the CMake use less stressfull.
#
# Note: to debug cmake, always delete ./build directory first and then
# pass `--trace` or `--trace-expand` flag to a configuration command.
#
# To debug `vcpkg`, pass `-DVCPKG_INSTALL_OPTIONS=--debug` flag.
# To include env vars into the dump, add `--debug-env` flag
#
# Since it may add support of `vcpkg`, the file must be included before the
# first call to `project()`

include("${CMAKE_CURRENT_LIST_DIR}/option_list.cmake")

#including now, but note it must be called strictly after the first project() call
include("${CMAKE_CURRENT_LIST_DIR}/generic_compiler_setup.cmake")

option_list(ENABLE_CLANG_TIDY OFF "Enable clang-tidy linting" ON)
if(ENABLE_CLANG_TIDY)
    message(STATUS "Enabling clang-tidy validation")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy-12;-extra-arg=-std=c++23")
endif()

# CMake sets CMAKE_BUILD_TYPE to Debug (with a command set(CMAKE_BUILD_TYPE_INIT Debug)) if no
# "-DCMAKE_BUILD_TYPE=.." CLI parameter is specified only for some platforms
# (https://github.com/Kitware/CMake/search?q=CMAKE_BUILD_TYPE_INIT) but not for the other including
# Linux. So it breaks every possible $<CONFIG> generator expressions and every other stuff, that
# relies on CMAKE_BUILD_TYPE variable having any value set.

# Note that quoting of the first two arguments seem unnecessary, but there was very strange
# unexplained error observed with unquoted form in cmake 3.16 & 3.21 on some machine that wasn't
# replicatable on others, so it's just simpler to set it.
option_list("CMAKE_BUILD_TYPE" "Debug" "Set build type" "Release;MinSizeRel;RelWithDebInfo")

if (NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build type is '${CMAKE_BUILD_TYPE}'.")
endif()

## setting some common variables for all code built within the repo
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
# If GCC complains it can't make library without -fPIC, turing it on for a concrete local target
# with `set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)`
# Generally PIC is desirable for shared libraries. Executables don't necessary need it, and
# omitting it could make a compiler to produce a better and more efficient code:
# According to https://stackoverflow.com/a/31332516 : "On some architectures,
# including x86, -fPIC generates much worse code (i.e. a function call) for
# loads/stores of data. While this is tolerable for libraries, it is
# undesirable for executables.", therefore it's better to do that precisely for the library instead
# of global `set(CMAKE_POSITION_INDEPENDENT_CODE ON)`

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)

include("${CMAKE_CURRENT_LIST_DIR}/setup_vcpkg.cmake")

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" REALPATH)
# note, ^^ yield no slash `/` at the end of the path
