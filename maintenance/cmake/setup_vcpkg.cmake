# This script prepares CMake & environment to use vcpkg.
# vcpkg doesn't have to be preinstalled, however, if it is, set VCPKG_ROOT
# environment variable to an absolute path to the vcpkg topmost folder.
#
# Discovery algo:
# - If VCPKG_ROOT env variable is found, this script will use it. Otherwise:
# - will try to read path from ../scripts/vcpkg_env.path, otherwise:
# - execute ../scripts/setup_vcpkg.sh and use environment setup files it'll create.

if( (NOT DEFINED ENV{VCPKG_ROOT}) OR ("$ENV{VCPKG_ROOT}" STREQUAL "") )
	message(STATUS "VCPKG_ROOT environment variable not found. Trying to use path file ../scripts/vcpkg_env.path")

	# sanity check
	if(DEFINED VCPKG_ROOT)
		message(FATAL_ERROR "Variable VCPKG_ROOT shouldn't be defined here!")
	endif()

	# reading from file
	set(MY_PATH_FILE "${CMAKE_CURRENT_LIST_DIR}/../scripts/vcpkg_env.path")
	if(EXISTS "${MY_PATH_FILE}")
		file(STRINGS "${MY_PATH_FILE}" VCPKG_ROOT LIMIT_COUNT 1)
	endif()

	if( (NOT DEFINED VCPKG_ROOT) OR ("${VCPKG_ROOT}" STREQUAL "") )
		message(STATUS "Path file not found. Executing scripts/setup_vcpkg.sh")

		execute_process(COMMAND "${CMAKE_CURRENT_LIST_DIR}/../scripts/setup_vcpkg.sh"
			RESULT_VARIABLE INSTALL_VCPKG_RESULT)

		if (NOT "${INSTALL_VCPKG_RESULT}" STREQUAL "0")
			message(FATAL_ERROR "scripts/setup_vcpkg.sh returned error: ${INSTALL_VCPKG_RESULT}")
		endif()

		if (NOT EXISTS "${MY_PATH_FILE}")
			message(FATAL_ERROR "Can't find ${MY_PATH_FILE} path file")
		endif()
		set(VCPKG_ROOT "")
		file(STRINGS "${MY_PATH_FILE}" VCPKG_ROOT LIMIT_COUNT 1)

		if( (DEFINED VCPKG_ROOT) AND (NOT "${VCPKG_ROOT}" STREQUAL ""))
			message(STATUS "Using installed VCPKG_ROOT=${VCPKG_ROOT}")
			set(ENV{VCPKG_ROOT} "${VCPKG_ROOT}")
		else()
			message(FATAL_ERROR "scripts/setup_vcpkg.sh made wrong path file ${MY_PATH_FILE}")
		endif()
	else()
		message(STATUS "Read VCPKG_ROOT=${VCPKG_ROOT}")
		set(ENV{VCPKG_ROOT} "${VCPKG_ROOT}")
	endif()

	# cleanup
	unset(MY_PATH_FILE)
	if(DEFINED VCPKG_ROOT)
		unset(VCPKG_ROOT)
	endif()
else()
	message(STATUS "Using pre-set VCPKG_ROOT=$ENV{VCPKG_ROOT}")
endif()

# Configure default CMake toolchain file for vcpkg if one is available before
# the top-level `project()` call.
set(VCPKG_TOOLCHAIN "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
if( (DEFINED CMAKE_TOOLCHAIN_FILE) AND (NOT "${VCPKG_TOOLCHAIN}" STREQUAL "${CMAKE_TOOLCHAIN_FILE}") )
	message(FATAL_ERROR "pre-set CMAKE_TOOLCHAIN_FILE is not supported (defined to ${CMAKE_TOOLCHAIN_FILE})")
endif()

if (EXISTS "${VCPKG_TOOLCHAIN}")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_TOOLCHAIN}" CACHE STRING "")
else()
	message(FATAL_ERROR "ENV{VCPKG_ROOT} does not correspond to a valid VCPKG directory")
endif()
unset(VCPKG_TOOLCHAIN)

