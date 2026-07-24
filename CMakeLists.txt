# -----------------------------------------------------------------------------
# PyHTM-core -- root build switchboard
#
# This repository holds TWO engines:
#
#   Single-HTM-Core/     the single-HTM machinery (SDR, SpatialPooler,
#                        TemporalMemory, Connections, encoders) -- the trimmed
#                        and optimized fork of Numenta's htm.core.
#
#   Pyramid-Model-Core/  the pyramid runtime: full multi-layer pyramid
#                        construction, encoding, merging, thread-hive
#                        execution and record streaming, all in C++.
#
# The build is driven exactly as before (htm_install.py + scikit-build-core),
# in two stages that both enter through THIS file:
#
#   Stage 1 (htm_install.py):   cmake -DBINDING_BUILD=CPP_Only
#       -> compiles the Single-HTM-Core static library into build/Release/.
#   Stage 2 (python -m build):  cmake -DBINDING_BUILD=Python3
#       -> compiles the pybind11 modules for BOTH cores against that library
#          and packages them into the `htm` wheel:
#              htm.bindings.sdr / .encoders / .algorithms   (Single-HTM-Core)
#              htm.bindings.pyramid                          (Pyramid-Model-Core)
#
# Nothing about the entry points changed: `python htm_install.py` (or PyHTM's
# main-setup.py, which wraps it) still performs the whole build.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.  See LICENSE.txt.
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)

project(htm LANGUAGES C CXX)

# Optional link-time optimization (opt-in, default OFF -- zero effect unless
# explicitly requested; lets Release builds inline across translation units,
# e.g. the Connections <-> SP/TM hot loops). Enable with -DHTM_LTO=ON on
# either build stage; applies to Release configurations only.
# Link-time (interprocedural) optimization. AUTO (the default) probes the
# toolchain with CMake's IPO check and enables LTO for Release builds of
# BOTH engines when supported -- cross-translation-unit and cross-library
# inlining (the pyramid runtime into htm_core's hot loops) with no source
# changes. Explicit ON forces it (configure fails if unsupported); OFF
# disables. FP semantics are preserved: no fast-math anywhere, and the
# pyramid target keeps -ffp-contract=off through compile AND link.
set(HTM_LTO "AUTO" CACHE STRING "Link-time optimization: AUTO|FULL|OFF")
set_property(CACHE HTM_LTO PROPERTY STRINGS AUTO FULL OFF)
if(DEFINED ENV{HTM_LTO} AND HTM_LTO STREQUAL "AUTO")
  set(HTM_LTO "$ENV{HTM_LTO}")
endif()
string(TOUPPER "${HTM_LTO}" _htm_lto)
set(HTM_LTO_PYRAMID OFF)
if(_htm_lto STREQUAL "OFF")
  message(STATUS "HTM_LTO: OFF")
else()
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _htm_ipo_ok OUTPUT _htm_ipo_msg LANGUAGES CXX)
  if(NOT _htm_ipo_ok)
    message(STATUS "HTM_LTO: disabled (toolchain lacks IPO support)")
  elseif(_htm_lto STREQUAL "FULL")
    # Whole-program LTO including the htm_core library. NOTE: new
    # cross-function inlining inside the library changes its floating-
    # point results at the ~1-ULP level (FP contraction at fresh inline
    # sites), so fixed-seed digests of the single core will differ from
    # non-FULL builds. Use only when re-baselining digests is acceptable.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(HTM_LTO_PYRAMID ON)
    message(STATUS "HTM_LTO: FULL (library + pyramid; library float "
                   "digests differ from non-FULL builds)")
  else()  # AUTO
    # Pyramid-scope LTO only: the pyramid runtime's translation units are
    # link-time-optimized together, while the htm_core library is built
    # exactly as before -- its binary behavior (and fixed-seed digests)
    # is unchanged. Verified bit-exact against the Python reference.
    set(HTM_LTO_PYRAMID ON)
    message(STATUS "HTM_LTO: AUTO -> pyramid-scope LTO "
                   "(library codegen unchanged)")
  endif()
endif()


# CommonCompilerConfig.cmake (shared toolchain flags for both cores) lives at
# the repository root; the single-core area is registered as REPOSITORY_DIR
# for the external/*.cmake modules that anchor on it.
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(REPOSITORY_DIR "${CMAKE_SOURCE_DIR}/Single-HTM-Core" ABSOLUTE)
set_property(GLOBAL PROPERTY REPOSITORY_DIR "${REPOSITORY_DIR}")
cmake_policy(SET CMP0135 NEW)   # sets timestamp downloading along with dependancy download.

if(BINDING_BUILD STREQUAL "Python2")
  message(FATAL_ERROR "The Extension library cannot be built for Python 2 using MSVC.  Use Python 3.x")
endif()

message(STATUS "REPOSITORY_DIR       = ${REPOSITORY_DIR}")
message(STATUS "CMAKE_BINARY_DIR     = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_BUILD_TYPE     = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "SKBUILD_DATA_DIR     = ${SKBUILD_DATA_DIR}")
message(STATUS "SKBUILD_PLATLIB_DIR  = ${SKBUILD_PLATLIB_DIR}")
message(STATUS "CMake Version:       = ${CMAKE_VERSION}")
message(STATUS "BINDING_BUILD        = ${BINDING_BUILD}")


IF (BINDING_BUILD STREQUAL "CPP_Only" )
    # ------------------------------------------------------------------
    # STAGE 1: compile the Single-HTM-Core C++ engine as a static library
    # into <root>/build/Release/ (lib + headers). Identical to the previous
    # single-folder layout, just rooted under Single-HTM-Core/.
    # ------------------------------------------------------------------

    # define ${core_library} for the entire project.
    set(core_library   htm_core)
	if(MSVC)
	    set(src_lib_shared)
	else()
	    set(src_lib_shared htm_core_shared)
	endif()

	# get the project version from pyproject.toml and write it to the file VERSION.
	# NOTE: this is the VERSION in pyproject.toml.
	#       copy this to the VERSION file in the project source.
	set(VERSION ${PROJECT_VERSION1})
	file(WRITE "${CMAKE_SOURCE_DIR}/VERSION" "${VERSION}")

	message(STATUS "VERSION              = ${VERSION}")
	message("")

	# Determine common toolchain settings, compiler and link flags
	include(CommonCompilerConfig)

	message(STATUS "CMAKE_CXX_COMPILER_ID= ${CMAKE_CXX_COMPILER_ID}")
	message(STATUS "CMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")
	message(STATUS "CMAKE_CXX_COMPILER   = ${CMAKE_CXX_COMPILER}")
	message(STATUS "CMAKE_CXX_STANDARD   = ${CMAKE_CXX_STANDARD}")
	message(STATUS "PLATFORM             = ${PLATFORM}")
	message(STATUS "BITNESS              = ${BITNESS}")
	message(STATUS "INTERNAL_CXX_FLAGS   = ${INTERNAL_CXX_FLAGS}")
	message(STATUS "INTERNAL_LINKER_FLAGS= ${INTERNAL_LINKER_FLAGS}")
	message(STATUS "INTERNAL_LINKER_FLAGS_STR= ${INTERNAL_LINKER_FLAGS_STR}")
	message(STATUS "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
	message(STATUS "COMMON_COMPILER_DEFINITIONS_STR=${COMMON_COMPILER_DEFINITIONS_STR}")
	message(STATUS "COMMON_OS_LIBS       = ${COMMON_OS_LIBS}")

	# Set up builds of external dependencies and get their exports.
	# (see individual external/*.cmake modules for exported settings and functions)
	include(Single-HTM-Core/external/bootstrap.cmake)

	# Build the C++ library
	add_subdirectory(Single-HTM-Core/src)
else()
    # ------------------------------------------------------------------
    # STAGE 2: compile the pybind11 extension modules for BOTH cores
    # against the stage-1 static library, for packaging into the wheel.
    # ------------------------------------------------------------------

	message(STATUS "VERSION              = ${VERSION}")
	# Determine common toolchain settings, compiler and link flags
	include(CommonCompilerConfig)
	set(HTM_CORE_LIB_DIR "${CMAKE_SOURCE_DIR}/build/Release/lib" CACHE PATH "Path to the built htm_core library")
    list(APPEND CORE_LIB_INCLUDES "${CMAKE_SOURCE_DIR}/build/Release/include")

	message(STATUS "INTERNAL_CXX_FLAGS   = ${INTERNAL_CXX_FLAGS}")
	message(STATUS "INTERNAL_LINKER_FLAGS= ${INTERNAL_LINKER_FLAGS}")
    message(STATUS "HTM_CORE_LIB_DIR     = ${HTM_CORE_LIB_DIR}")

	# Build the Python extension interface using pybind11.
	# Single-HTM-Core: htm.bindings.sdr / .encoders / .algorithms
	add_subdirectory(Single-HTM-Core/bindings/py/cpp_src)
	# Pyramid-Model-Core: htm.bindings.pyramid_engine + htm.pyramid
	add_subdirectory(Pyramid-Model-Core)
endif()