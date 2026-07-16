# -----------------------------------------------------------------------------
# Single-HTM-Core -- STANDALONE build entry point.
#
# This file makes the Single-HTM-Core engine buildable on its own, with this
# directory as the CMake source root:
#
#     cmake -S Single-HTM-Core -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build
#
# It is intended for using / contributing the single-HTM engine independently
# (e.g. a pull request back to upstream htm.core). It is NOT used by
# PyHTM-core's own build: htm_install.py drives the repository-root
# CMakeLists.txt, which compiles Single-HTM-Core/src and the bindings
# directly and never configures this file. The two paths are independent.
#
# Copies of CommonCompilerConfig.cmake and DetectMarch.cmake live in this
# directory so the subtree carries everything it needs to compile alone.
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
project(htm_core CXX C)

option(BINDING_BUILD "Python3 to build pybind11 modules, CPP_Only for the C++ library only" "CPP_Only")

# This directory is both the module path (for CommonCompilerConfig /
# DetectMarch / external modules) and the source root the sub-CMakeLists
# anchor on via PROJECT_SOURCE_DIR.
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(REPOSITORY_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set_property(GLOBAL PROPERTY REPOSITORY_DIR "${REPOSITORY_DIR}")
cmake_policy(SET CMP0135 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Version: prefer an explicit -DPROJECT_VERSION1=..., else the VERSION file,
# else a placeholder. (Upstream htm.core supplies this via its own flow.)
if(DEFINED PROJECT_VERSION1)
  set(VERSION ${PROJECT_VERSION1})
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION)
  string(STRIP "${VERSION}" VERSION)
else()
  set(VERSION "0.0.0")
endif()
message(STATUS "Single-HTM-Core standalone build -- VERSION ${VERSION}")

# Shared toolchain flags + adaptive SIMD (same modules the repo root uses).
include(CommonCompilerConfig)

set(core_library htm_core)
if(MSVC)
  set(src_lib_shared)
else()
  set(src_lib_shared htm_core_shared)
endif()

# External dependencies (cereal, MurmurHash3) -- anchored on REPOSITORY_DIR.
# (The repo-root build uses external/bootstrap.cmake, which hardcodes the
# nested layout; standalone we include the same two modules directly so the
# subtree needs no path edits.)
include(${CMAKE_CURRENT_SOURCE_DIR}/external/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/external/cereal.cmake)
set(EXTERNAL_INCLUDES "")
list(APPEND EXTERNAL_INCLUDES "${cereal_INCLUDE_DIR}")
list(APPEND EXTERNAL_INCLUDES "${common_INCLUDE_DIR}")

# The C++ library itself.
add_subdirectory(src)

# Optionally also build the pybind11 modules against the just-built library.
if(BINDING_BUILD STREQUAL "Python3")
  set(HTM_CORE_LIB_DIR "${CMAKE_BINARY_DIR}/Release/lib"
      CACHE PATH "Path to the built htm_core library")
  list(APPEND CORE_LIB_INCLUDES "${CMAKE_BINARY_DIR}/Release/include")
  add_subdirectory(bindings/py/cpp_src)
endif()
