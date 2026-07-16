# -----------------------------------------------------------------------------
# Pyramid-Model-Core :: CMakeLists.txt
#
# Builds `htm.bindings.pyramid_engine` -- the C++ pyramid runtime -- against
# the stage-1 htm_core static library, exactly like the Single-HTM-Core
# binding modules (sdr / encoders / algorithms) that this file sits beside
# in the Python3 build stage.  Also installs the pure-Python support package
# (`htm.pyramid`) that feeds the engine batched data.
#
# Included from the root dispatcher ONLY when BINDING_BUILD=Python3;
# CommonCompilerConfig and HTM_CORE_LIB_DIR / CORE_LIB_INCLUDES are already
# set by the root scope.
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
project(pyramid_model_core CXX)

message(STATUS "Configuring Pyramid-Model-Core (htm.bindings.pyramid_engine)")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(htm_core_lib_path
    "${HTM_CORE_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}htm_core${CMAKE_STATIC_LIBRARY_SUFFIX}")
message(STATUS "pyramid: htm_core_lib_path = ${htm_core_lib_path}")

set(src_pyramid_files
    bindings/py_PyramidRuntime.cpp
    src/pyramid/merge/SdrMerge.cpp
    src/pyramid/encoders/DateMath.cpp
    src/pyramid/encoders/FeatureEncoders.cpp
    src/pyramid/encoders/EncoderBuild.cpp
    src/pyramid/data/CsvSource.cpp
    src/pyramid/runtime/HtmNode.cpp
    src/pyramid/runtime/PyramidRuntime.cpp
	src/pyramid/runtime/PipelineRun.cpp
    src/pyramid/threading/WorkerPool.cpp
    )

set(pyramid_shared_lib pyramid_engine)
pybind11_add_module(${pyramid_shared_lib} ${src_pyramid_files})
target_link_libraries(${pyramid_shared_lib} PRIVATE
        ${htm_core_lib_path}
        ${COMMON_OS_LIBS}
        Threads::Threads
        )
target_compile_options(${pyramid_shared_lib} PUBLIC ${INTERNAL_CXX_FLAGS})
# The pyramid's numpy-compat paths (softmax, quantile, cumulative sums)
# reproduce numpy's double-precision results bit-for-bit. FMA contraction
# (enabled by -march levels with FMA) fuses a*b+c into a single rounding,
# which numpy's own binaries do not do in these code paths -- so contraction
# is disabled for THIS target only. MSVC defaults to /fp:precise (no
# contraction), so no flag is needed there.
# Pyramid-scope link-time optimization (see HTM_LTO at the repo root):
# this target's TUs are LTO'd together; the htm_core library is linked
# as-is unless HTM_LTO=FULL.
if(HTM_LTO_PYRAMID)
	set_property(TARGET ${pyramid_shared_lib} PROPERTY
	             INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
if(NOT MSVC)
	# Bind this module's references to its OWN copy of the statically
	# linked htm_core (and do not export those symbols): other Python
	# extensions in the same process (htm.bindings.algorithms) carry
	# their own copy, and cross-module symbol interposition between
	# differently-built copies corrupts memory.
	target_link_options(${pyramid_shared_lib} PRIVATE
	                    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)
	target_compile_options(${pyramid_shared_lib} PRIVATE -ffp-contract=off)
	# LTO re-optimizes at link; keep contraction off there as well.
	target_link_options(${pyramid_shared_lib} PRIVATE -ffp-contract=off)
endif()
target_compile_definitions(${pyramid_shared_lib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${pyramid_shared_lib} PRIVATE
        ${CORE_LIB_INCLUDES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        SYSTEM ${PYTHON_INCLUDE_DIRS}
        SYSTEM ${pybind11_INCLUDE_DIRS}
        )

# The compiled engine joins the other binding modules under htm/bindings.
install(TARGETS ${pyramid_shared_lib}
        DESTINATION htm/bindings COMPONENT python)

# The pure-Python support package installs as htm.pyramid (DatasetReader,
# the batched feeder consumed by the engine's PyReaderSource).
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/py/pyramid_support/dataset_reader.py
        DESTINATION htm/pyramid COMPONENT python)
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/py/pyramid_support/__init__.py
        DESTINATION htm/pyramid COMPONENT python)
