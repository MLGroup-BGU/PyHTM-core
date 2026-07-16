#!/usr/bin/env python3
"""
Single-HTM-Core :: standalone build/install script.

Compiles the single-HTM engine (Spatial Pooler, Temporal Memory,
Connections, anomaly, SDR, encoders) on its own, with this directory as the
CMake source root. Automatic decisions -- no manual flags required:

  * SIMD level (``HTM_MARCH=auto``): the CMake probe (CPUID + XGETBV, see
    ``DetectMarch.cmake``) selects x86-64-v2/v3/v4 for the build host; the
    library is capped at v3 for numeric invariance, with an AVX-512 integer
    kernel opting in per-file on v4 hosts.
  * OS / compiler: CMake auto-detects the platform toolchain (MSVC on
    Windows, gcc/clang on Linux/macOS).

Usage:
    python htm_install.py                 # build the C++ library only
    python htm_install.py --bindings      # also build the Python modules
    python htm_install.py --march v3      # force a SIMD level (override auto)
    python htm_install.py --build-type Debug

This script is NOT used by PyHTM-core's own build (that drives the
repository-root CMakeLists). It exists so the Single-HTM-Core subtree can be
built -- and contributed upstream (e.g. a PR to htm.core) -- on its own.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")


def _run(cmd, **kw):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def _have(exe):
    return shutil.which(exe) is not None


def preflight():
    print("== Single-HTM-Core build: pre-flight checks ==")
    print(f"   python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"   platform : {sys.platform}")
    if not _have("cmake"):
        sys.exit("ERROR: cmake not found on PATH. Install CMake >= 3.21 "
                 "(e.g. `pip install cmake`).")
    # a compiler check: MSVC is located by CMake itself on Windows; on
    # POSIX we can sanity-check for a C++ compiler up front.
    if sys.platform != "win32" and not (_have("c++") or _have("g++")
                                        or _have("clang++")):
        sys.exit("ERROR: no C++ compiler found (g++/clang++). Install a "
                 "C++17 toolchain.")
    print("   toolchain: OK")


def configure(args):
    cfg = [
        "cmake", "-S", HERE, "-B", BUILD_DIR,
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DBINDING_BUILD={'Python3' if args.bindings else 'CPP_Only'}",
    ]
    env = dict(os.environ)
    if args.march:
        env["HTM_MARCH"] = args.march   # override the auto-probe
    # HTM_MARCH unset -> CMake auto-probes this host.
    _run(cfg, env=env)


def build(args):
    _run(["cmake", "--build", BUILD_DIR, "--config", args.build_type,
          "-j", str(os.cpu_count() or 2)])


def main():
    ap = argparse.ArgumentParser(description="Build Single-HTM-Core standalone.")
    ap.add_argument("--bindings", action="store_true",
                    help="also build the pybind11 Python modules")
    ap.add_argument("--march", default=None,
                    help="SIMD level override (off/x86-64/v2/v3/v4/native); "
                         "default is auto-probe")
    ap.add_argument("--build-type", default="Release",
                    help="CMake build type (default: Release)")
    ap.add_argument("--clean", action="store_true",
                    help="remove the build directory first")
    args = ap.parse_args()

    if args.clean:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    preflight()
    configure(args)
    build(args)

    lib_hint = os.path.join(BUILD_DIR, "src")
    print("\n== build complete ==")
    print(f"   library : {lib_hint} (libhtm_core.a / htm_core.lib)")
    if args.bindings:
        print("   modules : htm.bindings.sdr / .encoders / .algorithms")
    print("   (this subtree is self-contained: it carries its own "
          "CommonCompilerConfig.cmake + DetectMarch.cmake)")


if __name__ == "__main__":
    main()
