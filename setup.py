"""Build script for py-kaldi-asr.

Compiler and linker flags come from the ``kaldi-asr`` pkg-config file
installed with Kaldi (e.g. pguyot/kaldi release tarballs), so
``pkg-config --cflags --libs kaldi-asr`` must work.

BLAS flavour (required by Kaldi's matrix/kaldi-blas.h):
  * OpenBLAS bundled with Kaldi (``<kaldi_root>/tools/OpenBLAS/install``),
    used by pguyot/kaldi >= 0ff452b;
  * otherwise ATLAS (``pkg-config atlas`` / ``blas-atlas``, or libatlas3-base
    in the usual library directories), used by older Kaldi builds.
Set ``KALDI_BLAS=openblas`` or ``KALDI_BLAS=atlas`` to force one.
"""

import os
import subprocess

from Cython.Build import cythonize
from setuptools import Extension, setup

FLAG_MAP = {"-I": "include_dirs", "-L": "library_dirs", "-l": "libraries"}


def pkg_config(*args):
    """Return pkg-config output as a list of tokens, or None on failure."""
    try:
        out = subprocess.run(
            ["pkg-config", *args], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.split()


def add_tokens(kw, tokens):
    for token in tokens:
        key = FLAG_MAP.get(token[:2])
        if key is None:
            kw.setdefault("extra_link_args", []).append(token)
        else:
            kw.setdefault(key, []).append(token[2:])


def find_openblas(kaldi_root):
    install = os.path.join(kaldi_root, "tools", "OpenBLAS", "install")
    if os.path.isfile(os.path.join(install, "include", "cblas.h")):
        return install
    return None


def find_atlas(kw):
    for name in ("blas-atlas", "atlas"):
        tokens = pkg_config("--libs", "--cflags", name)
        if tokens is not None:
            add_tokens(kw, tokens)
            return True
    for libdir in (
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/arm-linux-gnueabihf",
    ):
        if os.path.isfile(os.path.join(libdir, "libatlas.so.3")):
            kw.setdefault("library_dirs", []).append(libdir)
            kw.setdefault("libraries", []).extend(
                ["atlas", "cblas", "f77blas", "lapack_atlas"]
            )
            for incdir in (
                "/usr/include/atlas",
                "/usr/include/x86_64-linux-gnu/atlas",
            ):
                if os.path.isdir(incdir):
                    kw.setdefault("include_dirs", []).append(incdir)
            return True
    return False


def find_dependencies():
    kw = {}

    tokens = pkg_config("--libs", "--cflags", "kaldi-asr")
    if tokens is None:
        raise RuntimeError("*** failed to find pkg-config file for kaldi-asr")
    add_tokens(kw, tokens)

    kaldi_root = (pkg_config("--variable=kaldi_root", "kaldi-asr") or ["/opt/kaldi"])[0]
    blas = os.environ.get("KALDI_BLAS", "").lower()
    openblas = find_openblas(kaldi_root)

    if blas == "openblas" or (blas == "" and openblas):
        if openblas is None:
            raise RuntimeError(
                "KALDI_BLAS=openblas but no OpenBLAS found under %s" % kaldi_root
            )
        print("py-kaldi-asr: using OpenBLAS from %s" % openblas)
        kw.setdefault("include_dirs", []).append(os.path.join(openblas, "include"))
        kw.setdefault("define_macros", []).append(("HAVE_OPENBLAS", None))
    elif blas in ("", "atlas"):
        if not find_atlas(kw):
            raise RuntimeError(
                "No BLAS found: install Kaldi with its bundled OpenBLAS, "
                "or ATLAS (libatlas-base-dev)."
            )
        print("py-kaldi-asr: using ATLAS")
        kw.setdefault("define_macros", []).append(("HAVE_ATLAS", None))
    else:
        raise RuntimeError("Unsupported KALDI_BLAS=%s" % blas)

    return kw


EXTRA_COMPILE_ARGS = [
    "-Wall",
    "-pthread",
    "-std=c++17",
    "-DKALDI_DOUBLEPRECISION=0",
    "-Wno-sign-compare",
    "-Wno-unused-local-typedefs",
    "-Winit-self",
    "-DHAVE_EXECINFO_H=1",
    "-DHAVE_CXXABI_H",
    "-g0",
    "-O2",
]

deps = find_dependencies()

ext_modules = [
    Extension(
        "kaldiasr.%s" % name,
        sources=["kaldiasr/%s.pyx" % name, "kaldiasr/%s_wrappers.cpp" % name],
        language="c++",
        extra_compile_args=EXTRA_COMPILE_ARGS,
        **deps,
    )
    for name in ("nnet3", "gmm")
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
)
