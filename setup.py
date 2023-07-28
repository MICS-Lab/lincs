# Copyright 2023 Vincent Jacques

import glob
import itertools
import os
import shutil
import sys
import setuptools
import setuptools.command.build_ext


version = "0.6.0"

with open("README.rst") as f:
    long_description = f.read()
for file in ["COPYING", "COPYING.LESSER"]:
    long_description = long_description.replace(f" <{file}>`_", f" <https://github.com/MICS-Lab/lincs/blob/v{version}/{file}>`_")
for lang in ["yaml", "shell", "text", "diff"]:
    long_description = long_description.replace(f".. highlight:: {lang}", "")

with open("requirements.txt") as f:
    install_requires = f.readlines()


has_nvcc = os.environ.get("LINCS_DEV_FORBID_NVCC", "false") != "true" and shutil.which("nvcc") is not None

coverage_compile_args = []
coverage_link_args = []
if os.environ.get("LINCS_DEV_COVERAGE", "false") == "true":
    coverage_compile_args = ["--coverage", "-O0"]
    coverage_link_args = ["--coverage"]

# Method for building an extension with CUDA code extracted from https://stackoverflow.com/a/13300714/905845
# @todo Consider using scikit-build:
# it should make it easier to compile CUDA code using nvcc and to run C++ unit tests during build.
# Note that pybind11 comes with an example of building using scikit-build.
# (see also https://www.benjack.io/hybrid-python/c-packages-revisited/)
def customize_compiler_for_nvcc(self):
    self.src_extensions.append(".cu")

    default_compiler_so = self.compiler_so
    default_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            self.set_executable("compiler_so", "nvcc")
            postargs = extra_postargs["nvcc"]
        elif "/vendored/" in src:
            postargs = extra_postargs["gcc"] + ["-w", "-DQUIET", "-DNBUILD"]
        else:
            postargs = extra_postargs["gcc"] + coverage_compile_args

        default_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        setuptools.command.build_ext.build_ext.build_extensions(self)

optional_libraries = []

if has_nvcc:
    optional_libraries.append("cudart")

if sys.platform == "linux":
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]
    # Weirdly required because of BoostPython:
    optional_libraries.append(f"python{sys.version_info.major}.{sys.version_info.minor}{'m' if sys.hexversion < 0x03080000 else ''}")
elif sys.platform == "darwin":
    omp_compile_args = ["-Xclang", "-fopenmp"]
    omp_link_args = ["-lomp"]
else:
    assert False, f"Unsupported platform: {sys.platform}"

liblincs = setuptools.Extension(
    "liblincs",
    sources=list(itertools.chain.from_iterable(
        glob.glob(f"lincs/liblincs/**/*.{ext}", recursive=True)
        for ext in ["c", "cc", "cpp"] + (["cu"] if has_nvcc else [])
    )),
    libraries=[
        f"boost_python{sys.version_info.major}{sys.version_info.minor}",
        "ortools",
        "yaml-cpp",
    ] + optional_libraries,
    define_macros=[("DOCTEST_CONFIG_DISABLE", None)] + ([("LINCS_HAS_NVCC", None)] if has_nvcc else []),
    # @todo Support several versions of CUDA?
    include_dirs=["/usr/local/cuda-12.1/targets/x86_64-linux/include"],
    library_dirs=["/usr/local/cuda-12.1/targets/x86_64-linux/lib"],
    # Non-standard: the dict is accessed in `customize_compiler_for_nvcc`
    # to get the standard form for `extra_compile_args`
    extra_compile_args={
        "gcc": ["-std=c++17", "-Werror=switch"] + omp_compile_args,
        "nvcc": ["-std=c++17", "-Xcompiler", "-fopenmp,-fPIC,-Werror=switch"],
    },
    extra_link_args=omp_link_args + coverage_link_args,
)

setuptools.setup(
    name="lincs",
    version=version,
    description="Learn and Infer Non Compensatory Sortings",
    license="LGPLv3",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/MICS-Lab/lincs",
    author="Vincent Jacques",
    author_email="vincent@vincent-jacques.net",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "lincs = lincs.command_line_interface:main",
        ],
    },
    ext_modules=[liblincs],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={
        "build_ext": custom_build_ext
    },
)
