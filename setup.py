# Copyright 2023 Vincent Jacques

import glob
import itertools
import os
import sys
import setuptools
import setuptools.command.build_ext


version = "0.5.0"

with open("README.rst") as f:
    long_description = f.read()
for file in ["COPYING", "COPYING.LESSER"]:
    long_description = long_description.replace(f" <{file}>`_", f" <https://github.com/MICS-Lab/lincs/blob/v{version}/{file}>`_")
for lang in ["yaml", "shell", "text", "diff"]:
    long_description = long_description.replace(f".. highlight:: {lang}", "")

with open("requirements.txt") as f:
    install_requires = f.readlines()


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
        else:
            postargs = extra_postargs["gcc"]
        if "/vendored/" in src:
            postargs = postargs + ["-w"]

        default_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        setuptools.command.build_ext.build_ext.build_extensions(self)


liblincs = setuptools.Extension(
    "liblincs",
    sources=list(itertools.chain.from_iterable(
        glob.glob(f"lincs/liblincs/**/*.{ext}", recursive=True)
        for ext in ["c", "cc", "cpp", "cu"]
    )),
    libraries=[
        f"boost_python{sys.version_info.major}{sys.version_info.minor}",
        "ortools",
        f"python{sys.version_info.major}.{sys.version_info.minor}{'m' if sys.hexversion < 0x03080000 else ''}",  # Weirdly required because of BoostPython
        "yaml-cpp",
        "cudart",
    ],
    define_macros=[("DOCTEST_CONFIG_DISABLE", None)],
    # @todo Support building without CUDA (required on macOS)
    # @todo Support several versions of CUDA?
    include_dirs=["/usr/local/cuda-12.1/targets/x86_64-linux/include"],
    library_dirs=["/usr/local/cuda-12.1/targets/x86_64-linux/lib"],
    # Non-standard: the dict is accessed in `customize_compiler_for_nvcc`
    # to get the standard form for `extra_compile_args`
    extra_compile_args={
        "gcc": ["-std=c++17", "-fopenmp", "-Werror=switch"],
        "nvcc": ["-std=c++17", "-Xcompiler", "-fopenmp,-fPIC,-Werror=switch"],
    },
    extra_link_args=["-fopenmp"],
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
