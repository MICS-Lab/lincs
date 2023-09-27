# Copyright 2023 Vincent Jacques

import glob
import itertools
import os
import setuptools
import setuptools.command.build_ext
import shutil
import subprocess
import sys


version = "0.8.4"


with open("README.rst") as f:
    long_description = f.read()
for file in ["COPYING", "COPYING.LESSER"]:
    long_description = long_description.replace(f" <{file}>`_", f" <https://github.com/MICS-Lab/lincs/blob/v{version}/{file}>`_")
for lang in ["yaml", "shell", "text", "diff"]:
    long_description = long_description.replace(f".. highlight:: {lang}", "")


with open("requirements.txt") as f:
    install_requires = f.readlines()


# Method for building an extension with CUDA code extracted from https://stackoverflow.com/a/13300714/905845
# @todo(Project management, later) Consider using scikit-build:
# it should make it easier to compile CUDA code using nvcc and to run C++ unit tests during build.
# Note that pybind11 comes with an example of building using scikit-build.
# (see also https://www.benjack.io/hybrid-python/c-packages-revisited/)
def customize_compiler_for_nvcc(self):
    default_compile = self._compile

    if self.compiler_type == "unix":
        self.src_extensions += [".cu"]

        default_compiler_so = self.compiler_so

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == ".cu":
                self.set_executable("compiler_so", "nvcc")
                postargs = extra_postargs["cuda"]
            elif "/vendored/" in src:
                postargs = extra_postargs["vendored-c++"]
            else:
                postargs = extra_postargs["c++"]

            default_compile(obj, src, ext, cc_args, postargs, pp_opts)
            self.compiler_so = default_compiler_so

        self._compile = _compile
    else:
        assert False, f"Unsupported compiler type: {self.compiler_type}"


class custom_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        setuptools.command.build_ext.build_ext.build_extensions(self)


def make_liblincs_extension():
    define_macros = [("NDEBUG", None), ("DOCTEST_CONFIG_DISABLE", None)]

    sources = []
    sources += glob.glob(f"lincs/liblincs/**/*.c", recursive=True)
    sources += glob.glob(f"lincs/liblincs/**/*.cc", recursive=True)
    sources += glob.glob(f"lincs/liblincs/**/*.cpp", recursive=True)

    # Non-standard: the dict is accessed in `customize_compiler_for_nvcc`
    # to get the standard form for `extra_compile_args`
    extra_compile_args = {}

    libraries = []
    include_dirs = []
    library_dirs = []
    extra_link_args=[]

    if os.environ.get("LINCS_DEV_FORBID_NVCC", "false") != "true" and shutil.which("nvcc") is not None:
        sources += glob.glob(f"lincs/liblincs/**/*.cu", recursive=True)
        libraries += ["cudart"]
        define_macros += [("LINCS_HAS_NVCC", None)]
        # @todo(Project management, later) Support several versions of CUDA?
        include_dirs += ["/usr/local/cuda-12.1/targets/x86_64-linux/include"]
        library_dirs += ["/usr/local/cuda-12.1/targets/x86_64-linux/lib"]
        extra_compile_args["cuda"] = ["-std=c++17", "-Xcompiler", "-fopenmp,-fPIC,-Werror=switch"]
    else:
        if os.environ.get("LINCS_DEV_FORCE_NVCC", "false") == "true":
            raise Exception("nvcc is not available but LINCS_DEV_FORCE_NVCC is true")
        else:
            print("WARNING: 'nvcc' was not found, lincs will be compiled without CUDA support", file=sys.stderr)

    try:
        chrones_dir = subprocess.run(
            ["chrones", "instrument", "c++", "header-location"], capture_output=True, universal_newlines=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        chrones_dir = None

    if os.environ.get("LINCS_DEV_FORBID_CHRONES", "false") != "true" and chrones_dir is not None:
        include_dirs += [chrones_dir]
        define_macros += [("LINCS_HAS_CHRONES", None)]
    else:
        if os.environ.get("LINCS_DEV_FORCE_CHRONES", "false") == "true":
            raise Exception("Chrones is not available but LINCS_DEV_FORCE_CHRONES is true")
        else:
            print("WARNING: 'chrones' was not found, lincs will be compiled without Chrones", file=sys.stderr)

    if sys.platform == "linux":
        extra_compile_args["c++"] = ["-std=c++17", "-Werror=switch", "-fopenmp"]
        extra_compile_args["vendored-c++"] = ["-std=c++17", "-Werror=switch", "-w", "-DQUIET", "-DNBUILD", "-DNCONTRACTS"]
        extra_link_args += ["-fopenmp"]
        libraries += [
            f"boost_python{sys.version_info.major}{sys.version_info.minor}",
            "ortools",
            # Weirdly required because of BoostPython:
            f"python{sys.version_info.major}.{sys.version_info.minor}{'m' if sys.hexversion < 0x03080000 else ''}",
        ]
        if os.environ.get("LINCS_DEV_COVERAGE", "false") == "true":
            extra_compile_args["c++"] += ["--coverage", "-O0"]
            extra_link_args += ["--coverage"]
    elif sys.platform == "darwin":
        extra_compile_args["c++"] = ["-std=c++17", "-Werror=switch", "-Xclang", "-fopenmp"]
        extra_compile_args["vendored-c++"] = ["-std=c++17", "-Werror=switch", "-w", "-DQUIET", "-DNBUILD", "-DNCONTRACTS"]
        extra_link_args += ["-lomp"]
        libraries += [
            f"boost_python{sys.version_info.major}{sys.version_info.minor}",
            "ortools",
        ]
    else:
        assert False, f"Unsupported platform: {sys.platform}"

    return setuptools.Extension(
        name="liblincs",
        sources=sources,
        libraries=libraries,
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
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
    ext_modules=[make_liblincs_extension()],
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
        "build_ext": custom_build_ext,
    },
)
