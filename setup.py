# Copyright 2023 Vincent Jacques

import glob
import setuptools


version = "0.3.3"

with open("README.md") as f:
    long_description = f.read()
for image in ["model", "alternatives"]:
    long_description = long_description.replace(f"]({image}.png)", f"](https://github.com/MICS-Lab/lincs/raw/v{version}/{image}.png)")
for file in ["COPYING", "COPYING.LESSER"]:
    long_description = long_description.replace(f"]({file})", f"](https://github.com/MICS-Lab/lincs/blob/v{version}/{file})")

with open("requirements.txt") as f:
    install_requires = f.readlines()


# @todo Consider using scikit-build:
# it should make it easier to compile CUDA code using nvcc and to run C++ unit tests during build.
# Note that pybind11 comes with an example of building using scikit-build.
# (see also https://www.benjack.io/hybrid-python/c-packages-revisited/)
liblincs = setuptools.Extension(
    "liblincs",
    sources=glob.glob("lincs/liblincs/**/*.cpp", recursive=True),
    libraries=[
        "boost_python310",
        "ortools",
        "python3.10",  # Make the Python module usable as a C++ shared library without -lpython3.10 (still linked, but implicitly)
        "yaml-cpp",
    ],
    define_macros=[("DOCTEST_CONFIG_DISABLE", None)],
    include_dirs=["/usr/local/cuda-12.1/targets/x86_64-linux/include"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setuptools.setup(
    name="lincs",
    version=version,
    description="Learn and Infer Non Compensatory Sortings",
    license="LGPLv3",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
)
