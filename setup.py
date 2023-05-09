import setuptools


version = "0.2.3-dev"

with open("README.md") as f:
    long_description = f.read()
for image in ["model", "alternatives"]:
    long_description = long_description.replace(f"]({image}.png)", f"](https://github.com/jacquev6/lincs/raw/v{version}/{image}.png)")

with open("requirements.txt") as f:
    install_requires = f.readlines()


# @todo Consider using scikit-build:
# it should make it easier to compile CUDA code using nvcc and to run C++ unit tests during build.
# Note that pybind11 comes with an example of building using scikit-build.
# (see also https://www.benjack.io/hybrid-python/c-packages-revisited/)
# @todo Run unit tests for the Python and C++ parts of the code
# See https://github.com/doctest/doctest for C++ (compare to GoogleTest)
liblincs = setuptools.Extension(
    "liblincs",
    sources=[
        "lincs/liblincs/liblincs_module.cpp",
        "lincs/liblincs/classification.cpp",
        "lincs/liblincs/generation.cpp",
        "lincs/liblincs/io.cpp",
    ],
    libraries=[
        "boost_python310",
        "python3.10",  # Make the Python module usable as a C++ shared library without -lpython3.10 (still linked, but implicitly)
        "yaml-cpp",
    ],
)

setuptools.setup(
    name="lincs",
    version=version,
    description="MCDA algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacquev6/lincs",
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
        # Note: no license classifier yet
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
