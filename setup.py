import setuptools


version = "0.2.2"

with open("README.md") as f:
    long_description = f.read()
for image in ["model", "alternatives"]:
    long_description = long_description.replace(f"{image}.png", f"https://github.com/jacquev6/lincs/raw/v{version}/{image}.png")

with open("requirements.txt") as f:
    install_requires = f.readlines()


liblincs = setuptools.Extension(
    "liblincs",
    sources=[
        "lincs/liblincs/liblincs_module.cpp",
        "lincs/liblincs/lincs.cpp",
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
