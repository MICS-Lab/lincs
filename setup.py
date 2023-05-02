import setuptools


version = "0.1.0"

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.readlines()


libplad = setuptools.Extension(
    "libplad",
    sources=[
        "plad/libplad/libplad_module.cpp",
        "plad/libplad/plad.cpp",
    ],
    libraries=[
        "boost_python310",
    ],
)

setuptools.setup(
    name="plad",
    version=version,
    description="MCDA algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacquev6/plad",
    author="Vincent Jacques",
    author_email="vincent@vincent-jacques.net",
    license="MIT",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "plad = plad.command_line_interface:main",
        ],
    },
    ext_modules=[libplad],
)
