#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import os
import textwrap


def main():
    # With plad NOT installed
    #########################

    subprocess.run([f"pip3", "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL, check=True)

    # With plad installed
    #####################

    subprocess.run([f"pip3", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    os.chdir(os.path.expanduser("~"))

    # Use as a standalone command-line tool
    subprocess.run(["plad", "hello", "World"], check=True)
    # Use as an executable Python module
    subprocess.run(["python3", "-m", "plad", "hello", "World"], check=True)
    # Use as a Python package
    subprocess.run(["python3", "-c", "import plad; print(plad.hello('World'))"], check=True)
    # Use as a C++ library
    source = textwrap.dedent("""
        #include <plad.hpp>

        #include <iostream>

        int main() {
            std::cout << plad::hello("World") << std::endl;
        }
    """)
    subprocess.run(
        [
            "g++",
            "-x", "c++", "-",
            "-I/home/user/.local/lib/python3.10/site-packages/plad/libplad",
            "-L/home/user/.local/lib/python3.10/site-packages", "-lplad.cpython-310-x86_64-linux-gnu", "-lpython3.10",
        ],
        check=True,
        input=source, universal_newlines=True,
    )
    subprocess.run(["./a.out"], check=True, env={"LD_LIBRARY_PATH": "/home/user/.local/lib/python3.10/site-packages"})


if __name__ == "__main__":
    main()
