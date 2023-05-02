from __future__ import annotations
import shutil

import subprocess
import os
import textwrap


def main():
    # With plad NOT installed
    #########################

    subprocess.run([f"pip3", "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL, check=True)

    # Install plad
    ##############

    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("plad.egg-info", ignore_errors=True)
    subprocess.run([f"pip3", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    # With plad installed
    #####################

    os.chdir(os.path.expanduser("~"))

    # Use as a standalone command-line tool
    subprocess.run(["plad", "generate", "classification-domain", "3", "2", "-"], check=True)
    # Use as an executable Python module
    subprocess.run(["python3", "-m", "plad", "generate", "classification-domain", "3", "2", "-"], check=True)
    # Use as a Python package
    subprocess.run(["python3", "-c", "import io; import plad; buf = io.StringIO(); plad.Domain().dump(buf); print(buf.getvalue())"], check=True)
    # Use as a C++ library
    source = textwrap.dedent("""
        #include <plad.hpp>

        #include <iostream>

        int main() {
            plad::Domain().dump(std::cout);
            std::cout << std::endl;
        }
    """)
    subprocess.run(
        [
            "g++",
            "-x", "c++", "-",
            "-I/home/user/.local/lib/python3.10/site-packages/plad/libplad",
            "-L/home/user/.local/lib/python3.10/site-packages", "-lplad.cpython-310-x86_64-linux-gnu",
        ],
        check=True,
        input=source, universal_newlines=True,
    )
    subprocess.run(["./a.out"], check=True, env={"LD_LIBRARY_PATH": "/home/user/.local/lib/python3.10/site-packages"})
    with open("/wd/plad help-all.txt", "w") as f:
        subprocess.run(["plad", "help-all"], stdout=f, check=True)


if __name__ == "__main__":
    main()
