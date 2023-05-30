# Copyright 2023 Vincent Jacques

from __future__ import annotations
import glob
import multiprocessing
import re
import shutil

import subprocess
import os
import textwrap


def main():
    # @todo Collect failures in each step, print them at the end, add an option --keep-running à la GNU make

    shutil.rmtree("build", ignore_errors=True)

    # With lincs not installed
    ##########################

    print("Building extension module in debug mode")
    print("=======================================")
    print(flush=True)
    subprocess.run(
        [
            f"python3", "setup.py", "build_ext",
            "--inplace", "--debug", "--undef", "NDEBUG,DOCTEST_CONFIG_DISABLE",
            "--parallel", str(multiprocessing.cpu_count() - 1),
        ],
        check=True,
    )
    print()

    print("Running C++ unit tests")
    print("======================")
    print(flush=True)
    subprocess.run(
        [
            "g++",
            "-L.", "-llincs.cpython-310-x86_64-linux-gnu",  # Contains the `main` function
            "-o", "/tmp/lincs-tests",
        ],
        check=True,
    )
    subprocess.run(
        [
            "/tmp/lincs-tests",
        ],
        check=True,
        env=dict(os.environ, LD_LIBRARY_PATH="."),
    )
    print()

    print("Running Python unit tests")
    print("=========================")
    print(flush=True)
    run_python_tests()
    print()

    print("Making integration tests from README.md")
    print("=======================================")
    print(flush=True)

    make_example_integration_test_from_readme()

    print("Building Sphinx documentation")
    print("=============================")
    print(flush=True)

    shutil.rmtree("docs", ignore_errors=True)
    subprocess.run(
        [
            "sphinx-build",
            "-b", "html",
            "--jobs", str(multiprocessing.cpu_count() - 1),
            "docs-source", "docs",
        ],
        check=True,
    )

    # Install lincs
    ###############

    shutil.rmtree("build", ignore_errors=True)

    print("Installing *lincs*")
    print("==================")
    print(flush=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)
    subprocess.run([f"pip3", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    # With lincs installed
    ######################

    run_integration_tests()


def run_python_tests():
    subprocess.run(
        [
            "python3", "-m", "unittest", "discover",
            "--pattern", "*.py",
            "--start-directory", "lincs", "--top-level-directory", ".",
        ],
        check=True,
    )


def make_example_integration_test_from_readme():
    with open("README.md") as f:
        lines = f.readlines()

    files = {}
    current_file_name = None
    for line in lines:
        line = line.rstrip()
        m = re.fullmatch(r"(?:-->)?<!-- STOP -->", line)
        if m:
            assert current_file_name
            current_file_name = None
        if current_file_name:
            m = re.fullmatch(r"<!-- APPEND-TO-LAST-LINE( .+) -->", line)
            if m:
                assert files[current_file_name]
                files[current_file_name][-1] += m.group(1)
            else:
                files[current_file_name].append(line)
        m = re.fullmatch(r"<!-- (START|EXTEND) (.+) -->(?:<!--)?", line)
        if m:
            current_file_name = m.group(2)
            if m.group(1) == "START":
                files[current_file_name] = []
    assert current_file_name is None, current_file_name

    shutil.rmtree("integration-tests/readme", ignore_errors=True)
    for file_name, file_contents in files.items():
        file_path = os.path.join("integration-tests", "readme", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(textwrap.dedent("\n".join(file_contents)) + "\n")
    with open("integration-tests/readme/.gitignore", "w") as f:
        f.write("*\n")


def run_integration_tests():
    print("Running integration tests")
    print("=========================")
    print()
    ok = True
    for test_file_name in glob.glob("integration-tests/**/run.sh", recursive=True):
        test_name = test_file_name[18:-7]
        print(test_name)
        print("-" * len(test_name), flush=True)
        try:
            subprocess.run(
                ["bash", "run.sh"],
                cwd=os.path.dirname(test_file_name),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"{test_name}: FAILED")
            ok = False
        else:
            print()
    if not ok:
        exit(1)


if __name__ == "__main__":
    main()
