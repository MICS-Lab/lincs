# Copyright 2023 Vincent Jacques

from __future__ import annotations
import glob
import multiprocessing
import os
import re
import shutil
import subprocess
import textwrap

import click


@click.command()
@click.option(
    "--with-docs", is_flag=True,
    help=textwrap.dedent("""\
        Build the documentation.
        The built documentation is published at https://mics-lab.github.io/lincs/ using GitHub Pages.
        So, it should only be pushed to GitHub when a new version of the package is published.
        Use this option to see the impact of your changes on the documentation, but do not commit them.
    """)
)
@click.option(
    "--skip-long", is_flag=True,
    help="Skip long tests. We all know what it is to be in a hurry. But please run the full development cycle at least once before submitting your changes.",
)
@click.option(
    "--stop-after-unit", is_flag=True,
    help=textwrap.dedent("""\
        Stop before installing the package.
        For when you're even more in a hurry.
        Or when you've changed the dependencies in the Dockerfile but not yet in the "Getting started" guide.
    """),
)
def main(with_docs, skip_long, stop_after_unit):
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

    if stop_after_unit:
        pass
    else:
        print("Making integration tests from documentation")
        print("===========================================")
        print(flush=True)

        make_example_integration_test_from_doc()
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

        run_integration_tests(skip_long=skip_long)

    if with_docs:
        print("Building Sphinx documentation")
        print("=============================")
        print(flush=True)
        build_sphinx_documentation()
        print()
    else:
        subprocess.run(["git", "checkout", "--", "docs"], check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd", "docs"], check=True, capture_output=True)


def run_python_tests():
    subprocess.run(
        [
            "python3", "-m", "unittest", "discover",
            "--pattern", "*.py",
            "--start-directory", "lincs", "--top-level-directory", ".",
        ],
        check=True,
    )


def make_example_integration_test_from_doc():
    lines = []
    for file_name in ["README.rst"] + glob.glob("doc-sources/*.rst"):
        with open(file_name) as f:
            lines += f.readlines()

    files = {}
    current_file_name = None
    for line in lines:
        line = line.rstrip()
        m = re.fullmatch(r".. STOP", line)
        if m:
            assert current_file_name
            current_file_name = None
        if current_file_name:
            m = re.fullmatch(r".. APPEND-TO-LAST-LINE( .+)", line)
            if m:
                assert files[current_file_name]
                last_line_index = -1
                while files[current_file_name][last_line_index] == "":
                    last_line_index -= 1
                files[current_file_name][last_line_index] += m.group(1)
            elif line.startswith("    "):
                files[current_file_name].append(line)
            elif line == "" and files[current_file_name]:
                files[current_file_name].append("")
        m = re.fullmatch(r".. (START|EXTEND) (.+)", line)
        if m:
            current_file_name = m.group(2)
            if m.group(1) == "START":
                files[current_file_name] = []
    assert current_file_name is None, current_file_name

    shutil.rmtree("integration-tests/from-documentation", ignore_errors=True)
    for file_name, file_contents in files.items():
        file_path = os.path.join("integration-tests", "from-documentation", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        while file_contents and file_contents[-1] == "":
            file_contents.pop()
        with open(file_path, "w") as f:
            f.write(textwrap.dedent("\n".join(file_contents)) + "\n")
    with open("integration-tests/from-documentation/.gitignore", "w") as f:
        f.write("*\n")


def build_sphinx_documentation():
    with open("README.rst") as f:
        original_content = f.read()

    content = original_content
    content = re.sub(r"`(.*) <https://mics-lab.github.io/lincs/(.*)\.html>`_", r":doc:`\1 <\2>`", content)

    with open("README.rst", "w") as f:
        f.write(content)

    shutil.rmtree("docs", ignore_errors=True)
    subprocess.run(
        [
            "sphinx-build",
            "-b", "html",
            "--jobs", str(multiprocessing.cpu_count() - 1),
            "doc-sources", "docs",
        ],
        check=True,
    )

    with open("README.rst", "w") as f:
        f.write(original_content)


def run_integration_tests(skip_long):
    print("Running integration tests")
    print("=========================")
    print()
    ok = True
    for test_file_name in glob.glob("integration-tests/**/run.sh", recursive=True):
        test_name = test_file_name[18:-7]
        print(test_name)
        print("-" * len(test_name), flush=True)

        if skip_long and os.path.isfile(os.path.join(os.path.dirname(test_file_name), "is-long")):
            print(f"{test_name}: SKIPPED")
            print(flush=True)
            continue

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
