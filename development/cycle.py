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
    "--unit-coverage", is_flag=True,
    help="Measure coverage of unit tests, stop right after that. Quite long.",
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
@click.option(
    "--forbid-gpu", is_flag=True,
    help=textwrap.dedent("""\
        Skip all tests that use the GPU.
        This is done automatically (with a warning) if your machine does not have a GPU.
        Using this option explicitly avoids the warning.
    """),
)
def main(with_docs, unit_coverage, skip_long, stop_after_unit, forbid_gpu):
    python_versions = os.environ["LINCS_DEV_PYTHON_VERSIONS"].split(" ")
    if skip_long or unit_coverage:
        python_versions = [python_versions[0]]

    shutil.rmtree("build", ignore_errors=True)

    # With lincs not installed
    ##########################

    if unit_coverage:
        os.environ["LINCS_DEV_COVERAGE"] = "true"

    for python_version in python_versions:
        print_title(f"Building extension module in debug mode for Python {python_version}")
        subprocess.run(
            [
                f"python{python_version}", "setup.py", "build_ext",
                "--inplace", "--debug", "--undef", "NDEBUG,DOCTEST_CONFIG_DISABLE",
                "--parallel", str(multiprocessing.cpu_count() - 1),
            ],
            check=True,
        )
        print()

    print_title("Running C++ unit tests")
    run_cpp_tests(python_version=python_versions[0], skip_long=skip_long, forbid_gpu=forbid_gpu)
    print()

    for python_version in python_versions:
        print_title(f"Running Python {python_version} unit tests")
        run_python_tests(python_version=python_version, forbid_gpu=forbid_gpu)
        print()

    if unit_coverage:
        print_title(f"Making report of unit tests coverage")
        gcovr = ["gcovr", "--exclude-directories", "ccache", "--html-details", "build/coverage.html", "--print-summary", "--sort-uncovered"]
        # 'gcovr --exclude' doesn't work as advertised, so I'm using many '--filter' instead
        for source_name in glob.glob("lincs/liblincs/**/*.*", recursive=True):
            if not source_name.startswith("lincs/liblincs/vendored/") and not source_name.endswith("/liblincs_module.cpp"):
                gcovr += ["--filter", source_name]
        subprocess.run(gcovr, check=True)

        # Remove branch coverage (unreliable in C++ due to exception handling)
        with open("build/coverage.html") as f:
            lines = f.readlines()
        with open("build/coverage.html", "w") as f:
            f.writelines(line for line in lines if not "branch-coverage" in line)
        return

    if stop_after_unit:
        pass
    else:
        print_title("Making integration tests from documentation")
        make_integration_tests_from_doc()

        # Install lincs
        ###############

        shutil.rmtree("build", ignore_errors=True)
        for python_version in python_versions:
            print_title(f"Installing *lincs* for Python {python_version}")
            shutil.rmtree("lincs.egg-info", ignore_errors=True)
            subprocess.run([f"python{python_version}", "-m", "pip", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

        # With lincs installed
        ######################

        print_title("Running integration tests")
        run_integration_tests(python_versions=python_versions, skip_long=skip_long, forbid_gpu=forbid_gpu)

    if with_docs:
        print_title("Building Sphinx documentation")
        build_sphinx_documentation()
    else:
        subprocess.run(["git", "checkout", "--", "docs"], check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd", "docs"], check=True, capture_output=True)


def print_title(title, under="="):
    print(title)
    print(under * len(title))
    print(flush=True)


def run_cpp_tests(*, python_version, skip_long, forbid_gpu):
    suffix = "m" if int(python_version.split(".")[1]) < 8 else ""
    subprocess.run(
        [
            "g++",
            "-L.", f"-llincs.cpython-{python_version.replace('.', '')}{suffix}-x86_64-linux-gnu",  # Contains the `main` function
            "-o", "/tmp/lincs-tests",
        ],
        check=True,
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "."
    if skip_long:
        env["LINCS_DEV_SKIP_LONG"] = "true"
    if forbid_gpu:
        env["LINCS_DEV_FORBID_GPU"] = "true"
    subprocess.run(["/tmp/lincs-tests"], check=True, env=env)


def run_python_tests(*, python_version, forbid_gpu):
    env = dict(os.environ)
    if forbid_gpu:
        env["LINCS_DEV_FORBID_GPU"] = "true"
    subprocess.run(
        [
            f"python{python_version}", "-m", "unittest", "discover",
            "--pattern", "*.py",
            "--start-directory", "lincs", "--top-level-directory", ".",
        ],
        check=True,
        env=env,
    )


def make_integration_tests_from_doc():
    output_files = {}
    current_prefix = ""
    current_output_file_name = None
    for input_file_name in ["README.rst"] + glob.glob("doc-sources/*.rst"):
        with open(input_file_name) as f:
            lines = f.readlines()
        for line in lines:
            line = line.rstrip()

            if line == f"{current_prefix}.. STOP":
                assert current_output_file_name
                current_output_file_name = None
            if current_output_file_name:
                m = re.fullmatch(r".. APPEND-TO-LAST-LINE( .+)", line)
                if m:
                    assert output_files[current_output_file_name]
                    last_line_index = -1
                    while output_files[current_output_file_name][last_line_index] == "":
                        last_line_index -= 1
                    output_files[current_output_file_name][last_line_index] += m.group(1)
                elif line.startswith(current_prefix + "    "):
                    output_files[current_output_file_name].append(line)
                elif line == "" and output_files[current_output_file_name]:
                    output_files[current_output_file_name].append("")

            m = re.fullmatch(r"( *).. (START|EXTEND) (.+)", line)
            if m:
                assert current_output_file_name is None, (input_file_name, current_output_file_name)
                current_prefix = m.group(1)
                current_output_file_name = m.group(3)
                if m.group(2) == "START":
                    output_files[current_output_file_name] = []
        assert current_output_file_name is None, (input_file_name, current_output_file_name)

    shutil.rmtree("integration-tests/from-documentation", ignore_errors=True)
    for file_name, file_contents in output_files.items():
        file_path = os.path.join("integration-tests", "from-documentation", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        while file_contents and file_contents[-1] == "":
            file_contents.pop()
        with open(file_path, "w") as f:
            f.write(textwrap.dedent("\n".join(file_contents) + "\n"))
    with open("integration-tests/from-documentation/.gitignore", "w") as f:
        f.write("*\n")


def build_sphinx_documentation():
    subprocess.run([f"python3", "-m", "pip", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    with open("README.rst") as f:
        original_content = f.read()

    content = original_content
    content = re.sub(r"`(.*) <https://mics-lab.github.io/lincs/(.*)\.html>`_", r":doc:`\1 <\2>`", content)

    with open("README.rst", "w") as f:
        f.write(content)

    with open("doc-sources/problem-schema.yml", "w") as f:
        subprocess.run(["python3", "-c", "import lincs; print(lincs.Problem.JSON_SCHEMA, end='')"], check=True, stdout=f)

    with open("doc-sources/model-schema.yml", "w") as f:
        subprocess.run(["python3", "-c", "import lincs; print(lincs.Model.JSON_SCHEMA, end='')"], check=True, stdout=f)

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

    os.unlink("doc-sources/problem-schema.yml")
    os.unlink("doc-sources/model-schema.yml")
    with open("README.rst", "w") as f:
        f.write(original_content)


def run_integration_tests(*, python_versions, skip_long, forbid_gpu):
    env = dict(os.environ)
    env["LINCS_DEV_PYTHON_VERSIONS"] = " ".join(python_versions)

    for test_file_name in glob.glob("integration-tests/**/run.sh", recursive=True):
        test_name = test_file_name[18:-7]

        if skip_long and os.path.isfile(os.path.join(os.path.dirname(test_file_name), "is-long")):
            print_title(f"{test_name}: SKIPPED (is long)", '-')
            continue

        if forbid_gpu and os.path.isfile(os.path.join(os.path.dirname(test_file_name), "uses-gpu")):
            print_title(f"{test_name}: SKIPPED (uses GPU)", '-')
            continue

        print_title(test_name, '-')

        subprocess.run(
            ["bash", "run.sh"],
            cwd=os.path.dirname(test_file_name),
            check=True,
            env=env,
        )
        print()


if __name__ == "__main__":
    main()
