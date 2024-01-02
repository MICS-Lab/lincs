# Copyright 2023 Vincent Jacques

from __future__ import annotations
import copy
import glob
import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import textwrap
import time

import click
import jinja2
import joblib


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
    "--single-python-version", is_flag=True,
    help="Run tests under a single Python version to save time. Please run the full development cycle at least once before submitting your changes.",
)
@click.option(
    "--unit-coverage", is_flag=True,
    help="Measure coverage of unit tests, stop right after that. Implies --single-python-version. Quite long.",
)
@click.option(
    "--skip-unit", is_flag=True,
    help="Skip unit tests to save time.",
)
@click.option(
    "--skip-long-unit", is_flag=True,
    help="Skip long unit tests to save time.",
)
@click.option(
    "--skip-cpp-unit", is_flag=True,
    help="Skip C++ unit tests to save time.",
)
@click.option(
    "--skip-python-unit", is_flag=True,
    help="Skip Python unit tests to save time.",
)
@click.option(
    "--skip-notebooks", is_flag=True,
    help="Skip notebooks to save time.",
)
@click.option(
    "--skip-unchanged-notebooks", is_flag=True,
    help="Skip notebooks that have no 'git diff' to save time.",
)
@click.option(
    "--forbid-gpu", is_flag=True,
    help=textwrap.dedent("""\
        Skip all tests that use the GPU.
        This is done automatically (with a warning) if your machine does not have a GPU.
        Using this option explicitly avoids the warning.
    """),
)
@click.option(
    "--forbid-chrones", is_flag=True,
    help="Build lincs without Chrones.",
)
@click.option(
    "--doctest-option", multiple=True,
    help="Pass an option verbatim to doctest. Can be used multiple times.",
)
def main(
    with_docs,
    single_python_version,
    unit_coverage,
    skip_unit,
    skip_long_unit,
    skip_cpp_unit,
    skip_python_unit,
    skip_notebooks,
    skip_unchanged_notebooks,
    forbid_gpu,
    forbid_chrones,
    doctest_option,
):
    if forbid_gpu:
        os.environ["LINCS_DEV_FORBID_GPU"] = "true"
        os.environ["LINCS_DEV_FORBID_NVCC"] = "true"
    else:
        os.environ["LINCS_DEV_FORCE_NVCC"] = "true"
    if forbid_chrones:
        os.environ["LINCS_DEV_FORBID_CHRONES"] = "true"
    else:
        os.environ["LINCS_DEV_FORCE_CHRONES"] = "true"
    if unit_coverage:
        single_python_version = True

    python_versions = os.environ["LINCS_DEV_PYTHON_VERSIONS"].split(" ")
    if single_python_version:
        python_versions = [python_versions[0]]  # Use the lowest version to ensure backward compatibility
        os.environ["LINCS_DEV_PYTHON_VERSIONS"] = python_versions[0]

    shutil.rmtree("build", ignore_errors=True)

    # With lincs not installed
    ##########################

    if not skip_unit:
        if unit_coverage:
            os.environ["LINCS_DEV_COVERAGE"] = "true"

        for file_name in glob.glob("liblincs.cpython-*-x86_64-linux-gnu.so"):
            os.unlink(file_name)
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

        if not skip_cpp_unit:
            print_title("Running C++ unit tests")
            run_cpp_tests(python_version=python_versions[0], skip_long=skip_long_unit, doctest_options=doctest_option)
            print()

        if not skip_python_unit:
            for python_version in python_versions:
                print_title(f"Running Python {python_version} unit tests")
                run_python_tests(python_version=python_version)
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

    if not skip_notebooks:
        # Install lincs
        ###############

        shutil.rmtree("build", ignore_errors=True)
        # Install in reverse order to ensure that the lowest version is installed last,
        # so that the installed 'lincs' command is the one from the lowest version,
        # to keep the same behavior when we test with several versions.
        # At the time of writing:
        # - 'integration-tests/help-all' displays more things with recent Python versions
        # - images created by 'doc-sources/**/*.ipynb' differ because of the version of matplotlib
        for python_version in reversed(python_versions):
            print_title(f"Installing *lincs* for Python {python_version}")
            shutil.rmtree("lincs.egg-info", ignore_errors=True)
            subprocess.run([f"python{python_version}", "-m", "pip", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

        # With lincs installed
        ######################

        print_title("Running Jupyter notebooks (integration tests, documentation sources)")
        run_notebooks(forbid_gpu=forbid_gpu, skip_unchanged_notebooks=skip_unchanged_notebooks)

    print_title("Updating templates (documentation sources)")
    update_templates()

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


def run_cpp_tests(*, python_version, skip_long, doctest_options):
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
    command = ["/tmp/lincs-tests"]
    if skip_long:
        env["LINCS_DEV_SKIP_LONG"] = "true"
    else:
        command += ["-d"]
    command += list(doctest_options)
    subprocess.run(command, check=True, env=env)


def run_python_tests(*, python_version):
    subprocess.run(
        [
            f"python{python_version}", "-m", "unittest", "discover",
            "--pattern", "*.py",
            "--start-directory", "lincs", "--top-level-directory", ".",
        ],
        check=True,
    )


def build_sphinx_documentation():
    env = dict(os.environ)
    env["LINCS_DEV_FORBID_GPU"] = "false"
    env["LINCS_DEV_FORBID_NVCC"] = "false"
    env["LINCS_DEV_FORCE_NVCC"] = "true"
    for file_name in glob.glob("liblincs.cpython-*-x86_64-linux-gnu.so"):
        os.unlink(file_name)
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)
    subprocess.run([f"python3", "-m", "pip", "install", "--user", "."], env=env, stdout=subprocess.DEVNULL, check=True)

    with open("README.rst") as f:
        readme_content = f.read()

    readme_content = re.sub(r"`(.*) <https://mics-lab.github.io/lincs/(.*)\.html>`_", r":doc:`\1 <\2>`", readme_content)

    with open("doc-sources/README.rst", "w") as f:
        f.write(readme_content)

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
    os.unlink("doc-sources/README.rst")

    shutil.copy("COPYING", "docs/")
    shutil.copy("COPYING.LESSER", "docs/")
    shutil.copy("doc-sources/get-started/get-started.ipynb", "docs/")


def run_notebooks(*, forbid_gpu, skip_unchanged_notebooks):
    def run_notebook(notebook_path):
        # Work around race condition where two Jupyter instances try to open the same TCP port,
        # resulting in a zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:39787')
        time.sleep(random.random() * 5)

        original_cell_sources = {}

        # Ensure perfect reproducibility
        with open(notebook_path) as f:
            notebook = json.load(f)
        for (i, cell) in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                original_cell_sources[i] = copy.deepcopy(cell["source"])
                for (i, append) in enumerate(cell["metadata"].get("append_to_source", [])):
                    if i < len(cell["source"]):
                        if append != "":
                            cell["source"][i] = cell["source"][i].rstrip() + " " + append + "\n"
                    else:
                        cell["source"][-1] += "\n"
                        cell["source"].append(append + "\n")
        with open(notebook_path, "w") as f:
            json.dump(notebook, f, indent=1, sort_keys=True)
            f.write("\n")

        subprocess.run(
            ["git", "clean", "-fXd", os.path.dirname(notebook_path)],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--log-level=WARN", notebook_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print_title(f"{notebook_path}: FAILED", '-')
            print(e.stdout.decode())
            print(e.stderr.decode())
            return False
        finally:
            # Reduce git diff
            with open(notebook_path) as f:
                notebook = json.load(f)
            for (i, cell) in enumerate(notebook["cells"]):
                cell["metadata"].pop("execution", None)
                if cell["cell_type"] == "code":
                    cell["source"] = original_cell_sources[i]
                original_outputs = cell["outputs"]
                if original_outputs:
                    new_output = original_outputs[0]
                    if new_output["output_type"] == "stream":
                        for output in original_outputs[1:]:
                            assert output["name"] == new_output["name"]
                            assert output["output_type"] == new_output["output_type"]
                            new_output["text"] += output["text"]
                        cell["outputs"] = [new_output]
                    else:
                        assert len(original_outputs) == 1, original_outputs
                        assert new_output["output_type"] == "display_data", original_outputs
            with open(notebook_path, "w") as f:
                json.dump(notebook, f, indent=1, sort_keys=True)
                f.write("\n")

        print_title(f"{notebook_path}: OK", '-')
        return True

    jobs = []

    for notebook_path in sorted(glob.glob("**/*.ipynb", recursive=True)):
        if notebook_path.startswith("docs/"):
            continue

        if forbid_gpu and os.path.isfile(os.path.join(os.path.dirname(notebook_path), "uses-gpu")):
            print_title(f"{notebook_path}: SKIPPED (uses GPU)", '-')
            continue

        if skip_unchanged_notebooks:
            has_diff = (
                subprocess.run(["git", "diff", "--", notebook_path], check=True, capture_output=True).stdout
                or
                subprocess.run(["git", "diff", "--staged", "--", notebook_path], check=True, capture_output=True).stdout
            )
            if not has_diff:
                print_title(f"{notebook_path}: SKIPPED (unchanged)", '-')
                continue

        jobs.append(joblib.delayed(run_notebook)(notebook_path))

    results = joblib.Parallel(n_jobs=multiprocessing.cpu_count() - 1)(jobs)

    if not all(results):
        print("Some notebooks FAILED")
        exit(1)


def update_templates():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("doc-sources"),
        keep_trailing_newline=True,
    )

    env.globals["notebooks"] = {}
    for notebook_path in glob.glob("**/*.ipynb", recursive=True):
        with open(notebook_path) as f:
            env.globals["notebooks"][notebook_path] = json.load(f)

    for template_path in glob.glob("**/*.tmpl", recursive=True):
        output_path = template_path[:-5]
        print(template_path, "->", output_path)
        template = env.get_template(os.path.basename(template_path))
        with open(output_path, "w") as f:
            if output_path.endswith(".rst"):
                f.write(f".. WARNING: this file is generated from '{template_path}'. MANUAL EDITS WILL BE LOST.\n\n")
            else:
                assert False, "Unknown extension for warning comment"
            f.write(template.render())


if __name__ == "__main__":
    main()
