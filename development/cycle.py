# Copyright 2023-2024 Vincent Jacques

from __future__ import annotations
import copy
import glob
import inspect
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
import lark
import yaml


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
    "--python-versions", default=os.environ["LINCS_DEV_PYTHON_VERSIONS"],
    help="Run tests under the specified Python versions. Space-separated list.",
)
@click.option(
    "--unit-coverage", is_flag=True,
    help="Measure coverage of unit tests, stop right after that. Implies --single-python-version. Quite long.",
)
@click.option(
    "--skip-build", is_flag=True,
    help="Skip build to save time.",
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
    "--skip-wpb-unit", is_flag=True,
    help="Skip WPB learnings unit tests to save time.",
)
@click.option(
    "--skip-wpb-glop-unit", is_flag=True,
    help="Skip WPB learnings using GLOP unit tests to save time.",
)
@click.option(
    "--skip-wpb-alglib-unit", is_flag=True,
    help="Skip WPB learnings using Alglib unit tests to save time.",
)
@click.option(
    "--skip-wpb-custom-unit", is_flag=True,
    help="Skip WPB learnings using custom LP solver unit tests to save time.",
)
@click.option(
    "--skip-sat-unit", is_flag=True,
    help="Skip SAT-based learnings unit tests to save time.",
)
@click.option(
    "--skip-max-sat-unit", is_flag=True,
    help="Skip Max-SAT-based learnings unit tests to save time.",
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
    "--skip-install", is_flag=True,
    help="Skip installation to save time.",
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
    python_versions,
    unit_coverage,
    skip_build,
    skip_unit,
    skip_long_unit,
    skip_wpb_unit,
    skip_wpb_glop_unit,
    skip_wpb_alglib_unit,
    skip_wpb_custom_unit,
    skip_sat_unit,
    skip_max_sat_unit,
    skip_cpp_unit,
    skip_python_unit,
    skip_install,
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

    python_versions = python_versions.split(" ")
    if single_python_version:
        python_versions = [python_versions[0]]  # Use the lowest version to ensure backward compatibility
    os.environ["LINCS_DEV_PYTHON_VERSIONS"] = " ".join(python_versions)

    # With lincs not installed
    ##########################

    if not skip_unit:
        if unit_coverage:
            os.environ["LINCS_DEV_COVERAGE"] = "true"

        if not skip_build:
            shutil.rmtree("build", ignore_errors=True)
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
            run_cpp_tests(
                python_version=python_versions[0],
                skip_long=skip_long_unit,
                skip_wpb=skip_wpb_unit,
                skip_wpb_glop=skip_wpb_glop_unit,
                skip_wpb_alglib=skip_wpb_alglib_unit,
                skip_wpb_custom=skip_wpb_custom_unit,
                skip_sat=skip_sat_unit,
                skip_max_sat=skip_max_sat_unit,
                doctest_options=doctest_option,
            )
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

    # Install lincs
    ###############

    if not skip_install:
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

    if not skip_install and not skip_notebooks:
        print_title("Running Jupyter notebooks (integration tests, documentation sources)")
        run_notebooks(forbid_gpu=forbid_gpu, skip_unchanged_notebooks=skip_unchanged_notebooks)

    print_title("Updating templates (documentation sources)")
    update_templates()
    convert_notebooks()
    if not skip_install:
        make_python_reference()
    print()

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


def run_cpp_tests(*, python_version, skip_long, skip_wpb, skip_wpb_glop, skip_wpb_alglib, skip_wpb_custom, skip_sat, skip_max_sat, doctest_options):
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
    command = [
        # "gdb", "--eval-command=run", "--eval-command=quit",
        "/tmp/lincs-tests",
    ]
    if skip_wpb:
        env["LINCS_DEV_SKIP_WPB"] = "true"
    if skip_wpb_glop:
        env["LINCS_DEV_SKIP_WPB_GLOP"] = "true"
    if skip_wpb_alglib:
        env["LINCS_DEV_SKIP_WPB_ALGLIB"] = "true"
    if skip_wpb_custom:
        env["LINCS_DEV_SKIP_WPB_CUSTOM"] = "true"
    if skip_sat:
        env["LINCS_DEV_SKIP_SAT"] = "true"
    if skip_max_sat:
        env["LINCS_DEV_SKIP_MAX_SAT"] = "true"
    if skip_long:
        env["LINCS_DEV_SKIP_LONG"] = "true"
    else:
        command += ["-d"]
    command += list(doctest_options)
    before = time.monotonic()
    subprocess.run(command, check=True, env=env)
    print(f"[doctest] Duration: {time.monotonic() - before:.1f}s")


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
        subprocess.run(["python3", "-c", "import lincs; print(lincs.classification.Problem.JSON_SCHEMA, end='')"], check=True, stdout=f)

    with open("doc-sources/model-schema.yml", "w") as f:
        subprocess.run(["python3", "-c", "import lincs; print(lincs.classification.Model.JSON_SCHEMA, end='')"], check=True, stdout=f)

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
            for (cell_index, cell) in enumerate(notebook["cells"]):
                cell["metadata"].pop("execution", None)
                cell["metadata"].pop("jp-MarkdownHeadingCollapsed", None)
                if cell["cell_type"] == "code":
                    cell["source"] = original_cell_sources[cell_index]
                if "outputs" in cell:
                    new_outputs = {}
                    for (output_index, output) in enumerate(cell["outputs"]):
                        key = [output["output_type"]]
                        if output["output_type"] == "stream":
                            key.append(output["name"])
                            key = tuple(key)
                            if key in new_outputs:
                                new_outputs[key]["text"] += output["text"]
                            else:
                                new_outputs[key] = output
                        else:
                            assert output["output_type"] in ["display_data", "execute_result"], output
                            key.append(output_index)
                            key = tuple(key)
                            new_outputs[key] = output
                    cell["outputs"] = [item[1] for item in sorted(new_outputs.items())]
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


def convert_notebooks():
    for name in ["python-api"]:
        input_path = f"doc-sources/{name}/{name}.ipynb"
        output_path = f"doc-sources/{name}.md"
        print(input_path, "->", output_path)

        shutil.rmtree(f"doc-sources/{name}_files", ignore_errors=True)

        subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown", input_path, "--output-dir", "doc-sources"],
            check=True,
            capture_output=True,
        )

        with open(output_path) as f:
            old_lines = [line.rstrip() for line in f.readlines()]

        new_lines = [
            f"<!-- WARNING: this file is generated from '{input_path}'. MANUAL EDITS WILL BE LOST. -->",
            "",
        ]
        in_block = False
        in_python = False
        for line in old_lines:
            line = re.sub(r"\[(.*?)\]\(https://mics-lab.github.io/lincs/(.*?)\.html\)", r"{doc}`\1 <\2>`", line)
            if in_python:
                if line.startswith("```"):
                    in_python = False
            else:
                if line.startswith("```python"):
                    in_python = True
            if not in_python:
                if not in_block:
                    if line.startswith("    kind:"):
                        new_lines.append("```yaml")
                        in_block = True
                    elif line.startswith("    "):
                        new_lines.append("```text")
                        in_block = True
                if in_block:
                    if line.startswith("    "):
                        line = line[4:]
                    else:
                        in_block = False
                        new_lines.append("```")
            new_lines.append(line)

        with open(output_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")


def make_python_reference():
    def directive(kind, name, doc, **options):
        yield f".. {kind}:: {name}"
        for k, v in options.items():
            if v is True:
                yield f"    :{k}:"
            elif v:
                yield f"    :{k}: {v}"
        yield ""
        for line in doc.splitlines():
            yield f"    {line.strip()}"
        yield ""

    signature_parser = lark.Lark(
        r"""
        signature: CNAME "(" parameters ")" "->" TNAME

        parameters: [parameter ("," parameter)*]

        parameter: CNAME ":" TNAME ["=" default_value]

        default_value: "None" -> none
                    | "True" -> true
                    | "False" -> false
                    | SIGNED_NUMBER -> number
                    | "[]" -> empty_list
                    | "<" CNAME "." CNAME ":" SIGNED_NUMBER ">" -> enum
                    | "[" default_value "]" -> list

        TNAME: /[a-zA-Z0-9.\[\],]+[a-zA-Z0-9.\[\]]/
        %import common.CNAME
        %import common.SIGNED_NUMBER

        %import common.WS
        %ignore WS
        """,
        start="signature",
    )

    class SignatureTransformer(lark.Transformer):
        def signature(self, args):
            return (args[1], args[2])

        def parameters(self, args):
            return args

        def none(self, _):
            return "None"

        def number(self, args):
            return args[0].value
        
        def true(self, _):
            return "True"

        def false(self, _):
            return "False"

        def empty_list(self, _):
            return "[]"

        def enum(self, args):
            return f"{args[0]}.{args[1]}"

        def list(self, args):
            return f"[{', '.join(args)}]"

        def parameter(self, args):
            return (args[0].value, args[1].value, args[2])

    def fix_signature(path, signature):
        signature = signature.replace("[float, float]", "[float,float]")
        signature = signature.replace("[int, int]", "[int,int]")

        parsed = signature_parser.parse(signature)
        parameters, return_type = SignatureTransformer().transform(parsed)

        if parameters[0][0] == "self":
            assert parameters[0][1].split(".")[-1] == path[-2], (parameters[0][1], path[-2])
            parameters = parameters[1:]

        for parameter in parameters:
            assert not parameter[0].startswith("arg"), f"Set parameter names in {path}"

        text_parameters = []
        for parameter in parameters:
            (name, type, default) = parameter

            type = type.replace("liblincs.", "")
            type = type.replace("Criterion.", "")
            type = type.replace("LearnMrsortByWeightsProfilesBreed.", "")
            type = type.replace("AcceptedValues.", "")
            type = type.replace("SufficientCoalitions.", "")
            type = type.replace("Performance.", "")
            type = type.replace("[float,float]", "[float, float]")
            type = type.replace("[int,int]", "[int, int]")

            if default is None:
                text_parameters.append(f"{name}: {type}")
            else:
                text_parameters.append(f"{name}: {type}={default}")

        if return_type == "None":
            return_type = ""
        else:
            return_type = return_type.replace("liblincs.", "")
            return_type = f" -> {return_type}"

        return f"{path[-1]}({', '.join(text_parameters)}){return_type}"

    def walk(path, parent, node, description):
        assert isinstance(description, dict)
        assert description.get("show", True), path

        name = path[-1]
        class_name = node.__class__.__name__

        description_doc = description.get("doc", ".. @to" + f"do(Documentation, v1.1) Document {'.'.join(path)} in doc-sources/reference/lincs.yml.")
        assert description_doc.endswith("."), (path, description_doc)
        docstring = getattr(node, "__doc__", None)
        if docstring:
            docstring = docstring.strip()
        else:
            docstring = ".. @to" + f"do(Documentation, v1.1) Add a docstring to {'.'.join(path)}."
        if class_name != "builtin_function_or_method":
            if not docstring.endswith("."):
                docstring += ". @to" + f"do(Documentation, v1.1) Add a dot at the end of the docstring of {'.'.join(path)}."
        do_walk = True
        if class_name == "module":
            yield from directive("module", ".".join(path), docstring)
        elif class_name == "pybind11_type":
            yield from directive("class", name, docstring)
        elif class_name == "type" and name.endswith("Exception"):
            yield from directive("exception", name, description_doc)
        elif class_name == path[-2]:
            yield from directive("property", name, description_doc, classmethod=True, type=".".join(path[:-1]))
            do_walk = False
        elif class_name == "property":
            yield from directive("property", name, docstring, type=description.get("type", "@to" + f"do(Documentation, v1.1) Add type to {'.'.join(path)} in doc-sources/reference/lincs.yml"))
            do_walk = False
        elif class_name in ["instancemethod" , "builtin_function_or_method"]:
            parent_class_name = parent.__class__.__name__
            if parent_class_name == "pybind11_type":
                directive_name = "method"
            elif parent_class_name == "module":
                directive_name = "function"
            else:
                directive_name = "@to" + f"do(Documentation, v1.1) Handle parent {'.'.join(path[:-1])} (of type {parent_class_name}) in the ad-hoc generator"
            if docstring.splitlines()[1] == "Overloaded function.":
                docstring = docstring.splitlines()[3:]
                for i, (signature, doc) in enumerate(zip(docstring[0::4], docstring[2::4])):
                    yield from directive(directive_name, fix_signature(path, signature.split(". ", 1)[1]), doc, noindex=i > 0)
            else:
                signature, doc = docstring.split("\n\n", 1)
                if not doc:
                    doc = ".. @to" + f"do(Documentation, v1.1) Add a docstring to {'.'.join(path)}."
                if not doc.endswith("."):
                    doc += ". @to" + f"do(Documentation, v1.1) Add a dot at the end of the docstring of {'.'.join(path)}."
                yield from directive(directive_name, fix_signature(path, signature), doc, staticmethod=description.get("staticmethod", False))
            do_walk = False
        elif class_name == "function":
            signature = inspect.signature(node)
            yield from directive("function", f"{name}{signature}".replace("liblincs.", "lincs.classification."), docstring)
        elif class_name in ["bool", "str"]:
            yield from directive("data", name, description_doc, type=class_name)
            do_walk = False
        else:
            yield ".. @to" + f"do(Documentation, v1.1) Handle {'.'.join(path)} (of type {class_name}) in the ad-hoc generator"
            yield ""

        if do_walk:
            undocumented_children_names = set(dir(node))
            for child_description in description.get("children", []):
                child_name = child_description["name"]
                if child_name in undocumented_children_names:
                    undocumented_children_names.remove(child_name)
                    child = getattr(node, child_name)
                    if child_description.get("show", True):
                        for line in walk(path + [child_name], node, child, child_description):
                            yield f"    {line}"
                else:
                    yield ".. @to" + f"do(Documentation, v1.1) Remove {'.'.join(path + [child_name])} from doc-sources/reference/lincs.yml"
            for child_name in sorted(undocumented_children_names):
                if child_name.startswith("_") and child_name not in ["__init__", "__call__"]:
                    continue
                yield "    .. @to" + f"do(Documentation, v1.1) Include or exclude explicitly {'.'.join(path)}.{child_name} in doc-sources/reference/lincs.yml"
                yield ""

    print("doc-sources/reference/lincs.yml -> doc-sources/reference/lincs.rst")

    import lincs

    with open("doc-sources/reference/lincs.yml") as f:
        description = yaml.safe_load(f)

    with open("doc-sources/reference/lincs.rst", "w") as f:
        f.write(".. WARNING: this file is generated from 'doc-sources/reference/lincs.yml'. MANUAL EDITS WILL BE LOST.\n\n")
        for line in walk(["lincs"], None, lincs, description):
            f.write(line.rstrip() + "\n")


if __name__ == "__main__":
    main()
