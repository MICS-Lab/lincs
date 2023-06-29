# Copyright 2023 Vincent Jacques

from __future__ import annotations
import glob
import os
import shutil
import subprocess
import textwrap

import auditwheel.wheel_abi
import click
import semver

from cycle import build_sphinx_documentation, print_title


python_versions = os.environ["LINCS_DEV_PYTHON_VERSIONS"].split(" ")


@click.command()
@click.argument("level", type=click.Choice(["patch", "minor", "major", "dry-run"]))
def main(level):
    dry_run = level == "dry-run"
    if not dry_run:
        check_cleanliness()
    new_version = bump_version(level)
    if not dry_run:
        update_changelog(new_version)
        build_sphinx_documentation()
    build_archives(new_version)
    if not dry_run:
        publish(new_version)
        prepare_next_version(new_version)


def check_cleanliness():
    not_indexed = subprocess.run(["git", "diff", "--stat", "--exit-code"])
    if not_indexed.returncode != 0:
        print("ERROR: there are non-staged changes. Please 'git add' them and retry.")
        exit(1)

    branch = subprocess.run(["git", "branch", "--show-current"], stdout=subprocess.PIPE, universal_newlines=True, check=True)
    if branch.stdout.strip() != "main":
        input("WARNING: you're not on branch 'main'. Press enter to proceed, Ctrl+C to cancel.")

    not_commited = subprocess.run(["git", "diff", "--stat", "--staged", "--exit-code"], stdout=subprocess.DEVNULL)
    if not_commited.returncode != 0:
        input("WARNING: Some changes are staged but not committed. They will be included in the publication commit. Press enter to proceed, Ctrl+C to cancel.")


def bump_version(level):
    with open("setup.py") as f:
        setup_lines = f.readlines()
    for line in setup_lines:
        if line.startswith("version = "):
            dev_version = semver.VersionInfo.parse(line[11:-2])

    assert dev_version.prerelease == "dev"
    assert dev_version.build is None

    if level == "dry-run":
        return str(dev_version).replace("-dev", ".dev0")

    print("Development version:", dev_version)
    if level == "patch":
        new_version = dev_version.replace(prerelease=None)
    elif level == "minor":
        new_version = dev_version.bump_minor()
    elif level == "major":
        new_version = dev_version.bump_major()
    else:
        assert False
    print("New version:", new_version)

    with open("setup.py", "w") as f:
        for line in setup_lines:
            if line.startswith("version = "):
                f.write(f"version = \"{new_version}\"\n")
            else:
                f.write(line)

    return new_version


def update_changelog(new_version):
    tags = subprocess.run(
        ["git", "tag"],
        stdout=subprocess.PIPE, universal_newlines=True,
        check=True,
    ).stdout.splitlines()
    last_tag = None
    for tag in tags:
        assert tag.startswith("v")
        if last_tag is None or semver.compare(tag[1:], last_tag[1:]) > 0:
            last_tag = tag

    log_lines = subprocess.run(
        ["git", "log", "--oneline", "--no-decorate", f"{last_tag}.."],
        stdout=subprocess.PIPE, universal_newlines=True,
        check=True,
    ).stdout.splitlines()

    with open("doc-sources/changelog.rst") as f:
        lines = [line.rstrip() for line in f.readlines()]

        header_length = 6

        title = f"Version {new_version}"
        lines = lines[:header_length] + [
            title,
            "=" * len(title),
            ""
        ] + [
            f"- {log_line.split(' ', 1)[1]}"
            for log_line in log_lines
        ] + [
            ""
        ] + lines[header_length:]

    with open("doc-sources/changelog.rst", "w") as f:
        f.write("\n".join(lines) + "\n")

    input("Please edit 'doc-sources/changelog.rst' then press enter to proceed, Ctrl+C to cancel.")


def build_archives(new_version):
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)
    for path in glob.glob("/tmp/lincs-*"):
        shutil.rmtree(path)

    print_title("Building source archive and Docker image")
    subprocess.run(["python3", "-m", "build", "--sdist"], check=True)
    shutil.rmtree("lincs.egg-info")
    source_dist = f"dist/lincs-{new_version}.tar.gz"
    subprocess.run(["twine", "check", source_dist], check=True)
    # @todo Make ccache work in the following build
    subprocess.run([
        "sudo", "docker", "run",
        "--mount", f"type=bind,src={os.environ['HOST_ROOT_DIR']}/dist,dst=/dist,ro",
        "--rm",
        "lincs-development",
        "sh", "-c", textwrap.dedent(f"""\
            set -o errexit

            pip3 install --no-binary lincs --find-links /dist lincs=={new_version}
            python3 -m lincs --help
            lincs --help
        """),
    ], check=True)
    print()

    platform = None
    for python_version in python_versions:
        print_title(f"Building wheel for Python {python_version}")
        shutil.rmtree(f"/tmp/lincs-{new_version}", ignore_errors=True)
        subprocess.run(["tar", "xf", os.path.abspath(source_dist)], cwd="/tmp", check=True)

        subprocess.run([f"python{python_version}", "-m", "build", "--wheel", f"/tmp/lincs-{new_version}"], check=True)
        suffix = "m" if int(python_version.split(".")[1]) < 8 else ""
        linux_wheel = f"/tmp/lincs-{new_version}/dist/lincs-{new_version}-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}{suffix}-linux_x86_64.whl"

        if platform is None:
            # Inspired from the code of the 'auditwheel show' command, to get just what we need from its output
            platform = auditwheel.wheel_abi.analyze_wheel_abi(linux_wheel).sym_tag
        subprocess.run(["auditwheel", "repair", "--plat", platform, "--wheel-dir", "dist", "--strip", linux_wheel], check=True)
        repaired_wheel = f"dist/lincs-{new_version}-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}{suffix}-{platform}.whl"

        subprocess.run(["twine", "check", repaired_wheel], check=True)

        docker_tag = f"jacquev6/lincs:{new_version}-python{python_version}"
        os.mkdir("docker/dist")
        shutil.copy(repaired_wheel, "docker/dist")
        try:
            subprocess.run(
                [
                    "sudo", "docker", "build",
                    "--tag", docker_tag,
                    "--build-arg", f"PYTHON_VERSION={python_version}",
                    "--build-arg", f"LINCS_VERSION={new_version}",
                    "docker",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "sudo", "docker", "run", "--rm", docker_tag, "lincs", "--help",
                ],
                check=True,
            )
        finally:
            shutil.rmtree("docker/dist")
        print()


def publish(new_version):
    print_title("Publishing to PyPI")
    # The --repository option on next line assumes ~/.pypirc contains:
    # [distutils]
    #   index-servers=
    #     ...
    #     lincs
    # [lincs]
    #   repository = https://upload.pypi.org/legacy/
    #   username = __token__
    #   password = ... a token for package lincs
    subprocess.run(["twine", "upload", "--repository", "lincs"] + glob.glob("dist/*"), check=True)
    print()

    print_title("Publishing to Docker Hub")
    subprocess.run(["sudo", "docker", "tag", f"jacquev6/lincs:{new_version}-python{python_versions[-1]}", f"jacquev6/lincs:{new_version}"], check=True)
    subprocess.run(["sudo", "docker", "tag", f"jacquev6/lincs:{new_version}", "jacquev6/lincs:latest"], check=True)
    subprocess.run(["sudo", "docker", "push", f"jacquev6/lincs:{new_version}"], check=True)
    subprocess.run(["sudo", "docker", "push", "jacquev6/lincs:latest"], check=True)
    print()

    print_title("Publishing to GitHub")
    subprocess.run(["git", "add", "setup.py", "doc-sources/changelog.rst", "docs"], check=True)
    subprocess.run(["git", "commit", "-m", f"Publish version {new_version}"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "tag", f"v{new_version}"], check=True)
    subprocess.run(["git", "push", f"--tags"], check=True)


def prepare_next_version(new_version):
    with open("setup.py") as f:
        setup_lines = f.readlines()

    next_version = new_version.bump_patch().replace(prerelease="dev")

    with open("setup.py", "w") as f:
        for line in setup_lines:
            if line.startswith("version = "):
                f.write(f"version = \"{next_version}\"\n")
            else:
                f.write(line)

    subprocess.run(["git", "add", "setup.py"], check=True)
    subprocess.run(["git", "commit", "-m", f"Start working on next version"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "push"], check=True)


if __name__ == "__main__":
    main()
