# Copyright 2023 Vincent Jacques

from __future__ import annotations
import glob
import shutil
import subprocess
import sys
import time

import click
import semver

from cycle import build_sphinx_documentation


@click.command()
@click.argument("level", type=click.Choice(["patch", "minor", "major"]))
def main(level):
    check_cleanliness()
    new_version = bump_version(level)
    update_changelog(new_version)
    build_sphinx_documentation()
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


def publish(new_version):
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)

    # @todo Create a manylinux wheel and upload it (https://stackoverflow.com/a/59586096/905845)
    # Remove --sdist, upload the produced wheels as well as the produced tar.gz sdist.
    subprocess.run(["python3", "-m", "build", "--sdist"], check=True)
    subprocess.run(["twine", "check"] + glob.glob("dist/*.tar.gz"), check=True)

    # The --repository option on next line assumes ~/.pypirc contains:
    # [distutils]
    #   index-servers=
    #     ...
    #     lincs
    # [lincs]
    #   repository = https://upload.pypi.org/legacy/
    #   username = __token__
    #   password = ... a token for package lincs
    subprocess.run(["twine", "upload", "--repository", "lincs"] + glob.glob("dist/*.tar.gz"), check=True)

    # Give PyPI some time to process the publication
    for i in range(12):
        try:
            subprocess.run(["pip3", "download", "--no-deps", f"lincs=={new_version}"], check=True)
            break
        except subprocess.CalledProcessError:
            print("PyPI is not ready yet, retrying soon...")
            time.sleep(10)

    subprocess.run([
        "sudo", "docker", "build",
        "--build-arg", f"LINCS_VERSION={new_version}",
        "--tag", f"jacquev6/lincs:{new_version}",
        "--tag", "jacquev6/lincs:latest",
        "docker"
    ], check=True)
    subprocess.run(["sudo", "docker", "push", f"jacquev6/lincs:{new_version}"], check=True)
    subprocess.run(["sudo", "docker", "push", "jacquev6/lincs:latest"], check=True)

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
