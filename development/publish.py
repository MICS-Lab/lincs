# Copyright 2023 Vincent Jacques

from __future__ import annotations
import glob
import shutil

import subprocess
import sys

import semver


def main(args):
    assert len(args) == 2
    check_cleanliness()
    new_version = bump_version(args[1])
    update_changelog(new_version)
    publish_to_pypi(new_version)
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


def bump_version(part):
    with open("setup.py") as f:
        setup_lines = f.readlines()
    for line in setup_lines:
        if line.startswith("version = "):
            dev_version = semver.VersionInfo.parse(line[11:-2])

    assert dev_version.prerelease == "dev"
    assert dev_version.build is None

    print("Development version:", dev_version)
    if part == "patch":
        new_version = dev_version.replace(prerelease=None)
    elif part == "minor":
        new_version = dev_version.bump_minor()
    elif part == "major":
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
    # @todo(later) Fix order of changelog: put recent versions first

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

    with open("CHANGELOG.md", "a") as f:
        f.write(f"\n# Version {new_version}\n\n")
        for line in log_lines:
            f.write(f"- {line.split(' ', 1)[1]}\n")

    input("Please edit CHANGELOG.md then press enter to proceed, Ctrl+C to cancel.")


def publish_to_pypi(new_version):
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)

    # @todo Create a manylinux wheel and upload it (https://stackoverflow.com/a/59586096/905845)
    # Remove --sdist, upload the produced wheels as well as the produced tar.gz sdist.
    subprocess.run(["python3", "-m", "build", "--sdist"], check=True)
    subprocess.run(["twine", "check"] + glob.glob("dist/*.tar.gz"), check=True)
    subprocess.run(["twine", "upload"] + glob.glob("dist/*.tar.gz"), check=True)

    subprocess.run(["git", "add", "setup.py", "CHANGELOG.md"], check=True)
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
    main(sys.argv)
