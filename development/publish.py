# Copyright 2023 Vincent Jacques

from __future__ import annotations
import glob
import subprocess

import click
import semver

from cycle import build_sphinx_documentation, print_title


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
    dev_version = read_version()

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

    write_version(dev_version, new_version)

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
    print_title("Publishing to GitHub")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Publish version {new_version}"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "tag", f"v{new_version}"], check=True)
    subprocess.run(["git", "push", f"--tags"], check=True)


def prepare_next_version(new_version):
    next_version = new_version.bump_patch().replace(prerelease="dev")

    write_version(new_version, next_version)

    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Start working on next version"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "push"], check=True)


def read_version():
    with open("lincs/__init__.py") as f:
        for line in f.readlines():
            if line.startswith("__version__ = "):
                return semver.VersionInfo.parse(line[15:-2])


def write_version(old_version, new_version):
    old_version = str(old_version)
    new_version = str(new_version)

    with open("lincs/__init__.py") as f:
        lines = f.readlines()
    with open("lincs/__init__.py", "w") as f:
        for line in lines:
            if line.startswith("__version__ = "):
                f.write(f"__version__ = \"{new_version}\"\n")
            else:
                f.write(line)

    for file_name in glob.glob("doc-sources/*.rst"):
        with open(file_name) as f:
            lines = f.readlines()
        lines = [line.replace(old_version, new_version) for line in lines]
        with open(file_name, "w") as f:
            for line in lines:
                f.write(line)


if __name__ == "__main__":
    main()
