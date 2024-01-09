# Copyright 2023 Vincent Jacques

from __future__ import annotations
import datetime
import subprocess

import click
import parver

from cycle import build_sphinx_documentation, print_title


@click.command()
# @todo(Project management, v1.1) Support publishing dev and alpha/beta/rc versions?
@click.argument("level", type=click.Choice(["patch", "minor", "major"]))
@click.option("--dry-run", is_flag=True)
def main(level, dry_run):
    check_cleanliness()
    dev_version = read_version()
    public_version = make_public_version(level, dev_version)
    next_dev_version = make_next_dev_version(level, public_version)
    print("Current development version:", dev_version)
    print("New public version:", public_version)
    print("Next development version:", next_dev_version)
    write_version(dev_version, public_version)
    update_changelog(public_version)
    build_sphinx_documentation()
    if not dry_run:
        push_public_version(public_version)
        write_version(public_version, next_dev_version)
        push_next_dev_version()


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


def read_version():
    with open("lincs/__init__.py") as f:
        for line in f.readlines():
            if line.startswith("__version__ = "):
                return parver.Version.parse(line[15:-2], strict=True)


def make_public_version(level, dev_version):
    assert dev_version.pre is None
    assert dev_version.dev == 0

    if level == "patch":
        return dev_version.replace(dev=None)
    elif level == "minor":
        return dev_version.replace(dev=None).bump_release(index=1)
    elif level == "major":
        return dev_version.replace(dev=None).bump_release(index=0)
    else:
        assert False


def make_next_dev_version(level, public_version):
    return public_version.bump_release(index=2).bump_dev()


for (level, dev_version, public_version, next_dev_version) in [
    ("patch", "1.0.1.dev0", "1.0.1", "1.0.2.dev0"),
    ("minor", "1.0.1.dev0", "1.1.0", "1.1.1.dev0"),
    ("major", "1.0.1.dev0", "2.0.0", "2.0.1.dev0"),
]:
    dev_version = parver.Version.parse(dev_version, strict=True)
    public_version = parver.Version.parse(public_version, strict=True)
    next_dev_version = parver.Version.parse(next_dev_version, strict=True)
    assert make_public_version(level, dev_version) == public_version, f"make_public_version({level}, {dev_version}) == {make_public_version(level, dev_version)} != {public_version}"
    assert make_next_dev_version(level, public_version) == next_dev_version, f"make_next_dev_version({level}, {public_version}) == {make_next_dev_version(level, public_version)} != {next_dev_version}"


def update_changelog(public_version):
    tags = subprocess.run(
        ["git", "tag"],
        stdout=subprocess.PIPE, universal_newlines=True,
        check=True,
    ).stdout.splitlines()
    last_tag = None
    for tag in tags:
        assert tag.startswith("v")
        if last_tag is None or parver.Version.parse(tag[1:], strict=True) > parver.Version.parse(last_tag[1:], strict=True):
            last_tag = tag

    log_lines = subprocess.run(
        ["git", "log", "--oneline", "--no-decorate", f"{last_tag}.."],
        stdout=subprocess.PIPE, universal_newlines=True,
        check=True,
    ).stdout.splitlines()

    with open("doc-sources/changelog.rst") as f:
        lines = [line.rstrip() for line in f.readlines()]

        header_length = 6

        title = f"Version {public_version} ({datetime.date.today().isoformat()})"
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
    input("Please edit 'doc-sources/roadmap.rst' then press enter to proceed, Ctrl+C to cancel.")


def push_public_version(public_version):
    print_title("Publishing to GitHub")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Publish version {public_version}"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "tag", f"v{public_version}"], check=True)
    subprocess.run(["git", "push", f"--tags"], check=True)


def push_next_dev_version():
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Start working on next version"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "push"], check=True)


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

    for file_name in subprocess.run(["git", "ls-files"], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.splitlines():
        if file_name.endswith(".png"):
            continue
        if file_name.startswith("docs/"):
            continue
        if file_name.startswith("vendored/"):
            continue
        if file_name.startswith("lincs/liblincs/vendored/"):
            continue

        with open(file_name) as f:
            lines = f.readlines()
        lines = [line.replace(f"(with lincs version {old_version})", f"(with lincs version {new_version})") for line in lines]
        with open(file_name, "w") as f:
            for line in lines:
                f.write(line)


if __name__ == "__main__":
    main()
