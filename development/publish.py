# Copyright 2023-2024 Vincent Jacques

from __future__ import annotations
import datetime
import subprocess

import click
import parver

from cycle import build_sphinx_documentation, print_title


@click.group()
def main():
    pass

@main.command()
@click.argument("level", type=click.Choice(["patch", "minor", "major"]))
@click.option("--dry-run", is_flag=True)
def quick(level, dry_run):
    publish(*make_quick_release_versions(level, read_dev_version()), "main", dry_run)

@main.command()
@click.argument("level", type=click.Choice(["patch", "minor", "major"]))
@click.argument("pre", type=click.Choice(["alpha", "beta", "rc"]))
@click.option("--dry-run", is_flag=True)
def first_pre(level, pre, dry_run):
    publish(*make_first_pre_release_versions(level, pre, read_dev_version()), "develop", dry_run)

@main.command()
@click.option("--dry-run", is_flag=True)
def next_pre(dry_run):
    publish(*make_next_pre_versions(read_dev_version()), "develop", dry_run)

@main.command()
@click.option("--dry-run", is_flag=True)
def finalize(dry_run):
    publish(*make_final_release_versions(read_dev_version()), "main", dry_run)


def make_quick_release_versions(level, dev_version):
    assert dev_version.dev == 0
    assert dev_version.pre is None

    if level == "patch":
        public_version = dev_version.replace(dev=None)
    elif level == "minor":
        public_version = dev_version.replace(dev=None).bump_release(index=1)
    elif level == "major":
        public_version = dev_version.replace(dev=None).bump_release(index=0)
    else:
        assert False

    next_dev_version = public_version.bump_release(index=2).bump_dev()

    return dev_version, public_version, next_dev_version

for (level, dev_version, public_version, next_dev_version) in [
    ("patch", "3.4.5.dev0", "3.4.5", "3.4.6.dev0"),
    ("minor", "3.4.5.dev0", "3.5.0", "3.5.1.dev0"),
    ("major", "3.4.5.dev0", "4.0.0", "4.0.1.dev0"),
]:
    dev_version = parver.Version.parse(dev_version, strict=True)
    public_version = parver.Version.parse(public_version, strict=True)
    next_dev_version = parver.Version.parse(next_dev_version, strict=True)
    expected = f"({dev_version}, {public_version}, {next_dev_version})"
    actual = f"({', '.join(str(v) for v in make_quick_release_versions(level, dev_version))})"
    assert actual == expected, f"make_quick_release_versions({level}, {dev_version}) == {actual} != {expected}"


def make_first_pre_release_versions(level, pre, dev_version):
    assert dev_version.dev == 0

    if pre == "alpha":
        assert dev_version.pre_tag is None
        base_version = dev_version
        pre_tag = "a"
    elif pre == "beta":
        assert dev_version.pre_tag in [None, "a"]
        base_version = dev_version.replace(pre=None)
        pre_tag = "b"
    elif pre == "rc":
        assert dev_version.pre_tag in [None, "a", "b"]
        base_version = dev_version.replace(pre=None)
        pre_tag = "rc"
    else:
        assert False

    assert base_version.pre_tag is None
    assert base_version.pre is None

    if level == "patch":
        public_version = base_version.replace(dev=None).bump_pre(pre_tag)
    elif level == "minor":
        public_version = base_version.replace(dev=None).bump_pre(pre_tag).bump_release(index=1)
    elif level == "major":
        public_version = base_version.replace(dev=None).bump_pre(pre_tag).bump_release(index=0)
    else:
        assert False

    next_dev_version = public_version.bump_pre().bump_dev()

    return dev_version, public_version, next_dev_version

for (level, pre, dev_version, public_version, next_dev_version) in [
    ("patch", "alpha", "3.4.5.dev0", "3.4.5a0", "3.4.5a1.dev0"),
    ("patch", "beta", "3.4.5.dev0", "3.4.5b0", "3.4.5b1.dev0"),
    ("patch", "beta", "3.4.5a6.dev0", "3.4.5b0", "3.4.5b1.dev0"),
    ("patch", "rc", "3.4.5.dev0", "3.4.5rc0", "3.4.5rc1.dev0"),
    ("patch", "rc", "3.4.5a6.dev0", "3.4.5rc0", "3.4.5rc1.dev0"),
    ("patch", "rc", "3.4.5b6.dev0", "3.4.5rc0", "3.4.5rc1.dev0"),
    ("minor", "alpha", "3.4.5.dev0", "3.5.0a0", "3.5.0a1.dev0"),
    ("minor", "beta", "3.4.5.dev0", "3.5.0b0", "3.5.0b1.dev0"),
    ("minor", "beta", "3.4.5a6.dev0", "3.5.0b0", "3.5.0b1.dev0"),
    ("minor", "rc", "3.4.5.dev0", "3.5.0rc0", "3.5.0rc1.dev0"),
    ("minor", "rc", "3.4.5a6.dev0", "3.5.0rc0", "3.5.0rc1.dev0"),
    ("minor", "rc", "3.4.5b3.dev0", "3.5.0rc0", "3.5.0rc1.dev0"),
    ("major", "alpha", "3.4.5.dev0", "4.0.0a0", "4.0.0a1.dev0"),
    ("major", "beta", "3.4.5.dev0", "4.0.0b0", "4.0.0b1.dev0"),
    ("major", "beta", "3.4.5a4.dev0", "4.0.0b0", "4.0.0b1.dev0"),
    ("major", "rc", "3.4.5.dev0", "4.0.0rc0", "4.0.0rc1.dev0"),
    ("major", "rc", "3.4.5a5.dev0", "4.0.0rc0", "4.0.0rc1.dev0"),
    ("major", "rc", "3.4.5b6.dev0", "4.0.0rc0", "4.0.0rc1.dev0"),
]:
    dev_version = parver.Version.parse(dev_version, strict=True)
    public_version = parver.Version.parse(public_version, strict=True)
    next_dev_version = parver.Version.parse(next_dev_version, strict=True)
    expected = f"({dev_version}, {public_version}, {next_dev_version})"
    actual = f"({', '.join(str(v) for v in make_first_pre_release_versions(level, pre, dev_version))})"
    assert actual == expected, f"make_first_pre_release_versions({level}, {pre}, {dev_version}) == {actual} != {expected}"


def make_next_pre_versions(dev_version):
    assert dev_version.dev == 0
    assert dev_version.pre_tag is not None
    assert dev_version.pre is not None

    public_version = dev_version.replace(dev=None)
    next_dev_version = public_version.bump_pre().bump_dev()

    return dev_version, public_version, next_dev_version

for (dev_version, public_version, next_dev_version) in [
    ("3.4.5a6.dev0", "3.4.5a6", "3.4.5a7.dev0"),
    ("3.4.5b6.dev0", "3.4.5b6", "3.4.5b7.dev0"),
    ("3.4.5rc6.dev0", "3.4.5rc6", "3.4.5rc7.dev0"),
]:
    dev_version = parver.Version.parse(dev_version, strict=True)
    public_version = parver.Version.parse(public_version, strict=True)
    next_dev_version = parver.Version.parse(next_dev_version, strict=True)
    expected = f"({dev_version}, {public_version}, {next_dev_version})"
    actual = f"({', '.join(str(v) for v in make_next_pre_versions(dev_version))})"
    assert actual == expected, f"make_next_pre_versions({dev_version}) == {actual} != {expected}"


def make_final_release_versions(dev_version):
    assert dev_version.dev == 0
    assert dev_version.pre_tag is not None
    assert dev_version.pre is not None

    public_version = dev_version.replace(dev=None).replace(pre=None)
    next_dev_version = public_version.bump_release(index=2).bump_dev()

    return dev_version, public_version, next_dev_version

for (dev_version, public_version, next_dev_version) in [
    ("3.4.5a6.dev0", "3.4.5", "3.4.6.dev0"),
]:
    dev_version = parver.Version.parse(dev_version, strict=True)
    public_version = parver.Version.parse(public_version, strict=True)
    next_dev_version = parver.Version.parse(next_dev_version, strict=True)
    expected = f"({dev_version}, {public_version}, {next_dev_version})"
    actual = f"({', '.join(str(v) for v in make_final_release_versions(dev_version))})"
    assert actual == expected, f"make_final_release_versions({dev_version}) == {actual} != {expected}"


def read_dev_version():
    with open("lincs/__init__.py") as f:
        for line in f.readlines():
            if line.startswith("__version__ = "):
                return parver.Version.parse(line[15:-2], strict=True)


def publish(dev_version, public_version, next_dev_version, expected_branch, dry_run):
    print("Current development version:", dev_version)
    print("New public version:", public_version)
    print("Next development version:", next_dev_version)
    input("Please check the above versions. Press enter to proceed, Ctrl+C to cancel.")
    check_cleanliness(expected_branch)
    write_version(dev_version, public_version)
    update_changelog(public_version)
    input("Please edit 'doc-sources/changelog.rst' then press enter to proceed, Ctrl+C to cancel.")
    input("Please edit 'doc-sources/roadmap.rst' then press enter to proceed, Ctrl+C to cancel.")
    build_sphinx_documentation()
    input("Please check built documentation 'docs/index.html' then press enter to proceed, Ctrl+C to cancel.")
    if not dry_run:
        push_public_version(public_version)
        if expected_branch == "main":
            subprocess.run(["git", "checkout", "develop"], check=True)
            subprocess.run(["git", "merge", "main", "--ff-only"], check=True)
        write_version(public_version, next_dev_version)
        push_next_dev_version()


def check_cleanliness(expected_branch):
    not_indexed = subprocess.run(["git", "diff", "--stat", "--exit-code"])
    if not_indexed.returncode != 0:
        print("ERROR: there are non-staged changes. Please 'git add' them and retry.")
        exit(1)

    branch = subprocess.run(["git", "branch", "--show-current"], stdout=subprocess.PIPE, universal_newlines=True, check=True)
    if branch.stdout.strip() != expected_branch:
        print(f"ERROR: you're not on branch '{expected_branch}'.")
        exit(1)

    not_committed = subprocess.run(["git", "diff", "--stat", "--staged", "--exit-code"], stdout=subprocess.DEVNULL)
    if not_committed.returncode != 0:
        input("WARNING: Some changes are staged but not committed. They will be included in the publication commit. Press enter to proceed, Ctrl+C to cancel.")


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


def push_public_version(public_version):
    print_title("Publishing to GitHub")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Publish version {public_version}"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["git", "tag", f"v{public_version}"], check=True)
    subprocess.run(["git", "push", f"--tags"], check=True)
    subprocess.run(["git", "push"], check=True)


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
