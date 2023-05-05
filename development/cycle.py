from __future__ import annotations
import glob
import re
import shutil

import subprocess
import os
import textwrap


def main():
    # With lincs not installed
    ##########################

    print("Making integration tests from README.md")
    print("=======================================")
    print(flush=True)

    make_example_integration_test_from_readme()

    # Install lincs
    ###############

    print("Installing *lincs*")
    print("==================")
    print(flush=True)

    # Next line costs ~15s per cycle, but seems necessary because the package is not always rebuilt when C++ parts change.
    # Feel free to comment it out if you only modify Python parts.
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("lincs.egg-info", ignore_errors=True)
    subprocess.run([f"pip3", "install", "--user", "."], stdout=subprocess.DEVNULL, check=True)

    # With lincs installed
    ######################

    run_integration_tests()


def make_example_integration_test_from_readme():
    with open("README.md") as f:
        lines = f.readlines()

    files = {}
    current_file_name = None
    for line in lines:
        line = line.rstrip()
        m = re.fullmatch(r"(?:-->)?<!-- STOP -->", line)
        if m:
            assert current_file_name
            current_file_name = None
        if current_file_name:
            m = re.fullmatch(r"<!-- APPEND-TO-LAST-LINE( .+) -->", line)
            if m:
                assert files[current_file_name]
                files[current_file_name][-1] += m.group(1)
            else:
                files[current_file_name].append(line)
        m = re.fullmatch(r"<!-- (START|EXTEND) (.+) -->(?:<!--)?", line)
        if m:
            current_file_name = m.group(2)
            if m.group(1) == "START":
                files[current_file_name] = []
    assert current_file_name is None, current_file_name

    shutil.rmtree("integration-tests/readme", ignore_errors=True)
    for file_name, file_contents in files.items():
        file_path = os.path.join("integration-tests", "readme", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(textwrap.dedent("\n".join(file_contents)) + "\n")
    with open("integration-tests/readme/.gitignore", "w") as f:
        f.write("*\n")


def run_integration_tests():
    print("Running integration tests")
    print("=========================")
    print()
    ok = True
    for test_file_name in glob.glob("integration-tests/**/run.sh", recursive=True):
        test_name = test_file_name[18:-7]
        print(test_name)
        print("-" * len(test_name), flush=True)
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
    return ok


if __name__ == "__main__":
    main()
