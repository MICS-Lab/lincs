# Copyright 2021 Vincent Jacques

import sys
import os.path


source_file_name = sys.argv[1]
(source_file_base_name, source_file_extension) = os.path.splitext(source_file_name)
assert source_file_extension in [".cu", ".cpp"]

line_numbers_with_tests = []
with open(source_file_name) as f:
    for line_number, line_text in enumerate(f):
        if "EXPECT_COMPILE_ERROR" in line_text:
            line_numbers_with_tests.append(line_number + 1)

if line_numbers_with_tests:
    print(f"test: {source_file_name[:-9]}-non-compilation-tests")
    print()
    print(f".PHONY: {source_file_name[:-9]}-non-compilation-tests")
    for line_number in line_numbers_with_tests:
        print(f"{source_file_name[:-9]}-non-compilation-tests: build/tests/{source_file_name[:-9]}-non-compilation-test-{line_number}.ok")

    for line_number in line_numbers_with_tests:
        print()
        print(f"build/tests/{source_file_name[:-9]}-non-compilation-test-{line_number}.ok: {source_file_name}")
        if source_file_extension == ".cu":
            print(f'\t@echo "nvcc -c $< -DEXPECT_COMPILE_ERROR={line_number}"')
            print("\t@mkdir -p $(dir $@)")
            print("\t@nvcc $(NVCC_COMPILE_OPTIONS) $< -o $@-base.o")
            print(f'\t@if nvcc $(NVCC_COMPILE_OPTIONS) -DEXPECT_COMPILE_ERROR={line_number} $< -o $@-test.o 2>/dev/null; then echo "{source_file_name}:{line_number}: non-compilation test failed"; else touch $@; fi')
        else:
            print(f'\t@echo "g++  -c $< -DEXPECT_COMPILE_ERROR={line_number}"')
            print("\t@mkdir -p $(dir $@)")
            print("\t@g++ $(GPP_COMPILE_OPTIONS) $< -o $@-base.o")
            print(f'\t@if g++ $(GPP_COMPILE_OPTIONS) -DEXPECT_COMPILE_ERROR={line_number} $< -o $@-test.o 2>/dev/null; then echo "{source_file_name}:{line_number}: non-compilation test failed"; else touch $@; fi')
