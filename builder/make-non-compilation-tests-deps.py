# Copyright 2021 Vincent Jacques

import sys

source_file_name = sys.argv[1]
assert source_file_name.endswith("-tests.cu")

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
        print(f"build/tests/{source_file_name[:-9]}-non-compilation-test-{line_number}.ok: {source_file_name} build/obj/{source_file_name[:-3]}.o")
        print(f'\t@echo "nvcc -c $< -DEXPECT_COMPILE_ERROR={line_number}"')
        print("\t@mkdir -p $(dir $@)")
        print(f'\t@if nvcc -std=c++17 --expt-relaxed-constexpr -c $< -DEXPECT_COMPILE_ERROR={line_number} -o foo.o 2>/dev/null; then echo "{source_file_name}:{line_number}: non-compilation test failed"; else touch $@; fi')
