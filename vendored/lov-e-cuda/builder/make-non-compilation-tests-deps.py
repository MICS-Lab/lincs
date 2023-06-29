#!/usr/bin/env python3
# Copyright 2021-2022 Vincent Jacques

import sys
import os.path


source_file_name = sys.argv[1]
(source_file_base_name, source_file_extension) = os.path.splitext(source_file_name)

line_numbers_with_tests = []
with open(source_file_name) as f:
    for line_number, line_text in enumerate(f):
        if "EXPECT_COMPILE_ERROR" in line_text and not line_text.lstrip().startswith("//"):
            line_numbers_with_tests.append(line_number + 1)

if line_numbers_with_tests:
    for line_number in line_numbers_with_tests:
        print()
        print(f"non-compilation-tests: build/debug/{source_file_name}.non-compilation-test-{line_number}.ok")
        print()
        print(f"build/debug/{source_file_name}.non-compilation-test-{line_number}.ok: {source_file_name} lov-e.hpp")
        if source_file_extension == ".cpp":
            print(f'\t@echo "g++ -c $< -DEXPECT_COMPILE_ERROR={line_number}"')
            print("\t@mkdir -p $(dir $@)")
            print(f"\t@   g++ -c $(gcc_flags) $< -o $@-base.o")
            print(f'\t@if g++ -c $(gcc_flags) $< -o $@-test.o -DEXPECT_COMPILE_ERROR={line_number} >$@.log 2>&1; then echo "{source_file_name}:{line_number}: non-compilation test failed"; false; else touch $@; fi')
        elif source_file_extension == ".cu":
            print(f'\t@echo "nvcc -dc $< -DEXPECT_COMPILE_ERROR={line_number}"')
            print("\t@mkdir -p $(dir $@)")
            print(f"\t@   nvcc -dc $(nvcc_flags) $< -o $@-base.o")
            print(f'\t@if nvcc -dc $(nvcc_flags) $< -o $@-test.o -DEXPECT_COMPILE_ERROR={line_number} >$@.log 2>&1; then echo "{source_file_name}:{line_number}: non-compilation test failed"; false; else touch $@; fi')
        else:
            assert False
