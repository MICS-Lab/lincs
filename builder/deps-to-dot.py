#!/usr/bin/env python3

import os
import sys


print("digraph G {")
print('  rankdir="LR";')
print('  node [shape="box"];')


def is_important(name):
    if name in {"cuda-utils", "matrix-view", "uint"}:
        return False
    if name.endswith("-tests"):
        return False
    if name.startswith("tools/generate-"):
        return False
    return True


def normalize_names(names):
    def gen():
        for name in names.split():
            yield os.path.splitext(remove_prefixes(os.path.normpath(name), ["build/obj/", "build/deps/", "library/"]))[0]

    return sorted(set(gen()))


def remove_prefixes(s, prefixes):
    for prefix in prefixes:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s

all_names = set()

for line in sys.stdin:
    (left_names, right_names) = map(normalize_names, line.strip().split(":"))

    all_names.update(left_names)
    all_names.update(right_names)

    for right_name in right_names:
        if is_important(right_name):
            print(f'  "{right_name}" [color="black", fontcolor="black"];')
        else:
            print(f'  "{right_name}" [color="gray40", fontcolor="gray40"];')

        for left_name in left_names:
            if left_name != right_name:
                if is_important(left_name) and is_important(right_name):
                    attributes = ' [color="black"]'
                else:
                    attributes = ' [color="gray40"]'
                print(f'  "{left_name}" -> "{right_name}"{attributes};')

print('  subgraph {')
print('    rank="same";')
for name in sorted(all_names):
    if name.startswith("tools/"):
        print(f'    "{name}";')
print("  }")

print("}")
