#!/usr/bin/env python3

import os
import sys


def normalize_names(names):
    def gen():
        for name in names.split():
            name = os.path.splitext(remove_prefixes(os.path.normpath(name), ["build/obj/", "build/deps/", "library/"]))[0]
            if not is_insignificant(name):
                yield name

    return sorted(set(gen()))


def is_insignificant(name):
    if name in {"test-utils"}:
        return True

    if name.endswith("-tests"):
        return True

    return False


def is_important(name):
    if name in {"cuda-utils", "uint"}:
        return False

    if name.startswith("tools/generate-"):
        return False

    return True


def remove_prefixes(s, prefixes):
    for prefix in prefixes:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def get_color(important):
    if important:
        return "black"
    else:
        return "gray40"



print("digraph G {")
print('  rankdir="LR";')
print('  node [shape="box"];')

all_names = set()
for line in sys.stdin:
    (left_names, right_names) = map(normalize_names, line.strip().split(":"))

    all_names.update(left_names)
    all_names.update(right_names)

    for right_name in right_names:
        color = get_color(is_important(right_name))
        print(f'  "{right_name}" [color="{color}", fontcolor="{color}"];')

        for left_name in left_names:
            if left_name != right_name:
                color = get_color(is_important(left_name) and is_important(right_name))
                print(f'  "{left_name}" -> "{right_name}" [color="{color}"];')

print('  subgraph {')
print('    rank="same";')
for name in sorted(all_names):
    if name.startswith("tools/"):
        print(f'    "{name}";')
print("  }")

print("}")
