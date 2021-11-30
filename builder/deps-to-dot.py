#!/usr/bin/env python3

import os
import sys


print("digraph G {")
print('  node [shape="box"];')

def is_important(name):
    if name in {"cuda-utils", "matrix-view", "stopwatch"}:
        return False
    if name.endswith("-tests"):
        return False
    if name.startswith("test-"):
        return False
    if name.startswith("generate-"):
        return False
    return True

for line in sys.stdin:
    (left, right) = line.strip().split(":")

    for right in sorted(set(os.path.splitext(os.path.basename(name))[0] for name in right.split())):
        if is_important(right):
            print(f'  "{right}" [color="black", fontcolor="black"];')
        else:
            print(f'  "{right}" [color="darkgrey", fontcolor="darkgrey"];')

        for left in sorted(set(os.path.splitext(os.path.basename(name))[0] for name in left.split())):
            if left != right:
                if is_important(left) and is_important(right):
                    attributes = ' [color="black"]'
                else:
                    attributes = ' [color="darkgrey"]'
                print(f'  "{left}" -> "{right}"{attributes};')

print("}")
