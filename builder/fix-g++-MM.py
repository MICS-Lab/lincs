# Copyright 2021 Vincent Jacques

import sys


dependents = " ".join(sys.argv[1:])

dependencies = " ".join(sorted(set(
    dep
    for line in sys.stdin
    for dep in line.split(":")[-1].strip().rstrip("\\").strip().split(" ")
    if dep != "" and not dep.startswith("/")
)))

print(f"{dependents}: {dependencies}")
