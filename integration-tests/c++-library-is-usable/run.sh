# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


for python_version in $LINCS_DEV_PYTHON_VERSIONS
do
  echo "Python $python_version"

  lib_version=$(python$python_version -c 'import sys; suffix="m" if sys.hexversion < 0x03080000 else ""; print(f"{sys.version_info.major}{sys.version_info.minor}{suffix}")')

  g++ \
    -std=c++17 \
    test.cpp \
    -I/home/user/.local/lib/python$python_version/site-packages/lincs/liblincs \
    -I/usr/local/cuda-12.1/targets/x86_64-linux/include \
    -L/home/user/.local/lib/python$python_version/site-packages -llincs.cpython-$lib_version-x86_64-linux-gnu \
    -o test

  LD_LIBRARY_PATH=/home/user/.local/lib/python$python_version/site-packages ./test >actual-$python_version.txt

  diff expected.txt actual-$python_version.txt
done
