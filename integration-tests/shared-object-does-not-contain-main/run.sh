# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set +o pipefail
trap 'echo "Error on line $LINENO"' ERR


for python_version in $LINCS_DEV_PYTHON_VERSIONS
do
  lib_version=$(python$python_version -c 'import sys; suffix="m" if sys.hexversion < 0x03080000 else ""; print(f"{sys.version_info.major}{sys.version_info.minor}{suffix}")')

  g++ -L/home/user/.local/lib/python$python_version/site-packages -llincs.cpython-$lib_version-x86_64-linux-gnu 2>&1 \
  | grep "undefined reference to \`main'" >/dev/null
done
