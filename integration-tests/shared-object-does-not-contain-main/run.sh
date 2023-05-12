# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


if g++ -L/home/user/.local/lib/python3.10/site-packages -llincs.cpython-310-x86_64-linux-gnu 2>/dev/null
then
  false
fi
