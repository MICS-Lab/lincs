# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


for python_version in $LINCS_DEV_PYTHON_VERSIONS
do
  python$python_version test.py >actual-$python_version.txt

  diff expected.txt actual-$python_version.txt
done
