# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


for python_version in $LINCS_DEV_PYTHON_VERSIONS
do
  echo "Python $python_version"

  python$python_version -m lincs generate classification-problem 4 3 >actual-$python_version.yml

  diff expected.yml actual-$python_version.yml
done
