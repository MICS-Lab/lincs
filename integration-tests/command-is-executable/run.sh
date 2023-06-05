# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 4 3 >actual.yml

diff expected.yml actual.yml
