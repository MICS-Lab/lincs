# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 4 3 --random-seed 42 | lincs generate classification-model - --random-seed 45 >actual.yml

diff <(sed "s/LINCS_VERSION/$(lincs --version)/" expected.yml) actual.yml
