# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 4 3 --denormalized-min-max --allow-decreasing-criteria --random-seed 42 >actual.yml

diff <(sed "s/LINCS_VERSION/$(lincs --version)/" expected.yml) actual.yml
