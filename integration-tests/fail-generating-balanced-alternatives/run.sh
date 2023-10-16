# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 3 4 --random-seed 42 >problem.yml
lincs generate classification-model problem.yml --random-seed 24 >model.yml

if lincs generate classified-alternatives problem.yml model.yml 100 \
  --max-imbalance 0 \
  --random-seed 42 \
  --output-classified-alternatives alternatives.csv \
  2>stderr.txt
then
  false
fi

diff <(sed "s/LINCS_VERSION/$(lincs --version)/" expected-stderr.txt) stderr.txt

if [ -e alternatives.csv ]
then
  false
fi
