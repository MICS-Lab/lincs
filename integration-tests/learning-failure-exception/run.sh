# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 2 2 --random-seed 42 >problem.yml
lincs generate classification-model problem.yml --random-seed 42 >model.yml
lincs generate classified-alternatives problem.yml model.yml 100 --misclassified-count 10 --random-seed 42 >learning-set.csv

if lincs learn classification-model problem.yml learning-set.csv \
  --model-type ucncs --ucncs.approach sat-by-coalitions \
  --output-model learned-model.yml \
  2>stderr.txt
then
  false
fi

diff expected-stderr.txt stderr.txt

if [ -e learned-model.yml ]
then
  false
fi
