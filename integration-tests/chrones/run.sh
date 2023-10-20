# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


rm -f run-result.json lincs.*.chrones.csv report.png

lincs generate classification-problem 4 3 --random-seed 57 >problem.yml
lincs generate classification-model problem.yml --random-seed 59 >model.yml
lincs generate classified-alternatives problem.yml model.yml 200 --random-seed 61 >learning-set.csv
lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.strategy sat-by-coalitions >learned-model-without-chrones.yml

if test -f run-result.json; then false; fi
if test -f lincs.*.chrones.csv; then false; fi

chrones run -- lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.strategy sat-by-coalitions >learned-model-with-chrones.yml

test -f run-result.json
test -f lincs.*.chrones.csv

chrones report

test -f report.png

diff learned-model-without-chrones.yml learned-model-with-chrones.yml
