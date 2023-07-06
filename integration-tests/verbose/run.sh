# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set +o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 6 3 --random-seed 42 --output-problem problem.yml
lincs generate classification-model problem.yml --random-seed 42 --output-model model.yml
lincs generate classified-alternatives problem.yml model.yml 200 --random-seed 42 --output-classified-alternatives learning-set.csv

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.verbose --output-model trained-model.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42 2>learning-log.txt
diff expected-learning-log.txt learning-log.txt
