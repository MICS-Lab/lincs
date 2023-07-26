# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 4 2 --output-problem problem.yml --random-seed 40
lincs generate classification-model problem.yml --output-model model.yml --random-seed 41
lincs generate classified-alternatives problem.yml model.yml 1000 --output-classified-alternatives learning-set.csv --random-seed 42

lincs learn classification-model problem.yml learning-set.csv --output-model actual-trained-model.yml --model-type ucncs --ucncs.approach sat-by-separation

diff expected-trained-model.yml actual-trained-model.yml
