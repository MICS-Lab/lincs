# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set +o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 6 2 --random-seed 42 --output-problem problem.yml

lincs generate classification-model problem.yml --random-seed 42 --output-model model.yml

lincs generate classified-alternatives problem.yml model.yml 2000 --random-seed 42 --output-alternatives learning-set.csv

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 2 --output-model trained-model--2-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 1991/2000) <(lincs classification-accuracy problem.yml trained-model--2-iterations.yml learning-set.csv)

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 3 --output-model trained-model--3-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 1996/2000) <(lincs classification-accuracy problem.yml trained-model--3-iterations.yml learning-set.csv)

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 4 --output-model trained-model--4-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 2000/2000) <(lincs classification-accuracy problem.yml trained-model--4-iterations.yml learning-set.csv)
