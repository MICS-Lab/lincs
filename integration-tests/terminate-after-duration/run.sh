# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set +o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 6 3 --random-seed 42 --output-problem problem.yml

lincs generate classification-model problem.yml --random-seed 42 --output-model model.yml

lincs generate classified-alternatives problem.yml model.yml 2000 --random-seed 42 --output-classified-alternatives learning-set.csv

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-duration 15 --output-model trained-model--finished.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo '# Termination condition: target accuracy reached') <(grep 'Termination condition: ' trained-model--finished.yml)
diff <(echo 2000/2000) <(lincs classification-accuracy problem.yml trained-model--finished.yml learning-set.csv)

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-duration 2 --output-model trained-model--interrupted.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo '# Termination condition: maximum total duration reached') <(grep 'Termination condition: ' trained-model--interrupted.yml)
if diff <(echo 2000/2000) <(lincs classification-accuracy problem.yml trained-model--interrupted.yml learning-set.csv) >/dev/null
then
  false
fi
