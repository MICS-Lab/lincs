# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set +o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 6 2 --random-seed 42 --output-problem problem.yml

lincs generate classification-model problem.yml --random-seed 42 --output-model model.yml

lincs generate classified-alternatives problem.yml model.yml 2000 --random-seed 42 --output-classified-alternatives learning-set.csv

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 2 --output-model trained-model--2-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 1987/2000) <(lincs classification-accuracy problem.yml trained-model--2-iterations.yml learning-set.csv)

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 9 --output-model trained-model--9-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 1999/2000) <(lincs classification-accuracy problem.yml trained-model--9-iterations.yml learning-set.csv)

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 19 --output-model trained-model--19-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 1999/2000) <(lincs classification-accuracy problem.yml trained-model--19-iterations.yml learning-set.csv)

lincs learn classification-model problem.yml learning-set.csv --mrsort.weights-profiles-breed.max-iterations 20 --output-model trained-model--20-iterations.yml --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 42
diff <(echo 2000/2000) <(lincs classification-accuracy problem.yml trained-model--20-iterations.yml learning-set.csv)
