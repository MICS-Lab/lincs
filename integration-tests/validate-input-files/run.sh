# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set +o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-problem 2 2 | lincs generate classification-model - --output-model model-2-2.yml
lincs generate classification-problem 1 2 --output-problem problem-1-2.yml

if lincs generate classified-alternatives problem-1-2.yml model-2-2.yml 100 >/dev/null 2>stderr.txt
then
  false
fi

diff  <(sed "s/LINCS_VERSION/$(lincs --version)/" expected-stderr.txt) stderr.txt
