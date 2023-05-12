# Copyright 2023 Vincent Jacques

set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


python3 -m lincs  generate classification-domain 4 3 >actual.yml

diff expected.yml actual.yml
