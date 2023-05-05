set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs generate classification-domain 4 3 >actual.yml

diff expected.yml actual.yml
