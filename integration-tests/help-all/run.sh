set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


lincs help-all >actual.txt

diff expected.txt actual.txt
