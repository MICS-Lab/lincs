set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR

g++ -c -O3 lib.cpp -o lib.o
g++ -O3 no-virtual.cpp lib.o -o no-virtual
g++ -O3 yes-virtual.cpp lib.o -o yes-virtual

time ./no-virtual
time ./yes-virtual
