set -o errexit
set -o nounset
set -o pipefail
trap 'echo "Error on line $LINENO"' ERR


g++ \
  test.cpp \
  -I/home/user/.local/lib/python3.10/site-packages/lincs/liblincs \
  -I/usr/local/cuda-12.1/targets/x86_64-linux/include \
  -L/home/user/.local/lib/python3.10/site-packages -llincs.cpython-310-x86_64-linux-gnu \
  -o test

LD_LIBRARY_PATH=/home/user/.local/lib/python3.10/site-packages ./test >actual.txt

diff expected.txt actual.txt
