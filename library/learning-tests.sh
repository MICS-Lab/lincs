#!/bin/bash

set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

$BUILD_DIR/tools/bin/generate-model 4 3 57 >model.txt
$BUILD_DIR/tools/bin/generate-learning-set model.txt 500 42 >learning-set.txt

# We don't reach 100% accuracy in less than 1s
if $BUILD_DIR/tools/bin/learn \
  --max-duration-seconds 1 \
  --models-count 15 \
  --target-accuracy 100 \
  learning-set.txt
then
  false
fi

# We do reach 100% accuracy in less than 10s
time $BUILD_DIR/tools/bin/learn \
  --max-duration-seconds 10 \
  --models-count 15 \
  --target-accuracy 100 \
  learning-set.txt