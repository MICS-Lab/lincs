#!/bin/bash

set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

$BUILD_DIR/tools/bin/generate-model 4 3 57
$BUILD_DIR/tools/bin/generate-learning-set model-4-3-57.txt 2500 42

sed '3s/.*/0 0 0 0/' model-4-3-57.txt >model.txt

time $BUILD_DIR/tools/bin/test-improve-weights learning-set--model-4-3-57--2500-42.txt model.txt
