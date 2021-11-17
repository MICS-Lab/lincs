#!/bin/bash

set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

seed=57

$BUILD_DIR/tools/bin/generate-model 4 5 $seed

# Sum of weights == 2 to match test-improve-profiles.cu
sed '3s/.*/0.5 0.5 0.5 0.5/' model-4-5-$seed.txt >model.txt

$BUILD_DIR/tools/bin/generate-learning-set model.txt 250 $seed

$BUILD_DIR/tools/bin/test-improve-profiles learning-set--model--250-$seed.txt
