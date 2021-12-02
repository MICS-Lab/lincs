#!/bin/bash

set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

$BUILD_DIR/tools/bin/generate-model 4 3 57 >model.txt
$BUILD_DIR/tools/bin/generate-learning-set model.txt 500 42 >learning-set.txt

time $BUILD_DIR/tools/bin/test-learn learning-set.txt >reconstructed-model.txt
