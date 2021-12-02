#!/bin/bash

set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

$BUILD_DIR/tools/bin/generate-model 4 5 57 \
| sed -e '3s/.*/0.25 0.25 0.25 0.25/' -e '4s/.*/0.5/' `# Homogeneous weights to match test-improve-profiles.cu` \
>model.txt

$BUILD_DIR/tools/bin/generate-learning-set model.txt 250 42 >learning-set.txt

echo "On CPU:"
time $BUILD_DIR/tools/bin/test-improve-profiles learning-set.txt 79

echo "On GPU:"
time $BUILD_DIR/tools/bin/test-improve-profiles learning-set.txt 79 --device
