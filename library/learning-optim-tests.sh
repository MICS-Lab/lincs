set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

$BUILD_DIR/tools/bin/generate-model 6 3 57 >model.txt
$BUILD_DIR/tools/bin/generate-learning-set model.txt 4000 42 >learning-set.txt

time $BUILD_DIR/tools/bin/learn \
  --force-gpu \
  --max-iterations 4 \
  --models-count 56 \
  learning-set.txt

time $BUILD_DIR/tools/bin/learn \
  --forbid-gpu \
  --max-iterations 4 \
  --models-count 56 \
  learning-set.txt
