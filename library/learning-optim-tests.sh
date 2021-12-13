set -o errexit
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

$BUILD_DIR/tools/bin/generate-model 4 3 57 >model.txt
$BUILD_DIR/tools/bin/generate-learning-set model.txt 2000 42 >learning-set.txt

time $BUILD_DIR/tools/bin/learn \
  --force-gpu \
  --max-iterations 4 \
  --models-count 28 \
  learning-set.txt

time $BUILD_DIR/tools/bin/learn \
  --forbid-gpu \
  --max-iterations 4 \
  --models-count 28 \
  learning-set.txt
