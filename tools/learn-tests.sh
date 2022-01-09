# Quick tests of the command-line interface of learn.cpp
# For more in-depth tests of the learning process, see ../library/learning-tests.*

set -o errexit
set -o pipefail
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

function run_learn () {
  echo "learn $@:"
  echo "learn $@:" | sed 's/./-/g'
  $BUILD_DIR/tools/bin/learn $@ 2>&1 | sed 's/^/> /'
  echo
}

function fail_learn () {
  echo "learn $@:"
  echo "learn $@:" | sed 's/./-/g'
  if $BUILD_DIR/tools/bin/learn $@ 2>&1 | sed 's/^/> /'
  then
    echo "tools/bin/learn returned 0"
    exit 1
  fi
  echo
}

run_learn --help

fail_learn

fail_learn not-a-learning-set.txt

$BUILD_DIR/tools/bin/generate-model 4 3 57 >model.txt
$BUILD_DIR/tools/bin/generate-learning-set model.txt 500 42 >learning-set.txt

fail_learn learning-set.txt --force-gpu --forbid-gpu

fail_learn learning-set.txt --target-accuracy 100.01

fail_learn learning-set.txt --weights-optimization-strategy not-a-strategy

fail_learn learning-set.txt --profiles-improvement-strategy not-a-strategy

time run_learn learning-set.txt --force-gpu --weights-optimization-strategy glop --profiles-improvement-strategy heuristic

time run_learn learning-set.txt --forbid-gpu --target-accuracy 89.9 --weights-optimization-strategy glop-reuse --profiles-improvement-strategy heuristic-midpoints
