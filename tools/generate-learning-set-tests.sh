set -o errexit
set -o pipefail
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

function run_generate () {
  echo "generate-learning-set $@:"
  echo "generate-learning-set $@:" | sed 's/./-/g'
  $BUILD_DIR/tools/bin/generate-learning-set $@ 2>&1 | sed 's/^/> /'
  echo
}

function fail_generate () {
  echo "generate-learning-set $@:"
  echo "generate-learning-set $@:" | sed 's/./-/g'
  if $BUILD_DIR/tools/bin/generate-learning-set $@ 2>&1 | sed 's/^/> /'
  then
    echo "tools/bin/generate-learning-set returned 0"
    exit 1
  fi
  echo
}

run_generate --help

fail_generate

$BUILD_DIR/tools/bin/generate-model 4 5 42 >model.txt

fail_generate model.txt

fail_generate model.txt 250

run_generate model.txt 15 57
