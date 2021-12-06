set -o errexit
set -o pipefail
trap 'echo "Error on ${BASH_SOURCE[0]}:$LINENO"' ERR

function run_generate () {
  echo "generate-model $@:"
  echo "generate-model $@:" | sed 's/./-/g'
  $BUILD_DIR/tools/bin/generate-model $@ 2>&1 | sed 's/^/> /'
  echo
}

function fail_generate () {
  echo "generate-model $@:"
  echo "generate-model $@:" | sed 's/./-/g'
  if $BUILD_DIR/tools/bin/generate-model $@ 2>&1 | sed 's/^/> /'
  then
    echo "tools/bin/generate-model returned 0"
    exit 1
  fi
  echo
}

run_generate --help

fail_generate

fail_generate 4

fail_generate 4 5

run_generate 4 5 42
