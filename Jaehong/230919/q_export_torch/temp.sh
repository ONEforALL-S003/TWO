#!/bin/bash

# This script verifies the torch_qparams_exporter behaviors
#
# HOW TO USE
#
# ./export.sh <path/to/bin_dir> <path/to/work_dir> <path/to/venv_dir> <path/to/intp_dir>
#                  <TEST 1> <TEST 2> ...
# bin_dir  : build directory of q-implant (ex: build/compiler/q-implant)
# venv_dir : python virtual environment home directory
# tflite2circle_path : exporter needs to convert tflite to circle, include executable

VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/test.py"
BINDIR="$1"; shift
VIRTUALENV="$1"; shift
TFLITE2CIRCLE_PATH="$1"; shift
PYTORCHEXAMPLES_DIR="$1"; shift

echo ${PYTORCHEXAMPLES_DIR}

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do

  echo "${TESTCASE}"
  TESTED+=("${TESTCASE}")

  TEST_RESULT_FILE="${BINDIR}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex

    source "${VIRTUALENV}/bin/activate"

    "${VIRTUALENV}/bin/python" "${VERIFY_SCRIPT_PATH}"

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("${TESTCASE}")
  else
    FAILED+=("${TESTCASE}")
  fi
done

if [[ ${#TESTED[@]} -ne ${#PASSED[@]} ]]; then
  echo "FAILED"
  for TEST in "${FAILED[@]}"
  do
    echo "- ${TEST}"
  done
  exit 255
fi

echo "PASSED"
exit 0
