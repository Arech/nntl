#!/bin/bash

additional_args=

ARGS=$(getopt -o c -l check -- "$@") || exit 1
eval set -- "${ARGS}"
while :; do
  case "$1" in
    -c|--check)
      additional_args="--dry-run --Werror"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      exit 2
      ;;
  esac
done

run_clang_format () {
  target_files=$(find "$@" -name build -prune -o -regex '.*\.\(h\|hpp\|cc\|cpp\)' -print) || exit 1
  if [ "${target_files}" != "" ]; then
    clang-format-12 -i ${additional_args} ${target_files}
  fi
}

if [ $# -ge 1 ]; then
  run_clang_format "$@"
else
  run_clang_format .
fi
