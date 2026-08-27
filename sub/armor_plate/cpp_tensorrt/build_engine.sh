#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 detector.onnx detector_fp16.engine [workspace_MiB]" >&2
  exit 2
fi
onnx="$1"
engine="$2"
workspace="${3:-2048}"
trtexec --onnx="$onnx" --saveEngine="$engine" --fp16 --memPoolSize="workspace:${workspace}" \
  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640
echo "created $engine"
