#!/usr/bin/env bash
set -euo pipefail

mkdir -p models
curl -L --fail --retry 3 \
  "https://huggingface.co/microsoft/dml-ai-hub-models/resolve/1ce165a7b2eba2c8e7689004eb426811bc3d6586/googlenet/googlenet.onnx" \
  -o models/googlenet.onnx
echo "Downloaded models/googlenet.onnx"
