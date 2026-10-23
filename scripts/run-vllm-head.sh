#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:=deepseek-ai/DeepSeek-R1}"
: "${TENSOR_PARALLEL_SIZE:=4}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${MAX_MODEL_LEN:=524288}"
: "${GPU_MEMORY_UTILIZATION:=0.82}"
: "${SERVED_MODEL_NAME:=deepseek-r1}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --host 0.0.0.0 \
  --port 8000
