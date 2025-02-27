#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Common parameters
BASE_CMD="inspect eval eval.py"
BASE_URL="http://0.0.0.0:8000/v1"
LOG_DIR="prontoqa_logs_feb_2"
API_KEY="token-abc123"
MAX_CONN=100

# Define model families
declare -a MODELS=(
    # DeepSeek Qwen models
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # DeepSeek Llama models
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # Meta Llama models
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"

    # Qwen models
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"

    # Google Gemma models
#    "google/gemma-2-2b-it"
#    "google/gemma-2-9b-it"
#    "google/gemma-2-27b-it"
)

# Run evaluations for each model
for MODEL in "${MODELS[@]}"; do
    # Create model-specific log directory
    MODEL_LOG_DIR="${LOG_DIR}/$(echo $MODEL | tr '/' '_')"

    # Run with CoT
    echo "Running evaluation for $MODEL with CoT=True"
    $BASE_CMD \
        --model "vllm/$MODEL" \
        --log-dir "$MODEL_LOG_DIR" \
        -M "tensor_parallel_size=4" \
        --max-connections "$MAX_CONN" \
        -T "use_cot=True"

    # Run without CoT
    echo "Running evaluation for $MODEL with CoT=False"
    $BASE_CMD \
        --model "vllm/$MODEL" \
        --log-dir "$MODEL_LOG_DIR" \
        -M "tensor_parallel_size=4" \
        --max-connections "$MAX_CONN" \
        -T "use_cot=False"
done

#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=True -T fewshot=0
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=True -T fewshot=1
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=True -T fewshot=2
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=True -T fewshot=4
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=True -T fewshot=8
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=False -T fewshot=0
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=False -T fewshot=1
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=False -T fewshot=2
#inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 --log-dir prontoqa_logs_qwen32 -M api-key=token-abc123 --max-connections 100 -T use_cot=False -T fewshot=4
#inspect eval faithfulness.py --model vllm/Qwen/Qwen_Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7_test --max-connections 100 -T use_cot=True -T truncate_frac=0.5
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
## deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
## deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
#
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B
## deepseek-ai/DeepSeek-R1-Distill-Llama-70B
#
## meta-llama/Llama-3.2-1B-Instruct
## meta-llama/Llama-3.2-3B-Instruct
## meta-llama/Llama-3.1-8B-Instruct
## meta-llama/Llama-3.1-70B-Instruct
#
## Qwen/Qwen2.5-0.5B-Instruct
## Qwen/Qwen2.5-1.5B-Instruct
## Qwen/Qwen2.5-3B-Instruct
## Qwen/Qwen2.5-7B-Instruct
## Qwen/Qwen2.5-14B-Instruct
## Qwen/Qwen2.5-32B-Instruct
## Qwen/Qwen2.5-72B-Instruct
#
## google/gemma-2-2b-it
## google/gemma-2-9b-it
## google/gemma-2-27b-it
