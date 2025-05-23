#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.0
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.1
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.2
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.3
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.4
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.5
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.6
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.7
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.8
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.9
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=1.0

# for frac in $(seq 0 0.1 1.0); do
#     inspect eval faithfulness.py \
#         --model vllm/Qwen/Qwen2.5-7B-Instruct \
#         --log-dir unfaithful_logs_qwen7_5 \
#         --max-connections 400 \
#         -T use_cot=True \
#         -T truncate_frac=$frac
# done

# inspect eval paired_faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir paired_faithfulness_logs_qwen7 --max-connections 100

# inspect eval ../src/loadbearingcot/faithfulness/paired_faithfulness.py --model vllm/Qwen/Qwen2.5-72B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/paired_faithfulness_qwen72b --max-connections 100 --model-base-url http://localhost:8090/v1 -M api_key=token-abc123

#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.0
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.1
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.2
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.3
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.4
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.5
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.6
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.7
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.8
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=0.9
#inspect eval faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir unfaithful_logs_qwen7 --max-connections 100 -T use_cot=True -T truncate_frac=1.0

# for frac in $(seq 0 0.1 1.0); do
#     inspect eval faithfulness.py \
#         --model vllm/Qwen/Qwen2.5-7B-Instruct \
#         --log-dir unfaithful_logs_qwen7_5 \
#         --max-connections 400 \
#         -T use_cot=True \
#         -T truncate_frac=$frac
# done

# inspect eval paired_faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir paired_faithfulness_logs_qwen7 --max-connections 100

# inspect eval ../src/loadbearingcot/faithfulness/paired_faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/paired_faithfulness_qwen7b --max-connections 100 --model-base-url http://localhost:8091/v1 -M api_key=token-abc123

# inspect eval ../src/loadbearingcot/faithfulness/paired_faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/paired_faithfulness_qwen7b --max-connections 100 -M device=0,1 -M dtype=half --log-level debug

# inspect eval ../src/loadbearingcot/faithfulness/paired_faithfulness.py --model sglang/Qwen/Qwen2.5-7B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/paired_faithfulness_qwen7b --max-connections 100 -M device=0,1 --log-level debug


# inspect eval ../src/loadbearingcot/faithfulness/paired_faithfulness.py --model vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --log-dir ~/GitHub/LoadBearingCoT/logs/paired_faithfulness_qwenr1 --max-connections 100 --model-base-url http://localhost:8090/v1 -M api_key=token-abc123

# inspect eval ../src/loadbearingcot/faithfulness/basic_eval.py --model vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --log-dir ~/GitHub/LoadBearingCoT/logs/basic_eval_original_qwenr1 --max-connections 100 --model-base-url http://localhost:8090/v1 -M api_key=token-abc123

# inspect eval ../src/loadbearingcot/faithfulness/hint_faithfulness.py --model vllm/Qwen/Qwen2.5-72B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/hint_faithfulness_qwen72b --max-connections 100 --model-base-url http://localhost:8090/v1 -M api_key=token-abc123

inspect eval ../src/loadbearingcot/faithfulness/paired_faithfulness.py --model vllm/Qwen/Qwen2.5-7B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/paired_slim_faithfulness_qwen7b --max-connections 100 --model-base-url http://localhost:8091/v1 -M api_key=token-abc123



# for frac in $(seq 0 0.1 1.0); do
#     inspect eval ../src/loadbearingcot/faithfulness/truncated_faithfulness_v2.py \
#         --model vllm/Qwen/Qwen2.5-7B-Instruct \
#         --model-base-url http://localhost:8091/v1 \
#         -M api_key=token-abc123 \
#         --log-dir ~/GitHub/LoadBearingCoT/logs/truncated_faithfulness_k250_qwen7 \
#         --max-connections 400 \
#         -T truncation_fraction=$frac
# done

# inspect eval ../src/loadbearingcot/faithfulness/basic_eval.py --model openai/Qwen/Qwen2.5-7B-Instruct --log-dir ~/GitHub/LoadBearingCoT/logs/paired_data_test/single_no_distractors --max-connections 100 --model-base-url http://localhost:8090/v1 -M api_key=token-abc123


# for frac in $(seq 0 0.1 1.0); do
#     inspect eval ../src/loadbearingcot/faithfulness/paraphrase_faithfulness.py \
#         --model vllm/Qwen/Qwen2.5-7B-Instruct \
#         --log-dir ~/GitHub/LoadBearingCoT/logs/paraphrase_faithfulness_logs_qwen7_2 \
#         --max-connections 400 \
#         -T truncate_frac=$frac
# done