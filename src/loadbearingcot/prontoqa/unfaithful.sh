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
for frac in $(seq 0 0.1 1.0); do
    inspect eval faithfulness.py \
        --model vllm/Qwen/Qwen2.5-7B-Instruct \
        --log-dir unfaithful_logs_qwen7_5 \
        --max-connections 400 \
        -T use_cot=True \
        -T truncate_frac=$frac
done