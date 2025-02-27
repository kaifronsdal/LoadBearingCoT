# Load Bearing CoT



## Installation

```bash
git clone git@github.com:kaifronsdal/SabotageEvaluations.git
cd SabotageEvaluations
pip install -e .
```

## Run

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --api-key token-abc123 --tensor-parallel-size 1
```

```bash
inspect eval eval.py --model openai/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 -M api-key=token-abc123 --max-connections 100
```