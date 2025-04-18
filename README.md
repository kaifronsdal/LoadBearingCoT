# Load Bearing CoT

WIP Writeup: https://docs.google.com/document/d/1RDbzRxDCNC3GQh1LRI6yY9Hp-09vZra7z2wTmEy_oVc/edit?usp=sharing


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
inspect eval src/loadbearingcot/faithfulness/paired_faithfulness.py --model vllm/Qwen/Qwen2.5-32B-Instruct --model-base-url http://0.0.0.0:8000/v1 -M api-key=token-abc123 --max-connections 100
```

To see what other evaluations are available examine the `src/loadbearingcot/faithfulness` directory.

Note: untill https://github.com/UKGovernmentBEIS/inspect_ai/pull/1621 is merged, you need to install inspect from https://github.com/kaifronsdal/inspect_ai

Note: for R1 models you need to use the chat template in `src/loadbearingcot/chat_templates/r1_prefill.jinja` as the default template does not handle prefilling the assistent message correctly.

## Fine-tuning

See `src/loadbearingcot/finetune/finetune.py` for an example of how to fine-tune a model on our paired ProntoQA dataset
