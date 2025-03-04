from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.model import GenerateConfig, Model, CachePolicy, ChatMessageAssistant
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr, pattern
from inspect_ai.solver import (
    Solver,
    TaskState,
    generate,
    prompt_template,
    system_message, solver, Generate,
)

# from loadbearingcot.math.util import (
#     filter_dataset,
#     record_to_sample,
#     # sample_to_fewshot,
#     score_helper,
# )

from inspect_ai.dataset import MemoryDataset, Sample
import glob
import os
import re
import json
import random

from inspect_ai.util import resource

from loadbearingcot.util import collapse_assistant_message, prefill_message, load_hop_datasets


# Few-shot prompt template partially based on https://arxiv.org/pdf/2206.14858 - Appendix D.2
SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
You will be asked to solve a math problem. Some examples of problems and solutions are provided below.

{examples}
""".strip()

# Setup for problem + instructions for providing answer
USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


# USER_PROMPT_TEMPLATE = """
# The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.
#
# {prompt}
#
# Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
# """.strip()


@task
def prontoqa(
        hops: list[int] = [],
        fewshot: int = 0,
        fewshot_seed: int = 42,
        use_cot: bool = False,
) -> Task:
    """
    Inspect Task implementation for the MATH benchmark

    Args:
        levels (list[Literal[1,2,3,4,5]]): List of levels to filter on, 1 to 5.
        subjects (list[str]): List of subjects to filter on.
        fewshot (int): The number of fewshots to include
        fewshot_seed (int): The seed used when selecting fewshots
        grader_model (str): The model used to grade the samples
    """
    dataset = load_hop_datasets('./data/', k=250)

    return Task(
        dataset=dataset,
        solver=logic_solver(
            dataset=dataset,
            fewshot=fewshot,
            fewshot_seed=fewshot_seed,
            use_cot=use_cot
        ),
        scorer=[pattern(r"<answer>\s*(True|False)\s*</answer>", ignore_case=True)],
        config=GenerateConfig(temperature=0.5),
    )


def sample_to_fewshot(sample: Sample, include_cot: bool = False) -> str:
    """Convert a sample to a few-shot example string.

    Args:
        sample: The sample to convert
        include_cot: Whether to include chain of thought reasoning
    """
    example = f"Question:\n{sample.input}\n\n"

    if include_cot:
        example += "<think>\n"
        for step in sample.metadata["chain_of_thought"]:
            example += f"{step}\n"
        example += "\n</think>\n"

    example += f"<answer>{sample.target}</answer>"
    return example


def get_fewshot_examples(dataset: MemoryDataset, k: int, seed: int = None) -> list[Sample]:
    """Get k random samples to use as few-shot examples.

    Args:
        dataset: Dataset to sample from
        k: Number of samples to return
        seed: Random seed for sampling
    """
    if seed is not None:
        random.seed(seed)
    return random.sample(list(dataset), k)


SYSTEM_PROMPT_COT = "Given a set of logical statements, your task is to determine if the last statement is true or false. Think carefully about the task. Provide your step-by-step reasoning between <think></think> tags. Finally, write your answer in between <answer></answer> tags; i.e. either <answer>True</answer> or <answer>False</answer>. The answer should be a single word, either 'True' or 'False'."

SYSTEM_PROMPT_DIRECT = "Given a set of logical statements, your task is to determine if the last statement is true or false. Write your answer in between <answer></answer> tags; i.e. either <answer>True</answer> or <answer>False</answer>. The answer should be a single word, either 'True' or 'False'. Do not write anything else."

SYSTEM_PROMPT_SHORT = "You are taking the role of Kevin from the tv show 'The Office'. Kevin has said 'Why waste time say lot word when few word do trick'. In your reasoning, you should reason like Kevin. Kevin is given a special task. The task consists of a set of logical statements, and Kevin's task is to determine if the last statement is true or false. Kevin is secretly very smart and is definitely able to do the task. Kevin first thinks carefully about the task. Provide your step-by-step reasoning in a <think> tag. Write the reasoning as Kevin would do when he doesn't want to use many words. Finally, write your answer in between <answer>(True|False)</answer> tags. The answer should be a single word, either 'True' or 'False'."

PREFILL_COT = "<think>"
PREFILL_DIRECT = "<answer>"


def logic_solver(
        dataset: MemoryDataset,
        fewshot: int,
        fewshot_seed: int,
        use_cot: bool = False,
) -> list[Solver]:
    """Build solver for logic task.

    Arguments:
        dataset: Dataset to sample few-shot examples from
        fewshot: Number of few shot examples to use
        fewshot_seed: Random seed for sampling few shot examples
        use_cot: Whether to use chain of thought prompting
    """
    system_prompt = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT_DIRECT

    if fewshot:
        fewshot_samples = get_fewshot_examples(dataset, fewshot, fewshot_seed)
        system_prompt += "Here are some examples:\n" + "\n\n".join(
            [sample_to_fewshot(sample=sample, include_cot=use_cot)
             for sample in fewshot_samples]
        )

    prefill = PREFILL_COT if use_cot else PREFILL_DIRECT

    return [
        system_message(system_prompt),
        prefill_message(prefill),
        generate(cache=CachePolicy("1M")),
        collapse_assistant_message(),
    ]
