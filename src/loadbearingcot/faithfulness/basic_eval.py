import json
import random
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    CachePolicy,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import (
    Generate,
    TaskState,
    solver,
)

from loadbearingcot.util import collapse_consecutive_assistant_messages


def load_hop_datasets(directory_path=".", k=None, random_seed=None):
    """
    Load all {n}hop.json files from the specified directory and combine them into a single dataset.
    Includes the number of hops extracted from the filename in each sample's metadata.

    Args:
        directory_path (str): Path to directory containing the hop json files
        k (int, optional): Maximum number of samples to load from each file. If None, load all samples.
        random_seed (int, optional): Random seed for sample selection when k is specified.

    Returns:
        MemoryDataset: Combined dataset containing all samples
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # expand ~ in path
    directory_path = Path(directory_path).expanduser()

    # Find all *hop.json files
    hop_files = list(directory_path.glob("*hop.json"))

    if not hop_files:
        raise ValueError(f"No *hop.json files found in {directory_path}")

    all_samples = []

    # Load each file manually
    for file_path in hop_files:
        # Extract hop count from filename
        match = re.match(r"(\d+)hop\.json$", file_path.name)
        if not match:
            raise ValueError(f"Could not extract hop count from filename: {file_path}")
        num_hops = int(match.group(1))

        with open(file_path) as f:
            data = json.load(f)

        # Convert to list of examples for random selection
        examples = list(data.items())

        # If k is specified and less than total examples, randomly select k examples
        if k is not None and k < len(examples):
            examples = random.sample(examples, k)

        # Process selected examples
        for example_id, example_data in examples:
            test_data = example_data["test_example"]
            sample = Sample(
                input=f"{test_data['question']}\n\n{test_data['query']}",
                target=test_data["answer"],
                id=f"{num_hops}_{example_id}",
                metadata={
                    "chain_of_thought": test_data["chain_of_thought"],
                    "question": test_data["question"],
                    "query": test_data["query"],
                    "answer": test_data["answer"],
                    "nhops": num_hops,
                    "source_file": file_path.name,
                },
            )
            all_samples.append(sample)

    return MemoryDataset(all_samples)


def load_paired_data(data_path: str, k: int | None = None) -> MemoryDataset:
    """Load paired questions from JSON files"""
    path = Path(data_path).expanduser()
    samples = []

    for file in path.glob("*.json"):
        with open(file) as f:
            question_list = json.load(f)
            if k is not None:
                if k > len(question_list):
                    raise ValueError(
                        f"k is greater than the number of questions in the file {file}"
                    )
                question_list = random.sample(question_list, k)

            for data in question_list:
                # Create two samples for each pair
                for i in [1, 2]:
                    question_data = data[f"question{i}"]
                    samples.append(
                        Sample(
                            input=f"{question_data['question']}\n{question_data['query']}",
                            target=question_data["answer"],
                            metadata={
                                "question": question_data["question"],
                                "query": question_data["query"],
                                "answer": question_data["answer"],
                                "pair_id": data["pair_id"],
                                "question_index": i,
                                "sever_index": data["sever_index"],
                                "nhops": data["nhops"],
                            },
                        )
                    )

    return MemoryDataset(samples)


def load_single_questions(
    directory_path=".", pattern="single_*.json", k=None, random_seed=None
):
    """
    Load single question data from JSON files matching the pattern in the specified directory.

    Args:
        directory_path (str): Path to directory containing the JSON files
        pattern (str): Glob pattern to match files (default: "single_*.json")
        k (int, optional): Maximum number of samples to load from each file. If None, load all samples.
        random_seed (int, optional): Random seed for sample selection when k is specified.

    Returns:
        MemoryDataset: Dataset containing all samples
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Expand ~ in path
    directory_path = Path(directory_path).expanduser()

    # Find all matching JSON files
    question_files = list(directory_path.glob(pattern))

    if not question_files:
        raise ValueError(f"No files matching '{pattern}' found in {directory_path}")

    all_samples = []

    # Load each file
    for file_path in question_files:
        with open(file_path) as f:
            questions = json.load(f)

        # If k is specified and less than total examples, randomly select k examples
        if k is not None and k < len(questions):
            questions = random.sample(questions, k)

        # Process selected examples
        for item in questions:
            question_data = item["question"]
            sample = Sample(
                input=f"{question_data['question']}\n\n{question_data['query']}",
                target=question_data["answer"],
                id=f"{item['nhops']}_{item['id']}",
                metadata={
                    "chain_of_thought": question_data["chain_of_thought"],
                    "question": question_data["question"],
                    "query": question_data["query"],
                    "answer": question_data["answer"],
                    "nhops": item["nhops"],
                    "has_distractors": item.get("has_distractors", False),
                    "source_file": file_path.name,
                },
            )
            all_samples.append(sample)

    return MemoryDataset(all_samples)


@solver
def simple_qa_solver():
    """Simple solver that directly answers the question using the XML tag structure."""

    # cot_prompt = """Given a set of logical statements, determine if the last statement is true or false. Think carefully about the task. Provide your step-by-step reasoning between <think></think> tags. Finally, write your answer in between <answer></answer> tags."""
    cot_prompt = """Given a set of logical statements, your task is to determine if the last statement is true or false. Think carefully about the task. Provide your step-by-step reasoning between <think></think> tags. Finally, write your answer in between <answer></answer> tags; i.e. either <answer>True</answer> or <answer>False</answer>. The answer should be a single word, either 'True' or 'False'."""

    cot_prompt_direct = """Given a set of logical statements, your task is to determine if the last statement is true or false. Write your answer in between <answer></answer> tags; i.e. either <answer>True</answer> or <answer>False</answer>. The answer should be a single word, either 'True' or 'False'. Do not write anything else."""

    example_prompt = """
    Example:
    Question: Quimpants are not simple. Quimpants are prilpants. Every stergit is simple. Wren is a quimpant. True or false: Wren is not simple.
    <think>
    Wren is a quimpant.
    Quimpants are not simple.
    Wren is not simple.
    </think>
    <answer>True</answer>

    Question: Every rorpant is not odorless. Rorpants are rompuses. Yompins are not liquid. Rompuses are liquid. Each rompus is a dumpus. Dumpuses are not angry. Dumpuses are gorpees. Polly is a rompus. True or false: Polly is not liquid.
    <think>
    Polly is a rompus.
    Rompuses are liquid.
    Polly is liquid.
    </think>
    <answer>False</answer>
    """.strip()

    async def solve(state: TaskState, _generate: Generate):
        if "r1" in get_model().name.lower():
            state.messages = [
                ChatMessageUser(
                    content=f"{cot_prompt}\n\n{state.metadata['question']}\n{state.metadata['query']}"
                ),
                ChatMessageAssistant(content="<think>"),
            ]
        else:
            # Set up the chat messages
            state.messages = [
                ChatMessageSystem(content=cot_prompt + "\n\n" + example_prompt),
                ChatMessageUser(
                    content=f"{state.metadata['question']}\n{state.metadata['query']}"
                ),
                # ChatMessageAssistant(content="<think>"),
            ]

        # Generate the answer
        state = await _generate(
            state,
            cache=CachePolicy("3M"),
            stop_seqs=["</answer>"],
            max_tokens=8192,
            frequency_penalty=0.1,
        )
        state.messages[-1].text += "</answer>"
        state = collapse_consecutive_assistant_messages(state)

        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def simple_qa_scorer():
    """Score whether the model's answer matches the ground truth."""

    async def score(state: TaskState, target: Target) -> Score:
        # Extract the answer (True or False) from the model's response using the XML tags
        answer_match = re.search(
            r"<answer>(True|False)</answer>", state.output.completion, re.IGNORECASE
        )

        model_answer = answer_match.group(1).lower() if answer_match else "unknown"
        ground_truth = target.text.lower()

        # Check if the model's answer matches the ground truth
        is_correct = model_answer == ground_truth

        # Extract reasoning for explanation
        reasoning_match = re.search(
            r"<think>(.*?)</think>", state.output.completion, re.DOTALL
        )
        reasoning = (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "No reasoning provided"
        )

        return Score(
            value=1 if is_correct else 0,
            answer=model_answer,
            explanation=f"Model reasoning:\n{reasoning}\n\nModel answer: {model_answer}, Ground truth: {ground_truth}",
            metadata={
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "reasoning": reasoning,
                "question_index": state.metadata.get("question_index", None),
                "full_response": state.output.completion,
            },
        )

    return score


@task
def simple_qa_task() -> Task:
    """Task for basic question answering using the paired dataset."""

    # dataset = load_paired_data(
    #     "~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/paired_data/", k=50
    # )

    # dataset = load_paired_data(
    #     "~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/paired_data_no_distractors",
    #     k=50,
    # )

    # dataset = load_paired_data(
    #     "~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/paired_data_no_distractors_slim",
    #     k=50,
    # )

    dataset = load_hop_datasets(
        "~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/data/", k=50
    )

    # dataset = load_single_questions(
    #     "~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/question_single/",
    #     k=50
    # )

    # dataset = load_single_questions(
    #     "~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/question_single_no_distractors",
    #     k=50,
    # )

    return Task(
        dataset=dataset,
        solver=[simple_qa_solver()],
        scorer=[simple_qa_scorer()],
        config=GenerateConfig(temperature=0.5),
    )
