import functools
import glob
import json
import os
import random
import re
from typing import Type

from inspect_ai._util.content import ContentText
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessage, ChatMessageAssistant
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.util import resource


def collapse_consecutive_assistant_messages(
        messages: list[ChatMessage],
) -> list[ChatMessage]:
    return functools.reduce(assistant_message_reducer, messages, [])


def assistant_message_reducer(
        messages: list[ChatMessage],
        message: ChatMessage,
) -> list[ChatMessage]:
    return consecutive_message_reducer(messages, message, ChatMessageAssistant)


def consecutive_message_reducer(
        messages: list[ChatMessage],
        message: ChatMessage,
        message_type: Type[ChatMessage],
) -> list[ChatMessage]:
    if (
            isinstance(message, message_type)
            and len(messages) > 0
            and isinstance(messages[-1], message_type)
    ):
        messages[-1] = combine_messages(messages[-1], message, message_type)
    else:
        messages.append(message)
    return messages


def combine_messages(
        a: ChatMessage, b: ChatMessage, message_type: Type[ChatMessage]
) -> ChatMessage:
    if isinstance(a.content, str) and isinstance(b.content, str):
        return message_type(content=f"{a.content}{b.content}")
    elif isinstance(a.content, list) and isinstance(b.content, list):
        return message_type(content=a.content + b.content)
    elif isinstance(a.content, str) and isinstance(b.content, list):
        return message_type(content=[ContentText(text=a.content), *b.content])
    elif isinstance(a.content, list) and isinstance(b.content, str):
        return message_type(content=a.content + [ContentText(text=b.content)])
    else:
        raise TypeError(
            f"Cannot combine messages with invalid content types: {a.content!r}, {b.content!r}"
        )


@solver
def prefill_message(template: str, **params) -> Solver:
    """Solver which prefills the assistant message with a template.

    Args:
      template (str): Template for prefill message.
      **params (dict[str,Any]): Parameters to fill into the template.

    Returns:
      A solver that inserts the parameterised assistant prefill message.
    """
    # read template
    content = resource(template)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwargs = state.metadata | params
        state.messages.append(ChatMessageAssistant(content=content.format(**kwargs)))
        return state

    return solve


@solver
def collapse_assistant_message() -> Solver:
    """Solver which collapses the assistant message.

    Returns:
      A solver that collapses the assistant message.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = collapse_consecutive_assistant_messages(state.messages)
        state.output.completion = state.messages[-1].text
        return state

    return solve


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

    def get_hop_count(filename):
        """Extract the number of hops from filename."""
        match = re.match(r'(\d+)hop\.json$', os.path.basename(filename))
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract hop count from filename: {filename}")

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Find all *hop.json files
    hop_files = glob.glob(os.path.join(directory_path, "*hop.json"))

    if not hop_files:
        raise ValueError(f"No *hop.json files found in {directory_path}")

    all_samples = []

    # Load each file manually
    for file_path in hop_files:
        num_hops = get_hop_count(file_path)

        with open(file_path, 'r') as f:
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
                    "raw_question": test_data["question"],
                    "raw_query": test_data["query"],
                    "num_hops": num_hops,
                    "source_file": os.path.basename(file_path)
                }
            )
            all_samples.append(sample)

    return MemoryDataset(all_samples)
