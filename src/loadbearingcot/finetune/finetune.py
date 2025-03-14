import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig


def get_chat_format(example: Dict) -> List[Dict]:
    """
    Format the example as a chat format compatible with chat templates.

    Args:
        example: Single example from the dataset

    Returns:
        List of message dictionaries in chat format
    """
    # Create the user message with question and query
    user_content = f"{example['question']}\n{example['query']}"

    # Create the assistant message with chain of thought and answer
    cot = "\n".join(example["chain_of_thought"])
    assistant_content = (
        f"<think>\n{cot}\n</think>\n\n<answer>{example['answer']}</answer>"
    )

    # Return formatted messages
    return [
        {
            "role": "system",
            "content": "Given a set of logical statements, determine if the last statement is true or false. Think carefully about the task. Provide your step-by-step reasoning between <think></think> tags. Finally, write your answer in between <answer></answer> tags.",
        },
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def load_hops_dataset(data_path: str, split_ratio: float = 0.9) -> DatasetDict:
    """
    Load the hops dataset from JSON files in a directory and split it into train/validation sets.

    Args:
        data_path: Path to the directory containing hops dataset JSON files
        split_ratio: Ratio for train/validation split

    Returns:
        A DatasetDict containing train and validation splits
    """
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found at {path}")

    # Process the data into a format suitable for the model
    processed_data = []

    # Find all JSON files in the directory
    json_files = list(path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {path}")

    # Load and process each file
    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Process each item in the data
        for item in data:
            # Process both questions in the pair
            for q_key in ["question1", "question2"]:
                q_data = item[q_key]
                # Combine the question, chain of thought, and answer
                entry = {
                    "question": q_data["question"],
                    "query": q_data["query"],
                    "chain_of_thought": q_data["chain_of_thought"],
                    "answer": q_data["answer"],
                    "nhops": item["nhops"],
                    "pair_id": item["pair_id"],
                    "source_file": file_path.name,
                }
                processed_data.append(entry)

    # Create dataset and split into train/validation
    dataset = Dataset.from_dict(
        {k: [d.get(k) for d in processed_data] for k in processed_data[0].keys()}
    )

    # Split into train and validation
    train_test_split = dataset.train_test_split(train_size=split_ratio, seed=42)
    return DatasetDict(
        {"train": train_test_split["train"], "valid": train_test_split["test"]}
    )


def get_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """
    Load the base model and tokenizer without applying PEFT.

    Args:
        model_name: Name or path of the pretrained model

    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding side to left for flash attention 2
    tokenizer.padding_side = "left"

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_response_template_ids(tokenizer):
    # Create dummy messages to extract the template
    dummy_user_message = "dummy user message"
    dummy_assistant_message = "dummy assistant message"

    # Format as chat messages
    messages = [
        {"role": "user", "content": dummy_user_message},
        {"role": "assistant", "content": dummy_assistant_message},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)

    # Find the position where assistant message starts
    user_end_pos = formatted.find(dummy_user_message) + len(dummy_user_message)
    assistant_start_pos = formatted.find(dummy_assistant_message)

    # Extract the template between user and assistant content
    response_template = formatted[user_end_pos:assistant_start_pos]

    # Encode the template
    return tokenizer.encode(response_template, add_special_tokens=False)


def plot_token_length_distribution(tokenized_dataset, output_dir=None):
    """
    Create and save a plot showing the distribution of token lengths in the dataset.

    Args:
        tokenized_dataset: Dictionary containing tokenized datasets (train/valid)
        output_dir: Directory to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))

    # Collect token lengths for each split
    for split, dataset in tokenized_dataset.items():
        token_lengths = [len(example["input_ids"]) for example in dataset]

        # Plot histogram
        plt.hist(
            token_lengths,
            bins=50,
            alpha=0.7,
            label=f"{split} (mean: {np.mean(token_lengths):.1f})",
        )

        # Print stats
        print(f"{split} token length stats:")
        print(f"  Min: {min(token_lengths)}")
        print(f"  Max: {max(token_lengths)}")
        print(f"  Mean: {np.mean(token_lengths):.2f}")
        print(f"  Median: {np.median(token_lengths):.2f}")
        print(f"  95th percentile: {np.percentile(token_lengths, 95):.2f}")
        print(f"  99th percentile: {np.percentile(token_lengths, 99):.2f}")

    plt.xlabel("Token Length")
    plt.ylabel("Number of Examples")
    plt.title("Distribution of Token Lengths")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "token_length_distribution.png"))
        print(
            f"Token length distribution plot saved to {os.path.join(output_dir, 'token_length_distribution.png')}"
        )
    else:
        plt.show()


def main(args):
    """
    Main training function.

    Args:
        args: Command line arguments
    """
    # Load base model and tokenizer without PEFT
    model, tokenizer = get_model_and_tokenizer(args.model)

    # Create PEFT config if needed
    peft_config = None
    if not args.no_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",  # Using literal value instead of string
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    # Load dataset
    dataset = load_hops_dataset(args.dataset_path)

    # Define tokenization function using chat template
    def tokenize(element):
        formatted = tokenizer.apply_chat_template(
            get_chat_format(element), tokenize=False
        )
        outputs = tokenizer(formatted)
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # Tokenize the dataset
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(tokenize)

    # Plot token length distribution if requested
    if args.plot_token_lengths:
        plot_token_length_distribution(tokenized_dataset, args.output_dir)
        if not args.train_after_plot:
            return

    # Create data collator for completion-only LM
    response_template_ids = get_response_template_ids(tokenizer)
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    # Set up SFT configuration
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=None,
        # torch_compile=True,
    )

    # Initialize SFT trainer with PEFT config
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        peft_config=peft_config,  # Pass the PEFT config here
        processing_class=tokenizer,
        data_collator=collator,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir + "/final_model")
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    # Final evaluation
    print("Performing final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {final_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on the hops dataset"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="~/GitHub/LoadBearingCoT/src/loadbearingcot/data/prontoqa/paired_data/",
        help="Path to the dataset directory containing JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for the model",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--epochs", type=int, default=0.02, help="Number of training epochs"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    # Add PEFT-related arguments
    parser.add_argument(
        "--no_peft",
        action="store_true",
        help="Disable PEFT/LoRA (use full fine-tuning)",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--plot_token_lengths",
        action="store_true",
        help="Plot distribution of token lengths",
    )
    parser.add_argument(
        "--train_after_plot",
        action="store_true",
        help="Continue with training after plotting",
    )

    args = parser.parse_args()
    main(args)
