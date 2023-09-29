import copy
import torch
import datasets
from llama_recipes.datasets.utils import Concatenator

TEST_MODE = True

SEED = 2023

NEED_DEV_SET = False

use_concat = False

IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

def get_preprocessed_gsm8k(dataset_config, tokenizer, split):
    if split == "train":
        dataset = datasets.load_dataset("gsm8k", "main", split="train")
        if NEED_DEV_SET:
            print("Reducing training set")
            split_ds = dataset.train_test_split(test_size=1319, seed=SEED)
            dataset = split_ds.pop("train")
    elif split == "test":
        if TEST_MODE:
            print("Using the real test set")
            dataset = datasets.load_dataset("gsm8k", "main", split="test")
        else:
            print("Using a portion of the training set for validation")
            dataset = datasets.load_dataset("gsm8k", "main", split="train")
            split_ds = dataset.train_test_split(test_size=1319, seed=SEED)
            dataset = split_ds.pop("test")
    else:
        raise ValueError("Unknow split type")

    def process_sample(sample):
        prompt = f"Question: {{question}}.\nAnswer:".format(
            question=sample["question"],
        )
        example = f"Question: {{question}}.\nAnswer: {{answer}}{{eos_token}}".format(
            question=sample["question"],
            answer=sample["answer"],
            eos_token=tokenizer.eos_token,
        )
        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
        example = torch.tensor(tokenizer.encode(example), dtype=torch.int64)
        max_seq_len = 1024  # tokenizer.max_position_embeddings
        padding = max_seq_len - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: max_seq_len]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
    dataset = dataset.map(
        lambda sample: process_sample(sample),
        batched=False,
        remove_columns=list(dataset.features),
    )
    return dataset
