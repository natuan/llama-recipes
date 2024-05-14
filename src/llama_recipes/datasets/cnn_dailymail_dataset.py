import copy

import torch

from datasets import load_dataset

IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

DEBUG = False
NSAMPLES = 15000


def get_preprocessed_cnn_dailymail(
    dataset_config,
    tokenizer,
    split="train",
):
    assert split in ["train", "validation", "test"]
    dataset = load_dataset("cnn_dailymail", dataset_config.subset, split=split)
    if DEBUG and split == "train":
        dataset = dataset.select(range(NSAMPLES))

    max_seq_len = dataset_config.max_seq_len
    
    def process_sample(sample):
        prompt = f"{{bos_token}}Article:\n{{article}}\n\n### Summarization:\n".format(
            bos_token=tokenizer.bos_token,
            article=sample["article"],
        )
        example = prompt + f"{{highlights}}{{eos_token}}".format(
            highlights=sample["highlights"],
            eos_token=tokenizer.eos_token,
        )
        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
        example = torch.tensor(tokenizer.encode(example), dtype=torch.int64)

        padding = max_seq_len - example.shape[0]
        if padding > 0:
            example = torch.cat(
                (example, torch.zeros(padding, dtype=torch.int64) - 1)
            )
        elif padding < 0:
            example = example[:max_seq_len]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = tokenizer.pad_token_id
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        res = {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
        return res

    dataset = dataset.map(
        lambda sample: process_sample(sample),
        batched=False,
        num_proc=64,
        remove_columns=list(dataset.features),
    )

    return dataset
