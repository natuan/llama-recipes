import datasets
from llama_recipes.datasets.utils import Concatenator

TEST_MODE = True

SEED = 2023

NEED_DEV_SET = False

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

    def apply_prompt_template(sample):
        return {
            "text": f"Question: {{question}}.\nAnswer: {{answer}}".format(
                question=sample["question"],
                answer=sample["answer"],
                eos_token=tokenizer.eos_token,
            )
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
