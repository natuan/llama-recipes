# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class gsm8k_dataset:
    dataset: str =  "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "test"
    max_seq_len: int = 512

@dataclass
class gsm8k_v2_dataset:
    dataset: str =  "gsm8k_v2_dataset"
    train_split: str = "train"
    test_split: str = "test"
    max_seq_len: int = 512


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class cnn_dailymail_dataset:
    dataset: str = "cnn_dailymail"
    subset: str = "3.0.0"
    max_seq_len: int = 2048
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
