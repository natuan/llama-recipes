# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str = "PATH/to/LLAMA/7B"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    gradient_accumulation_steps: int = 1
    use_gradient_clipping: bool = False
    gradient_clipping_thresh: float = 1.0
    max_position_embeddings: int = 1024
    num_epochs: int = 3
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    lr_scheduler: str = "linear"
    sparse_training: bool = False
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    gamma: float = 0.85
    dev_set_seed: float = 2023
    test_as_dev: bool = True
    use_custom_loss: bool = False
    result_loss_weight: float = 0.5  # Specific for GSM8K; Todo: make this more general
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "samsum_dataset"
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = (
        "PATH/to/save/FSDP/model"  # will be used if using FSDP
    )
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels


@dataclass
class kd_config:
    teacher_model_path: str = None
    layerwise: bool = False
    layerwise_loss: str = "mse_normalized"
    temperature: float = 1.0
    # loss = hardness_ce x CrossEntropyLoss + hardness_kd_out x KLDivLoss + hardness_kd_layerwise x LayerwiseTokenL2Loss
    hardness_ce: float = 1.0
    hardness_output: float = 1.0
    hardness_layerwise: float = 0.5
