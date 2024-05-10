from dataclasses import dataclass

@dataclass
class kd_config:
    enabled: bool = False
    layerwise: bool = True
    output: bool = False
    temperature: float = 1.0
    # loss = hardness_ce x CrossEntropyLoss + hardness_kd_output x KLDivLoss + hardness_kd_layerwise x LayerwiseTokenL2Loss
    hardness_ce: float = 1.0
    hardness_kd_output: float = 0.0
    hardness_kd_layerwise: float = 1.0
    enable_fsdp: bool = False
    teacher_model_path: str = None
    layerwise_loss: str = 'mse_normalized'
