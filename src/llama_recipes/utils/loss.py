import copy
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from transformers import LlamaConfig, LlamaForCausalLM


def register_custom_loss(model: LlamaForCausalLM, loss_key: str, **kwargs):
    if not hasattr(model, "custom_loss"):
        raise RuntimeError("Model must have custom_loss attribute to register")
    custom_loss_map = {
        "GSM8K_AccuracyAwareLoss": GSM8K_AccuracyAwareLoss
    }
    if loss_key not in custom_loss_map:
        raise ValueError(f"Unknown loss class. Supported: {list(custom_loss_map.keys())}")
    custom_loss = custom_loss_map[loss_key](model.config, **kwargs)
    setattr(model, "custom_loss", custom_loss)


class GSM8K_AccuracyAwareLoss(Module):
    def __init__(self, config: LlamaConfig, result_loss_weight: float = None) -> None:
        super().__init__()
        self.config = config
        if result_loss_weight is not None and not (0.0 < result_loss_weight < 1.0):
            raise ValueError("Invalid weight for result part in the loss")
        self.result_loss_weight = result_loss_weight

    def forward(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        result_mask = kwargs.get("result_mask", None)
        if result_mask is None:
            loss = self._cross_entropy_loss(logits, labels)
        else:
            result_labels = copy.deepcopy(labels)
            result_labels[~result_mask] = -100  # Ignored by CE loss
            result_loss = self._cross_entropy_loss(logits, result_labels)

            chain_of_thoughts_labels = copy.deepcopy(labels)
            chain_of_thoughts_labels[result_mask] = -100  # Ignored by CE loss
            chain_of_thoughts_loss = self._cross_entropy_loss(logits, chain_of_thoughts_labels)

            total_loss = result_loss * self.result_loss_weight + chain_of_thoughts_loss * (1.0 - self.result_loss_weight)
            loss = {
                "result_loss": result_loss,
                "chain_of_thoughts_loss": chain_of_thoughts_loss,
                "loss": total_loss
            }
        return loss

    def _cross_entropy_loss(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss
