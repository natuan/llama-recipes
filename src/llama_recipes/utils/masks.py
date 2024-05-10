import torch


@torch.no_grad()
def apply_masks(model):
    def mask_weights(module):
        if hasattr(module, "mask"):
            # print("Weight shape: {}, mask shape: {}".format(module.weight.size(), module.mask.size()))
            module.weight *= module.mask

    model.apply(mask_weights)


def attach_masks(model, to_layer=torch.nn.Linear, debug=False):
    for name, module in model.named_children():
        # we should make this more specific to avoid masking of unpruned layers
        # e.g.: project_in and project_out in OPT models
        if isinstance(module, to_layer):
            ## Only for debugging purposes, set sparsity to 10%
            # module.weight.data[torch.rand_like(module.weight) < 0.10] = 0

            mask = torch.where(
                module.weight == 0,
                torch.tensor(0, dtype=torch.uint8),
                torch.tensor(1, dtype=torch.uint8),
            )
            module.register_buffer("mask", mask, persistent=False)
            if debug:
                print(
                    f"[Debugging] attaching mask to {name} with sparsity = {torch.sum(mask == 0)/mask.numel()}"
                )
        else:
            attach_masks(module, to_layer)