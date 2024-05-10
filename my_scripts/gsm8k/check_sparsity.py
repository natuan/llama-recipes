from safetensors import safe_open
import os

model_dir = "/home/tuan/models02/llama3/gsm8k_v2/llama-recipes/sparse_finetuned/ongoing/Src14891@linear@lr5e-5@B@GrAcc1@W0.1@ep1@GPUs8@WD0.0@ID28682-/hf"
tensors_file = "model-00007-of-00007.safetensors"

tensors = {}
with safe_open(os.path.join(model_dir, tensors_file), framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
import pdb; pdb.set_trace()
t=0