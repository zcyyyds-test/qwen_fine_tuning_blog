import torch
from modelscope import snapshot_download
import os
model_dir = snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', cache_dir='.', revision='master')