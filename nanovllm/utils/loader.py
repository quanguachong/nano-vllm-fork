import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器：直接将读取的 tensor 拷贝到 parameter.data"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从指定目录下的 safetensors 文件加载权重到模型。

    主要逻辑：
    1. 遍历 path 下所有 *.safetensors 文件，逐个打开并读取其中的权重名与张量。
    2. 对每个权重名做两类处理：
       - 打包权重（packed）：若权重名匹配模型定义的 packed_modules_mapping（例如 HF
         将 q_proj/k_proj/v_proj 存为 qkv_proj，或 gate_proj/up_proj 存为 gate_up_proj），
         则用「映射后的参数名」找到对应 Parameter，并调用该参数上的 weight_loader(param, tensor, shard_id)，
         由 weight_loader 负责按 shard_id 拆解或按张量并行分片写入（见 Linear 层中的 QKVParallelLinear、ColumnParallelLinear 等）。
       - 普通权重：直接用权重名 get_parameter，再调用参数上的 weight_loader（若有）或 default_weight_loader，
         将 f.get_tensor(weight_name) 写入 param.data（张量并行时 weight_loader 会做切分与拷贝）。

    这样实现可以同时支持：HF 的「多模块打包成一个文件」的格式、以及张量并行下按 rank 只加载本 rank 所需分片。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否为「打包」权重（HF 中多个 proj 合并为一个 key）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重：按权重名取参数，用自定义或默认的 weight_loader 写入（含张量并行切分）
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
