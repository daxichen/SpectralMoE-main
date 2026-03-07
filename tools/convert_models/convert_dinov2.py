import torch
import os.path as osp
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
import sys
import numpy as np
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("pretrained", type=str)
    args.add_argument("converted", type=str)
    args.add_argument("--kernel", default=16, type=int)
    args.add_argument("--height", default=512, type=int)
    args.add_argument("--width", default=512, type=int)
    return args.parse_args()


def load_weight(pretrained_path):
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"{pretrained_path} dont exist(absolute path: {osp.abspath(pretrained_path)})"
        )
    weight = torch.load(pretrained_path, map_location="cpu")
    if len(weight.keys()) <= 10:
        print(f"The read weights may be abnormal, as shown below:")
        print(weight.keys())
        raise KeyError()
    return weight

def interpolate_patch_embed_(weight, key="patch_embed.proj.weight", kernel_conv=16):
    assert key in weight, f"{key} must in {weight.keys()}"
    ori_shape = weight[key].shape
    weight[key] = F.interpolate(
        weight[key].float(),
        size=(kernel_conv, kernel_conv),
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = weight[key].shape
    print(f"Convert conv kernel in patch embed layer: {ori_shape} -> {dst_shape}")

#--------------------------------------------------------------------------------------------------------
# def interpolate_patch_embed_(weight, key="patch_embed.proj.weight", kernel_conv=16, in_channels=4):
#     assert key in weight, f"{key} must in {weight.keys()}"
#     ori_shape = weight[key].shape
#     original_weight = weight[key].float()
#     new_weight = torch.zeros(1024, in_channels, 14, 14)
#     for i in range(in_channels):
#         new_weight[:, i, :, :] = original_weight[:, i % 3, :, :]
#     new_weight = F.interpolate(
#         new_weight.float(),
#         size=(kernel_conv, kernel_conv),
#         mode="bicubic",
#         align_corners=False,
#     )
#     weight[key] = new_weight
#     dst_shape = weight[key].shape
#     print(f"Convert conv kernel in patch embed layer: {ori_shape} -> {dst_shape}")
    
#--------------------------------------------------------------------------------------------------------
    

# def interpolate_patch_embed_(weight, key="patch_embed.proj.weight", kernel_conv=16, in_channels=4):
#     assert key in weight, f"{key} must in {weight.keys()}"
#     ori_shape = weight[key].shape
#     original_weight = weight[key].float()
    
#     # 创建通道映射规则 [Nir, R, G, B] -> [R, R, G, B]
#     channel_mapping = [0, 0, 1, 2]  # 将前两个通道映射到原始R通道
    
#     new_weight = torch.zeros(1024, in_channels, 14, 14)
#     for i in range(in_channels):
#         # 使用预定义的通道映射规则
#         original_channel = channel_mapping[i]
#         new_weight[:, i, :, :] = original_weight[:, original_channel, :, :]
    
#     new_weight = F.interpolate(
#         new_weight.float(),
#         size=(kernel_conv, kernel_conv),
#         mode="bicubic",
#         align_corners=False,
#     )
#     weight[key] = new_weight
#     dst_shape = weight[key].shape
#     print(f"Convert conv kernel in patch embed layer: {ori_shape} -> {dst_shape}")

# def interpolate_patch_embed_(
#     weight, 
#     key="patch_embed.proj.weight", 
#     kernel_conv=16, 
#     in_channels=4
# ):
#     assert key in weight, f"{key} must in {weight.keys()}"
#     original = weight[key].float()  # [out_c, orig_in_c, h, w]
#     out_c, orig_in_c, h, w = original.shape
    
#     # 通道维度插值（使用1D线性插值）
#     # 重组张量为 [out_c, h, w, orig_in_c]
#     channel_wise = original.permute(0, 2, 3, 1)  # 通道维度放在最后
    
#     # 转换为伪3D张量 [out_c*h*w, 1, orig_in_c]
#     channel_wise = channel_wise.reshape(-1, 1, orig_in_c)
    
#     # 1D插值扩展到目标通道数
#     channel_interp = F.interpolate(
#         channel_wise,
#         size=in_channels,
#         mode='linear',  # 1D线性插值
#         align_corners=False
#     )
    
#     # 恢复形状 [out_c, h, w, in_channels]
#     channel_interp = channel_interp.reshape(out_c, h, w, in_channels)
    
#     # 转回标准卷积权重格式 [out_c, in_channels, h, w]
#     weight_3d = channel_interp.permute(0, 3, 1, 2)
    
#     # 空间维度插值（保持原有逻辑）
#     weight[key] = F.interpolate(
#         weight_3d,
#         size=(kernel_conv, kernel_conv),
#         mode='bicubic',
#         align_corners=False
#     )
    
#     print(f"Convert conv kernel: {original.shape} -> {weight[key].shape}")

def interpolate_pos_embed_(
    weight: dict, key="pos_embed", crop_size=(512, 512), kernel_conv=16
):
    pos_cls, pos_tokens = weight[key][:, :1, :], weight["pos_embed"][:, 1:, :]
    embed_dim = pos_tokens.shape[-1]
    orig_size = int(pos_tokens.shape[-2] ** 0.5)
    orig_shape = (-1, orig_size, orig_size, embed_dim)
    crop_size = tuple(L // kernel_conv for L in crop_size)
    resized_pos_tokens = F.interpolate(
        pos_tokens.reshape(*orig_shape).permute(0, 3, 1, 2),
        size=crop_size,
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = resized_pos_tokens.shape
    resized_pos_tokens = resized_pos_tokens.permute(0, 2, 3, 1).reshape(
        -1, np.prod(crop_size), embed_dim
    )
    weight[key] = torch.cat((pos_cls, resized_pos_tokens), dim=1)
    print(
        f"Convert pos embedding: {pos_tokens.shape} -> {orig_shape} -> {dst_shape} -> {resized_pos_tokens.shape}"
    )


def main():
    args = parse_args()
    pretrained_path = args.pretrained
    converted_path = args.converted
    kernel_conv = args.kernel
    crop_size = (args.height, args.width)
    weight = load_weight(pretrained_path)
    print("Load from", pretrained_path)
    interpolate_patch_embed_(weight, kernel_conv=kernel_conv)
    interpolate_pos_embed_(weight, crop_size=crop_size, kernel_conv=kernel_conv)
    torch.save(weight, converted_path)
    print("Save to", converted_path)
    return args


# Check if the script is run directly (and not imported)
if __name__ == "__main__":
    main()
