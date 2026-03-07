from mmseg.models.builder import BACKBONES, MODELS

from .depthmoe import DepthMoE
from .dino_v3 import DinoVisionTransformerv3
from .utils import set_requires_grad, set_train

import sys
from pathlib import Path

current_dir = Path(__file__).parent

depth_anything_dir = current_dir / "third_party" / "PromptDA"

sys.path.insert(0, str(depth_anything_dir))

from promptda.promptda import PromptDA


# depth_anything_dir = current_dir / "third_party" / "Depth-Anything-V2"

# sys.path.insert(0, str(depth_anything_dir))

# from depth_anything_v2.dpt import DepthAnythingV2

import torch
import torch.nn.functional as F

import types

def forward_features_extra(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            out.append(x)
        return out

@BACKBONES.register_module()
class DepthMoEDinoVisionTransformerV3(DinoVisionTransformerv3):
    def __init__(
        self,
        depthmoe_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depthmoe: DepthMoE = MODELS.build(depthmoe_config)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.depth_anything = PromptDA().to(DEVICE).eval()
        self.depth_anything.pretrained.forward_features_extra = types.MethodType(forward_features_extra, self.depth_anything.pretrained)
        
        # model_configs = {
        #     "vits": {
        #         "encoder": "vits",
        #         "features": 64,
        #         "out_channels": [48, 96, 192, 384],
        #     },
        #     "vitb": {
        #         "encoder": "vitb",
        #         "features": 128,
        #         "out_channels": [96, 192, 384, 768],
        #     },
        #     "vitl": {
        #         "encoder": "vitl",
        #         "features": 256,
        #         "out_channels": [256, 512, 1024, 1024],
        #     },
        #     "vitg": {
        #         "encoder": "vitg",
        #         "features": 384,
        #         "out_channels": [1536, 1536, 1536, 1536],
        #     },
        # }
        
        # self.depth_anything = DepthAnythingV2(**model_configs["vitl"])
        # self.depth_anything = self.depth_anything.to(DEVICE).eval()
        # self.depth_anything.pretrained.forward_features_extra = types.MethodType(forward_features_extra, self.depth_anything.pretrained)
        

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        rgb = x[:, 1:, :, :] # MSI Images
        # rgb = x # RGB Images
        depth_features = self.depth_anything.pretrained.forward_features_extra(rgb)
        
        H, W = h // self.patch_size, w // self.patch_size
        x_tokens, (grid_H, grid_W) = self.prepare_tokens_with_masks(x, masks)
        
        outs = []
        lora_loss_list = []
        for idx, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=grid_H, W=grid_W)
            else:
                rope_sincos = None
            
            x_tokens = blk(x_tokens, rope_sincos)
            x_tokens, cur_loss_moe = self.depthmoe.forward(
                x_tokens,
                depth_features[idx],
                idx,
                has_cls_token=True,
            )
            lora_loss_list.append(cur_loss_moe)
            if idx in self.out_indices:
                patch_tokens = x_tokens[:, self.n_storage_tokens + 1 :]
                feature_map = patch_tokens.permute(0, 2, 1).reshape(
                    B, -1, H, W
                )
                outs.append(feature_map.contiguous())
        
        lora_loss = torch.mean(torch.stack(lora_loss_list))
        return self.depthmoe.return_auto(outs), lora_loss
    
    def forward(self, x, masks=None):
        features, lora_loss = self.forward_features(x, masks)

        if isinstance(features[0], torch.Tensor):
            # 处理不同层次的特征图缩放
            features[0] = F.interpolate(
                features[0], scale_factor=4, mode="bilinear", align_corners=False
            )
            features[1] = F.interpolate(
                features[1], scale_factor=2, mode="bilinear", align_corners=False
            )
            features[3] = F.interpolate(
                features[3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
        else:
            features[0][0] = F.interpolate(
                features[0][0], scale_factor=4, mode="bilinear", align_corners=False
            )
            features[0][1] = F.interpolate(
                features[0][1], scale_factor=2, mode="bilinear", align_corners=False
            )
            features[0][3] = F.interpolate(
                features[0][3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
            
        return features, lora_loss

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["depthmoe"])
        set_train(self, ["depthmoe"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "depthmoe" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
