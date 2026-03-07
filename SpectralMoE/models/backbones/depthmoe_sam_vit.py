from mmseg.models.builder import BACKBONES, MODELS
from .depthmoe import DepthMoE
from .sam_vit import SAMViT
from .utils import set_requires_grad, set_train

import sys
from pathlib import Path

current_dir = Path(__file__).parent

depth_anything_dir = current_dir / "third_party" / "PromptDA"

sys.path.insert(0, str(depth_anything_dir))

from promptda.promptda import PromptDA


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
class DepthMoESAMViT(SAMViT):
    def __init__(
        self,
        depthmoe_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depthmoe_enabled_layers: list = kwargs.get("global_attn_indexes")

        self.depthmoe: DepthMoE = MODELS.build(depthmoe_config)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.depth_anything = PromptDA().to(DEVICE).eval()
        self.depth_anything.pretrained.forward_features_extra = types.MethodType(forward_features_extra, self.depth_anything.pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        rgb = x[:, 1:, :, :]
        depth_features = self.depth_anything.pretrained.forward_features_extra(rgb)
        
        x = self.patch_embed(x)
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if self.pos_embed is not None:
            x = x + self.pos_embed
        features = []
        lora_loss_list = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            B, H, W, C = x.shape
            if idx in self.depthmoe_enabled_layers:
                x, cur_loss_moe = self.depthmoe.forward(
                    x.view(B, -1, C),
                    depth_features[idx],
                    self.depthmoe_enabled_layers.index(idx),
                    has_cls_token=False,
                )
                x = x.view(B, H, W, C)
                lora_loss_list.append(cur_loss_moe)
            # 4,32,32,768
            
            if idx in self.out_indices:
                features.append(x.permute(0, 3, 1, 2))
        lora_loss = torch.mean(torch.stack(lora_loss_list))
        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        
        return self.depthmoe.return_auto(features), lora_loss

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
