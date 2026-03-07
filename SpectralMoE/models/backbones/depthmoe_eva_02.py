from .eva_02 import EVA2
from mmseg.models.builder import BACKBONES, MODELS
from .depthmoe import DepthMoE
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from .utils import set_requires_grad, set_train

import sys
from pathlib import Path

current_dir = Path(__file__).parent

depth_anything_dir = current_dir / "third_party" / "PromptDA"

sys.path.insert(0, str(depth_anything_dir))

from promptda.promptda import PromptDA

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
class DepthMoEEVA2(EVA2):
    def __init__(self, depthmoe_config=None, **kwargs):
        super().__init__(**kwargs)
        self.depthmoe: DepthMoE = MODELS.build(depthmoe_config)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.depth_anything = PromptDA().to(DEVICE).eval()
        self.depth_anything.pretrained.forward_features_extra = types.MethodType(forward_features_extra, self.depth_anything.pretrained)

    def forward_features(self, x):
        B, C, H, W = x.shape
        rgb = x[:, 1:, :, :]
        depth_features = self.depth_anything.pretrained.forward_features_extra(rgb)
        
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        lora_loss_list = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            x, cur_loss_moe = self.depthmoe.forward(
                x,
                depth_features[i],
                i,
                has_cls_token=True,
            )
            lora_loss_list.append(cur_loss_moe)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        lora_loss = torch.mean(torch.stack(lora_loss_list))
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
