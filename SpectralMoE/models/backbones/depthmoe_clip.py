from mmseg.models.builder import BACKBONES, MODELS
from .depthmoe import DepthMoE
from .clip import CLIPVisionTransformer
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
class DepthMoECLIPVisionTransformer(CLIPVisionTransformer):
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

    def forward(self, x: torch.Tensor):
        
        rgb = x[:, 1:, :, :]

        depth_features = self.depth_anything.pretrained.forward_features_extra(rgb)
        
        # depth_features = self.depth_anything.pretrained.forward_features_extra(x) # RGB Images

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:,]
            .reshape(1, self.spatial_size, self.spatial_size, C)
            .permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        lora_loss_list = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            x, cur_loss_moe = self.depthmoe.forward(x, depth_features[i].permute(1, 0, 2).contiguous(), i, has_cls_token=True)
            lora_loss_list.append(cur_loss_moe)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        lora_loss = torch.mean(torch.stack(lora_loss_list))
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = (
                x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
            )  # B C H W

            features.append([global_embedding, visual_embedding])
        # features[0] = F.interpolate(
        #     features[0], scale_factor=4, mode="bilinear", align_corners=False
        # )
        # features[1] = F.interpolate(
        #     features[1], scale_factor=2, mode="bilinear", align_corners=False
        # )
        # features[3] = F.interpolate(
        #     features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        # )
        return self.depthmoe.return_auto(features), lora_loss

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["fpn", "depthmoe"])
        set_train(self, ["fpn", "depthmoe"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if ("depthmoe" not in k) and ('fpn' not in k)]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
