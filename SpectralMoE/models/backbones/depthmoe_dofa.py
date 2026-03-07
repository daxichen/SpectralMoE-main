from mmseg.models.builder import BACKBONES, MODELS
from .depthmoe import DepthMoE
from .dofa import DOFAVisionTransformer
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
class DepthMoEDOFAVisionTransformer(DOFAVisionTransformer):
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
        

    def forward_features(self, x):

        # rgb = x[:, [1, 6, 14], :, :] # WHU-OHS
        rgb = x[:, 1:, :, :] # GF2
        # rgb = x[:, :3, :, :] # FLAIR
        depth_features = self.depth_anything.pretrained.forward_features_extra(rgb)
        
        wavelist = torch.tensor(self.wave_list, device=x.device).float()
        self.waves = wavelist
        
        x, _ = self.patch_embed(x, self.waves)
        
        hw = self.img_size // self.patch_embed.kernel_size
        hw_shape = (hw, hw)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out_features = []
        lora_loss_list = []

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x, cur_loss_moe = self.depthmoe.forward(
                x,
                depth_features[idx],
                idx,
                has_cls_token=True,
            )
            lora_loss_list.append(cur_loss_moe)
            if idx in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(out)
        lora_loss = torch.mean(torch.stack(lora_loss_list))
        return self.depthmoe.return_auto(out_features), lora_loss

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
