from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from mmengine.model import BaseModule
from timm.models.vision_transformer import Block
from .DOFA.wave_dynamic_layer import Dynamic_MLP_OFA


@BACKBONES.register_module()
class DOFAVisionTransformer(BaseModule):
    """DOFA Vision Transformer for semantic segmentation tasks with MMSegmentation framework"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=45,
        global_pool=True,
        drop_path_rate=0.0,
        wv_planes=128,
        out_indices=[3, 5, 7, 11],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        wave_list=[0.83, 0.665, 0.56, 0.49],
        init_cfg=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim
            drop_path_rate (float): stochastic depth rate
            wv_planes (int): wave planes for dynamic MLP
            out_indices (list): output indices for multi-scale features
            norm_layer (nn.Module): normalization layer
            init_cfg (dict): initialization config
        """
        super().__init__(init_cfg)
        
        self.wv_planes = wv_planes
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.wave_list = wave_list

        # Dynamic patch embedding with wave information
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=wv_planes, 
            inter_dim=128, 
            kernel_size=patch_size, 
            embed_dim=embed_dim
        )
        
        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding (fixed sin-cos embedding)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # self.norm = norm_layer(embed_dim)

    def forward_features(self, x):
        # embed patches
        wavelist = torch.tensor(self.wave_list, device=x.device).float()
        self.waves = wavelist
        
        if x.shape[1] != len(self.wave_list):
            raise ValueError(
                f"Input channels ({x.shape[1]}) don't match wave_list length ({len(self.wave_list)}). "
                f"Expected {len(self.wave_list)} channels for wavelengths {self.wave_list}"
            )

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

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(out)

        return out_features

    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            wave_list (list): List of wavelength information
            
        Returns:
            list: Multi-scale features for segmentation
        """
            
        features = self.forward_features(x)
        
        # Apply interpolation for multi-scale features
        if len(features) >= 4:
            features[0] = F.interpolate(
                features[0], scale_factor=4, mode="bilinear", align_corners=False
            )
            features[1] = F.interpolate(
                features[1], scale_factor=2, mode="bilinear", align_corners=False
            )
            # features[2] remains unchanged (base scale)
            features[3] = F.interpolate(
                features[3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
        
        return features


# def dofa_vit_base_patch16(**kwargs):
#     """DOFA ViT-Base model with patch size 16"""
#     model = DOFAVisionTransformer(
#         out_indices=[3, 5, 7, 11],
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model


# def dofa_vit_large_patch16(**kwargs):
#     """DOFA ViT-Large model with patch size 16"""
#     model = DOFAVisionTransformer(
#         out_indices=[7, 11, 15, 23],
#         patch_size=16,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model


# def dofa_vit_huge_patch14(**kwargs):
#     """DOFA ViT-Huge model with patch size 14"""
#     model = DOFAVisionTransformer(
#         out_indices=[7, 15, 23, 31],
#         patch_size=14,
#         embed_dim=1280,
#         depth=32,
#         num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model
