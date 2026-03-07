from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import MODELS
from mmseg.structures.seg_data_sample import SegDataSample

@MODELS.register_module()
class DepthMoEMask2FormerHead(Mask2FormerHead):
    def __init__(self, replace_query_feat=False, **kwargs):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

    def forward(
        self, x: Tuple[Tuple[List[Tensor], List[Tensor]], Tensor], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, lora_loss = x
        x, query_embed = x # x: list[0, 1, 2, 3]--[4, 1024, 32, 32], query_embed: [100, 256]
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1) # [4, 100, 256]
        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x) # mask_features: [4, 1024, 32, 32], multi_scale_memorys: [4, 1024, 32, 32], [4, 1024, 16, 16], [4, 1024, 8, 8], [4, 1024, 4, 4]
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i]) # [4, 1024, 32, 32]
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1) # [4, 1024, 1024]
            level_embed = self.level_embed.weight[i].view(1, 1, -1) # [1, 1, 256]
            decoder_input = decoder_input + level_embed # [4, 1024, 1024]
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            ) # [4, 32, 32]
            decoder_positional_encoding = self.decoder_positional_encoding(mask) # [4, 1024, 32, 32]
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1) # [4, 1024, 1024]
            decoder_inputs.append(decoder_input) # [4, 1024, 1024]
            decoder_positional_encodings.append(decoder_positional_encoding) # [4, 1024, 1024]
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed) # [4, 100, 256] -- [4, 100, 256]
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1)) # [4, 100, 256]

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        ) # cls_pred: [4, 100, num_classes], mask_pred: [4, 100, 32, 32], attn_mask: [4, 100, 1024]
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            ) # [4, 100, 256]
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            ) # cls_pred: [4, 100, num_classes], mask_pred: [4, 100, 32, 32], attn_mask: [4, 100, 1024]

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, lora_loss#, orthogonal_loss
    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds , lora_loss = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        
        losses['loss_lora'] = lora_loss

        return losses
    
    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # size = size[:2] # test tuxiang no label
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits
