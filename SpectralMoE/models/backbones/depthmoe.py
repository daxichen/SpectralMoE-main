from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from torch import nn, einsum
from einops import rearrange, repeat
import random
from math import sqrt
import numpy as np
from typing import Tuple, Union, List, Any

@MODELS.register_module()
class DepthMoE(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        num_experts: int = 6,
        lora_dims: list = [16, 16, 16, 16, 16, 16],
        noisy_gating: bool = True,
        top_k: int = 1,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        gating: str = 'softmax',  # 'softmax', 'laplace', 'gaussian'
        num_modalities: int = 2  # MSI & Depth
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.num_experts = num_experts
        self.lora_dims = lora_dims
        self.noisy_gating = noisy_gating
        self.top_k = top_k
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.gating = gating
        self.num_modalities = num_modalities
        self.create_moe_model()

    def create_moe_model(self):
        self.experts_a = nn.ModuleList([
        nn.ParameterList([  
            nn.Parameter(  # [token_len, r_i]
                torch.empty(self.token_length, self.lora_dims[expert_idx]))
                for expert_idx in range(self.num_experts)
            ])
            for _ in range(self.num_layers)  
        ])
    
        self.experts_b = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.empty(self.lora_dims[expert_idx], self.embed_dims)) 
                for expert_idx in range(self.num_experts)
            ])
            for _ in range(self.num_layers)
        ])
        
        
        for layer_idx in range(self.num_layers):
            for expert_idx in range(self.num_experts):
                
                nn.init.kaiming_uniform_(
                    self.experts_a[layer_idx][expert_idx],
                    a=math.sqrt(5),       
                    mode='fan_in',        
                    nonlinearity='leaky_relu'
                )
                nn.init.zeros_(self.experts_b[layer_idx][expert_idx])
        
        # Local Gate & Noise Parameters for each modality
        self.w_gate = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(self.embed_dims, self.num_experts), requires_grad=True),  # MSI
                nn.Parameter(torch.zeros(self.embed_dims, self.num_experts), requires_grad=True)   # Depth
            ])
            for _ in range(self.num_layers)
        ])
        self.w_noise = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.zeros(self.embed_dims, self.num_experts), requires_grad=True),  # MSI
                nn.Parameter(torch.zeros(self.embed_dims, self.num_experts), requires_grad=True)   # Depth
            ])
            for _ in range(self.num_layers)
        ])
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        
        self.transform = nn.Linear(self.embed_dims, self.query_dims) # [1024, 256]
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims) # [768, 256]
        
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        assert(self.top_k <= self.num_experts)
        
    def get_layer_tokens(self, layer_idx: int) -> Tensor:
        """获取指定层所有专家Token的聚合结果（平均或求和）"""
        # 直接收集所有专家的Token到列表
        expert_tokens_list = [
            self.get_expert_tokens(layer_idx, expert_idx)
            for expert_idx in range(self.num_experts)
        ]
        
        # 堆叠后按维度求聚合 [num_experts, token_len, embed_dim] → [token_len, embed_dim]
        all_tokens = torch.stack(expert_tokens_list, dim=0)  
        return all_tokens.mean(dim=0)
    
    def get_all_tokens(self) -> Tensor:
        """获取所有层Token的堆叠 [num_layers, token_len, embed_dim]"""
        return torch.stack([
            self.get_layer_tokens(layer_idx)
            for layer_idx in range(self.num_layers)
        ], dim=0)
        
    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_all_tokens()).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats
        
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
        
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = torch.distributions.normal.Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def _get_logits(self, x, noise_epsilon, layer_idx, idx=None):
        
        w_gate = self.w_gate[layer_idx][idx].to(x.device)
        w_noise = self.w_noise[layer_idx][idx].to(x.device)

        if self.gating == 'softmax':
            clean_logits = x @ w_gate
        elif self.gating == 'laplace':
            clean_logits = -torch.cdist(x, torch.t(w_gate))
        elif self.gating == 'gaussian':
            clean_logits = -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)

        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noise_stddev = None
            noisy_logits = clean_logits
        return logits, clean_logits, noisy_logits, noise_stddev
    
    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, k):
        top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]
        if self.gating == 'softmax':
            top_k_gates = self.softmax(top_k_logits)
        elif self.gating == 'laplace' or self.gating == 'gaussian':
            top_k_gates = torch.exp(top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True))

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.training and k < self.num_experts and noise_stddev is not None:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    
    def noisy_top_k_gating(self, msi_feats, depth_feats, layer_idx, noise_epsilon=1e-2):
        """
        双路由网络门控
        Args:
            msi_feats: MSI特征 [B*N, D]
            depth_feats: Depth特征 [B*N, D]  
            layer_idx: 当前层索引
            noise_epsilon: 噪声epsilon
        Returns:
            msi_gates, depth_gates: 两个模态的门控值
            msi_load, depth_load: 两个模态的负载
        """
        
        # MSI路由网络 (idx=0)
        msi_logits = self._get_logits(msi_feats, noise_epsilon, layer_idx, idx=0)
        msi_gates, msi_load = self._top_k_gating(*msi_logits, self.top_k)
        
        # Depth路由网络 (idx=1)  
        depth_logits = self._get_logits(depth_feats, noise_epsilon, layer_idx, idx=1)
        depth_gates, depth_load = self._top_k_gating(*depth_logits, self.top_k)
        
        return msi_gates, depth_gates, msi_load, depth_load
    
    def get_expert_tokens(self, layer_idx: int, expert_idx: int) -> Tensor:
        """获取指定层指定专家的低秩token表示"""
        # 确保层索引有效
        assert layer_idx < self.num_layers, f"Invalid layer idx {layer_idx}"
        
        # 提取对应层的参数
        a = self.experts_a[layer_idx][expert_idx]  # 先选层：[num_layers, token, r_i] → [token, r_i]
        b = self.experts_b[layer_idx][expert_idx]  # 同理 [r_i, dim]
        
        # 生成实际token参数
        return a @ b  # [token_len, dim]

    def forward_delta_feat(self, msi_feats: Tensor, depth_feats: Tensor, layer: int, loss_coef=1e3) -> Tensor:
        
        B, N, D = msi_feats.shape
        msi_flat = msi_feats.reshape(-1, D)    # [B*N, D]
        depth_flat = depth_feats.reshape(-1, D)  # [B*N, D]
        
        msi_gates, depth_gates, msi_load, depth_load = self.noisy_top_k_gating(
            msi_flat, depth_flat, layer
        )
        
        msi_importance = msi_gates.sum(0)
        depth_importance = depth_gates.sum(0)
        
        msi_loss = self.cv_squared(msi_importance) + self.cv_squared(msi_load)
        depth_loss = self.cv_squared(depth_importance) + self.cv_squared(depth_load)
        loss = (msi_loss + depth_loss) * loss_coef
        
        # MSI路由处理
        msi_dispatcher = SparseDispatcher(self.num_experts, msi_gates)
        msi_expert_inputs = msi_dispatcher.dispatch(msi_flat)
        
        msi_expert_outputs = []
        for expert_idx in range(self.num_experts):
            if len(msi_expert_inputs[expert_idx]) == 0:
                msi_expert_outputs.append(torch.empty(0, D, device=msi_feats.device))
                continue
            
            tokens = self.get_expert_tokens(layer, expert_idx) # [token_len, D]
            msi_attn = torch.einsum("bc,td->bt", msi_expert_inputs[expert_idx], tokens)
            if self.use_softmax:
                msi_attn = msi_attn * (self.embed_dims ** -0.5)
                msi_attn = F.softmax(msi_attn, dim=-1)
                
            delta_msi = torch.einsum("bt,td->bd", msi_attn[:, 1:], self.mlp_token2feat(tokens[1:, :]))
            msi_expert_outputs.append(delta_msi)
            
        msi_output = msi_dispatcher.combine(msi_expert_outputs)
        msi_output = msi_output.reshape(B, N, D)
        
        
        # Depth路由处理
        depth_dispatcher = SparseDispatcher(self.num_experts, depth_gates)
        depth_expert_inputs = depth_dispatcher.dispatch(depth_flat)
        
        depth_expert_outputs = []
        for expert_idx in range(self.num_experts):
            if len(depth_expert_inputs[expert_idx]) == 0:
                depth_expert_outputs.append(torch.empty(0, D, device=depth_feats.device))
                continue
                
            tokens = self.get_expert_tokens(layer, expert_idx)
            depth_attn = torch.einsum("bc,td->bt", depth_expert_inputs[expert_idx], tokens)
            if self.use_softmax:
                depth_attn = depth_attn * (self.embed_dims ** -0.5)
                depth_attn = F.softmax(depth_attn, dim=-1)

            delta_depth = torch.einsum("bt,td->bd", depth_attn[:, 1:], self.mlp_token2feat(tokens[1:, :]))
            depth_expert_outputs.append(delta_depth)
        
        depth_output = depth_dispatcher.combine(depth_expert_outputs)
        depth_output = depth_output.reshape(B, N, D)
        
        attn = torch.einsum("bnc,bmc->bnm", depth_output, msi_output)  # [B, N, N]
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)
            
        delta = torch.einsum("bnn,bnc->bnc", attn, msi_output)
        
        delta_f = self.mlp_delta_f(delta+msi_feats)
        
        return delta_f, loss
        
    def forward(self, feats: Tensor, depth_features: Tensor, layer: int, has_cls_token=True) -> Tensor:
        if has_cls_token:
            # cls_token, feats = torch.tensor_split(feats, [1], dim=1) # dinov2
            cls_token, storage_tokens, feats = torch.tensor_split(feats, [1, 5], dim=1) # dinov3
            _, depth_features = torch.tensor_split(depth_features, [1], dim=1)
        # _, depth_features = torch.tensor_split(depth_features, [1], dim=1) # SAM Model
        
        # 计算混合专家增量
        delta_feat, loss_moe = self.forward_delta_feat(feats, depth_features, layer)
        delta_feat *= self.scale
        
        feats = feats + delta_feat
        if has_cls_token:
            # feats = torch.cat([cls_token, feats], dim=1) # dinov2
            feats = torch.cat([cls_token, storage_tokens, feats], dim=1) # DINOv3
        return feats, loss_moe
    
    
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)