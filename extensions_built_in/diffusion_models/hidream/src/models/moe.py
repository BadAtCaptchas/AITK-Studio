import math
import torch
from torch import nn
import torch.nn.functional as F
from .attention import FeedForwardSwiGLU
import torch.distributed as dist


def _mean_across_workers(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.aux_loss_shape = aux_loss.shape
        ctx.aux_loss_device = aux_loss.device
        ctx.aux_loss_dtype = aux_loss.dtype
        ctx.aux_loss_requires_grad = aux_loss.requires_grad
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_aux_loss = None
        if ctx.aux_loss_requires_grad:
            grad_aux_loss = torch.ones(
                ctx.aux_loss_shape,
                device=ctx.aux_loss_device,
                dtype=ctx.aux_loss_dtype,
            )
        return grad_output, grad_aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.last_routing_stats = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def _load_balancing_loss(self, scores, topk_idx, bsz, seq_len):
        if self.alpha <= 0.0:
            self.last_routing_stats = None
            return None

        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores.view(bsz, seq_len, -1)
            expert_usage = torch.zeros(
                bsz,
                self.n_routed_experts,
                device=scores.device,
                dtype=scores.dtype,
            )
            expert_usage.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(
                    bsz,
                    seq_len * self.top_k,
                    device=scores.device,
                    dtype=scores.dtype,
                ),
            ).div_(seq_len * self.top_k / self.n_routed_experts)
            router_prob = scores_for_seq_aux.mean(dim=1)
            load_balance = (expert_usage.detach() * router_prob).sum(dim=1).mean()
        else:
            expert_mask = F.one_hot(
                topk_idx_for_aux_loss.reshape(-1),
                num_classes=self.n_routed_experts,
            ).to(scores.dtype)
            expert_usage = expert_mask.mean(0) * self.n_routed_experts
            expert_usage = _mean_across_workers(expert_usage)
            router_prob = scores.mean(0)
            load_balance = (router_prob * expert_usage.detach()).sum()

        aux_loss = load_balance * self.alpha
        with torch.no_grad():
            stats_usage = expert_usage.detach()
            stats_prob = router_prob.detach()
            self.last_routing_stats = {
                "aux_loss": aux_loss.detach(),
                "expert_usage_min": stats_usage.min(),
                "expert_usage_max": stats_usage.max(),
                "expert_usage_std": stats_usage.float().std(unbiased=False),
                "router_prob_min": stats_prob.min(),
                "router_prob_max": stats_prob.max(),
                "router_prob_std": stats_prob.float().std(unbiased=False),
            }
        if not torch.is_grad_enabled():
            return None
        return aux_loss

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = self._load_balancing_loss(scores, topk_idx, bsz, seq_len)
        return topk_idx, topk_weight, aux_loss

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
    ):
        super().__init__()
        self.shared_experts = FeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.ModuleList([FeedForwardSwiGLU(dim, hidden_dim) for i in range(num_routed_experts)])
        self.gate = MoEGate(
            embed_dim = dim, 
            num_routed_experts = num_routed_experts, 
            num_activated_experts = num_activated_experts
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x) 
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        if aux_loss is not None:
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        return y
    
    # @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount(
            minlength=len(self.experts)
        ).cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out = expert_out * flat_expert_weights[idxs[start_idx:end_idx]]
            
            # for fp16 and other dtype
            if expert_cache.dtype != expert_out.dtype:
                expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache = expert_cache.index_add(0, exp_token_idx, expert_out)
        return expert_cache
