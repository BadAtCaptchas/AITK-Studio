"""
NOTE: This is experimental and under active development; expect breaking changes
and bugs. Feedback welcome.
"""

from typing import List, Optional

import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A single learning rate is kept per parameter tensor. Each step nudges that
    lr according to whether per-element update directions flipped versus the
    previous step, then gently pulls each tensor lr toward the global average.
    The flip signal gives the optimizer a real edge-of-stability equilibrium,
    and the global pull keeps layers from drifting to opposite extremes.

    With ``fused=True`` (default), each parameter is updated from a
    ``register_post_accumulate_grad_hook`` as soon as autograd finishes
    accumulating into it. ``.step()`` therefore only refreshes reporting state.
    This keeps peak VRAM low, but it bypasses trainer-side gradient clipping and
    nan-skip logic because those run after backward.

    With ``fused=False``, it behaves like a traditional optimizer: grads
    accumulate across backward passes and the update happens in ``.step()``.
    Low-precision grads and parameter write-backs use stochastic rounding so
    small updates are not lost to repeated round-to-nearest casts.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        lr_bump_rate: float = 0.1,
        lr_pull: float = 0.025,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        lr_smoothing_steps: int = 3,
        fused: bool = True,
    ):
        # Accepted for backwards compatibility with earlier local Automagic3
        # configs. Final v3 uses mean reversion instead of hard lr rails.
        _ = (min_lr, max_lr)
        if lr > 1e-3:
            print(f"Warning! Start lr {lr} is very high; forcing to 1e-6.")
            lr = 1e-6

        lr_smoothing_steps = max(1, int(lr_smoothing_steps))
        defaults = dict(
            lr=lr,
            lr_bump_rate=lr_bump_rate,
            lr_pull=max(0.0, float(lr_pull)),
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            lr_smoothing_steps=lr_smoothing_steps,
            dir_beta=lr_smoothing_steps / (lr_smoothing_steps + 1.0),
        )
        super().__init__(params, defaults)

        self.fused = fused
        self._avg_lr = float(lr)
        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.fused:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)
                elif p.dtype != torch.float32:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_accum_hook()
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training paramiters: {total:,}")

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @staticmethod
    def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
        as_int = v_fp32.view(torch.int32)
        as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
        as_int.bitwise_and_(-(1 << drop_bits))
        return v_fp32

    @staticmethod
    def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        finfo = torch.finfo(dtype)
        absv = v.abs().clamp_(min=finfo.tiny)
        ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
        noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
        return v.add_(noise).to(dtype)

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))

    def _make_accum_hook(self):
        def _hook(p: torch.Tensor):
            if p.grad is None:
                return
            if hasattr(p, "_accum_grad"):
                acc = p._accum_grad.to(torch.float32).add_(p.grad.to(torch.float32))
                self._stochastic_copy_(p._accum_grad, acc)
            else:
                p._accum_grad = p.grad.clone()
            p.grad = None

        return _hook

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.tensor(
            float(group["lr"]), dtype=torch.float32, device=p.device
        )
        state["prev_sign"] = None
        state["dir_ema"] = torch.zeros((), dtype=torch.float32, device=p.device)
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=p.dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return

        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic3 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
            update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v = v_state
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            else:
                v = v_state.to(torch.float32)
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                v_state.copy_(v.to(v_state.dtype))
            update = v.add(eps).rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        cur_sign = update.sign().to(torch.int8)
        prev_sign = state["prev_sign"]
        lr_t = state["lr"]

        if prev_sign is not None:
            prod = cur_sign * prev_sign
            den = prod.count_nonzero().clamp_(min=1)
            log_dir = prod.sum().float().div_(den.float()).mul_(
                group["lr_bump_rate"]
            )
            ema = state["dir_ema"]
            beta = group["dir_beta"]
            ema.mul_(beta).add_(log_dir, alpha=1.0 - beta)
            lr_t.mul_(torch.exp(ema))

            pull = group["lr_pull"]
            if pull > 0.0 and self._avg_lr > 0.0:
                lr_t.mul_(lr_t.reciprocal().mul_(self._avg_lr).pow_(pull))

        state["prev_sign"] = cur_sign
        state["step"] += 1

        wd = group["weight_decay"]

        if p.dtype == torch.float32:
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_t, value=-1.0)
        else:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_t, value=-1.0)
            self._stochastic_copy_(p, new_p_fp32)

        p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.fused:
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    accum = getattr(p, "_accum_grad", None)
                    if accum is not None:
                        p.grad = accum
                        del p._accum_grad
                    if p.grad is None:
                        continue
                    self._update_param(p, group)

        self._refresh_avg_lr()
        return loss

    def _all_lrs(self) -> list:
        return [
            st["lr"]
            for group in self.param_groups
            for p in group["params"]
            if (st := self.state.get(p)) is not None and "lr" in st
        ]

    def _refresh_avg_lr(self) -> None:
        lrs = self._all_lrs()
        if lrs:
            self._avg_lr = float(torch.stack(lrs).log_().mean().exp_())

    def get_learning_rates(self) -> List[float]:
        out = []
        for group in self.param_groups:
            lrs = [
                self.state[p]["lr"]
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(float(torch.stack(lrs).mean()) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                if isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
                if "prev_sign" in st:
                    st["prev_sign"] = None
                if isinstance(st.get("dir_ema"), torch.Tensor):
                    st["dir_ema"] = torch.zeros_like(
                        st["dir_ema"], dtype=torch.float32
                    )
        self._refresh_avg_lr()
