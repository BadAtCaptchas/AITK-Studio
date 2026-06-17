import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np
from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme


def get_shift_for_sequence_length(
    seq_length: int,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    m = (max_shift - min_shift) / (max_tokens - min_tokens)
    b = min_shift - m * min_tokens
    return m * seq_length + b


def sample_shifted_logit_normal(
    batch_size: int,
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    std: float = 1.0,
    eps: float = 1e-3,
    uniform_prob: float = 0.1,
) -> torch.Tensor:
    mu = get_shift_for_sequence_length(seq_length)
    normal_999_percentile = 3.0902 * std
    normal_005_percentile = -2.5758 * std

    normal_samples = torch.randn((batch_size,), device=device, dtype=dtype) * std + mu
    logitnormal_samples = torch.sigmoid(normal_samples)

    percentile_999 = torch.sigmoid(
        torch.tensor(mu + normal_999_percentile, device=device, dtype=dtype)
    )
    percentile_005 = torch.sigmoid(
        torch.tensor(mu + normal_005_percentile, device=device, dtype=dtype)
    )

    zero_terminal_raw = (logitnormal_samples - percentile_005) / (
        percentile_999 - percentile_005
    )
    stretched_logit = torch.where(
        zero_terminal_raw >= eps,
        zero_terminal_raw,
        2 * eps - zero_terminal_raw,
    )
    stretched_logit = torch.clamp(stretched_logit, 0, 1)

    uniform = (1 - eps) * torch.rand((batch_size,), device=device, dtype=dtype) + eps
    prob = torch.rand((batch_size,), device=device, dtype=dtype)
    return torch.where(prob > uniform_prob, stretched_logit, uniform)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 1
            timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item()
                        for t in timesteps]

        # Get the weights for the timesteps
        if timestep_type == "weighted":
            weights = torch.tensor(
                [default_weighing_scheme[i] for i in step_indices],
                device=timesteps.device,
                dtype=timesteps.dtype
            )
        elif v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def _get_sampling_sequence_length(self, latents=None, patch_size=1, seq_length=None) -> int:
        if seq_length is not None:
            return max(1, int(seq_length))
        if latents is None:
            return 1024

        if latents.dim() == 5:
            _, _, frames, height, width = latents.shape
            return max(1, int(frames * height * width // max(1, patch_size**2)))
        if latents.dim() == 4:
            _, _, height, width = latents.shape
            return max(1, int(height * width // max(1, patch_size**2)))
        if latents.dim() == 3:
            return max(1, int(latents.shape[1]))
        return 1024

    def sample_shifted_logit_normal_timesteps(
        self,
        batch_size,
        device,
        latents=None,
        patch_size=1,
        seq_length=None,
        min_timestep=None,
        max_timestep=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        seq_length = self._get_sampling_sequence_length(
            latents=latents,
            patch_size=patch_size,
            seq_length=seq_length,
        )
        sigmas = sample_shifted_logit_normal(
            batch_size=batch_size,
            seq_length=seq_length,
            device=device,
            dtype=dtype,
        )
        timesteps = sigmas * self.config.num_train_timesteps
        if min_timestep is not None or max_timestep is not None:
            if torch.is_tensor(min_timestep):
                min_timestep = min_timestep.detach().to(device=device, dtype=dtype)
            elif min_timestep is not None:
                min_timestep = torch.tensor(float(min_timestep), device=device, dtype=dtype)
            if torch.is_tensor(max_timestep):
                max_timestep = max_timestep.detach().to(device=device, dtype=dtype)
            elif max_timestep is not None:
                max_timestep = torch.tensor(float(max_timestep), device=device, dtype=dtype)
            low = torch.tensor(0.0, device=device, dtype=dtype) if min_timestep is None else min_timestep
            high = (
                torch.tensor(float(self.config.num_train_timesteps), device=device, dtype=dtype)
                if max_timestep is None
                else max_timestep
            )
            clamp_min = torch.minimum(low, high)
            clamp_max = torch.maximum(low, high)
            timesteps = torch.maximum(torch.minimum(timesteps, clamp_max), clamp_min)
        return timesteps

    def get_nearest_timestep_indices(self, timesteps: torch.Tensor) -> torch.Tensor:
        schedule_timesteps = self.timesteps.to(device=timesteps.device, dtype=timesteps.dtype)
        distances = torch.abs(timesteps[:, None] - schedule_timesteps[None, :])
        return torch.argmin(distances, dim=1).long()

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps.float() / 1000).to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        while t_01.dim() < original_samples.dim():
            t_01 = t_01.unsqueeze(-1)
        # forward ODE
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        # reverse ODE
        # noisy_model_input = (1 - t_01) * noise + t_01 * original_samples
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type='linear',
        latents=None,
        patch_size=1
    ):
        self.timestep_type = timestep_type
        if timestep_type in ['linear', 'weighted', 'shifted_logit_normal']:
            timesteps = torch.linspace(1000, 1, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        elif timestep_type == 'sigmoid':
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = ((1 - t) * 1000)

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            # matches inference dynamic shifting
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(
                    self.sigma_min), num_timesteps
            )

            sigmas = timesteps / self.config.num_train_timesteps

            if self.config.use_dynamic_shifting:
                if latents is None:
                    raise ValueError('latents is None')

                # for flux we double up the patch size before sending her to simulate the latent reduction
                h = latents.shape[2]
                w = latents.shape[3]
                image_seq_len = h * w // (patch_size**2)

                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            if self.config.shift_terminal:
                sigmas = self.stretch_shift_to_terminal(sigmas)

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)

            sigmas = torch.from_numpy(sigmas).to(
                dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps

            if self.config.invert_sigmas:
                sigmas = 1.0 - sigmas
                timesteps = sigmas * self.config.num_train_timesteps
                sigmas = torch.cat(
                    [sigmas, torch.ones(1, device=sigmas.device)])
            else:
                sigmas = torch.cat(
                    [sigmas, torch.zeros(1, device=sigmas.device)])

            self.timesteps = timesteps.to(device=device)
            self.sigmas = sigmas

            self.timesteps = timesteps.to(device=device)
            return timesteps

        elif timestep_type == 'lognorm_blend':
            # disgtribute timestepd to the center/early and blend in linear
            alpha = 0.75

            lognormal = LogNormal(loc=0, scale=0.333)

            # Sample from the distribution
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)

            # Scale and reverse the values to go from 1000 to 0
            t1 = ((1 - t1/t1.max()) * 1000)

            # add half of linear
            t2 = torch.linspace(1000, 1, int(
                num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")
