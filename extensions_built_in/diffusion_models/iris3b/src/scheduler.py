"""Iris training grid, isolated from the schedulers used by other models."""
import torch

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


class IrisTrainingScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def set_train_timesteps(self, num_timesteps, device, timestep_type="shift", latents=None, patch_size=1):
        if timestep_type != "shift":
            return super().set_train_timesteps(
                num_timesteps, device, timestep_type, latents, patch_size
            )
        self.timestep_type = timestep_type
        # The reference starts at zero and stops at .999 BEFORE shifting.
        # Do not use diffusers' already-shifted sigma_min/sigma_max here.
        base = 1.0 - torch.linspace(1.0, 0.001, num_timesteps, dtype=torch.float64)
        sigmas = (self.shift * base / (1 + (self.shift - 1) * base)).flip(0)
        times = sigmas * self.config.num_train_timesteps
        self.model_timesteps = times.to(torch.int64).to(device)
        self.timesteps = times.float().to(device)
        self.sigmas = torch.cat([sigmas.float().to(device), torch.zeros(1, device=device)])
        self._step_index = None
        self._begin_index = None
        return self.timesteps

    def training_model_times(self, timesteps):
        # Keep continuous sigma for noise interpolation, but condition the DiT
        # with the reference's integer-truncated model time. Retain the fp64
        # grid's truncation even where fp32 rounds a value up to an integer.
        result = timesteps.float().floor()
        if self.timestep_type == "shift" and hasattr(self, "model_timesteps"):
            grid = self.timesteps.flip(0).to(timesteps.device)
            indices = torch.searchsorted(grid, timesteps.contiguous()).clamp(max=len(grid) - 1)
            exact = grid[indices] == timesteps
            reference = self.model_timesteps.flip(0).to(timesteps.device)[indices].float()
            result = torch.where(exact, reference, result)
        return result
