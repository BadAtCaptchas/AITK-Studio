from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


FLUX2_LATENTS_MEAN = [-0.06761776655912399, -0.07152235507965088, -0.07534133642911911, -0.07449393719434738, 0.022278539836406708, 0.017995379865169525, 0.014197370037436485, 0.01836133562028408, -6.275518535403535e-05, -0.006251443177461624, -0.00021015340462327003, -0.0031394739635288715, -0.027202727273106575, -0.02810601517558098, -0.027645578607916832, -0.029033277183771133, -0.0768895298242569, -0.06717019528150558, -0.09018829464912415, -0.08921381831169128, 0.016836659982800484, 0.015206480398774147, 0.00790204294025898, 0.008579261600971222, 0.008347982540726662, 0.0015409095212817192, 0.0002583497844170779, -0.004281752277165651, -0.043877143412828445, -0.04189559817314148, -0.04378034919500351, -0.043148837983608246, -0.010246668942272663, -0.013186423107981682, -0.006620197091251612, -0.004766239318996668, -0.031062893569469452, -0.03055436909198761, -0.027904054149985313, -0.01795399747788906, 0.0030211929697543383, 0.001502539962530136, 0.012592565268278122, 0.0144742326810956, 0.034720875322818756, 0.03376586362719536, 0.033663298934698105, 0.02829528972506523, 0.0019797170534729958, 0.004728920292109251, 0.004654144402593374, 0.004963618237525225, 0.012272646650671959, 0.008096166886389256, 0.00805679615586996, 0.014576919376850128, 0.06810732930898666, 0.06790295243263245, 0.07665354013442993, 0.07318653911352158, -0.04621443152427673, -0.04739413782954216, -0.03918757662177086, -0.05109340697526932, -0.05277586728334427, -0.04773825407028198, -0.047003958374261856, -0.0517151840031147, -0.03170523792505264, -0.03163386881351471, -0.03446723148226738, -0.02825590781867504, 0.050968676805496216, 0.04450491443276405, 0.057813018560409546, 0.04580356180667877, -0.0411602221429348, -0.04582904279232025, -0.048741210252046585, -0.04673927649855614, -0.008838738314807415, -0.010627646930515766, -0.008805501274764538, -0.004613492637872696, -0.03758484125137329, -0.043219830840826035, -0.043574366718530655, -0.049890533089637756, 0.011846445500850677, 0.016636915504932404, 0.020284568890929222, 0.027899663895368576, 0.011271224357187748, 0.01290129590779543, 0.0015593513380736113, 0.007155619561672211, -0.01180021371692419, -0.0018362690461799502, -0.014141527935862541, -0.005370706785470247, -0.009097136557102203, -0.013795508071780205, -0.014467928558588028, -0.01869881898164749, 0.03225415572524071, 0.030501458793878555, 0.02587026357650757, 0.02995659038424492, 0.05399540066719055, 0.06144390255212784, 0.049539074301719666, 0.05898929387331009, -0.051080696284770966, -0.06032619997859001, -0.047775182873010635, -0.052397292107343674, -0.022676242515444756, -0.027419250458478928, -0.015365149825811386, -0.025462470948696136, -0.05720777437090874, -0.056476689875125885, -0.05176353082060814, -0.049556463956832886, 0.011585467495024204, 0.0054222596809268, 0.01630038022994995, 0.010384724475443363]
FLUX2_LATENTS_VAR = [3.2502119541168213, 3.163407325744629, 3.192434072494507, 3.1813714504241943, 3.1389076709747314, 3.0941381454467773, 3.1011831760406494, 3.0550901889801025, 3.0051753520965576, 3.0179455280303955, 3.0067572593688965, 3.0076351165771484, 3.4690163135528564, 3.432523727416992, 3.470231533050537, 3.45538592338562, 3.0949840545654297, 3.071377754211426, 3.0819239616394043, 3.091344118118286, 3.014709711074829, 3.027461051940918, 3.01198673248291, 3.0252928733825684, 3.0074563026428223, 2.9741339683532715, 3.024878978729248, 2.9940483570098877, 3.080418586730957, 3.0669093132019043, 3.0831477642059326, 3.058147430419922, 3.403618097305298, 3.4055330753326416, 3.44087290763855, 3.435497283935547, 3.326714277267456, 3.1730010509490967, 3.1874520778656006, 3.22017240524292, 3.2569847106933594, 3.1953234672546387, 3.130955457687378, 3.124211549758911, 3.1620266437530518, 3.1209557056427, 3.2129595279693604, 3.185375690460205, 3.090271472930908, 3.030029058456421, 3.0565788745880127, 3.0162465572357178, 3.225846767425537, 3.2391276359558105, 3.211076259613037, 3.21309494972229, 3.161032199859619, 3.149500846862793, 3.142376184463501, 3.150174379348755, 3.071641206741333, 3.0439963340759277, 3.1177477836608887, 3.0607917308807373, 3.1593689918518066, 3.139946222305298, 3.1729917526245117, 3.1730189323425293, 3.2984564304351807, 3.244508981704712, 3.248305559158325, 3.251725673675537, 3.0720319747924805, 3.00360369682312, 3.084465742111206, 3.056194543838501, 3.100954532623291, 3.064960479736328, 3.1261374950408936, 3.102006435394287, 3.120508909225464, 3.0782599449157715, 3.178100109100342, 3.141893148422241, 3.2024238109588623, 3.2396669387817383, 3.1909685134887695, 3.1540026664733887, 3.102187395095825, 3.106377601623535, 3.08341121673584, 3.0892975330352783, 3.1621134281158447, 3.1226611137390137, 3.1719861030578613, 3.168121337890625, 2.958735942840576, 2.9129180908203125, 2.980844497680664, 2.9209375381469727, 3.165689706802368, 3.08971905708313, 3.0632121562957764, 3.0465474128723145, 3.0928444862365723, 3.0622732639312744, 3.0709831714630127, 3.014193534851074, 3.103145122528076, 3.087780714035034, 3.042872667312622, 3.0380074977874756, 3.065497875213623, 3.10084867477417, 3.109544038772583, 3.101743698120117, 2.976869583129883, 2.935845136642456, 2.999986171722412, 2.9673469066619873, 3.1200692653656006, 3.105872631072998, 3.139338493347168, 3.12007999420166, 3.0474750995635986, 3.0419390201568604, 3.086534261703491, 3.072920083999634]


@dataclass
class I1PipelineOutput:
    images: list[Image.Image]


def _center_crop_square_spatial(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 3:
        raise ValueError("i1 tensor preparation expects at least 3 dimensions.")
    height, width = tensor.shape[-2:]
    if height == width:
        return tensor
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return tensor[..., top : top + side, left : left + side]


def _resize_spatial_square(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[-2:] == (size, size):
        return tensor
    if tensor.dim() not in (3, 4):
        raise ValueError("i1 tensor preparation expects CHW or BCHW tensors.")

    squeeze_batch = tensor.dim() == 3
    batch_tensor = tensor.unsqueeze(0) if squeeze_batch else tensor
    original_dtype = batch_tensor.dtype
    interpolate_tensor = batch_tensor
    if interpolate_tensor.device.type == "cpu" and interpolate_tensor.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        interpolate_tensor = interpolate_tensor.float()

    resized = F.interpolate(
        interpolate_tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.to(dtype=original_dtype)
    return resized.squeeze(0) if squeeze_batch else resized


def prepare_i1_image_tensor(image: torch.Tensor, resolution: int = 1024) -> torch.Tensor:
    image = _center_crop_square_spatial(image)
    return _resize_spatial_square(image, resolution)


def prepare_i1_latent_tensor(latents: torch.Tensor, latent_size: int = 128) -> torch.Tensor:
    latents = _center_crop_square_spatial(latents)
    return _resize_spatial_square(latents, latent_size)


def sample_i1_lognorm_timesteps(
    batch_size: int,
    device: torch.device,
    shift: float = 0.3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    t = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype))
    if shift != 0.0:
        t = (shift * t) / (1.0 + (shift - 1.0) * t)
    return t


def i1_rectified_flow_noisy_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    t = timesteps.to(device=latents.device, dtype=latents.dtype)
    if t.max() > 1.0:
        t = t / 1000.0
    while t.dim() < latents.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * noise + t * latents


def i1_velocity_target(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return latents - noise


def time_grid(
    num_steps: int,
    shift: float,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=dtype, device=device)
    if shift != 0.0:
        times = (shift * times) / (1.0 + (shift - 1.0) * times)
    return times


def _flux2_stats(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(
        FLUX2_LATENTS_MEAN, device=latents.device, dtype=latents.dtype
    ).reshape(1, -1, 1, 1)
    var = torch.tensor(
        FLUX2_LATENTS_VAR, device=latents.device, dtype=latents.dtype
    ).reshape(1, -1, 1, 1)
    return mean, torch.sqrt(var + 0.0001)


def _pixel_unshuffle_flux2(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("FLUX.2 latent scaling expects even latent height and width.")
    latents = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 1, 3, 5, 2, 4).reshape(
        batch, channels * 4, height // 2, width // 2
    )


def _pixel_shuffle_flux2(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if channels % 4 != 0:
        raise ValueError("FLUX.2 latent scaling expects channel count divisible by 4.")
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return latents.permute(0, 1, 4, 2, 5, 3).reshape(
        batch, channels // 4, height * 2, width * 2
    )


def scale_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    packed = _pixel_unshuffle_flux2(latents)
    mean, std = _flux2_stats(packed)
    return _pixel_shuffle_flux2((packed - mean) / std)


def reverse_scale_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    packed = _pixel_unshuffle_flux2(latents)
    mean, std = _flux2_stats(packed)
    return _pixel_shuffle_flux2(packed * std + mean)


def prepare_cfg_conditioning(
    model,
    text: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, cond_len, _ = text.shape
    uncond = model.text_encoder_adapter.learnable_null_caption.to(
        device=text.device, dtype=text.dtype
    )
    if uncond.shape[0] == 1 and batch > 1:
        uncond = uncond.repeat(batch, 1, 1)
    uncond_len = uncond.shape[1]
    if uncond_len < cond_len:
        pad = torch.zeros(
            batch,
            cond_len - uncond_len,
            uncond.shape[2],
            device=text.device,
            dtype=text.dtype,
        )
        uncond = torch.cat([uncond, pad], dim=1)
        uncond_mask = mask & (torch.arange(cond_len, device=text.device)[None] < uncond_len)
    else:
        uncond = uncond[:, :cond_len]
        uncond_mask = mask
    return torch.cat([text, uncond], dim=0), torch.cat([mask, uncond_mask], dim=0)


def apply_i1_cfg_rescale(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float | torch.Tensor,
    cfg_rescale: Optional[float],
) -> torch.Tensor:
    velocity = cond + (guidance_scale - 1.0) * (cond - uncond)
    if cfg_rescale is not None:
        axes = tuple(range(1, velocity.ndim))
        std_c = torch.std(cond.float(), dim=axes, keepdim=True)
        std_g = torch.std(velocity.float(), dim=axes, keepdim=True)
        factor = (std_c / (std_g + 1e-8)).to(dtype=velocity.dtype)
        velocity = velocity * (1.0 - cfg_rescale + cfg_rescale * factor)
    return velocity


def randn_i1_latents(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    randn_device = device
    if generator is not None:
        generator_device = getattr(generator, "device", None)
        if generator_device is not None:
            generator_device = torch.device(generator_device)
            if generator_device.type != device.type:
                randn_device = generator_device
    randn_dtype = dtype if randn_device.type != "cpu" else torch.float32
    latents = torch.randn(
        shape,
        generator=generator,
        device=randn_device,
        dtype=randn_dtype,
    )
    return latents.to(device=device, dtype=dtype)


class I1Pipeline:
    vae_scale_factor = 8

    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        transformer,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.dtype = dtype

    @property
    def device(self) -> torch.device:
        if hasattr(self.transformer, "device"):
            return self.transformer.device
        try:
            return next(self.transformer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(self, device: torch.device | str):
        self.transformer.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        return self

    def set_progress_bar_config(self, *args, **kwargs):
        return None

    def encode_prompt(
        self,
        prompts: str | Sequence[str],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        device = device or self.device
        tokenized = self.tokenizer(
            list(prompts),
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        with torch.inference_mode():
            outputs = self.text_encoder(**inputs)
        hidden = outputs.last_hidden_state.float()
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
        return hidden, mask.bool()

    def denoise_latents(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float = 12.0,
        guidance_rescale: Optional[float] = 1.0,
        inference_timestep_shift: float = 0.3,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = self.device
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        if latent_height != self.transformer.input_size or latent_width != self.transformer.input_size:
            raise ValueError(
                "i1-3B 1024-resolution checkpoint expects 1024x1024 samples."
            )
        batch = prompt_embeds.shape[0]
        if latents is None:
            latents = randn_i1_latents(
                (batch, 32, latent_height, latent_width),
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
        else:
            latents = latents.to(device=device, dtype=self.dtype)

        text = prompt_embeds.to(device=device, dtype=self.dtype)
        mask = prompt_attention_mask.to(device=device, dtype=torch.bool)
        cfg_text, cfg_mask = prepare_cfg_conditioning(self.transformer, text, mask)
        forward_cache = self.transformer.prepare_forward_cache(
            cfg_text, cfg_mask, self.transformer.hw * self.transformer.hw
        )
        times = time_grid(
            num_inference_steps,
            inference_timestep_shift,
            device,
            dtype=self.dtype,
        )
        guidance = torch.full(
            (batch, 1, 1, 1),
            guidance_scale,
            device=device,
            dtype=self.dtype,
        )

        for idx in range(num_inference_steps):
            t = times[idx].expand(batch)
            latent_input = torch.cat([latents, latents], dim=0)
            t_input = torch.cat([t, t], dim=0)
            velocity = self.transformer(
                latent_input,
                t_input,
                cfg_text,
                cfg_mask,
                forward_cache,
            )
            cond, uncond = velocity.chunk(2, dim=0)
            velocity = apply_i1_cfg_rescale(
                cond,
                uncond,
                guidance,
                guidance_rescale,
            )
            latents = latents + (times[idx + 1] - times[idx]) * velocity
        return latents

    def decode_latents_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        latents = reverse_scale_flux2_latents(latents.to(self.vae.device, dtype=self.dtype))
        decoded = self.vae.decode(latents).sample
        return (decoded / 2 + 0.5).clamp(0, 1)

    def decode_latents(self, latents: torch.Tensor) -> list[Image.Image]:
        image_tensor = self.decode_latents_tensor(latents)
        image_batch = (
            (image_tensor.permute(0, 2, 3, 1) * 255)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return [Image.fromarray(image) for image in image_batch]

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Optional[str | Sequence[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 12.0,
        guidance_rescale: Optional[float] = 1.0,
        inference_timestep_shift: float = 0.3,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> I1PipelineOutput:
        del kwargs
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("Either prompt or prompt_embeds must be provided.")
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                prompt_embeds.shape[:2],
                dtype=torch.bool,
                device=prompt_embeds.device,
            )
        latents = self.denoise_latents(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            inference_timestep_shift=inference_timestep_shift,
            latents=latents,
            generator=generator,
        )
        return I1PipelineOutput(images=self.decode_latents(latents))
