# Supported models

[← Back to AITK Studio](../README.md#documentation)

Browse the model integrations included in AITK Studio. Available training modes and memory requirements vary by model; start with the [example configurations](../config/examples).

- [Image](#image)
- [Instruction / Edit](#instruction--edit)
- [Video](#video)
- [Audio](#audio)
- [Multimodal text](#multimodal-text)
- [MiniMax H3](#minimax-h3)
- [Experimental](#experimental)

## Image

- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [Lakonik/AsymFLUX.2-klein-9B](https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B) (AsymFLUX.2-klein-9B, experimental)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen Image 2.1; Studio defaults to the [Comfy-Org repack](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), ConvRot int8, and guidance 1.0)
- [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) (GLM-Image)
- [Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) (Boogu-Image Base)
- [Boogu/Boogu-Image-0.1-Turbo](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo) (Boogu-Image Turbo, experimental training)
- [zlab-princeton/i1-3B](https://huggingface.co/zlab-princeton/i1-3B) (i1-3B)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream-O1)
- [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) (Anima)
- [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4) (Ideogram 4 NF4)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)
- [Comfy-Org/Ideogram-4](https://huggingface.co/Comfy-Org/Ideogram-4) (Ideogram 4 Comfy NVFP4/FP8 weights)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [RunDiffusion/Juggernaut-Z-Image](https://huggingface.co/RunDiffusion/Juggernaut-Z-Image) (Juggernaut Z)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P) (Z-Image L2P)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRX Pixel)
- [microsoft/Mage-Flow-Base](https://huggingface.co/microsoft/Mage-Flow-Base) (Mage-Flow)

## Instruction / Edit

- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (same model as text-to-image; add dataset control paths or sample reference images for editing). Use dimensions divisible by 32. Up to 10 ordered sample references are supported; enable `model_kwargs.rgba` for RGBA output. See [Qwen Image 2.1 controls and configuration](qwen-image-2.1.md).
- [Boogu/Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) (Boogu-Image Edit)
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2 Edit Training)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo Edit Training)
- [microsoft/Mage-Flow-Edit-Base](https://huggingface.co/microsoft/Mage-Flow-Edit-Base) (Mage-Flow Edit)

## Video

- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (LTX-2.5)

## Audio

- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)
- [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) (experimental YuE2 music generation and training, with the community audio tokenizer and SheetSage conditioning). Stem-only/separation training is disabled in this integration.

## Multimodal text

- [ai-toolkit/Qwen2.5-Omni-7B](https://huggingface.co/ai-toolkit/Qwen2.5-Omni-7B) (Qwen2.5-Omni instruction tuning and text generation from image, audio, or video input). Training requires batch size 1; gradient accumulation is supported. The media encoders remain frozen.

## MiniMax H3

- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (MiniMax H3 and MiniMax H3 Ref2VA, joint video/audio and single-image training)
  - Supports T2V, first-frame I2V, image-only training, joint audio/video training, 24 fps defaults, automatic frame counts, and the H3 `17n+5` frame grid.
  - Supports pruned and non-pruned checkpoints, local-first Comfy weights under `MODELS_PATH`, ConvRot/NVFP4 prequantized components, layer offloading, VSA sparse attention when available, and a dense-attention fallback.
  - The shared base download is approximately 43 GB. Ref2VA supports multiple reference images/videos, picture or static-video presentation, the v1 training adapter, contrastive guidance, and D-OPSD self-distillation.

## Experimental

- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)
