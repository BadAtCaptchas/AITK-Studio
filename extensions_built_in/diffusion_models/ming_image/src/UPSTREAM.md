Ming-Image component provenance
==============================

The native transformer reuses differentiable blocks from Hugging Face
Diffusers' Apache-2.0 Z-Image implementation (Alibaba Z-Image Team and
The HuggingFace Team). Its Ming-specific conditioning, alignment padding,
reference-frame handling, and output-layer ordering are adapted from:

https://github.com/vllm-project/vllm-omni/pull/8021

Pinned source revision: `a62d2ec999ae8fa669e8c22194c67575c0f2d3dc`.

Relevant Apache-2.0 source files:

- `vllm_omni/diffusion/models/z_image/z_image_transformer.py`
- `vllm_omni/diffusion/models/ming_image/transformer.py`
- `vllm_omni/diffusion/models/ming_image/pipeline.py`
- `vllm_omni/diffusion/models/ming_image/condition.py`

Copyright contributors to the vLLM-Omni project. The port deliberately does
not depend on vLLM's inference kernels, process orchestration, or CUDA graphs.

Model weights remain separate downloads governed by their published licenses.

The frozen Bailing backbone in `bailing.py` is a focused adaptation of
`modeling_bailing_moe_v2.py` from inclusionAI/Ming, revision
`2a0c02ae3130190160c215f89fce7de3005db483`:

https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py

That source file carries the Apache-2.0 license and the notice
"Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved."
The surrounding Ming repository uses the MIT license; its notice is also
retained in `LICENSE-Ming`. Apache-2.0 is reproduced in `LICENSE-Apache-2.0`.

The port retains the native checkpoint names, modality-specific expert
routing, FP32 router probabilities, partial video rotary embeddings, and
causal attention. Native PyTorch SDPA replaces the original attention code;
generation, audio, distributed execution, and legacy Transformers interfaces
are omitted. `conditioning.py` implements the exact released image prompt,
reference positions, hidden-state selection, and connector semantics from
PR 8021, including its `ming_flash_omni/condition_encoder.py` and
`ming_flash_omni_thinker.py` dependencies. The actual Qwen vision encoder
and connector use Hugging Face Transformers implementations.
