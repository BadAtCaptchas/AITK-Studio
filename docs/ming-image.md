# Ming Image Design and Design-Layer (experimental)

AITK Studio provides native PyTorch transformer LoRA adapters for
[Ming Image Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
(`ming_image_design`) and
[Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)
(`ming_image_design_layer`). Inference behavior follows
[vLLM-Omni PR #8021](https://github.com/vllm-project/vllm-omni/pull/8021), pinned
to `a62d2ec999ae8fa669e8c22194c67575c0f2d3dc`. vLLM is not a runtime dependency.
The reference is an open PR and support remains experimental.

## Starting profiles

- [Design, 1024 pixels](../config/examples/train_lora_ming_design_16gb_experimental.yaml)
- [Design-Layer, 512 pixels and two target layers](../config/examples/train_lora_ming_design_layer_16gb_experimental.yaml)

These profiles target a 16 GB RTX 5080 with substantial CPU RAM. They use batch
size one, rank-16 LoRA, BF16 compute, FP8 frozen transformer weights, checkpointed
refiners/main blocks, and full block offloading. The frozen conditioning model
loads into CPU memory and stages individual components onto the GPU. Final
training and preview embeddings are cached before the conditioner is unloaded.
The approximately 191 GB RAM development host can accommodate this staging;
the same profiles may not fit hosts with much less CPU RAM.

**Hardware validation (2026-09-22):** both profiles passed actual-checkpoint
optimizer steps, exact LoRA save/reload, optimizer-state resume and 12-step PNG
preview generation on an RTX 5080 with 16 GB VRAM. Each test used batch one,
rank/alpha 16, FP32 adapters, BF16 compute, `ming_fp8` frozen weights, gradient
checkpointing, staged conditioning and all transformer blocks offloaded.

| Model | Tested canvas | Target layers | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- |
| Design | 1024 × 1024 | Single image | 3.50 GiB | 4.48 GiB |
| Design-Layer | 512 × 512 | 2, plus composite | 3.87 GiB | 5.03 GiB |

Figures are PyTorch CUDA peaks across loading, encoding, two optimizer steps,
reload and preview generation; they exclude other applications and driver memory.
The conditioning model was unloaded before denoising. These short tests verify
training and export mechanics, not convergence or generation quality. Larger
canvases, more layers and longer conditioning sequences were not tested.
Support remains experimental.

Only transformer-block LoRA weights train. The RGBA VAE, Bailing vision/language
model, learned queries, connector and conditioning projectors stay frozen.
Standard AITK LoRA checkpoint save/resume applies. Full model fine-tuning is not
supported. Cached profiles require fixed captions and geometry; leave caption
dropout, token shuffle, random crop/scale and random triggers disabled.
Keep `qtype: ming_fp8` for the tested offloading profile: it stores scaled FP8
weights as bytes and dequantizes for BF16 matrix operations. It does not require
vLLM or a custom CUDA kernel.

Training uses checkpoint flow shift 6. The adapter converts toolkit timesteps to
`1 - timestep / 1000` and negates the prediction for the `noise - clean` target.
Sampling uses the reference spatial-token shift and
`positive + guidance_scale * (positive - negative)`, with sequential predictions.
Defaults are 12 steps, guidance 1 for Design and 2 for Design-Layer. Negative
conditioning is zero; nonempty negative prompts are rejected. Image dimensions
must be divisible by 16. Every RGBA plane is encoded independently using posterior
mode and checkpoint VAE scaling.

## Layered documents

Open **Layered documents → Import PSD / ORA** in a dataset. One visible
top-level layer or group becomes one full-canvas RGBA target. Groups flatten their
visible descendants, preserving supported raster masks, opacity and offsets.
Targets are ordered bottom-to-top. Hidden and empty targets are excluded.

The importer accepts 8-bit RGB PSD raster content and embedded raster previews,
and PNG-based ORA with normal/source-over compositing. Unsupported adjustment,
clipping, effects, vector-only, external-resource and blend behavior is rejected
with the layer identified. Rasterize unsupported content in an editor first.
This is deliberately a limited renderer; see the
[PSD rendering documentation](https://psd-tools.readthedocs.io/en/stable/usage.html)
and [ORA stacking rules](https://www.openraster.org/baseline/layer-stack-spec.html).

Imports are sequential, staged, validated and published as complete samples.
Limits are 512 MiB/document, 32 megapixels, 256 source nodes, 32 exported targets,
and five minutes per conversion. Additional archive expansion, nesting and decoded
pixel limits prevent compressed documents from exhausting memory. Grouped encrypted
datasets are currently rejected. Ordinary Design image training retains the
existing encrypted dataset support: plaintext caches are disabled and the frozen
conditioner remains available for live encoding. That mode differs from the
measured cached profile above.

Each imported document appears once in Studio with a layer-count badge and ordered
previews. Edit the document caption and optional layer captions. Source names are
metadata, never automatically used as training instructions. A training prompt
begins with the actual layer count, followed by the supplied captions. Automatic
caption generation is outside this version.

## Manifest and lifecycle

Set `datasets[].type: layered_image`. A dataset contains a visible composite PNG
and sibling caption TXT; `.layers/<sample-id>/manifest.json` owns the ordered RGBA
targets. Paths are dataset-relative and must resolve canonically inside that
dataset. A version-one manifest looks like:

```json
{
  "format": "aitk.layered-image",
  "version": 1,
  "id": "sample-id",
  "composite": "poster.png",
  "caption": "poster.txt",
  "width": 512,
  "height": 512,
  "order": "bottom-to-top",
  "layers": [
    {"path": ".layers/sample-id/000.png", "name": "Background", "caption": "blue background"},
    {"path": ".layers/sample-id/001.png", "name": "Graphic", "caption": "yellow sun"}
  ]
}
```

Training targets are `[composite, layer1, ..., layerN]`. Prediction appends the
clean composite reference and removes that reference from the loss. Groups share
all spatial transforms, and buckets include their layer count. Cache identities
include ordered layer contents, captions and geometry. Ordinary image discovery
excludes hidden `.layers` assets. Dataset deletion, copy, combine, export and remote
transfer preserve complete groups; composite-only image edits are disabled.

## Previews

Design-Layer samples require a reference (`ctrl_img`) and accept a typed per-sample
`num_layers` from 1 to 32 (default 2). Start with two targets when testing memory.
The layer-count instruction is constructed from this value. Set `sample.format:
png`: a `LayeredImageOutput` saves the composite PNG and individual RGBA PNGs,
with a `.layers.json` sidecar for Studio's layer preview/download controls.
Layers are independent still images and do not enter animated-image handling.
The generic single-image `train.validation_config` path is unsupported for
Design-Layer; use layered sample previews.

## Reproducing the hardware check

The opt-in harness uses synthetic artwork and an already downloaded checkpoint.
It writes the adapter, previews, reference and phase-by-phase `report.json` into
a new output directory. On Windows, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m testing.ming_checkpoint_smoke --checkpoint <local-Design-snapshot> --variant design --resolution 1024 --output .tmp/ming-design-check
.\.venv\Scripts\python.exe -m testing.ming_checkpoint_smoke --checkpoint <local-Design-Layer-snapshot> --variant layer --resolution 512 --output .tmp/ming-layer-check
```

Validated checkpoint revisions were Design
`1cd7fac3b0dcb54196fe2cd12b80da09edf8fcf4` and Design-Layer
`650448783505b103af305ce347bf60d8889e655a`, using PyTorch 2.10.0+cu130,
Transformers 5.5.3 and the installed Diffusers 0.39 development version.
CUDA tests also compared resident and offloaded tiny native Ming modules with
identical inputs and adapters: outputs, LoRA gradients and two optimizer updates
matched exactly for both variants.

The regular `run.py` training path also passed two-step jobs for both profiles,
including disk caches, conditioner unloading, baseline/final previews and
checkpoint/optimizer saves. Design-Layer previews each contained two static
512 × 512 RGBA targets and their composite. Both jobs were then resumed through
`run.py` from step two to step three, reusing caches and restoring saved LoRA and
optimizer state.

Focused checks cover conditioning/routing, padding, flow time/sign, RGBA VAE
normalization, gradients/checkpointing, LoRA reload, group caches and lifecycle,
PSD/ORA conversion, malformed documents, containment, cancellation and atomic
publication. UI application/worker typechecks and public-class/registry runtime
imports passed. A real Node-to-Python ORA import was exercised; browser and live
remote-worker workflows were not manually exercised.

Broader existing config tests still have unrelated Orbit/LoKr and MiniMax
failures reproduced against the unchanged repository version. Legacy latent-cache
test fixtures also lack current cache-identity methods, and the existing DFE4
fixture omits `sd_ref`. These are separate from the passing focused Ming and
grouped-dataset checks.
