# Upstream integration: September 10–20, 2026

Reviewed `ostris/ai-toolkit` through [`8fa15e356939a922b8fd3307610bf03100c803f6`](https://github.com/ostris/ai-toolkit/commit/8fa15e356939a922b8fd3307610bf03100c803f6), the September 19 head available during the September 20 review. The [comparison from `f56b5a1d`](https://github.com/ostris/ai-toolkit/compare/f56b5a1d405f819c74724228564e99982624c186...8fa15e356939a922b8fd3307610bf03100c803f6) contains 47 subsequent commits. The September 10 HiDream x0 change at the base was already present in Studio, as were the EMA correction, file-extension recognition, test-output ignores, and stronger dataset path guards. No September 20 commit was present in the reviewed history.

This is a source port into Studio, with its storage, job lifecycle, authentication, remote-worker, and encrypted-dataset behavior retained. Upstream merge commits and version-only commits do not require separate code changes.

## Included

| Area | Integration |
| --- | --- |
| Training | Experimental YuE2, Qwen2.5-Omni multimodal instruction tuning, model cache versions, integer cache metadata preservation, fp32 audio encoding, prose caption preservation, duration and instruction controls |
| Models and quantization | Krea2 patch-size correction, streamed HiDream safetensors export, rotated embedding import, stochastic int8 requantization, compatible restoration of prequantized linear layers |
| Captioning | ACE-Step transcription/format improvements and optional vocal extraction, Qwen2.5-Omni, Qwen3-Omni audio input, MOSS music captioning, LoRA selection and application |
| Live inference | Resident engine, component reuse, model switching, LoRA/LoKr stacks, adapter uploads and metadata, reference uploads, streamed progress/previews, reconnect, cancellation, image/video/audio/text outputs |
| Studio UI | Installed extension model cards, job notes, sample range selection and bulk deletion, audio thumbnails, text sample viewer, two-space YAML editor |
| Utilities | MelBandRoformer conversion, Qwen conversion updates, model-loading and inference smoke-test utilities |

Studio keeps the main Native/ComfyUI Generate page and adds `/generate/live`. Engine credentials are derived per execution attempt and passed through the child environment; endpoint files contain no token. The proxy checks the local job, process, attempt, and loopback address. UI-created engines use the configured training root and managed job folder. Notes and sample deletion use job-aware path resolution and remote forwarding.

Installed extension UI modules are compiled from local `ui.tsx`/`ui.ts`/`ui.jsx`/`ui.js` files. Studio's existing model catalog provides the baseline; extension declarations extend or override it. Built-in cards were adapted into Studio's typed catalog instead of replacing it wholesale. See [extension UI documentation](../ui/src/extensions/README.md).

## Limits and validation

- YuE2 separation/stem-only training is rejected because the upstream commit explicitly describes it as not working as expected. Ordinary YuE2 training and the separation utilities used by captioners are included.
- Live inference currently supports local engines. Existing remote training, captioning, and Native/ComfyUI generation workflows remain available.
- Validation uses the installed project dependencies without downloading model weights or starting training. Passing import and unit checks does not establish full model training quality, generation correctness, or VRAM requirements.
- Focused Python tests cover checkpoint round trips/atomic publication, quantized embeddings and linear restoration, adapter math, audio caches/artwork, prose captions, model contracts, stream replay, HTTP auth/output containment, and engine lifecycle concurrency. UI tests cover framing, preview decoding, credentials, endpoint validation, extension TSX compilation, existing media paths, adapter library, and caption workflows.
- App and worker TypeScript checks and changed-module Python import/syntax/undefined-name checks are the integration gates. The manual scripts under `testing/` may load large models; they are not part of the weight-free checks.

Source changes are left in the working tree for review; this integration does not create a commit or push.

Validated on September 20: 38 focused Python tests and 55 UI/server tests, app and worker typechecking, runtime imports of the model registries, captioners, trainers and inference modules, and syntax checks for 76 changed Python files. No model weights were downloaded and no training or full model inference was run.
