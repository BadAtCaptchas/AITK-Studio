# Upstream integration: September 21–27, 2026

Status: complete. Implemented and verified locally on 2026-09-29 from Studio main `54c81ff7`. User requested a commit after reviewing completion.

## Objective and constraints

Integrate every applicable outstanding change from ostris/ai-toolkit main in the inclusive date range. Port behavior selectively into Studio; preserve global storage resolution, encrypted datasets, existing model improvements, saved explicit job settings, telemetry recovery, and environment backup/rollback. Do not modify runtime data or download model weights. Commit requested in follow-up; no push requested.

## Implementation plan

1. [x] Training and loader fixes: Qwen Edit Plus caller isolation, per-prompt batching/padding, stacked controls (`9b67238f`, `ae6ab907`, `f0731883`); unloaded text encoder device state and freeing model-owned encoders (`543a0d74`, `a35d833a`), preserving mllm aliases; Comfy attention metadata filtering (`6468a2ff`).
2. [x] Qwen Image 2.1 target-resolution matching (`07abdbed`): thread optional target size through prompt APIs, cache building, training/prior prediction, and sampling; version cache by reference sizing policy and bucket dimensions; retain official weights, precision, transparency, and multi-reference behavior.
3. [x] YuE2 safetensors heads/adapters and head-specific latent cache identity (`8f2777a4`, containing `d0ecde77`), preserving Studio's chain-of-thought cache identity.
4. [x] MiniMax/FastH3 (`a7763943`, `60d0c28c`): expose FastH3 8-Step V2 with matching weights, 0.8 sparsity, video shift 10, audio shift 3, and correct adapter. Preserve the legacy four-step class/config behavior via a separate v2 architecture. Update new base/Ref2VA job defaults to adapters v3/v2 without default contrastive guidance; keep explicitly saved settings.
5. [x] Ming (`77847d7f`, `460c29ba`, `cbc6bbdd`): add upstream Comfy-backed `ming_image` as a separate implementation/package, alongside Studio's `ming_image_design` and `ming_image_design_layer`. Port the final in-range checkpoint alias/vision fixes, quantized text encoder, live training adapter, and dynamic scheduler. Register in Studio's shared capability registry/UI and use configured model storage. Retain native/layered model implementation and tests.
6. [x] DGX (`60ca869b`, `5aa3d0c9`, `16ae8d29`): use normal Linux ARM64 requirements, managed Python with usable C headers, and non-destructive environment replacement; retire obsolete DGX instructions/pins with updated links. Report shared memory and GB10 power fallback without regressing telemetry. Add Linux NVIDIA integrated-device CPU-movement suppression with upstream opt-out, leaving Windows/ROCm/discrete GPUs and tensor CPU transfers unchanged.
7. [x] UI and housekeeping: stable table row IDs (`ecee894e`), ignore `/issue_review` (`11b65a39`), reconcile support-link change (`71bc8778`) with Studio README, bump Studio release version once instead of importing upstream's unrelated version numbers (`0bd34111` and embedded bumps).
8. [x] Verification and review: targeted behavior tests with tiny tensors/mocks, real runtime imports of changed classes and diffusion registry, undefined-name checks, relevant Node tests and full UI/worker typecheck. Review final diff and record results/limitations below. Stop any test servers/processes started.

## Already covered

- `3a39d8cf`: cache_text_encoder device preset and validation already implemented and tested locally.
- `d0ecde77`: duplicate branch history; integrate once via `8f2777a4`.
- Managed installer already selects TorchCodec 0.15.0; retain the profile-based dependency policy rather than adding a conflicting base requirement.

## Acceptance checks

- Qwen Edit inputs retain original controls; multi-item unequal-length prompts and stacked controls work.
- Encoder unloading frees model-owned/aliased encoders once and device state tolerates stubs.
- Qwen 2.1 cache/train/sample reference sizes agree, with matching off/default behavior and distinct cache keys.
- YuE2 original and safetensors formats work; different heads and cot modes do not share latent caches.
- Old FastH3 configs keep old schedule; v2 defaults and adapter selection agree; saved MiniMax options persist.
- Native Ming/layered behavior remains intact; new Comfy Ming imports/registers, maps checkpoint keys, handles vision metadata, and keeps assistant adapter out of sampling.
- DGX header recovery preserves prior environments; telemetry distinguishes shared memory; unified-memory changes are gated and preserve dtype casts and tensor transfers.
- Table deletion cannot transfer row state. Existing model/config, cache/security, manager, and monitor tests pass.

## Progress and verification log

- Baseline: clean working tree. Upstream patches inspected through authenticated GitHub read tools; exact commit data cached in tool session.

- Implementation complete for steps 1-7. Added `ming_image_comfy` / `ming_image` alongside native Ming, and `minimax_h3_vsa_v2` alongside legacy FastH3. Studio version is 1.9.4.
- Studio adaptations include configured model storage, ordered sample references, live preview hooks, preserved native Ming and global/encrypted dataset paths, target-aware Qwen cache keys, and non-destructive installer recovery.
- Corrected two upstream integration defects during porting: YuE2 merge retains `torch.no_grad`; Qwen prior target-size kwargs are passed to prompt encoding, not Tensor.to.
- Focused Python suite: 143 tests ran, 142 passed and 1 opt-in native Ming CUDA test skipped. Covers native Ming, Qwen source/loading, cache/unloader, manager/runtime, config contract, and September regressions. Tiny new Ming transformer forward/backward and sampling pass without weights.
- Follow-up Python checks passed after final review fixes: 22 tests for cache geometry/content identity, then 24 tests for the shared ordered-reference prompt kwargs and device state. A further real tiny Ming training-adapter test passed: the adapter changes output when active, restores base output when disabled, leaves base weights unchanged, and stays frozen.
- UI: 15 monitor tests passed; layered/preset suite ran 29 tests (28 passed, 1 case-sensitive-filesystem test skipped on Windows); MiniMax/table tests passed all 5 cases. Both app and worker typechecks passed. The layered suite needed sandbox escalation to refresh generated `ui/dist` files; the retry passed.
- Runtime imports passed for changed modules/public classes and the full diffusion registry, including both new architectures and preserved native/legacy variants. Python syntax and Pyflakes undefined-name checks passed. DGX spec smoke confirms standard requirements, Python 3.12, TorchCodec 0.15.0 and the manual compatibility alias. `git diff --check` passed; final source diff reviewed.
- No model weights downloaded or training jobs launched. No test servers started. Full pretrained-model GPU and physical DGX validation remain unperformed.

## Reproduce verification

From the repository root (the existing `.venv` was used):

```powershell
.\.venv\Scripts\python.exe -m unittest testing.test_unloader testing.test_base_model_device_state_presets tests.test_qwen_image_2 tests.test_qwen_image_2_loading tests.test_qwen_image_2_sources tests.test_september_upstream tests.test_upstream_sep21_27 tests.test_manager_python tests.test_manager_natten tests.test_runtime_profiles tests.test_config_contract testing.test_ming_cache_identity testing.test_ming_trainer testing.test_ming_integration testing.test_ming_components
```

From `ui/`:

```powershell
npm.cmd run typecheck
npm.cmd run test:monitor
npm.cmd run test:layered-images
node --test scripts/universalTable.test.mjs scripts/minimaxH3Options.test.mjs
```

The follow-up regressions are included in `tests/test_upstream_sep21_27.py`, so the combined Python command now discovers more tests than the initial 143-test run. Validation did not install packages into the working training environment. The user subsequently authorized committing the completed integration.
