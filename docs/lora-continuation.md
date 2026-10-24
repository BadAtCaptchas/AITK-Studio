# Further train a Krea 2 LoRA

In a new Krea 2 Raw or Turbo training job, enable **Further train this LoRA**. Choose a library entry or upload a `.safetensors` file, then choose **Inspect LoRA**. Uploads go to the selected compute destination: local storage for **Local worker**, or the selected remote worker. Remote uploads stream through the controller, verify SHA-256 at the destination, and require a worker advertising continuation version 1. The maximum source size is 2 GiB.

Inspection must succeed before saving a continuation configuration. It checks every learned tensor and every target against Krea's denoiser topology. Missing pairs, unknown tensors, incompatible shapes, duplicates and nonfinite values fail explicitly. Per-layer ranks, sparse targets and effective scaling come from the file. This workflow does not slice or pad ranks. An export without recoverable scaling requires an explicit alpha/rank declaration; rank alone cannot determine scaling.

The initial source supplies weights only. A new job starts at step zero with fresh optimizer and scheduler state. Once that job saves a checkpoint, normal checkpoint resume takes precedence. The original file stays unchanged. The output is a complete updated adapter: use it by itself with the deployment base. Teacher and Turbo training-adapter weights are excluded.

The separate **Use a LoRA as the training base** option retains its existing merge behavior. It is incompatible with continuation.

## Preservation

Add new training examples for the intended refinement or added capability. Choose what to preserve with explicit source prompts and reviewed images; a downloaded weight file cannot enumerate the original concepts, triggers or training examples.

| Setting | Preservation-batch loss |
| --- | --- |
| Off | No preservation batches |
| Curated replay | Ordinary image-training loss |
| Frozen-source teacher | Source/student velocity prediction matching |
| Replay + teacher | Both losses |

Curated replay is the recommended first preservation experiment. Teacher modes are opt-in and experimental, with coefficient 1.0 by default. Each mode uses a dedicated preservation dataset role, separate from regularization (`is_reg`). By default, the loader alternates three new-data batches and one preservation batch. The ratio counts microbatches. New checkpoints retain its position even when data order can only resume approximately.

Generate candidates using the original adapter, its triggers and preservation prompts. In the generation job's Samples tab, select suitable images and choose **Use selected samples for preservation**. Captions come from validated prompt metadata, not generation-flag `.txt` sidecars. The published manifest records source hash, selected images, captions and generation metadata. Training rejects changed or unlisted examples. Keep evaluation prompts and seeds outside this dataset.

New source samples include an image checksum so curation rejects an image/caption pair that changes during publication. Older samples without this checksum can be curated after their generation job completes. Retrying curation verifies the existing published snapshot before acknowledging it.

Preservation datasets must use their curated `.txt` captions. If the text encoder is unloaded, cache preservation text embeddings first; otherwise the trainer could use blank or trigger-only conditioning.

For a source uploaded to a remote worker, create generation candidates through that worker's Generate page. The controller's Generate page currently creates local generation jobs. Remote sample review can publish a preservation dataset on the owning worker; normal dataset transfer handles replay data for training.

The frozen teacher shares one frozen base with the trainable adapter. It runs first with dropout disabled under `no_grad`, using identical noisy latents, conditioning, masks and timesteps. Only the student remains active through its forward, backward and gradient-checkpoint recomputation. Existing base-model preservation settings have different semantics and cannot be combined with source preservation.

## Raw, Turbo and provenance

Select the checkpoint used for this continuation separately from the deployment base. A Turbo listing on a model card is not evidence of the source adapter's original training base. Use Raw for Raw continuation, and explicitly choose Turbo with its training adapter for direct Turbo continuation.

Source and auxiliary adapters are copied into job-owned `.continuation` assets outside checkpoint cleanup. Their hashes, selected base identity, preservation manifests and configuration participate in provenance/resume checks. Remote job import verifies bounded assets and rewrites paths; a source uploaded to one worker cannot silently resolve against another worker's filesystem.

A resumed job can reference a relocated base checkpoint once its identity has been pinned. The loader still verifies the checkpoint bytes against the original hash. Changing the declared Turbo training-adapter checksum requires a new job.

Turbo teacher and student share the training-adapter-fused base. The negative cancellation adapter stays inactive through training forward/backward and becomes active for sampling. Before training, a deterministic velocity probe compares ordinary source inference with fusion, quantization and cancellation. The recorded discrepancy is numerical runtime drift, not an image-retention score.

Use a separate Turbo deployment evaluation: **8 steps, AITK CFG 1, fixed `model.model_kwargs.schedule_mu: 1.15`**, with fixed held-out prompts and seeds. Training-space prediction matching does not establish preservation of distilled low-step inference. Raw training and Turbo deployment need separate validation.

## Evaluation and boundaries

Compare untouched source, unprotected continuation, replay, teacher and combined modes. Hold the number of new-data batches constant, and report added preservation batches, teacher forwards and elapsed compute. Review desired changes, original behavior, unrelated prompts and concept combinations. Use subject/style similarity and prompt adherence alongside visual review; CLIP alone is insufficient. Include repeated Turbo deployment checks and report failures.

The automated tests cover strict import, scaling/rank preservation, teacher isolation, checkpoint behavior, asset checksums, upload failures and sample curation. Short GPU smoke runs establish execution, not useful adaptation or preservation quality. Character/style benchmarks, long-run retention tradeoffs and a real second-machine remote deployment remain separate acceptance work.

Initial support is ordinary linear Krea 2 Raw/Turbo text-to-image denoiser LoRA, including text-fusion layers, in native AITK/Comfy/Kohya naming and explicitly mapped Diffusers names. Edit/KV-cache adapters, Kroma-specific checkpoints, external text-encoder adapters, DoRA/LoKr, rank changes and video are rejected. Residual-adapter composition and merge/reinitialize comparisons remain future experiments.
