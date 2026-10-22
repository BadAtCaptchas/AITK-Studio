# Checkpoints and interrupted training

AITK Studio saves each checkpoint as a complete generation inside the job's
`.aitk-checkpoints` directory. A generation contains its model files, any split
WAN stages, and the matching optimizer and training state. It becomes available
for resume only after all writers finish and the generation is published.

The familiar model filenames remain available in the job's output directory.
These are portable exports for using the trained model elsewhere. Keep the
complete job output, including `.aitk-checkpoints`, when backing up a run for
training continuation. A model file by itself does not contain optimizer moments,
pending gradients, or the complete random state.

## Interrupted or failed saves

A failed model or optimizer write leaves the previous committed checkpoint
available. Temporary `.tmp-*` generations are never considered for resume. A
portable export can fail to refresh, for example because another application has
the file open, while its complete generation remains available for resume. The
training log distinguishes that export warning from a failed checkpoint save.

Automatic resume checks complete generations in training-step order. It verifies
the declared files and checksums, including both files of a split checkpoint,
and falls back to an older valid checkpoint if the newest is incomplete or
corrupt. If checkpoints exist but none are valid, startup fails with an error
instead of silently restarting from step zero. An explicitly configured invalid
pretrained adapter also fails validation.

Legacy checkpoints remain supported, including model directories with
Safetensors or PyTorch tensor files. Directory checkpoints must contain real
weights and every shard named by their weight index. Legacy selection uses the
recorded training step, or the step in the filename when metadata is absent;
copying a file does not make it a later training step.

## What resume restores

New checkpoint generations pair optimizer moments with the complete exported
model set. Raw adapter parameters and EMA state are retained when applicable, so
portable LoRA/LoKr scaling and EMA export weights do not get mistaken for the live
parameters associated with the optimizer.

For supported runs, the training snapshot also records Python, NumPy, PyTorch and
CUDA random state; the current data order and position; bucket order and crop
geometry; scheduler and AMP scaler state; live training parameters; and pending
gradient accumulation. Sampling preserves the training random stream. Resume
continues after the saved training call, including a partially accumulated update.

The log explicitly reports either restored continuation or an **approximate
resume** with a reason. Exact state continuation currently requires:

- One training process, a standard data loader with zero worker processes, and
  serial batch loading.
- Supported AITK datasets and standard random or sequential sampling without a
  separate custom sampler generator.
- Unchanged model, network, training and dataset configuration and source-file
  identity. Save/sample cadence also matters when regularization data is used.
- No manual training phases, pooled or point-of-interest buckets, layered
  datasets, merge-on-save, or optimizer parameter swapping.

Legacy checkpoints, unsupported loader modes, configuration/data changes and an
explicit `start_step` use approximate continuation. They can still restore
compatible model and optimizer state, but do not promise the same next batch,
random draws, schedule position or pending accumulation as an uninterrupted run.
Extending the configured number of steps changes the training signature and is
therefore reported as approximate. Restoring captured state does not make
nondeterministic GPU kernels deterministic across hardware or library versions.

## Storage and retention

Positive `max_step_saves_to_keep` values bound retained committed generations;
zero or negative values leave retention unlimited. Model files,
optimizer state and the training snapshot are removed together. After recovery
from an invalid later checkpoint, the newly committed generation is protected
from retention even if its step is lower than the damaged generation's step.
Deletion failures are reported and leave extra files for a later cleanup attempt.

Portable exports use hard links when the filesystem supports them, with a copy
fallback. Treat exported checkpoints as read-only: modifying a hard-linked file
in place also modifies its generation, and checksum validation will reject it.
Training snapshots include raw trainable parameters and accumulated gradients,
so disk usage can exceed the portable model size, especially for full-model
training. Preserve the entire paired generation when moving or backing it up.
