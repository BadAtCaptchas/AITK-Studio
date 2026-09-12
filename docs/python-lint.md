# Python linting

Run Pylint with the existing training environment from the repository root:

```powershell
.\.venv\Scripts\python.exe -B scripts/lint.py
```

Install `requirements_dev.txt` in that environment if Pylint is unavailable. The
runner can also use an installed VS Code Pylint bundle, without allowing bundled
dependencies to override the project's installed packages.

The runner selects Git-visible Python sources, including new source files, and
uses explicit module names. Ignored runtime data, environments and generated
artifacts are outside that inventory.

## Scope and exceptions

The current profile enables correctness errors and practical warnings. Naming,
docstring coverage, complexity thresholds, import ordering, deferred imports and
framework override conventions are not enforced across bundled research code.
New public helpers should still have useful docstrings.

`scripts/pylint_inference.py` describes actual installed Torch/OpenCV native
exports to Astroid. Unknown export names remain errors. Diffusers `FrozenDict`
attributes are populated by configuration decorators and are exempt from member
inference checks.

Triton, XLA, FlashAttention, SageAttention, xformers, DAdaptation, LLaVA, Modal and
DashScope are optional or platform-specific dependencies. Their import/member
checks are exempt in this general environment; feature-specific runtime testing
is still required when using them.

## Current status

This is a cleanup profile in progress, not a passing CI gate. The initial default
run found 24,320 messages across 652 sources. Outstanding findings and reviewed
exceptions must be resolved before claiming a clean project-wide Pylint result.

The last focused validation passed syntax checks for 655 Python files,
the registered-model import gate with real dependencies, and 55 focused tests.
These checks did not download model weights or run training.
