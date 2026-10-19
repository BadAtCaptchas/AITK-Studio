# Python linting

Run Pylint with the existing training environment from the repository root:

```powershell
.\.venv\Scripts\python.exe -B scripts/lint.py
```

To check errors and fatal errors without warnings:

```powershell
.\.venv\Scripts\python.exe -B scripts/lint.py --errors-only
```

The default command retains the existing warning checks. Both modes accept
`--output-format` and `--output` for saving reports.

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
exports to Astroid, including OpenCV extension classes and their installed
public members. Torch native functions are described from the installed package;
their return types remain uninferred. Unknown export and class-member names
remain errors. The plugin also inspects installed Transformers configuration
constructors, tokenizer aliases, Gradio event signatures, Diffusers checkpoint
callbacks, and the native Torch context manager. Project model exports come from
the same declarations as the runtime registry. Bounded signatures retain their
keyword checks. Optional Triton inference never requires the native backend.

Transformers can populate its import search path with `Path` objects on Windows;
the plugin normalizes those entries to strings for Astroid compatibility.
Diffusers `FrozenDict`
attributes are populated by configuration decorators and are exempt from member
inference checks.

Triton, XLA, FlashAttention, SageAttention, xformers, DAdaptation, LLaVA, Modal and
DashScope are optional or platform-specific dependencies. Their import/member
checks are exempt in this general environment; feature-specific runtime testing
is still required when using them.

## Current status

The verified `--errors-only` run passed across all 803 Git-visible Python source
files. Warnings were outside this verification; the default command still
reports them.

The lint regression suite passed 15 tests, including installed framework APIs,
unknown exports and keywords, optional Triton availability, and the runner's
default behavior. The expanded constructor regression and PhotoMaker runtime
import also passed. These checks used the project environment without
downloading model weights or running training.
