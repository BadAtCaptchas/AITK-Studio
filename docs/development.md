# Development utilities

[← Back to AITK Studio](../README.md#documentation)

Tools for working on AITK Studio itself.

- [Local UI development](#local-ui-development)
- [Python CI groups](#python-ci-groups)
- [Sync local changes](#sync-local-changes)

## Local UI development

See the [UI README](../ui/README.md) for development commands and focused tests, and [Python linting](python-lint.md) for Python checks.

## Python CI groups

The full-runtime CPU job runs the pinned CUDA-profile packages with GPUs hidden
and Hugging Face offline flags enabled. It includes training configuration and
phases, Kromoss/distillation, adapter/optimizer checkpoints, inference parity,
dataset-cache identity, EMA/scheduler lifecycle, training batch math and trainer
paths. CUDA-only tests explicitly skip on this runner. The lightweight Python
behavior matrix remains separate and compiles both `tests/` and `testing/`.

Keep the runtime groups in `.github/workflows/ci.yml` in separate Python
processes when reproducing locally. Broad discovery is unsafe: some suites
install framework stubs in `sys.modules`, and Pylint/Astroid changes inference
state. Install `requirements_dev.txt` and verify Pylint 4.0.5 before running
`tests.test_pylint_regressions` in its own process, so missing development tooling
cannot silently skip native-inference coverage. MOSS batching tests build tiny
random Qwen3.5 encoders locally; no pretrained weights are needed.

The MongoDB job temporarily uses Ubuntu 22.04's 6.8 kernel because MongoDB 8.0
refuses startup on the newer runner kernel ([SERVER-121912](https://jira.mongodb.org/browse/SERVER-121912)).
Its SQLite native binding is rebuilt from source to match that runner's glibc.
Move it back to a current runner once the service/kernel combination is supported,
before [Ubuntu 22.04 runner retirement in April 2027](https://github.com/actions/runner-images/issues/14254).

## Sync local changes

Run the following commands from the repository root.

To test uncommitted local changes in another checkout without pushing a branch first, use the git-status sync helper:

```bash
python scripts/sync_local_changes.py /path/to/other/ai-toolkit
```

Or from the UI npm scripts:

```bash
npm --prefix ui run sync:local -- /path/to/other/ai-toolkit
```

On Windows, this also works with WSL UNC paths:

```powershell
$env:AITK_DEV_SYNC_TARGET='\\wsl.localhost\Ubuntu\home\<user>\ai-toolkit'
npm --prefix ui run sync:local -- --dry-run
npm --prefix ui run sync:local
```

The helper copies files reported by `git status`, including untracked files, and removes target files for local deletions or rename sources. Use `--tracked-only` to skip untracked files, `--no-delete` to leave target deletions alone, and `--no-verify-target` for non-git target directories.
