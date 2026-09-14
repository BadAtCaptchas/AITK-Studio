# Development utilities

[← Back to AITK Studio](../README.md#documentation)

Tools for working on AITK Studio itself.

- [Local UI development](#local-ui-development)
- [Sync local changes](#sync-local-changes)

## Local UI development

See the [UI README](../ui/README.md) for development commands and focused tests, and [Python linting](python-lint.md) for Python checks.

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
