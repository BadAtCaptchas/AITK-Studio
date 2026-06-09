#!/usr/bin/env python3
"""Copy this checkout's git-status changes into another checkout.

This is intended for development workflows where a second local checkout, WSL
share, container mount, or remote-synced folder needs the same uncommitted
working tree changes without pushing a branch first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


TARGET_ENV_VAR = "AITK_DEV_SYNC_TARGET"


@dataclass(frozen=True)
class GitChange:
    xy: str
    path: str
    original_path: str | None = None

    @property
    def is_untracked(self) -> bool:
        return self.xy == "??"

    @property
    def is_rename(self) -> bool:
        return "R" in self.xy


@dataclass
class SyncStats:
    copied: int = 0
    deleted: int = 0
    skipped: int = 0


def run_git(args: list[str], cwd: Path, *, decode: bool = True) -> str | bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    if decode:
        return proc.stdout.decode("utf-8", errors="surrogateescape")
    return proc.stdout


def repo_root() -> Path:
    root = run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    return Path(str(root).strip()).resolve()


def parse_status(output: bytes) -> list[GitChange]:
    text = output.decode("utf-8", errors="surrogateescape")
    fields = text.split("\0")
    if fields and fields[-1] == "":
        fields.pop()

    changes: list[GitChange] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if len(entry) < 4:
            continue

        xy = entry[:2]
        path = entry[3:]
        original_path = None

        # In porcelain v1 -z output, rename/copy records are emitted as:
        # "XY new-path\0old-path\0".
        if "R" in xy or "C" in xy:
            if index >= len(fields):
                raise RuntimeError(f"Malformed git status rename/copy record for {path}")
            original_path = fields[index]
            index += 1

        changes.append(GitChange(xy=xy, path=path, original_path=original_path))

    return changes


def git_status_changes(root: Path, *, include_untracked: bool) -> list[GitChange]:
    untracked_mode = "all" if include_untracked else "no"
    output = run_git(
        ["status", "--porcelain=v1", "-z", f"--untracked-files={untracked_mode}"],
        root,
        decode=False,
    )
    return parse_status(bytes(output))


def safe_relative_parts(git_path: str) -> tuple[str, ...]:
    path = PurePosixPath(git_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Refusing unsafe git path: {git_path!r}")
    return path.parts


def joined_path(root: Path, git_path: str) -> Path:
    return root.joinpath(*safe_relative_parts(git_path))


def remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if dry_run:
        return True
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def copy_path(source: Path, destination: Path, *, dry_run: bool) -> bool:
    if not source.exists() and not source.is_symlink():
        return False

    if dry_run:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir() and not source.is_symlink():
        if destination.exists() and not destination.is_dir():
            remove_path(destination, dry_run=False)
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
        return True

    if destination.exists() or destination.is_symlink():
        remove_path(destination, dry_run=False)
    shutil.copy2(source, destination, follow_symlinks=False)
    return True


def validate_target(target: Path, *, create_target: bool, verify_target: bool) -> None:
    if not target.exists():
        if create_target:
            target.mkdir(parents=True)
        else:
            raise ValueError(
                f"Target does not exist: {target}\n"
                f"Create it first, or rerun with --create-target."
            )

    if not target.is_dir():
        raise ValueError(f"Target is not a directory: {target}")

    if verify_target and not (target / ".git").exists():
        raise ValueError(
            f"Target does not look like a git checkout: {target}\n"
            "Use --no-verify-target if you intentionally want to sync into this directory."
        )


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve() == right.resolve()


def sync_changes(
    changes: list[GitChange],
    source_root: Path,
    target_root: Path,
    *,
    dry_run: bool,
    delete: bool,
) -> SyncStats:
    stats = SyncStats()

    for change in changes:
        if change.is_rename and change.original_path and delete:
            old_target = joined_path(target_root, change.original_path)
            if remove_path(old_target, dry_run=dry_run):
                print(f"delete {change.original_path}")
                stats.deleted += 1

        source = joined_path(source_root, change.path)
        target = joined_path(target_root, change.path)

        if copy_path(source, target, dry_run=dry_run):
            print(f"copy   {change.path}")
            stats.copied += 1
            continue

        if delete:
            if remove_path(target, dry_run=dry_run):
                print(f"delete {change.path}")
                stats.deleted += 1
            else:
                print(f"skip   {change.path} (missing locally and in target)")
                stats.skipped += 1
        else:
            print(f"skip   {change.path} (missing locally; deletes disabled)")
            stats.skipped += 1

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync git-status changes from this checkout into another checkout.",
        epilog=(
            f"Set {TARGET_ENV_VAR} to avoid passing TARGET each time. "
            "Only files reported by git status are copied; ignored files are not included."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=os.environ.get(TARGET_ENV_VAR),
        help=f"Destination checkout path. Defaults to ${TARGET_ENV_VAR}.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print the planned copy/delete operations without changing the target.",
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Exclude untracked files from the sync.",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not delete target files for local deletions or rename sources.",
    )
    parser.add_argument(
        "--create-target",
        action="store_true",
        help="Create the target directory if it does not already exist.",
    )
    parser.add_argument(
        "--no-verify-target",
        action="store_true",
        help="Allow syncing into a directory that does not contain a .git entry.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.target:
        parser.error(f"TARGET is required unless {TARGET_ENV_VAR} is set")

    source_root = repo_root()
    target_root = Path(args.target).expanduser()

    try:
        if same_path(source_root, target_root):
            raise ValueError("Target is the current checkout; choose a different destination.")

        validate_target(
            target_root,
            create_target=args.create_target,
            verify_target=not args.no_verify_target,
        )

        changes = git_status_changes(source_root, include_untracked=not args.tracked_only)
        if not changes:
            print("No local git-status changes to sync.")
            return 0

        if args.dry_run:
            print("Dry run; no files will be changed.")

        stats = sync_changes(
            changes,
            source_root,
            target_root,
            dry_run=args.dry_run,
            delete=not args.no_delete,
        )
        verb = "Would sync" if args.dry_run else "Synced"
        print(f"{verb} {stats.copied} copied, {stats.deleted} deleted, {stats.skipped} skipped.")
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
