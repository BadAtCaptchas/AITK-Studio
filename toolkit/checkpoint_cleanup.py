"""Checkpoint retention and non-destructive output permission checks."""

import errno
import glob
import json
import os
import re
import shutil
import stat
import tempfile
import time
from typing import Callable, Iterable


_NAME_HISTORY = ".checkpoint_names.json"
_RETRY_DELAYS = (0.1, 0.2)


def remove_old_save(path: str, warn: Callable[[str], None]) -> bool:
    """Best-effort retention must not turn a successful save into a failed job."""
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return True
        except OSError as error:
            if isinstance(error, FileNotFoundError) and not os.path.lexists(path):
                return True
            if attempt < len(_RETRY_DELAYS):
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            warn(
                f"Warning: Could not remove old save '{path}': {error}. "
                "Training will continue. Extra saves may use more disk space."
            )
            return False


def _check_windows_delete_access(path: str) -> None:
    # Request DELETE access, then close the handle without marking for deletion.
    # os.access does not check Windows ACLs or incompatible sharing modes.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    # DELETE, SHARE_READ | SHARE_WRITE | SHARE_DELETE, OPEN_EXISTING,
    # BACKUP_SEMANTICS (directories) | OPEN_REPARSE_POINT (do not follow links).
    handle = create_file(os.path.abspath(path), 0x10000, 0x7, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = path
        raise error
    close_handle(handle)


def _check_existing_save(path: str) -> None:
    info = os.lstat(path)
    is_directory = stat.S_ISDIR(info.st_mode)
    # Detect the filesystem capabilities at runtime so analyzers on Windows
    # also check the POSIX branch, where geteuid is available.
    attributes = getattr(info, "st_file_attributes", None)
    if attributes is not None:
        if not is_directory and attributes & stat.FILE_ATTRIBUTE_READONLY:
            raise PermissionError(errno.EACCES, "Checkpoint is read-only", path)
        _check_windows_delete_access(path)
    else:
        parent = os.path.dirname(path)
        if not os.access(parent, os.W_OK | os.X_OK):
            raise PermissionError(errno.EACCES, "Cannot remove files from this folder", path)
        parent_info = os.stat(parent)
        if parent_info.st_mode & stat.S_ISVTX:
            get_effective_uid = getattr(os, "geteuid", None)
            effective_uid = get_effective_uid() if get_effective_uid is not None else None
            if effective_uid not in (0, info.st_uid, parent_info.st_uid):
                raise PermissionError(errno.EACCES, "Cannot remove this checkpoint from a sticky folder", path)
    # Never traverse a checkpoint symlink or Windows junction during preflight.
    is_reparse = bool((attributes or 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    if is_directory and not is_reparse and not stat.S_ISLNK(info.st_mode):
        with os.scandir(path) as entries:
            for entry in entries:
                _check_existing_save(entry.path)


def check_save_permissions(save_root: str, prefixes: Iterable[str]) -> None:
    """Exercise actual create/write/rename/delete operations before model loading.

    Existing checkpoints are checked without changing their contents or permissions.
    This checks current access; retention still needs to tolerate later file locks.
    """
    probe = None
    try:
        os.makedirs(save_root, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".checkpoint-access-", dir=save_root, delete=False) as handle:
            probe = handle.name
            handle.write(b"checkpoint access check\n")
            handle.flush()
            os.fsync(handle.fileno())
        renamed = probe + ".renamed"
        os.replace(probe, renamed)
        probe = renamed
        os.remove(probe)
        probe = None

        paths = {os.path.join(save_root, name) for name in ("optimizer.pt", _NAME_HISTORY)}
        for prefix in prefixes:
            paths.update(glob.glob(os.path.join(save_root, glob.escape(prefix) + "*")))
        for path in sorted(paths):
            if os.path.lexists(path):
                _check_existing_save(path)
    except OSError as error:
        problem_path = error.filename or save_root
        raise PermissionError(
            f"Checkpoint access check failed for '{problem_path}': {error}. "
            "Training has not started. Close apps using the saves, check folder permissions "
            "and read-only files, or choose another training folder."
        ) from error
    finally:
        if probe is not None:
            try:
                os.remove(probe)
            except OSError:
                # The original failure names the file that could not be removed.
                pass


def load_checkpoint_names(save_root: str, name: str, storage_name: str) -> list[str]:
    """Keep resumes and retention working after a Studio job is renamed."""
    names = [name, storage_name]
    history_path = os.path.join(save_root, _NAME_HISTORY)
    try:
        with open(history_path, encoding="utf-8") as handle:
            history = json.load(handle)
    except FileNotFoundError:
        history = []
    if not isinstance(history, list) or any(
        not isinstance(item, str) or not item or item in (".", "..")
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', item)
        for item in history
    ):
        raise ValueError(f"Invalid checkpoint name history in '{history_path}'")
    return list(dict.fromkeys(names + history))


def save_checkpoint_names(save_root: str, names: list[str]) -> None:
    history_path = os.path.join(save_root, _NAME_HISTORY)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".checkpoint-names-", dir=save_root, delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(names, handle)
        os.replace(temporary_path, history_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)
