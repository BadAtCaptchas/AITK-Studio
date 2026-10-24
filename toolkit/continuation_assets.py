"""Publish immutable continuation metadata without exposing partial writes."""
import json
import os
from pathlib import Path
import tempfile


def pin_json(path, value, conflict_message):
    path = Path(path)

    def verify():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f'Corrupt continuation metadata: {path.name}') from error
        if existing != value:
            raise ValueError(conflict_message)

    if path.exists():
        verify()
        return
    # Serialize before creating a file, then publish the complete, flushed file
    # without replacing another writer's result.
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.metadata-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify()
    finally:
        os.unlink(temporary)
