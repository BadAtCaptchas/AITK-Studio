"""Bounded sibling-caption expansion, without recursive prompt substitution."""
import os
import re
import stat

TOKEN = re.compile(r'<caption\.([A-Za-z0-9_.-]+)>')
MAX_CAPTION_BYTES = 1024 * 1024


def expand_caption_tokens(prompt, file_path, *, caption_reader=None):
    """Expand local sibling files or use the caller's encrypted-caption reader.

    Missing, invalid, oversized or escaping files expand to blank. A supplied
    reader is authoritative: encrypted data must never fall back to plaintext.
    """
    if not prompt or '<caption.' not in prompt:
        return prompt
    cache = {}
    parent = os.path.realpath(os.path.dirname(os.path.abspath(file_path)))
    stem = os.path.splitext(os.path.basename(file_path))[0]

    def replace(match):
        extension = match.group(1)
        if extension in cache:
            return cache[extension]
        value = ''
        try:
            if caption_reader is not None:
                value = caption_reader(extension) or ''
                if not isinstance(value, str) or len(value.encode('utf-8')) > MAX_CAPTION_BYTES:
                    value = ''
            else:
                candidate = os.path.realpath(os.path.join(parent, f'{stem}.{extension}'))
                if os.path.normcase(os.path.dirname(candidate)) == os.path.normcase(parent) and os.path.isfile(candidate):
                    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
                    with os.fdopen(os.open(candidate, flags), 'rb') as source:
                        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                            return ''
                        data = source.read(MAX_CAPTION_BYTES + 1)
                    if len(data) <= MAX_CAPTION_BYTES:
                        value = data.decode('utf-8')
        except (OSError, UnicodeError, ValueError):
            value = ''
        cache[extension] = value.strip()
        return cache[extension]

    return TOKEN.sub(replace, prompt)
