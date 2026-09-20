"""Sequential safetensors output for filesystems that reject large mmap writes."""
import json
import os
import struct
import tempfile
from collections.abc import Mapping

import torch


_DTYPE_TAGS = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}
_CHUNK_BYTES = 256 * 1024 * 1024


def save_file_streamed(
    tensors: Mapping[str, torch.Tensor],
    path: str | os.PathLike,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write in bounded chunks, publishing only a complete checkpoint."""
    header = {}
    flat = {}
    offset = 0
    for name, tensor in tensors.items():
        if name == "__metadata__" or not isinstance(name, str):
            raise ValueError("Invalid safetensors tensor name")
        if tensor.dtype not in _DTYPE_TAGS or tensor.layout != torch.strided:
            raise ValueError(f"Unsupported tensor format for {name}")
        tensor = tensor.detach().cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _DTYPE_TAGS[tensor.dtype], "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        flat[name] = tensor
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)

    destination = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=os.path.dirname(destination))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(header_bytes)))
            handle.write(header_bytes)
            for tensor in flat.values():
                if tensor.numel():
                    data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
                    for start in range(0, len(data), _CHUNK_BYTES):
                        handle.write(data[start:start + _CHUNK_BYTES])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
