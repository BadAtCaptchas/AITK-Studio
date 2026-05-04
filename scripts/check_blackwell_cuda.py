import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.cuda_compat import check_blackwell_cuda_compatibility


def _run_cuda_smoke():
    device = torch.device("cuda:0")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")
    print(f"compiled arches: {torch.cuda.get_arch_list()}")

    a = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
    b = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"bf16 matmul: {tuple(c.shape)} {c.dtype}")

    embedding = torch.nn.Embedding(1024, 128, device=device, dtype=torch.bfloat16)
    tokens = torch.randint(0, 1024, (8, 32), device=device)
    embedded = embedding(tokens)
    torch.cuda.synchronize()
    print(f"bf16 embedding: {tuple(embedded.shape)} {embedded.dtype}")

    if hasattr(torch, "float8_e4m3fn"):
        x = torch.randn((128, 128), device=device, dtype=torch.float32)
        x8 = x.to(torch.float8_e4m3fn)
        torch.cuda.synchronize()
        print(f"float8 cast: {tuple(x8.shape)} {x8.dtype}")


def main():
    check_blackwell_cuda_compatibility(torch)
    if not torch.cuda.is_available():
        print("CUDA is not available; Blackwell smoke test skipped.")
        return
    _run_cuda_smoke()
    print("Blackwell CUDA smoke test passed.")


if __name__ == "__main__":
    main()

