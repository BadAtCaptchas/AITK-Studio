"""Exercise compiled NATTEN forward/backward kernels against reference results."""
import importlib.metadata
import json

import natten
import torch
from natten.backends.reference import reference_fna_generic
from torch.nn.functional import scaled_dot_product_attention


def compare(label, shape, dtype, actual_fn, reference_fn):
    inputs = [torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True) for _ in range(3)]
    refs = [x.detach().float().requires_grad_() for x in inputs]
    actual = actual_fn(*inputs)
    expected = reference_fn(*refs)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, refs, grad.float())
    tol = 0.035 if dtype == torch.bfloat16 else 0.006
    torch.testing.assert_close(actual.float(), expected, atol=tol, rtol=tol)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad.float(), expected_grad, atol=tol, rtol=tol)
    torch.cuda.synchronize()
    print(f'PASS {label} {dtype}: forward and q/k/v gradients', flush=True)


assert natten.HAS_LIBNATTEN, 'The wheel must contain working CUDA kernels'
assert torch.cuda.is_available()
torch.manual_seed(42)
print(json.dumps({
    'natten': importlib.metadata.version('natten'),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(),
    'capability': torch.cuda.get_device_capability(),
    'module': natten.__file__,
}), flush=True)
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for dims, shape in [(1, (1, 32, 2, 32)), (2, (1, 8, 8, 2, 32)), (3, (1, 5, 5, 5, 2, 32))]:
        op = getattr(natten, f'na{dims}d')
        for causal, dilation in [(False, 1), (True, 1), (False, 2)]:
            if dims == 3 and dilation == 2:
                continue
            kwargs = dict(kernel_size=3, dilation=dilation, is_causal=causal)
            compare(
                f'na{dims}d causal={causal} dilation={dilation}', shape, dtype,
                lambda q, k, v: op(q, k, v, backend='cutlass-fna', **kwargs),
                lambda q, k, v: reference_fna_generic(q, k, v, **kwargs),
            )
    for causal in (False, True):
        compare(
            f'attention causal={causal}', (1, 32, 2, 32), dtype,
            lambda q, k, v: natten.attention(q, k, v, is_causal=causal, backend='cutlass-fmha'),
            lambda q, k, v: scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
            ).transpose(1, 2),
        )
print('All 30 forward/backward comparisons passed.', flush=True)
