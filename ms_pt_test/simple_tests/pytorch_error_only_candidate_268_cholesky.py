"""Attempt repro for candidate index 268.

Target pattern:
- MindSpore succeeds
- PyTorch fails

Candidate config:
- api: mindspore.ops.cholesky / torch.linalg.cholesky
- shape: [3, 3]
- dtype: float32
- upper: True
"""

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def check_once(x_np: np.ndarray, upper: bool) -> tuple[bool, bool, str | None, str | None]:
    ms_ok = False
    pt_ok = False
    ms_error = None
    pt_error = None

    try:
        out_ms = ops.cholesky(ms.Tensor(x_np), upper=upper)
        _ = out_ms.asnumpy()
        ms_ok = True
    except Exception as error:  # noqa: BLE001
        ms_error = str(error)

    try:
        _ = torch.linalg.cholesky(torch.tensor(x_np), upper=upper)
        pt_ok = True
    except Exception as error:  # noqa: BLE001
        pt_error = str(error)

    return ms_ok, pt_ok, ms_error, pt_error


def main() -> None:
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    print("=== Candidate 268 attempt ===")

    upper = True
    fixed_cases = [
        np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, -1.0]], dtype=np.float32),
        np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]], dtype=np.float32),
    ]

    for idx, mat in enumerate(fixed_cases, start=1):
        ms_ok, pt_ok, ms_error, pt_error = check_once(mat, upper)
        print(f"fixed_case_{idx}: ms_ok={ms_ok}, pt_ok={pt_ok}")
        if ms_ok and (not pt_ok):
            print("FOUND with fixed case")
            print(mat)
            print("pytorch_error:", pt_error)
            return
        if (not ms_ok) and (not pt_ok):
            print("both failed")

    rng = np.random.default_rng(268)
    for trial in range(20000):
        mat = rng.uniform(-3.0, 3.0, size=(3, 3)).astype(np.float32)
        ms_ok, pt_ok, _, pt_error = check_once(mat, upper)
        if ms_ok and (not pt_ok):
            print("FOUND with random search, trial:", trial)
            print(mat)
            print("pytorch_error:", pt_error)
            return

    print("NOT_FOUND: no matrix reproduced 'ms success + pytorch failure' in this environment.")


if __name__ == "__main__":
    main()
