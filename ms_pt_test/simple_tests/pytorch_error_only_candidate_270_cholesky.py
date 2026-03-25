"""Attempt repro for candidate index 270.

Target pattern:
- MindSpore succeeds
- PyTorch fails

Candidate config:
- shape: [4, 4]
- dtype: float64
- upper: default(False)
"""

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def check_once(x_np: np.ndarray) -> tuple[bool, bool, str | None, str | None]:
    ms_ok = False
    pt_ok = False
    ms_error = None
    pt_error = None

    try:
        out_ms = ops.cholesky(ms.Tensor(x_np))
        _ = out_ms.asnumpy()
        ms_ok = True
    except Exception as error:  # noqa: BLE001
        ms_error = str(error)

    try:
        _ = torch.linalg.cholesky(torch.tensor(x_np, dtype=torch.float64))
        pt_ok = True
    except Exception as error:  # noqa: BLE001
        pt_error = str(error)

    return ms_ok, pt_ok, ms_error, pt_error


def main() -> None:
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    print("=== Candidate 270 attempt ===")

    fixed_cases = [
        np.diag([-1.0, 1.0, 2.0, 3.0]).astype(np.float64),
        np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0], [3.0, 0.0, -1.0, 2.0], [4.0, 1.0, 2.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 3.0], [0.0, 0.0, 3.0, 0.0]], dtype=np.float64),
    ]

    for idx, mat in enumerate(fixed_cases, start=1):
        ms_ok, pt_ok, _, pt_error = check_once(mat)
        print(f"fixed_case_{idx}: ms_ok={ms_ok}, pt_ok={pt_ok}")
        if ms_ok and (not pt_ok):
            print("FOUND with fixed case")
            print(mat)
            print("pytorch_error:", pt_error)
            return

    rng = np.random.default_rng(270)
    for trial in range(20000):
        mat = rng.uniform(-4.0, 4.0, size=(4, 4)).astype(np.float64)
        ms_ok, pt_ok, _, pt_error = check_once(mat)
        if ms_ok and (not pt_ok):
            print("FOUND with random search, trial:", trial)
            print(mat)
            print("pytorch_error:", pt_error)
            return

    print("NOT_FOUND: no matrix reproduced 'ms success + pytorch failure' in this environment.")


if __name__ == "__main__":
    main()
