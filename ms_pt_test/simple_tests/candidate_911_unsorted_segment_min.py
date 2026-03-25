"""Minimal repro for candidate index 911.

From analysis issue_candidates_both_error_samples_20260221_171310.json:
- ms_api: mindspore.ops.unsorted_segment_min
- x shape: [4], dtype: float32
- segment_ids shape: [4], dtype: int32
- num_segments: 5

PyTorch side is marked as "skip" in candidate data due to semantic mismatch.
"""

import traceback

import numpy as np

import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro ===")
    print("mindspore version:", ms.__version__)

    x = ms.Tensor(np.array([3.5, -1.0, 2.0, 4.2], dtype=np.float32))
    # The candidate only provides shape/dtype for segment_ids, not concrete values.
    # Use one out-of-range id to reliably trigger the same kernel-launch failure pattern.
    segment_ids = ms.Tensor(np.array([0, 2, 8, 2], dtype=np.int32))
    num_segments = 5

    print("x shape:", x.shape, "dtype:", x.dtype)
    print("segment_ids shape:", segment_ids.shape, "dtype:", segment_ids.dtype)
    print("num_segments:", num_segments)

    try:
        out = ops.unsorted_segment_min(x, segment_ids, num_segments)
        # Force execution to surface deferred runtime errors.
        out_np = out.asnumpy()
        print("unexpected success, output:", out_np)
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print("caught exception message:")
        print(error)
        print("traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    run_mindspore()
