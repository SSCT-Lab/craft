"""Minimal repro for tensorflow_error_only candidate index 225.

Runs MindSpore and TensorFlow in separate subprocesses to avoid runtime conflicts
when importing both frameworks in one process on Windows.
"""

import subprocess
import sys
import textwrap


def run_subprocess(label: str, code: str) -> None:
    print(f"\n=== {label} ===")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    print("exit_code:", completed.returncode)


def main() -> None:
    # Candidate 225 fixed values: segment_ids[0] = 5 with num_segments = 2.
    ms_code = textwrap.dedent(
        """
        import numpy as np
        import mindspore as ms
        from mindspore import ops

        x = np.array([3.0, -1.0, 2.0, 5.0, 0.5, -7.0], dtype=np.float32)
        segment_ids = np.array([5, 1, 0, 1, 0, 1], dtype=np.int32)
        num_segments = 2

        print('mindspore:', ms.__version__)
        print('api: mindspore.ops.unsorted_segment_min')
        print('x:', x)
        print('segment_ids:', segment_ids)
        print('num_segments:', num_segments)

        out = ops.unsorted_segment_min(ms.Tensor(x), ms.Tensor(segment_ids), num_segments)
        print('mindspore_success: True')
        print('output:', out.asnumpy())
        """
    )

    tf_code = textwrap.dedent(
        """
        import numpy as np
        import tensorflow as tf

        x = np.array([3.0, -1.0, 2.0, 5.0, 0.5, -7.0], dtype=np.float32)
        segment_ids = np.array([5, 1, 0, 1, 0, 1], dtype=np.int32)
        num_segments = 2

        print('tensorflow:', tf.__version__)
        print('api: tf.math.unsorted_segment_min')
        print('x:', x)
        print('segment_ids:', segment_ids)
        print('num_segments:', num_segments)

        try:
            out = tf.math.unsorted_segment_min(x, segment_ids, num_segments)
            print('tensorflow_success: True')
            print('output:', out.numpy())
        except Exception as e:
            print('tensorflow_success: False')
            print(type(e).__name__)
            print(e)
        """
    )

    run_subprocess("MindSpore Subprocess", ms_code)
    run_subprocess("TensorFlow Subprocess", tf_code)


if __name__ == "__main__":
    main()
