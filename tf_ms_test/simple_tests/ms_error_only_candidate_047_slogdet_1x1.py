import os
import traceback

import numpy as np
import tensorflow as tf

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import mindspore as ms
from mindspore import ops


def run_tf_case() -> tuple[bool, str]:
    x_np = np.array([[5.0]], dtype=np.float32)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    sign_tf, logabsdet_tf = tf.linalg.slogdet(x_tf)
    _ = sign_tf.numpy(), logabsdet_tf.numpy()
    return True, ""


def run_ms_case() -> tuple[bool, str]:
    x_np = np.array([[5.0]], dtype=np.float32)
    x_ms = ms.Tensor(x_np, ms.float32)
    sign_ms, logabsdet_ms = ops.slogdet(x_ms)
    _ = sign_ms.asnumpy(), logabsdet_ms.asnumpy()
    return True, ""


def main() -> None:
    print("Candidate 047: tf.linalg.slogdet vs mindspore.ops.slogdet")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"MindSpore version: {ms.__version__}")

    tf_success = True
    try:
        run_tf_case()
        print("[TF] success")
    except Exception:
        tf_success = False
        print("[TF] error")
        print(traceback.format_exc())

    ms_success = True
    try:
        run_ms_case()
        print("[MS] success")
    except Exception:
        ms_success = False
        print("[MS] error")
        print(traceback.format_exc())

    reproduced = tf_success and (not ms_success)
    print(f"REPRODUCED_MS_ERROR_ONLY={reproduced}")


if __name__ == "__main__":
    main()
