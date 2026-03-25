import traceback

import numpy as np
import paddle
import tensorflow as tf


def run_tf_case() -> None:
    x_np = np.empty((2, 0), dtype=np.int64)
    x_tf = tf.constant(x_np, dtype=tf.int64)
    y_tf = tf.math.argmin(x_tf, axis=-1)
    _ = y_tf.numpy()


def run_paddle_case() -> None:
    x_np = np.empty((2, 0), dtype=np.int64)
    x_pd = paddle.to_tensor(x_np, dtype="int64")
    y_pd = paddle.argmin(x_pd, axis=-1, keepdim=False, dtype="int64")
    _ = y_pd.numpy()


def main() -> None:
    print("Candidate 013: tf.math.argmin vs paddle.argmin")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Paddle version: {paddle.__version__}")

    tf_success = True
    try:
        run_tf_case()
        print("[TF] success")
    except Exception:
        tf_success = False
        print("[TF] error")
        print(traceback.format_exc())

    pd_success = True
    try:
        run_paddle_case()
        print("[PD] success")
    except Exception:
        pd_success = False
        print("[PD] error")
        print(traceback.format_exc())

    reproduced = (not tf_success) and pd_success
    print(f"REPRODUCED_TF_ERROR_ONLY={reproduced}")


if __name__ == "__main__":
    main()
