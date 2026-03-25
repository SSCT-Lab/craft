import traceback

import numpy as np
import paddle
import tensorflow as tf


def run_tf_case() -> None:
    x_np = (np.arange(1 * 3 * 32 * 32, dtype=np.float32) / 100.0).reshape(1, 3, 32, 32)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    y_tf = tf.nn.avg_pool(
        input=x_tf,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        data_format="NCHW",
    )
    _ = y_tf.numpy()


def run_paddle_case() -> None:
    x_np = (np.arange(1 * 3 * 32 * 32, dtype=np.float32) / 100.0).reshape(1, 3, 32, 32)
    x_pd = paddle.to_tensor(x_np, dtype="float32")
    y_pd = paddle.nn.functional.avg_pool2d(
        x=x_pd,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding="SAME",
        data_format="NCHW",
    )
    _ = y_pd.numpy()


def main() -> None:
    print("Candidate 033: tf.nn.avg_pool vs paddle.nn.functional.avg_pool2d")
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
