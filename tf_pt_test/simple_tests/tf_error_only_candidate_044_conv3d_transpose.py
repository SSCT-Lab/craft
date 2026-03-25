import traceback

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F


def run_tensorflow_case() -> tuple[bool, str]:
    x_np = (np.arange(2 * 3 * 4 * 3 * 2, dtype=np.float32) / 10.0).reshape(2, 3, 4, 3, 2)
    filters_np = (np.arange(3 * 3 * 3 * 2 * 2, dtype=np.float32) / 20.0).reshape(3, 3, 3, 2, 2)

    x_tf = tf.constant(x_np, dtype=tf.float32)
    filters_tf = tf.constant(filters_np, dtype=tf.float32)

    y_tf = tf.nn.conv3d_transpose(
        input=x_tf,
        filters=filters_tf,
        output_shape=[2, 6, 8, 6, 2],
        strides=[1, 2, 2, 2, 1],
        padding="VALID",
    )
    _ = y_tf.numpy()
    return True, ""


def run_pytorch_case() -> tuple[bool, str]:
    x_np = (np.arange(2 * 2 * 3 * 4 * 3, dtype=np.float32) / 10.0).reshape(2, 2, 3, 4, 3)
    weight_np = (np.arange(2 * 2 * 3 * 3 * 3, dtype=np.float32) / 20.0).reshape(2, 2, 3, 3, 3)

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    weight_torch = torch.tensor(weight_np, dtype=torch.float32)

    y_torch = F.conv_transpose3d(
        input=x_torch,
        weight=weight_torch,
        bias=None,
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        output_padding=(0, 0, 0),
        groups=1,
        dilation=(1, 1, 1),
    )
    _ = y_torch.detach().cpu().numpy()
    return True, ""


def main() -> None:
    print("Candidate 044: tf.nn.conv3d_transpose vs torch.nn.functional.conv_transpose3d")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    tf_success = True
    tf_error = ""
    try:
        tf_success, tf_error = run_tensorflow_case()
        print("[TF] success")
    except Exception:
        tf_success = False
        tf_error = traceback.format_exc()
        print("[TF] error")
        print(tf_error)

    torch_success = True
    torch_error = ""
    try:
        torch_success, torch_error = run_pytorch_case()
        print("[PT] success")
    except Exception:
        torch_success = False
        torch_error = traceback.format_exc()
        print("[PT] error")
        print(torch_error)

    reproduced = (not tf_success) and torch_success
    print(f"REPRODUCED_TF_ERROR_ONLY={reproduced}")


if __name__ == "__main__":
    main()
