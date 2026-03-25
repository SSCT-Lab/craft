import traceback

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F


def run_tensorflow_case() -> tuple[bool, str]:
    x_np = (np.arange(1 * 4 * 8 * 8 * 8, dtype=np.float32) / 50.0).reshape(1, 4, 8, 8, 8)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    y_tf = tf.nn.avg_pool3d(
        input=x_tf,
        ksize=[1, 2, 3, 3, 3],
        strides=[1, 2, 3, 3, 3],
        padding="VALID",
        data_format="NCDHW",
    )
    _ = y_tf.numpy()
    return True, ""


def run_pytorch_case() -> tuple[bool, str]:
    x_np = (np.arange(1 * 4 * 8 * 8 * 8, dtype=np.float32) / 50.0).reshape(1, 4, 8, 8, 8)
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    y_torch = F.avg_pool3d(
        x_torch,
        kernel_size=(2, 3, 3),
        stride=(2, 3, 3),
        padding=(0, 0, 0),
        count_include_pad=True,
    )
    _ = y_torch.detach().cpu().numpy()
    return True, ""


def main() -> None:
    print("Candidate 037: tf.nn.avg_pool3d vs torch.nn.functional.avg_pool3d")
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
