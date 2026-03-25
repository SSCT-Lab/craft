import numpy as np
import torch
import tensorflow as tf

# 来自样例: llm_enhanced_torch_matmul_20251216_000247.json_sample2.txt
input_np = np.array(
    [
        -0.3334232,
        -0.31540596,
        -0.88246304,
        -0.33851838,
        0.28093928,
        0.3300107,
        1.2412974,
        1.052649,
        1.2152904,
        1.4947938,
    ],
    dtype=np.float32,
).reshape(2, 1, 5)

other_np = np.array(
    [
        2.0505385,
        0.30406088,
        0.17450507,
        1.9595498,
        -1.5622088,
        0.2881979,
        -1.2475091,
        0.06584224,
        -1.1761891,
        -0.50481474,
    ],
    dtype=np.float32,
).reshape(2, 5, 1)

out_pt = torch.matmul(torch.tensor(input_np), torch.tensor(other_np)).detach().numpy()
out_tf = tf.linalg.matmul(tf.constant(input_np), tf.constant(other_np)).numpy()

max_diff = np.max(np.abs(out_pt - out_tf))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"TensorFlow output shape: {out_tf.shape}")
print(f"Maximum difference: {max_diff}")
