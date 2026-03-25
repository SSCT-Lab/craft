# import numpy as np
# import torch
# import tensorflow as tf

# input_data = [
#     [-0.44332579318682674, -0.14522185536296073, 0.14717903848811734],
#     [0.08460438987005668, -0.6951234296799749, 0.24125915763781158],
#     [0.34045975782091975, 0.6165841633753035, 0.5205773942639326]
# ]
# input_np = np.array(input_data, dtype=np.float64)

# input_pt = torch.tensor(input_np)
# out_pt = torch.logdet(input_pt)

# input_tf = tf.constant(input_np)
# out_tf = tf.linalg.logdet(input_tf)

# pt_val = out_pt.numpy()
# tf_val = out_tf.numpy()
# det_val = np.linalg.det(input_np)
# print(f"Matrix determinant: {det_val}")
# print(f"PyTorch logdet output: {pt_val}")
# print(f"TensorFlow logdet output: {tf_val}")
# print(f"Difference: {np.abs(pt_val - tf_val)}")

# import numpy as np
# import torch
# import tensorflow as tf

# np.random.seed(42)
# input_np = np.random.randn(4, 4, 2, 2).astype(np.float32)

# input_pt = torch.tensor(input_np)
# out_pt = torch.logdet(input_pt)

# input_tf = tf.constant(input_np)
# out_tf = tf.linalg.logdet(input_tf)

# pt_np_out = out_pt.numpy()
# tf_np_out = out_tf.numpy()
# max_diff = np.max(np.abs(pt_np_out - tf_np_out))
# neg_det_count = np.sum(np.linalg.det(input_np) <= 0)
# print(f"Number of matrices with non-positive determinant: {neg_det_count}")
# print(f"PyTorch logdet output: {pt_np_out}")
# print(f"TensorFlow logdet output: {tf_np_out}")
# print(f"Maximum difference: {max_diff}")

import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 3, 4, 4).astype(np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
neg_det_count = np.sum(np.linalg.det(input_np) <= 0)
print(f"Number of matrices with non-positive determinant: {neg_det_count}")
print(f"PyTorch logdet output: {pt_np_out}")
print(f"TensorFlow logdet output: {tf_np_out}")
print(f"Maximum difference: {max_diff}")