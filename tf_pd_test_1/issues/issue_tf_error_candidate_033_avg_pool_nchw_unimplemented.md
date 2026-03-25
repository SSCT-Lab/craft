Add a title
*
Title
tf.nn.avg_pool with data_format='NCHW' on CPU raises UnimplementedError while paddle avg_pool2d succeeds

Issue type
*
What type of issue would you like to report?
Bug

Have you reproduced the bug with TensorFlow Nightly?
*
Not yet (reproduced on TensorFlow 2.20.0 stable).

Source
*
TensorFlow installed from
pip (binary package)

TensorFlow version
*
e.g., tf 2.8
tf 2.20.0

Custom code
*
Yes

OS platform and distribution
e.g., Linux Ubuntu 16.04
Windows 10 (10.0.26200)

Mobile device
e.g., Linux Ubuntu 16.04
N/A

Python version
e.g., 3.9
3.10.18

Bazel version
If compiling from source
N/A

GCC/compiler version
If compiling from source
N/A

CUDA/cuDNN version
N/A (CPU only)

GPU model and memory
If compiling from source
N/A

Current behavior?
*
Also tell us, what did you expect to happen?
Current behavior:
`tf.nn.avg_pool` with `data_format="NCHW"` fails on CPU with `UnimplementedError` from MKL kernel:
`MaxPooling supports exactly one of pooling across depth or pooling across width/height.`

Expected behavior:
Either:
1. The op should run successfully for this valid NCHW 2D average pooling case, or
2. TensorFlow should raise a clear high-level validation error that directly explains unsupported CPU data format constraints (instead of low-level MKL message inconsistent with AvgPool context).

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
Tell us what you see!
Standalone code to reproduce the issue
*
Provide a reproducible test case that is the bare minimum necessary to generate the problem. Please share a link to Colab, Jupyter, or any notebook.
import numpy as np
import tensorflow as tf

print("tf:", tf.__version__)

x_np = (np.arange(1 * 3 * 32 * 32, dtype=np.float32) / 100.0).reshape(1, 3, 32, 32)
x_tf = tf.constant(x_np, dtype=tf.float32)

# TensorFlow path (fails)
y_tf = tf.nn.avg_pool(
    input=x_tf,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding="SAME",
    data_format="NCHW",
)
print(y_tf.numpy().shape)

Tell us what you see!
Relevant log output
2026-03-24 ... OP_REQUIRES failed at mkl_avgpooling_op.cc:68 : UNIMPLEMENTED: MaxPooling supports exactly one of pooling across depth or pooling across width/height.
tensorflow.python.framework.errors_impl.UnimplementedError: ... [Op:AvgPool]

Metadata
Assignees
No one assigned
Labels
No labels
Type
No type
Projects
No projects
Milestone
No milestone
Remember, contributions to this repository should follow its contributing guidelines, security policy and code of conduct.
