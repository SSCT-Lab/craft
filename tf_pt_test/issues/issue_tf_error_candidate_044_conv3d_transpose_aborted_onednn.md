Add a title
*
Title
tf.nn.conv3d_transpose on CPU aborts with oneDNN primitive descriptor error (candidate 044)

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
`tf.nn.conv3d_transpose` on CPU fails with `AbortedError` and oneDNN primitive descriptor creation failure (`mkl_conv_grad_input_ops.cc:546`).

Expected behavior:
The op should either execute correctly or raise a clear `InvalidArgumentError` describing which input/output shape constraint is violated. A backend abort-style error is difficult to diagnose.

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
print(y_tf.numpy().shape)

Tell us what you see!
Relevant log output
ABORTED: Operation received an exception:Status: 2, message: could not create a primitive descriptor for the convolution forward propagation primitive ...
tensorflow.python.framework.errors_impl.AbortedError: ... [Op:Conv3DBackpropInputV2]

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
