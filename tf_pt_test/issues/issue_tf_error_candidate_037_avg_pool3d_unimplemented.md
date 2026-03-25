Add a title
*
Title
tf.nn.avg_pool3d with NCDHW and nontrivial ksize/strides raises UnimplementedError on CPU (candidate 037)

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
`tf.nn.avg_pool3d` with `data_format="NCDHW"`, `ksize=[1,2,3,3,3]`, `strides=[1,2,3,3,3]`, `padding="VALID"` fails on CPU with `UnimplementedError` from MKL backend.

Expected behavior:
Either successful execution or clear user-facing error documenting unsupported argument combinations.

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

x_np = (np.arange(1 * 4 * 8 * 8 * 8, dtype=np.float32) / 50.0).reshape(1, 4, 8, 8, 8)
x_tf = tf.constant(x_np, dtype=tf.float32)

y_tf = tf.nn.avg_pool3d(
    input=x_tf,
    ksize=[1, 2, 3, 3, 3],
    strides=[1, 2, 3, 3, 3],
    padding="VALID",
    data_format="NCDHW",
)
print(y_tf.numpy().shape)

Tell us what you see!
Relevant log output
2026-03-24 ... UNIMPLEMENTED: AvgPooling3D supports exactly one of pooling across depth or pooling across depth/width/height.
tensorflow.python.framework.errors_impl.UnimplementedError: ... [Op:AvgPool3D]

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
