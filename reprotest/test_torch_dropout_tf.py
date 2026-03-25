"""
最小可复现代码：torch.nn.functional.dropout vs tf.nn.dropout
测试场景：Dropout 随机丢弃操作

来源文件: llm_enhanced_torch_nn_functional_dropout_20251215_230222.json
样例编号: 1, 2, 3

关键问题：Dropout 是随机操作，能否通过参数设置使输出一致？
"""

import numpy as np

# 创建确定性输入数据
def create_input(shape, sample_values, dtype=np.float32):
    total = np.prod(shape)
    data = []
    for i in range(total):
        data.append(sample_values[i % len(sample_values)])
    return np.array(data, dtype=dtype).reshape(shape)

print("=" * 70)
print("测试1: training=True 时的 Dropout (样例1)")
print("=" * 70)

shape1 = [2110, 16]
sample1 = [-0.398, -0.829, -0.289, -0.952, -2.413, -0.659, 1.701, -1.035, 0.032, 0.893]
input_data1 = create_input(shape1, sample1)

print(f"输入形状: {shape1}")
print(f"Dropout rate: 0.5")
print()

# PyTorch
import torch
torch.manual_seed(42)
pt_input1 = torch.tensor(input_data1)
pt_output1 = torch.nn.functional.dropout(pt_input1, p=0.5, training=True)
pt_result1 = pt_output1.numpy()

print(f"PyTorch (training=True):")
print(f"  输出前10个值: {pt_result1.flatten()[:10]}")
print(f"  零值比例: {(pt_result1 == 0).sum() / pt_result1.size:.2%}")
print()

# TensorFlow
import tensorflow as tf
tf.random.set_seed(42)
tf_input1 = tf.constant(input_data1)
tf_output1 = tf.nn.dropout(tf_input1, rate=0.5)
tf_result1 = tf_output1.numpy()

print(f"TensorFlow (rate=0.5):")
print(f"  输出前10个值: {tf_result1.flatten()[:10]}")
print(f"  零值比例: {(tf_result1 == 0).sum() / tf_result1.size:.2%}")
print()

diff1 = np.abs(pt_result1 - tf_result1).max()
print(f"最大差异: {diff1:.4f}")
print("结论: training=True 时，随机丢弃位置不同，数值必然不同")
print()

print("=" * 70)
print("测试2: PyTorch training=False vs TensorFlow (样例2)")
print("=" * 70)

shape2 = [2110, 16]
sample2 = [-1.473, 0.461, -0.635, 0.279, -1.528, -2.299, 0.759, -0.720, -0.168, 1.103]
input_data2 = create_input(shape2, sample2)

# PyTorch training=False
pt_input2 = torch.tensor(input_data2)
pt_output2 = torch.nn.functional.dropout(pt_input2, p=0.5, training=False)
pt_result2 = pt_output2.numpy()

print(f"PyTorch (training=False):")
print(f"  输出前10个值: {pt_result2.flatten()[:10]}")
print(f"  零值比例: {(pt_result2 == 0).sum() / pt_result2.size:.2%}")
print(f"  与输入相同: {np.allclose(pt_result2, input_data2)}")
print()

# TensorFlow 仍然执行 dropout
tf.random.set_seed(42)
tf_input2 = tf.constant(input_data2)
tf_output2 = tf.nn.dropout(tf_input2, rate=0.5)
tf_result2 = tf_output2.numpy()

print(f"TensorFlow (rate=0.5, 无training参数):")
print(f"  输出前10个值: {tf_result2.flatten()[:10]}")
print(f"  零值比例: {(tf_result2 == 0).sum() / tf_result2.size:.2%}")
print()

diff2 = np.abs(pt_result2 - tf_result2).max()
print(f"最大差异: {diff2:.4f}")
print("结论: PyTorch training=False 不丢弃，TensorFlow 仍丢弃，行为不一致!")
print()

print("=" * 70)
print("测试3: 如何使两者输出一致？")
print("=" * 70)

print("\n方案A: 两者都禁用 Dropout")
print("-" * 50)

# PyTorch: training=False
pt_input3 = torch.tensor(input_data2)
pt_output3 = torch.nn.functional.dropout(pt_input3, p=0.5, training=False)
pt_result3 = pt_output3.numpy()

# TensorFlow: rate=0 (相当于禁用)
tf_input3 = tf.constant(input_data2)
tf_output3 = tf.nn.dropout(tf_input3, rate=0.0)  # rate=0 不丢弃任何元素
tf_result3 = tf_output3.numpy()

print(f"PyTorch (training=False) 前5个值: {pt_result3.flatten()[:5]}")
print(f"TensorFlow (rate=0.0) 前5个值:    {tf_result3.flatten()[:5]}")
diff3a = np.abs(pt_result3 - tf_result3).max()
print(f"最大差异: {diff3a}")
print(f"完全一致: {np.allclose(pt_result3, tf_result3)}")
print()

print("\n方案B: 使用 Keras 的 training 参数")
print("-" * 50)

# Keras Dropout 层支持 training 参数
keras_dropout = tf.keras.layers.Dropout(rate=0.5)
tf_input4 = tf.constant(input_data2)
tf_output4_train = keras_dropout(tf_input4, training=True)
tf_output4_eval = keras_dropout(tf_input4, training=False)

print(f"Keras Dropout (training=True) 前5个值:  {tf_output4_train.numpy().flatten()[:5]}")
print(f"Keras Dropout (training=False) 前5个值: {tf_output4_eval.numpy().flatten()[:5]}")
print(f"training=False 时与输入一致: {np.allclose(tf_output4_eval.numpy(), input_data2)}")
print()

# 比较 PyTorch training=False 和 Keras training=False
diff3b = np.abs(pt_result3 - tf_output4_eval.numpy()).max()
print(f"PyTorch(training=False) vs Keras(training=False) 差异: {diff3b}")
print()

print("=" * 70)
print("测试4: 验证 Dropout 的缩放行为 (Inverted Dropout)")
print("=" * 70)

# Dropout 会对保留的值进行缩放: output = input / (1 - p) * mask
# 这样训练和推理时期望值一致

small_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

print(f"输入: {small_input}")
print(f"Dropout rate: 0.5")
print()

# 多次运行取平均，验证缩放
torch.manual_seed(0)
pt_outputs = []
for _ in range(1000):
    out = torch.nn.functional.dropout(torch.tensor(small_input), p=0.5, training=True)
    pt_outputs.append(out.numpy())
pt_mean = np.mean(pt_outputs, axis=0)

tf.random.set_seed(0)
tf_outputs = []
for _ in range(1000):
    out = tf.nn.dropout(tf.constant(small_input), rate=0.5)
    tf_outputs.append(out.numpy())
tf_mean = np.mean(tf_outputs, axis=0)

print(f"PyTorch 1000次平均: {pt_mean}")
print(f"TensorFlow 1000次平均: {tf_mean}")
print(f"原始输入: {small_input}")
print("结论: 两者都正确实现了 Inverted Dropout，期望值等于输入")
print()

print("=" * 70)
print("总结")
print("=" * 70)
print("""
Dropout 的语义一致性分析:

1. 随机性问题:
   - Dropout 是随机操作，即使输入相同，输出也会不同
   - 不同框架的随机数生成器不同，无法通过 seed 对齐

2. training 参数差异:
   - PyTorch: training=True 执行 dropout, training=False 不执行
   - TensorFlow tf.nn.dropout: 没有 training 参数，总是执行 dropout
   - TensorFlow Keras Dropout: 支持 training 参数

3. 如何使输出一致:
   方案A: PyTorch training=False + TensorFlow rate=0
   方案B: PyTorch training=False + Keras Dropout(training=False)
   
4. 结论:
   - 样例1,3 (training=True): 数值不同是预期行为，应验证统计特性
   - 样例2 (training=False): PyTorch 不丢弃，但 tf.nn.dropout 仍丢弃
     这是 API 语义差异，tf.nn.dropout 没有 training 参数
     
原始测试的"数值不匹配"对于 training=True 是误报（随机操作）
对于 training=False 是真正的 API 语义差异
""")
