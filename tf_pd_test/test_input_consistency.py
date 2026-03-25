#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试输入一致性验证脚本
验证TensorFlow和PaddlePaddle是否使用完全相同的输入数据
"""

import numpy as np
import tensorflow as tf
import paddle
from llm_enhanced_compare import LLMEnhancedComparator


def test_input_consistency():
    """测试输入一致性"""
    print("=" * 80)
    print("🧪 输入一致性验证测试")
    print("=" * 80)
    
    # 创建比较器实例
    comparator = LLMEnhancedComparator()
    
    # 测试用例1：简单的张量参数
    print("\n【测试1】简单张量参数")
    test_case_1 = {
        "api": "test_api",
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "y": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    }
    
    print("原始测试用例（numpy数组）:")
    print(f"  x: shape={test_case_1['x'].shape}, dtype={test_case_1['x'].dtype}")
    print(f"  x值:\n{test_case_1['x']}")
    print(f"  y: shape={test_case_1['y'].shape}, dtype={test_case_1['y'].dtype}")
    print(f"  y值:\n{test_case_1['y']}")
    
    # 准备TensorFlow参数
    tf_args_1, tf_kwargs_1 = comparator.prepare_arguments_tensorflow(test_case_1, "tf.add")
    print("\nTensorFlow转换后:")
    for i, arg in enumerate(tf_args_1):
        print(f"  args[{i}]: {arg.numpy()}")
    
    # 准备PaddlePaddle参数
    pd_args_1, pd_kwargs_1 = comparator.prepare_arguments_paddle(test_case_1, "paddle.add")
    print("\nPaddlePaddle转换后:")
    for i, arg in enumerate(pd_args_1):
        print(f"  args[{i}]: {arg.numpy()}")
    
    # 验证一致性
    if len(tf_args_1) == len(pd_args_1):
        all_match = True
        for i in range(len(tf_args_1)):
            tf_np = tf_args_1[i].numpy()
            pd_np = pd_args_1[i].numpy()
            if np.array_equal(tf_np, pd_np):
                print(f"  ✅ args[{i}] 完全一致")
            else:
                print(f"  ❌ args[{i}] 不一致！")
                all_match = False
        
        if all_match:
            print("\n✅ 测试1通过：输入完全一致")
        else:
            print("\n❌ 测试1失败：输入不一致")
            return False
    else:
        print(f"\n❌ 测试1失败：参数数量不一致（TF: {len(tf_args_1)}, Paddle: {len(pd_args_1)}）")
        return False
    
    # 测试用例2：包含dict格式（应该抛出异常）
    print("\n" + "=" * 80)
    print("【测试2】dict格式参数（应该抛出异常）")
    test_case_2 = {
        "api": "test_api",
        "x": {"shape": [2, 3], "dtype": "float32"}  # 这是错误的格式
    }
    
    print("测试用例（dict格式）:")
    print(f"  x: {test_case_2['x']}")
    
    try:
        tf_args_2, tf_kwargs_2 = comparator.prepare_arguments_tensorflow(test_case_2, "tf.abs")
        print("❌ 测试2失败：应该抛出ValueError但没有抛出")
        return False
    except ValueError as e:
        print(f"✅ 测试2通过：正确抛出异常")
        print(f"  异常信息: {str(e)[:200]}...")
    
    # 测试用例3：可变参数(varargs)
    print("\n" + "=" * 80)
    print("【测试3】可变参数")
    test_case_3 = {
        "api": "test_api",
        "*tensors": [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ]
    }
    
    print("测试用例（可变参数）:")
    for i, tensor in enumerate(test_case_3['*tensors']):
        print(f"  tensors[{i}]: shape={tensor.shape}, dtype={tensor.dtype}, 值={tensor}")
    
    tf_args_3, tf_kwargs_3 = comparator.prepare_arguments_tensorflow(test_case_3, "tf.concat")
    pd_args_3, pd_kwargs_3 = comparator.prepare_arguments_paddle(test_case_3, "paddle.concat")
    
    print("\nTensorFlow转换后:")
    for i, arg in enumerate(tf_args_3):
        print(f"  args[{i}]: {arg.numpy()}")
    
    print("\nPaddlePaddle转换后:")
    for i, arg in enumerate(pd_args_3):
        print(f"  args[{i}]: {arg.numpy()}")
    
    # 验证一致性
    if len(tf_args_3) == len(pd_args_3):
        all_match = True
        for i in range(len(tf_args_3)):
            tf_np = tf_args_3[i].numpy()
            pd_np = pd_args_3[i].numpy()
            if np.array_equal(tf_np, pd_np):
                print(f"  ✅ args[{i}] 完全一致")
            else:
                print(f"  ❌ args[{i}] 不一致！")
                all_match = False
        
        if all_match:
            print("\n✅ 测试3通过：可变参数输入完全一致")
        else:
            print("\n❌ 测试3失败：可变参数输入不一致")
            return False
    else:
        print(f"\n❌ 测试3失败：参数数量不一致（TF: {len(tf_args_3)}, Paddle: {len(pd_args_3)}）")
        return False
    
    # 测试用例4：prepare_shared_numpy_data是否正确转换
    print("\n" + "=" * 80)
    print("【测试4】prepare_shared_numpy_data转换")
    
    # 模拟MongoDB文档
    mock_document = {
        "api": "torch.add",
        "_id": "test_id",
        "x": [{"shape": [2, 3], "dtype": "float32"}],  # list of dicts
        "y": [{"shape": [2, 3], "dtype": "float32"}]
    }
    
    print("模拟MongoDB文档:")
    print(f"  x[0]: {mock_document['x'][0]}")
    print(f"  y[0]: {mock_document['y'][0]}")
    
    shared_data = comparator.prepare_shared_numpy_data(mock_document, case_index=0)
    
    print("\n转换后的共享数据:")
    for key, value in shared_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"    值的前5个元素: {value.flatten()[:5]}")
        else:
            print(f"  {key}: {value}")
    
    # 验证是否已经转换为numpy数组
    if isinstance(shared_data.get('x'), np.ndarray) and isinstance(shared_data.get('y'), np.ndarray):
        print("\n✅ 测试4通过：所有张量参数已转换为numpy数组")
    else:
        print("\n❌ 测试4失败：仍有dict格式的参数")
        return False
    
    # 测试用例5：可变参数中的dict应该被转换
    print("\n" + "=" * 80)
    print("【测试5】可变参数中的dict转换")
    
    mock_document_varargs = {
        "api": "torch.cat",
        "_id": "test_id",
        "*tensors": [
            [  # 第一个测试用例
                {"shape": [2, 3], "dtype": "float32"},
                {"shape": [2, 3], "dtype": "float32"}
            ]
        ]
    }
    
    print("模拟MongoDB文档（可变参数）:")
    print(f"  *tensors[0]: {mock_document_varargs['*tensors'][0]}")
    
    shared_data_varargs = comparator.prepare_shared_numpy_data(mock_document_varargs, case_index=0)
    
    print("\n转换后的共享数据:")
    for key, value in shared_data_varargs.items():
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            for i, item in enumerate(value):
                if isinstance(item, np.ndarray):
                    print(f"    [{i}]: numpy.ndarray, shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"    [{i}]: {type(item).__name__} = {item}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # 验证可变参数中的元素是否已转换为numpy数组
    varargs_value = shared_data_varargs.get('*tensors')
    if isinstance(varargs_value, list):
        all_numpy = all(isinstance(item, np.ndarray) for item in varargs_value)
        if all_numpy:
            print("\n✅ 测试5通过：可变参数中的所有元素已转换为numpy数组")
        else:
            print("\n❌ 测试5失败：可变参数中仍有非numpy数组元素")
            return False
    else:
        print(f"\n❌ 测试5失败：可变参数格式错误（类型={type(varargs_value)}）")
        return False
    
    print("\n" + "=" * 80)
    print("🎉 所有测试通过！输入一致性得到保证。")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_input_consistency()
    if not success:
        print("\n❌ 测试失败，请检查代码")
        exit(1)
    else:
        print("\n✅ 所有测试通过")
        exit(0)
