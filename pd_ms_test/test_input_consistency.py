#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试输入一致性验证脚本
验证PaddlePaddle和MindSpore是否使用完全相同的输入数据
"""

# ==== 重要：设置环境变量解决PaddlePaddle和MindSpore的protobuf版本冲突 ====
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# =========================================================================

import numpy as np
import paddle
import mindspore
from mindspore import Tensor, context

# 设置MindSpore运行模式为PyNative
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

from llm_enhanced_compare import LLMEnhancedComparator


def test_input_consistency():
    """测试输入一致性"""
    print("=" * 80)
    print("🧪 PaddlePaddle-MindSpore 输入一致性验证测试")
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
    
    # 准备PaddlePaddle参数
    pd_args_1, pd_kwargs_1 = comparator.prepare_arguments_paddle(test_case_1, "paddle.add")
    print("\nPaddlePaddle转换后:")
    for i, arg in enumerate(pd_args_1):
        print(f"  args[{i}]: {arg.numpy()}")
    
    # 准备MindSpore参数
    ms_args_1, ms_kwargs_1 = comparator.prepare_arguments_mindspore(test_case_1, "mindspore.ops.add")
    print("\nMindSpore转换后:")
    for i, arg in enumerate(ms_args_1):
        print(f"  args[{i}]: {arg.asnumpy()}")
    
    # 验证一致性
    if len(pd_args_1) == len(ms_args_1):
        all_match = True
        for i in range(len(pd_args_1)):
            pd_np = pd_args_1[i].numpy()
            ms_np = ms_args_1[i].asnumpy()
            if np.array_equal(pd_np, ms_np):
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
        print(f"\n❌ 测试1失败：参数数量不一致（PD: {len(pd_args_1)}, MS: {len(ms_args_1)}）")
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
        pd_args_2, pd_kwargs_2 = comparator.prepare_arguments_paddle(test_case_2, "paddle.abs")
        print("❌ 测试2失败：应该抛出ValueError但没有抛出")
        return False
    except ValueError as e:
        print(f"✅ 测试2通过：正确抛出异常")
        print(f"  异常信息: {str(e)[:200]}...")
    
    # 测试用例3：prepare_shared_numpy_data转换
    print("\n" + "=" * 80)
    print("【测试3】prepare_shared_numpy_data转换")
    
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
        print("\n✅ 测试3通过：所有张量参数已转换为numpy数组")
    else:
        print("\n❌ 测试3失败：仍有dict格式的参数")
        return False
    
    # 测试用例4：简单的算子调用测试
    print("\n" + "=" * 80)
    print("【测试4】算子基本调用测试")
    
    try:
        # 测试abs算子
        test_tensor = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
        
        pd_tensor = paddle.to_tensor(test_tensor.copy())
        ms_tensor = Tensor(test_tensor.copy())
        
        pd_result = paddle.abs(pd_tensor)
        ms_result = mindspore.ops.abs(ms_tensor)
        
        print(f"输入张量: {test_tensor}")
        print(f"PaddlePaddle paddle.abs结果: {pd_result.numpy()}")
        print(f"MindSpore mindspore.ops.abs结果: {ms_result.asnumpy()}")
        
        if np.allclose(pd_result.numpy(), ms_result.asnumpy()):
            print("\n✅ 测试4通过：PaddlePaddle和MindSpore结果一致")
        else:
            print("\n❌ 测试4失败：结果不一致")
            return False
    except Exception as e:
        print(f"\n❌ 测试4失败：{e}")
        return False
    
    # 测试用例5：算子对比执行测试
    print("\n" + "=" * 80)
    print("【测试5】完整算子对比执行测试")
    
    try:
        # 使用相同的numpy数据
        shared_input = np.array([[-1.5, 2.3], [-3.7, 4.1]], dtype=np.float32)
        
        pd_test_case = {"x": shared_input.copy(), "api": "paddle.abs"}
        ms_test_case = {"x": shared_input.copy(), "api": "mindspore.ops.abs"}
        
        result = comparator.execute_test_case(
            pd_test_case, ms_test_case,
            "paddle.abs", "mindspore.ops.abs"
        )
        
        print(f"执行结果:")
        print(f"  PaddlePaddle成功: {result['paddle_success']}")
        print(f"  MindSpore成功: {result['mindspore_success']}")
        print(f"  结果匹配: {result['results_match']}")
        print(f"  PaddlePaddle输出形状: {result['paddle_shape']}")
        print(f"  MindSpore输出形状: {result['mindspore_shape']}")
        
        if result['results_match']:
            print("\n✅ 测试5通过：完整测试执行成功，结果一致")
        else:
            print(f"\n❌ 测试5失败：{result.get('comparison_error', '未知错误')}")
            return False
    except Exception as e:
        print(f"\n❌ 测试5失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("🎉 所有测试通过！PaddlePaddle-MindSpore输入一致性得到保证。")
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
