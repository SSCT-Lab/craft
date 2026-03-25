# 该代码文件用于复现并比较 PyTorch 与 Paddle 在同语义 API 上的数值差异（此例为 as_strided）。
# 流程概览：
# 1) 根据 shape/dtype/sample_values 构造 numpy 输入数据；
# 2) 分别转换为 torch.tensor 与 paddle.Tensor；
# 3) 通过字符串反射调用对应 API（torch.as_strided 与 paddle.as_strided）；
# 4) 打印两端结果的形状，并在均成功时计算两者最大绝对差。

import torch
import paddle
import numpy as np

def get_input_data(shape, dtype, sample_values):
    # 根据提供的形状和样本值生成 numpy 数组：
    # - shape 为空视为标量，直接取 sample_values[0]
    # - 否则按总元素数填充或截断 sample_values，并 reshape 为指定形状
    if not shape:
        return np.array(sample_values[0], dtype=dtype)
    total_elements = np.prod(shape)
    if len(sample_values) < total_elements:
        repeats = int(np.ceil(total_elements / len(sample_values)))
        full_values = (sample_values * repeats)[:total_elements]
    else:
        full_values = sample_values[:total_elements]
    return np.array(full_values, dtype=dtype).reshape(shape)

def test_torch():
    # 构造并调用 Torch 端 API
    inputs = {}
    if 'input' in {"api": "torch.as_strided", "input": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.9080240726470947, -1.4123036861419678, 1.4656487703323364, -0.2257762998342514, 0.06752820312976837, -1.424748182296753, -0.5443827509880066, 0.11092258989810944, -1.1509935855865479]}, "size": [2, 2], "stride": [1, 2], "storage_offset": 1}:
        input_info = {"api": "torch.as_strided", "input": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.9080240726470947, -1.4123036861419678, 1.4656487703323364, -0.2257762998342514, 0.06752820312976837, -1.424748182296753, -0.5443827509880066, 0.11092258989810944, -1.1509935855865479]}, "size": [2, 2], "stride": [1, 2], "storage_offset": 1}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.as_strided", "input": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.9080240726470947, -1.4123036861419678, 1.4656487703323364, -0.2257762998342514, 0.06752820312976837, -1.424748182296753, -0.5443827509880066, 0.11092258989810944, -1.1509935855865479]}, "size": [2, 2], "stride": [1, 2], "storage_offset": 1}
    for k, v in params.items():
        if k == 'api' or k == 'input':
            continue
        if isinstance(v, dict) and 'shape' in v:
            inputs[k] = torch.tensor(get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            ), dtype=getattr(torch, v['dtype']))
        else:
            inputs[k] = v
    api_name = "torch.as_strided"
    try:
        # 通过字符串反射获取函数对象
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
        # 组装参数：若存在 input，则作为位置参数，其余作为关键字参数
        args = []
        kwargs = {}
        if 'input' in inputs:
            args.append(inputs['input'])
        for k, v in inputs.items():
            if k == 'input':
                continue
            kwargs[k] = v
        result = func(*args, **kwargs)
        print("Torch result shape:", result.shape)
        return result.detach().numpy()
    except Exception as e:
        print(f"Torch error: {e}")
        return None

def test_paddle():
    # 构造并调用 Paddle 端 API
    inputs = {}
    # params = {"api": "paddle.as_strided", "input": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.9080240726470947, -1.4123036861419678, 1.4656487703323364, -0.2257762998342514, 0.06752820312976837, -1.424748182296753, -0.5443827509880066, 0.11092258989810944, -1.1509935855865479]}, "shape": [2, 2], "stride": [1, 2]}
    # 对齐 Torch 参数：size -> shape，stride 保持一致，storage_offset -> offset
    params = {"api": "paddle.as_strided", "input": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.9080240726470947, -1.4123036861419678, 1.4656487703323364, -0.2257762998342514, 0.06752820312976837, -1.424748182296753, -0.5443827509880066, 0.11092258989810944, -1.1509935855865479]}, "shape": [2, 2], "stride": [1, 2], "offset": 1}
    for k, v in params.items():
        if k == 'api':
            continue
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = paddle.to_tensor(np_data, dtype=getattr(paddle, v['dtype']))
        else:
            inputs[k] = v
    api_name = "paddle.as_strided"
    try:
        # 通过层级 getattr 获取 Paddle 端函数对象，并优先使用命名参数调用
        parts = api_name.split('.')
        module = eval(parts[0])
        for part in parts[1:-1]:
            module = getattr(module, part)
        func = getattr(module, parts[-1])
        result = func(**inputs)
        print("Paddle result shape:", result.shape)
        return result.numpy()
    except Exception as e:
        print(f"Paddle error: {e}")
        try:
            # 若命名参数调用失败，尝试位置参数顺序调用
            print("Retrying with positional args...")
            result = func(*inputs.values())
            print("Paddle result shape:", result.shape)
            return result.numpy()
        except Exception as e2:
            print(f"Paddle retry error: {e2}")
            return None

if __name__ == "__main__":
    # 主入口：分别运行 Torch 与 Paddle 的测试
    print(f"Reproducing Case 1: torch.as_strided vs paddle.as_strided")
    torch_res = test_torch()
    pd_res = test_paddle()
    if torch_res is not None and pd_res is not None:
        try:
            # 计算数值差异：逐元素绝对差并求最大值
            diff = np.abs(torch_res - pd_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
