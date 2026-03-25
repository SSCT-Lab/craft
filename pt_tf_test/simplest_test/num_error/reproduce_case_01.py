# 该代码文件用于复现并比较 PyTorch 与 TensorFlow 在数值相关 API 上的行为差异（这里是 cholesky_solve）。
# 流程概览：
# 1) 按给定 shape/dtype/sample_values 构造输入的 numpy 数据；
# 2) 将 numpy 数据分别转换为 torch.tensor 与 tf.constant；
# 3) 调用对应 API（torch.cholesky_solve 与 tf.linalg.cholesky_solve）；
# 4) 打印结果形状，并在两者都成功返回时计算两者数值差异的最大值。

import torch
import tensorflow as tf
import numpy as np

def get_input_data(shape, dtype, sample_values):
    # 根据 shape 与 sample_values 生成一个 numpy 数组：
    # - 如果 shape 为空，则认为是标量，直接使用 sample_values[0]
    # - 否则将 sample_values 重复/截断以填满指定的形状，再 reshape
    if not shape:
        return np.array(sample_values[0], dtype=dtype)
    
    total_elements = np.prod(shape)
    if len(sample_values) < total_elements:
        # Repeat values to fill the shape
        repeats = int(np.ceil(total_elements / len(sample_values)))
        full_values = (sample_values * repeats)[:total_elements]
    else:
        full_values = sample_values[:total_elements]
        
    return np.array(full_values, dtype=dtype).reshape(shape)

def test_torch():
    print("Testing torch API...")
    # 构造 torch 端的输入参数字典
    inputs = {}
    
    # 处理主输入：此处通过检测固定字典是否包含 'input'（始终为 True），
    # 从中提取 input 的 shape/dtype/sample_values 来构造 torch.tensor
    if 'input' in {"api": "torch.cholesky_solve", "input": {"shape": [3, 2], "dtype": "float32", "sample_values": [-0.33505409955978394, 0.5129841566085815, 0.33704325556755066, 0.7546667456626892, -0.8685919642448425, -2.908078908920288]}, "input2": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.30667033791542053, -1.3762863874435425, 0.06743878871202469, 0.3087283968925476, 1.9169071912765503, 0.17984506487846375, -0.5016515254974365, -0.6793695688247681, 0.34692952036857605]}}:
        input_info = {"api": "torch.cholesky_solve", "input": {"shape": [3, 2], "dtype": "float32", "sample_values": [-0.33505409955978394, 0.5129841566085815, 0.33704325556755066, 0.7546667456626892, -0.8685919642448425, -2.908078908920288]}, "input2": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.30667033791542053, -1.3762863874435425, 0.06743878871202469, 0.3087283968925476, 1.9169071912765503, 0.17984506487846375, -0.5016515254974365, -0.6793695688247681, 0.34692952036857605]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
             inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
        else:
             # Handle scalar or other types if necessary
             pass
             
    # 处理其他参数（例如 input2 等），如果是带 shape 的字典，就同样转为 torch.tensor
    params = {"api": "torch.cholesky_solve", "input": {"shape": [3, 2], "dtype": "float32", "sample_values": [-0.33505409955978394, 0.5129841566085815, 0.33704325556755066, 0.7546667456626892, -0.8685919642448425, -2.908078908920288]}, "input2": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.30667033791542053, -1.3762863874435425, 0.06743878871202469, 0.3087283968925476, 1.9169071912765503, 0.17984506487846375, -0.5016515254974365, -0.6793695688247681, 0.34692952036857605]}}
    for k, v in params.items():
        if k == 'api': continue
        if k == 'input': continue
        
        if isinstance(v, dict) and 'shape' in v:
            inputs[k] = torch.tensor(get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            ), dtype=getattr(torch, v['dtype']))
        else:
            inputs[k] = v

    # 调用 torch API（此处使用字符串反射获取函数对象）
    # 约定：如果存在 'input' 键，通常作为第一个位置参数，其余作为 kwargs
    api_name = "torch.cholesky_solve"
    try:
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
            
        # 组装参数：将 inputs['input'] 放到位置参数，其余作为关键字参数
        args = []
        kwargs = {}
        
        if 'input' in inputs:
            args.append(inputs['input'])
            
        for k, v in inputs.items():
            if k == 'input': continue
            kwargs[k] = v
            
        result = func(*args, **kwargs)
        print("Torch result shape:", result.shape)
        return result.detach().numpy()
    except Exception as e:
        print(f"Torch error: {e}")
        return None

def test_tensorflow():
    print("\nTesting TensorFlow API...")
    # 构造 TensorFlow 端的输入参数字典
    inputs = {}
    
    # 处理所有参数：带 shape 的字典转换为 tf.constant
    params = {"api": "tf.linalg.cholesky_solve", "rhs": {"shape": [3, 2], "dtype": "float32", "sample_values": [-1.0670522451400757, -0.6584956645965576, -0.8646241426467896, 0.727580189704895, 0.7642821073532104, -1.552671194076538]}, "chol": {"shape": [3, 3], "dtype": "float32", "sample_values": [0.20300863683223724, -1.8740577697753906, -0.4296968877315521, 0.5470719933509827, -0.9220492839813232, -0.9780227541923523, 0.15461786091327667, 0.41276776790618896, -2.111048698425293]}}
    for k, v in params.items():
        if k == 'api': continue
        
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = tf.constant(np_data, dtype=getattr(tf, v['dtype']))
        else:
            inputs[k] = v

    # 调用 TF API：通过字符串路径逐层 getattr 获取函数对象
    api_name = "tf.linalg.cholesky_solve"
    try:
        if '.' in api_name:
            # Handle tf.experimental.numpy -> tf.experimental.numpy
            parts = api_name.split('.')
            module = tf
            for part in parts[1:-1]:
                module = getattr(module, part)
            func = getattr(module, parts[-1])
        else:
            func = eval(api_name)
            
        # 说明：TF 通常使用命名参数，此处直接以 kwargs 方式调用；
        # 如果失败，后续会尝试以位置参数方式重试
        
        args = []
        kwargs = {}
        
        result = func(**inputs)
        print("TensorFlow result shape:", result.shape)
        return result.numpy()
    except Exception as e:
        print(f"TensorFlow error: {e}")
        # 兜底：若命名参数方式失败，尝试按位置参数调用
        try:
             print("Retrying with positional args...")
             result = func(*inputs.values())
             print("TensorFlow result shape:", result.shape)
             return result.numpy()
        except Exception as e2:
             print(f"TensorFlow retry error: {e2}")
             return None

if __name__ == "__main__":
    # 主入口：分别运行 torch 与 TF 的测试，然后在两者都成功返回时计算差异
    print(f"Reproducing Case 1: torch.cholesky_solve vs tf.linalg.cholesky_solve")
    torch_res = test_torch()
    tf_res = test_tensorflow()

    if torch_res is not None and tf_res is not None:
        try:
            # 数值差异：对两个 numpy 数组做逐元素差的绝对值，并打印最大差异
            diff = np.abs(torch_res - tf_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
