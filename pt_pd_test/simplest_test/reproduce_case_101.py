
import torch
import paddle
import numpy as np

def get_input_data(shape, dtype, sample_values):
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
    inputs = {}
    if 'input' in {"input": {"shape": [3, 3, 3, 3], "dtype": "float64", "sample_values": [-1.2457574620067222, 0.01054231786414665, 0.5134221263242057, 0.3351189472390396, 0.8688013569227642, 0.7946271678206103, 1.180162213223083, 0.9224008867753111, 2.5362883814212744, -0.6446747448422923]}, "api": "torch.rand_like"}:
        input_info = {"input": {"shape": [3, 3, 3, 3], "dtype": "float64", "sample_values": [-1.2457574620067222, 0.01054231786414665, 0.5134221263242057, 0.3351189472390396, 0.8688013569227642, 0.7946271678206103, 1.180162213223083, 0.9224008867753111, 2.5362883814212744, -0.6446747448422923]}, "api": "torch.rand_like"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"input": {"shape": [3, 3, 3, 3], "dtype": "float64", "sample_values": [-1.2457574620067222, 0.01054231786414665, 0.5134221263242057, 0.3351189472390396, 0.8688013569227642, 0.7946271678206103, 1.180162213223083, 0.9224008867753111, 2.5362883814212744, -0.6446747448422923]}, "api": "torch.rand_like"}
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
    api_name = "torch.rand_like"
    try:
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
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
    inputs = {}
    params = {"input": {"shape": [3, 3, 3, 3], "dtype": "float64", "sample_values": [-1.2457574620067222, 0.01054231786414665, 0.5134221263242057, 0.3351189472390396, 0.8688013569227642, 0.7946271678206103, 1.180162213223083, 0.9224008867753111, 2.5362883814212744, -0.6446747448422923]}, "api": "torch.rand_like"}
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
    api_name = "torch.rand_like"
    try:
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
            print("Retrying with positional args...")
            result = func(*inputs.values())
            print("Paddle result shape:", result.shape)
            return result.numpy()
        except Exception as e2:
            print(f"Paddle retry error: {e2}")
            return None

if __name__ == "__main__":
    print(f"Reproducing Case 101: torch.rand_like vs torch.rand_like")
    torch_res = test_torch()
    pd_res = test_paddle()
    if torch_res is not None and pd_res is not None:
        try:
            diff = np.abs(torch_res - pd_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
