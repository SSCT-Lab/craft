
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
    if 'input' in {"api": "torch.split", "tensor": {"shape": [128, 116, 16, 16], "dtype": "float32", "sample_values": [0.6462712287902832, 1.0259859561920166, 1.6038140058517456, -0.5383261442184448, -0.44232356548309326, -0.4519813656806946, 0.8260343670845032, 0.7233507037162781, -0.8153083324432373, -0.19612039625644684]}, "split_size_or_sections": 58, "dim": 1}:
        input_info = {"api": "torch.split", "tensor": {"shape": [128, 116, 16, 16], "dtype": "float32", "sample_values": [0.6462712287902832, 1.0259859561920166, 1.6038140058517456, -0.5383261442184448, -0.44232356548309326, -0.4519813656806946, 0.8260343670845032, 0.7233507037162781, -0.8153083324432373, -0.19612039625644684]}, "split_size_or_sections": 58, "dim": 1}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.split", "tensor": {"shape": [128, 116, 16, 16], "dtype": "float32", "sample_values": [0.6462712287902832, 1.0259859561920166, 1.6038140058517456, -0.5383261442184448, -0.44232356548309326, -0.4519813656806946, 0.8260343670845032, 0.7233507037162781, -0.8153083324432373, -0.19612039625644684]}, "split_size_or_sections": 58, "dim": 1}
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
    api_name = "torch.split"
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
    params = {"api": "paddle.split", "x": {"shape": [128, 116, 16, 16], "dtype": "float32", "sample_values": [0.10416247695684433, 0.19478070735931396, 0.12136126309633255, -1.9196019172668457, 2.3189055919647217, -1.31280517578125, 0.13926167786121368, -0.2753335237503052, -0.009365061298012733, -0.3066876232624054]}, "num_or_sections": [58, 58], "axis": 1}
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
    api_name = "paddle.split"
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
    print(f"Reproducing Case 119: torch.split vs paddle.split")
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
