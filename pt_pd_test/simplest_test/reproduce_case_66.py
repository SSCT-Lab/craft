
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
    if 'input' in {"input": {"shape": [1, 32, 1080, 1080], "dtype": "float32", "sample_values": [0.09292271733283997, -0.09921056032180786, -0.06847439706325531, 1.4938929080963135, -0.4606103003025055, -0.9430948495864868, 1.7384998798370361, -0.38499560952186584, 0.21781684458255768, 0.917046844959259]}, "running_mean": null, "running_var": null, "weight": {"shape": [32], "dtype": "float32", "sample_values": [0.6005064249038696, -0.08665292710065842, -0.8042373657226562, 1.4362220764160156, -0.8646830320358276, 0.7210879921913147, 0.6112185716629028, -1.0599706172943115, -1.2758837938308716, 1.1291000843048096]}, "bias": {"shape": [32], "dtype": "float32", "sample_values": [1.0950727462768555, -1.4232069253921509, -0.008934719488024712, 0.6081317067146301, 1.0646874904632568, -0.1814490407705307, -0.1379966288805008, -0.009896107949316502, -0.17341892421245575, -0.3955577313899994]}, "use_input_stats": true, "momentum": 0.1, "eps": 1e-05, "api": "torch.nn.functional.instance_norm"}:
        input_info = {"input": {"shape": [1, 32, 1080, 1080], "dtype": "float32", "sample_values": [0.09292271733283997, -0.09921056032180786, -0.06847439706325531, 1.4938929080963135, -0.4606103003025055, -0.9430948495864868, 1.7384998798370361, -0.38499560952186584, 0.21781684458255768, 0.917046844959259]}, "running_mean": null, "running_var": null, "weight": {"shape": [32], "dtype": "float32", "sample_values": [0.6005064249038696, -0.08665292710065842, -0.8042373657226562, 1.4362220764160156, -0.8646830320358276, 0.7210879921913147, 0.6112185716629028, -1.0599706172943115, -1.2758837938308716, 1.1291000843048096]}, "bias": {"shape": [32], "dtype": "float32", "sample_values": [1.0950727462768555, -1.4232069253921509, -0.008934719488024712, 0.6081317067146301, 1.0646874904632568, -0.1814490407705307, -0.1379966288805008, -0.009896107949316502, -0.17341892421245575, -0.3955577313899994]}, "use_input_stats": true, "momentum": 0.1, "eps": 1e-05, "api": "torch.nn.functional.instance_norm"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"input": {"shape": [1, 32, 1080, 1080], "dtype": "float32", "sample_values": [0.09292271733283997, -0.09921056032180786, -0.06847439706325531, 1.4938929080963135, -0.4606103003025055, -0.9430948495864868, 1.7384998798370361, -0.38499560952186584, 0.21781684458255768, 0.917046844959259]}, "running_mean": null, "running_var": null, "weight": {"shape": [32], "dtype": "float32", "sample_values": [0.6005064249038696, -0.08665292710065842, -0.8042373657226562, 1.4362220764160156, -0.8646830320358276, 0.7210879921913147, 0.6112185716629028, -1.0599706172943115, -1.2758837938308716, 1.1291000843048096]}, "bias": {"shape": [32], "dtype": "float32", "sample_values": [1.0950727462768555, -1.4232069253921509, -0.008934719488024712, 0.6081317067146301, 1.0646874904632568, -0.1814490407705307, -0.1379966288805008, -0.009896107949316502, -0.17341892421245575, -0.3955577313899994]}, "use_input_stats": true, "momentum": 0.1, "eps": 1e-05, "api": "torch.nn.functional.instance_norm"}
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
    api_name = "torch.nn.functional.instance_norm"
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
    params = {"input": {"shape": [1, 32, 1080, 1080], "dtype": "float32", "sample_values": [0.09292271733283997, -0.09921056032180786, -0.06847439706325531, 1.4938929080963135, -0.4606103003025055, -0.9430948495864868, 1.7384998798370361, -0.38499560952186584, 0.21781684458255768, 0.917046844959259]}, "running_mean": null, "running_var": null, "weight": {"shape": [32], "dtype": "float32", "sample_values": [0.6005064249038696, -0.08665292710065842, -0.8042373657226562, 1.4362220764160156, -0.8646830320358276, 0.7210879921913147, 0.6112185716629028, -1.0599706172943115, -1.2758837938308716, 1.1291000843048096]}, "bias": {"shape": [32], "dtype": "float32", "sample_values": [1.0950727462768555, -1.4232069253921509, -0.008934719488024712, 0.6081317067146301, 1.0646874904632568, -0.1814490407705307, -0.1379966288805008, -0.009896107949316502, -0.17341892421245575, -0.3955577313899994]}, "use_input_stats": true, "momentum": 0.1, "eps": 1e-05, "api": "torch.nn.functional.instance_norm"}
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
    api_name = "torch.nn.functional.instance_norm"
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
    print(f"Reproducing Case 66: torch.nn.functional.instance_norm vs torch.nn.functional.instance_norm")
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
