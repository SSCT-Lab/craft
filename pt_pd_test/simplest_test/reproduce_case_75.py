
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
    if 'input' in {"api": "torch.nn.functional.mse_loss", "input": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.16318155825138092, 0.4904520809650421, -0.576161801815033, -1.0095715522766113, -0.38941505551338196, 0.6064612865447998, -0.1320561021566391, 0.14729851484298706, -0.6788693070411682, -1.149118423461914]}, "target": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.6876878142356873, -0.24527201056480408, 0.6903731822967529, 0.3563068211078644, -0.6550918817520142, 0.3678756058216095, 2.223078727722168, -1.0070582628250122, 0.44984549283981323, -0.15876135230064392]}, "reduction": "mean"}:
        input_info = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.16318155825138092, 0.4904520809650421, -0.576161801815033, -1.0095715522766113, -0.38941505551338196, 0.6064612865447998, -0.1320561021566391, 0.14729851484298706, -0.6788693070411682, -1.149118423461914]}, "target": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.6876878142356873, -0.24527201056480408, 0.6903731822967529, 0.3563068211078644, -0.6550918817520142, 0.3678756058216095, 2.223078727722168, -1.0070582628250122, 0.44984549283981323, -0.15876135230064392]}, "reduction": "mean"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.16318155825138092, 0.4904520809650421, -0.576161801815033, -1.0095715522766113, -0.38941505551338196, 0.6064612865447998, -0.1320561021566391, 0.14729851484298706, -0.6788693070411682, -1.149118423461914]}, "target": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.6876878142356873, -0.24527201056480408, 0.6903731822967529, 0.3563068211078644, -0.6550918817520142, 0.3678756058216095, 2.223078727722168, -1.0070582628250122, 0.44984549283981323, -0.15876135230064392]}, "reduction": "mean"}
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
    api_name = "torch.nn.functional.mse_loss"
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
    params = {"api": "paddle.nn.functional.mse_loss", "input": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.16318155825138092, 0.4904520809650421, -0.576161801815033, -1.0095715522766113, -0.38941505551338196, 0.6064612865447998, -0.1320561021566391, 0.14729851484298706, -0.6788693070411682, -1.149118423461914]}, "label": {"shape": [10000, 1], "dtype": "float32", "sample_values": [-0.42269760370254517, -1.6849075555801392, -0.18594197928905487, 1.864675760269165, -0.12367667257785797, -0.8393742442131042, -1.3443987369537354, 0.5883033871650696, -0.134107768535614, 0.3346792161464691]}, "reduction": "mean"}
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
    api_name = "paddle.nn.functional.mse_loss"
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
    print(f"Reproducing Case 75: torch.nn.functional.mse_loss vs paddle.nn.functional.mse_loss")
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
