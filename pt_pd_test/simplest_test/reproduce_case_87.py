
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
    if 'input' in {"api": "torch.nn.functional.mse_loss", "input": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-1.3435563138088207, -1.1768572503653942, -1.0919904280311878, 1.475686815897495, 0.6838452265299534, -0.13975759499110263, 1.324731841954669, -1.7753671009881435, -0.7662765044516577, -0.45998569941267886]}, "target": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-0.486932459314289, -1.6375918714008475, 1.601372172168578, -0.5034528058481812, -1.975401346861347, 0.3081219109527887, 0.34472569042110096, 0.908035789239144, 0.1635418051370828, 1.8620310770239623]}}:
        input_info = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-1.3435563138088207, -1.1768572503653942, -1.0919904280311878, 1.475686815897495, 0.6838452265299534, -0.13975759499110263, 1.324731841954669, -1.7753671009881435, -0.7662765044516577, -0.45998569941267886]}, "target": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-0.486932459314289, -1.6375918714008475, 1.601372172168578, -0.5034528058481812, -1.975401346861347, 0.3081219109527887, 0.34472569042110096, 0.908035789239144, 0.1635418051370828, 1.8620310770239623]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-1.3435563138088207, -1.1768572503653942, -1.0919904280311878, 1.475686815897495, 0.6838452265299534, -0.13975759499110263, 1.324731841954669, -1.7753671009881435, -0.7662765044516577, -0.45998569941267886]}, "target": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-0.486932459314289, -1.6375918714008475, 1.601372172168578, -0.5034528058481812, -1.975401346861347, 0.3081219109527887, 0.34472569042110096, 0.908035789239144, 0.1635418051370828, 1.8620310770239623]}}
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
    params = {"api": "paddle.nn.functional.mse_loss", "input": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-1.3435563138088207, -1.1768572503653942, -1.0919904280311878, 1.475686815897495, 0.6838452265299534, -0.13975759499110263, 1.324731841954669, -1.7753671009881435, -0.7662765044516577, -0.45998569941267886]}, "label": {"shape": [1, 3, 4], "dtype": "float64", "sample_values": [-0.532377229112641, -1.061709321565141, -0.8675898213272059, -0.4863423667936291, 0.2533523667130525, -0.02878991406530333, 0.3962521900215446, -1.5737120160913383, 0.10469026595492369, 0.3569335653529185]}}
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
    print(f"Reproducing Case 87: torch.nn.functional.mse_loss vs paddle.nn.functional.mse_loss")
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
