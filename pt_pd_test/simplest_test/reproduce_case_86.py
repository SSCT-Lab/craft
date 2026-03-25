
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
    if 'input' in {"api": "torch.nn.functional.mse_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [-2.7887885570526123, 2.6088523864746094, 0.6294252872467041, 0.5737127661705017, 1.4134799242019653, 1.0316283702850342]}, "target": {"shape": [2, 3], "dtype": "float32", "sample_values": [1.1558423042297363, -1.8055822849273682, 0.8483590483665466, -0.8965839147567749, -0.07416098564863205, -0.4314345121383667]}}:
        input_info = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [-2.7887885570526123, 2.6088523864746094, 0.6294252872467041, 0.5737127661705017, 1.4134799242019653, 1.0316283702850342]}, "target": {"shape": [2, 3], "dtype": "float32", "sample_values": [1.1558423042297363, -1.8055822849273682, 0.8483590483665466, -0.8965839147567749, -0.07416098564863205, -0.4314345121383667]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [-2.7887885570526123, 2.6088523864746094, 0.6294252872467041, 0.5737127661705017, 1.4134799242019653, 1.0316283702850342]}, "target": {"shape": [2, 3], "dtype": "float32", "sample_values": [1.1558423042297363, -1.8055822849273682, 0.8483590483665466, -0.8965839147567749, -0.07416098564863205, -0.4314345121383667]}}
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
    params = {"api": "paddle.nn.functional.mse_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [-2.7887885570526123, 2.6088523864746094, 0.6294252872467041, 0.5737127661705017, 1.4134799242019653, 1.0316283702850342]}, "label": {"shape": [2, 3], "dtype": "float32", "sample_values": [-0.19460156559944153, 0.26951324939727783, 1.274024486541748, 1.2193883657455444, -0.5848221778869629, 0.43588724732398987]}}
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
    print(f"Reproducing Case 86: torch.nn.functional.mse_loss vs paddle.nn.functional.mse_loss")
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
