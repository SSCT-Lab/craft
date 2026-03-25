
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
    if 'input' in {"api": "torch.lerp", "input": {"shape": [1, 3], "dtype": "float32", "sample_values": [0.840606689453125, -0.8252919316291809, -0.4929182827472687]}, "end": {"shape": [2, 3, 3], "dtype": "float32", "sample_values": [-0.08837328851222992, 2.2106897830963135, 0.8049333691596985, 2.1904170513153076, 0.8823971152305603, 1.0274792909622192, 1.97407865524292, 0.37726324796676636, -0.18722732365131378, 1.032657504081726]}, "weight": {"shape": [2, 1, 3], "dtype": "float32", "sample_values": [-0.6411649584770203, -0.2650649845600128, -0.9026543498039246, 2.698328733444214, 0.24207067489624023, -0.8509750962257385]}}:
        input_info = {"api": "torch.lerp", "input": {"shape": [1, 3], "dtype": "float32", "sample_values": [0.840606689453125, -0.8252919316291809, -0.4929182827472687]}, "end": {"shape": [2, 3, 3], "dtype": "float32", "sample_values": [-0.08837328851222992, 2.2106897830963135, 0.8049333691596985, 2.1904170513153076, 0.8823971152305603, 1.0274792909622192, 1.97407865524292, 0.37726324796676636, -0.18722732365131378, 1.032657504081726]}, "weight": {"shape": [2, 1, 3], "dtype": "float32", "sample_values": [-0.6411649584770203, -0.2650649845600128, -0.9026543498039246, 2.698328733444214, 0.24207067489624023, -0.8509750962257385]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.lerp", "input": {"shape": [1, 3], "dtype": "float32", "sample_values": [0.840606689453125, -0.8252919316291809, -0.4929182827472687]}, "end": {"shape": [2, 3, 3], "dtype": "float32", "sample_values": [-0.08837328851222992, 2.2106897830963135, 0.8049333691596985, 2.1904170513153076, 0.8823971152305603, 1.0274792909622192, 1.97407865524292, 0.37726324796676636, -0.18722732365131378, 1.032657504081726]}, "weight": {"shape": [2, 1, 3], "dtype": "float32", "sample_values": [-0.6411649584770203, -0.2650649845600128, -0.9026543498039246, 2.698328733444214, 0.24207067489624023, -0.8509750962257385]}}
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
    api_name = "torch.lerp"
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
    params = {"api": "paddle.lerp", "x": {"shape": [1, 3], "dtype": "float32", "sample_values": [0.8639235496520996, 0.7691357731819153, -0.18772701919078827]}, "y": {"shape": [2, 3, 3], "dtype": "float32", "sample_values": [-2.01021671295166, -1.3239511251449585, 1.8076307773590088, 1.6414285898208618, 0.0657886490225792, 1.6629314422607422, -0.16007448732852936, -0.24341030418872833, 0.056510649621486664, -0.12492159754037857]}, "weight": {"shape": [2, 1, 3], "dtype": "float32", "sample_values": [-0.6411649584770203, -0.2650649845600128, -0.9026543498039246, 2.698328733444214, 0.24207067489624023, -0.8509750962257385]}}
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
    api_name = "paddle.lerp"
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
    print(f"Reproducing Case 16: torch.lerp vs paddle.lerp")
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
