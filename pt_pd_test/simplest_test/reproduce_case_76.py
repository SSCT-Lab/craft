
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
    if 'input' in {"api": "torch.nn.functional.mse_loss", "input": {"shape": [256, 1], "dtype": "float32", "sample_values": [1.7293580770492554, -1.5765578746795654, 0.4986388683319092, 0.03928026556968689, -0.2689659893512726, 0.618901252746582, 1.8660428524017334, -0.7973401546478271, 0.240870863199234, 0.6156963109970093]}, "target": {"shape": [256, 1], "dtype": "float32", "sample_values": [0.7191405892372131, 1.3412795066833496, -0.6062544584274292, -0.8667584657669067, 0.2912656366825104, 0.2988748848438263, -0.7206003665924072, -0.15524709224700928, 0.031931012868881226, 1.541993260383606]}, "reduction": "mean"}:
        input_info = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [256, 1], "dtype": "float32", "sample_values": [1.7293580770492554, -1.5765578746795654, 0.4986388683319092, 0.03928026556968689, -0.2689659893512726, 0.618901252746582, 1.8660428524017334, -0.7973401546478271, 0.240870863199234, 0.6156963109970093]}, "target": {"shape": [256, 1], "dtype": "float32", "sample_values": [0.7191405892372131, 1.3412795066833496, -0.6062544584274292, -0.8667584657669067, 0.2912656366825104, 0.2988748848438263, -0.7206003665924072, -0.15524709224700928, 0.031931012868881226, 1.541993260383606]}, "reduction": "mean"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [256, 1], "dtype": "float32", "sample_values": [1.7293580770492554, -1.5765578746795654, 0.4986388683319092, 0.03928026556968689, -0.2689659893512726, 0.618901252746582, 1.8660428524017334, -0.7973401546478271, 0.240870863199234, 0.6156963109970093]}, "target": {"shape": [256, 1], "dtype": "float32", "sample_values": [0.7191405892372131, 1.3412795066833496, -0.6062544584274292, -0.8667584657669067, 0.2912656366825104, 0.2988748848438263, -0.7206003665924072, -0.15524709224700928, 0.031931012868881226, 1.541993260383606]}, "reduction": "mean"}
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
    params = {"api": "paddle.nn.functional.mse_loss", "input": {"shape": [256, 1], "dtype": "float32", "sample_values": [1.7293580770492554, -1.5765578746795654, 0.4986388683319092, 0.03928026556968689, -0.2689659893512726, 0.618901252746582, 1.8660428524017334, -0.7973401546478271, 0.240870863199234, 0.6156963109970093]}, "label": {"shape": [256, 1], "dtype": "float32", "sample_values": [0.3798743784427643, -0.16103680431842804, 0.1676994264125824, 0.5577983856201172, -0.1258053332567215, 0.201909601688385, -2.49698805809021, -0.5231993198394775, 0.09266702085733414, 0.00014091913180891424]}, "reduction": "mean"}
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
    print(f"Reproducing Case 76: torch.nn.functional.mse_loss vs paddle.nn.functional.mse_loss")
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
