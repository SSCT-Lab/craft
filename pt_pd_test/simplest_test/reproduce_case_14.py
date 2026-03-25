
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
    if 'input' in {"api": "torch.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.10346278984692345, -0.4895706470306716, 0.04189592115427648, 2.6366691419455885, 1.1535028497815438, 1.1625736681878402, 0.435546318992364, 0.16648527257714707, -2.3960822883304163, -0.6789803464631321]}, "end": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.43102295801089313, -0.11356256479045149, -0.05585218265281086, -0.3773865174622099, 1.3448540311806945, -0.7329260381085351, -0.7402322525487084, 2.1209935658451093, 0.9778475646543314, -2.3851136986930364]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.08378309988906465, 2.254038108921119, -0.20233890687008754, -0.18324524320928193, -1.4419103561118092, 0.9459295852703784, 0.9531762543293538, 0.7890752013999099, -0.6292206728473624, -0.7067067683730057]}}:
        input_info = {"api": "torch.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.10346278984692345, -0.4895706470306716, 0.04189592115427648, 2.6366691419455885, 1.1535028497815438, 1.1625736681878402, 0.435546318992364, 0.16648527257714707, -2.3960822883304163, -0.6789803464631321]}, "end": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.43102295801089313, -0.11356256479045149, -0.05585218265281086, -0.3773865174622099, 1.3448540311806945, -0.7329260381085351, -0.7402322525487084, 2.1209935658451093, 0.9778475646543314, -2.3851136986930364]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.08378309988906465, 2.254038108921119, -0.20233890687008754, -0.18324524320928193, -1.4419103561118092, 0.9459295852703784, 0.9531762543293538, 0.7890752013999099, -0.6292206728473624, -0.7067067683730057]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.10346278984692345, -0.4895706470306716, 0.04189592115427648, 2.6366691419455885, 1.1535028497815438, 1.1625736681878402, 0.435546318992364, 0.16648527257714707, -2.3960822883304163, -0.6789803464631321]}, "end": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.43102295801089313, -0.11356256479045149, -0.05585218265281086, -0.3773865174622099, 1.3448540311806945, -0.7329260381085351, -0.7402322525487084, 2.1209935658451093, 0.9778475646543314, -2.3851136986930364]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.08378309988906465, 2.254038108921119, -0.20233890687008754, -0.18324524320928193, -1.4419103561118092, 0.9459295852703784, 0.9531762543293538, 0.7890752013999099, -0.6292206728473624, -0.7067067683730057]}}
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
    params = {"api": "paddle.lerp", "x": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.09030979920643496, 1.1644117405352703, 0.9256767577138694, 0.03433338856883548, -0.36249900288206854, 0.9475119085755672, 0.6746949778086199, -0.61836502004413, -0.7948936470016904, -2.072355181695998]}, "y": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.0105457117410256, 1.1402505526712048, -1.18023173420037, 0.41870457829236835, -0.013973228476073308, -0.33348628096969524, 0.8852563208659603, 0.8790141545570074, -0.9464907959884281, -0.1821605838579241]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.08378309988906465, 2.254038108921119, -0.20233890687008754, -0.18324524320928193, -1.4419103561118092, 0.9459295852703784, 0.9531762543293538, 0.7890752013999099, -0.6292206728473624, -0.7067067683730057]}}
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
    print(f"Reproducing Case 14: torch.lerp vs paddle.lerp")
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
