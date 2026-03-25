
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
    if 'input' in {"api": "torch.lerp", "input": {"shape": [1, 1, 3], "dtype": "float64", "sample_values": [-0.9157550411335605, 0.2496094582097118, -2.440704014775084]}, "end": {"shape": [2, 4, 3], "dtype": "float64", "sample_values": [0.1487806348073052, -0.17125630482493448, 0.08739142892012171, -0.01605403924858619, 2.583573660166567, -1.330226255641338, -1.9899462698211248, 0.33237683351960967, -0.3764225545881953, 0.3160678697299512]}, "weight": {"shape": [2, 1, 3], "dtype": "float64", "sample_values": [-1.323649570842179, -0.435901469090433, 0.8775475534514207, 0.351662354974059, 0.5686446160127705, 0.38292171005726305]}}:
        input_info = {"api": "torch.lerp", "input": {"shape": [1, 1, 3], "dtype": "float64", "sample_values": [-0.9157550411335605, 0.2496094582097118, -2.440704014775084]}, "end": {"shape": [2, 4, 3], "dtype": "float64", "sample_values": [0.1487806348073052, -0.17125630482493448, 0.08739142892012171, -0.01605403924858619, 2.583573660166567, -1.330226255641338, -1.9899462698211248, 0.33237683351960967, -0.3764225545881953, 0.3160678697299512]}, "weight": {"shape": [2, 1, 3], "dtype": "float64", "sample_values": [-1.323649570842179, -0.435901469090433, 0.8775475534514207, 0.351662354974059, 0.5686446160127705, 0.38292171005726305]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.lerp", "input": {"shape": [1, 1, 3], "dtype": "float64", "sample_values": [-0.9157550411335605, 0.2496094582097118, -2.440704014775084]}, "end": {"shape": [2, 4, 3], "dtype": "float64", "sample_values": [0.1487806348073052, -0.17125630482493448, 0.08739142892012171, -0.01605403924858619, 2.583573660166567, -1.330226255641338, -1.9899462698211248, 0.33237683351960967, -0.3764225545881953, 0.3160678697299512]}, "weight": {"shape": [2, 1, 3], "dtype": "float64", "sample_values": [-1.323649570842179, -0.435901469090433, 0.8775475534514207, 0.351662354974059, 0.5686446160127705, 0.38292171005726305]}}
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
    params = {"api": "paddle.lerp", "x": {"shape": [1, 1, 3], "dtype": "float64", "sample_values": [1.6264307346088593, 0.23604306382437004, -0.386022797870698]}, "y": {"shape": [2, 4, 3], "dtype": "float64", "sample_values": [-0.18428900183271194, -0.20583985992134396, -1.364470853617281, -0.9282658386559959, 0.8096575301482413, 0.23071174907496259, 0.1017883337073275, 0.30634753393202263, 0.09262758794892355, 1.2428625628583145]}, "weight": {"shape": [2, 1, 3], "dtype": "float64", "sample_values": [-1.323649570842179, -0.435901469090433, 0.8775475534514207, 0.351662354974059, 0.5686446160127705, 0.38292171005726305]}}
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
    print(f"Reproducing Case 17: torch.lerp vs paddle.lerp")
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
