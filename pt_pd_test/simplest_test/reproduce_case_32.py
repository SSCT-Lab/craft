
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
    if 'input' in {"api": "torch.nn.functional.conv3d", "input": {"shape": [2, 3, 4, 4, 4], "dtype": "float32", "sample_values": [-1.0043760538101196, 0.866652250289917, 0.8314599394798279, -0.7042994499206543, -2.317051887512207, -0.361727774143219, 0.4425320327281952, -0.013195186853408813, 0.08048070967197418, 0.3495277166366577]}, "weight": {"shape": [33, 3, 3, 5, 2], "dtype": "float32", "sample_values": [0.9195599555969238, -1.3229821920394897, 0.8595151901245117, -0.17571453750133514, -0.831229567527771, -0.31459030508995056, 0.8882789611816406, 0.6087844371795654, 0.29731401801109314, 0.37848517298698425]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [1.19706392288208, -1.703310489654541, -0.7508948445320129, 0.19531649351119995, -0.029411356896162033, 1.4187215566635132, -1.1707854270935059, -0.3640933334827423, -1.9692845344543457, 0.13160037994384766]}, "stride": [2, 1, 1], "padding": [4, 2, 0]}:
        input_info = {"api": "torch.nn.functional.conv3d", "input": {"shape": [2, 3, 4, 4, 4], "dtype": "float32", "sample_values": [-1.0043760538101196, 0.866652250289917, 0.8314599394798279, -0.7042994499206543, -2.317051887512207, -0.361727774143219, 0.4425320327281952, -0.013195186853408813, 0.08048070967197418, 0.3495277166366577]}, "weight": {"shape": [33, 3, 3, 5, 2], "dtype": "float32", "sample_values": [0.9195599555969238, -1.3229821920394897, 0.8595151901245117, -0.17571453750133514, -0.831229567527771, -0.31459030508995056, 0.8882789611816406, 0.6087844371795654, 0.29731401801109314, 0.37848517298698425]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [1.19706392288208, -1.703310489654541, -0.7508948445320129, 0.19531649351119995, -0.029411356896162033, 1.4187215566635132, -1.1707854270935059, -0.3640933334827423, -1.9692845344543457, 0.13160037994384766]}, "stride": [2, 1, 1], "padding": [4, 2, 0]}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.conv3d", "input": {"shape": [2, 3, 4, 4, 4], "dtype": "float32", "sample_values": [-1.0043760538101196, 0.866652250289917, 0.8314599394798279, -0.7042994499206543, -2.317051887512207, -0.361727774143219, 0.4425320327281952, -0.013195186853408813, 0.08048070967197418, 0.3495277166366577]}, "weight": {"shape": [33, 3, 3, 5, 2], "dtype": "float32", "sample_values": [0.9195599555969238, -1.3229821920394897, 0.8595151901245117, -0.17571453750133514, -0.831229567527771, -0.31459030508995056, 0.8882789611816406, 0.6087844371795654, 0.29731401801109314, 0.37848517298698425]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [1.19706392288208, -1.703310489654541, -0.7508948445320129, 0.19531649351119995, -0.029411356896162033, 1.4187215566635132, -1.1707854270935059, -0.3640933334827423, -1.9692845344543457, 0.13160037994384766]}, "stride": [2, 1, 1], "padding": [4, 2, 0]}
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
    api_name = "torch.nn.functional.conv3d"
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
    params = {"api": "paddle.nn.functional.conv3d", "x": {"shape": [2, 3, 4, 4, 4], "dtype": "float32", "sample_values": [0.32378679513931274, 0.1927059292793274, 0.5059528350830078, -0.01159985363483429, 0.17404627799987793, -1.0132030248641968, -1.6180633306503296, -1.1178581714630127, -1.6870076656341553, 0.22443775832653046]}, "weight": {"shape": [33, 3, 3, 5, 2], "dtype": "float32", "sample_values": [0.9195599555969238, -1.3229821920394897, 0.8595151901245117, -0.17571453750133514, -0.831229567527771, -0.31459030508995056, 0.8882789611816406, 0.6087844371795654, 0.29731401801109314, 0.37848517298698425]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [1.19706392288208, -1.703310489654541, -0.7508948445320129, 0.19531649351119995, -0.029411356896162033, 1.4187215566635132, -1.1707854270935059, -0.3640933334827423, -1.9692845344543457, 0.13160037994384766]}, "stride": [2, 1, 1], "padding": [4, 2, 0]}
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
    api_name = "paddle.nn.functional.conv3d"
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
    print(f"Reproducing Case 32: torch.nn.functional.conv3d vs paddle.nn.functional.conv3d")
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
