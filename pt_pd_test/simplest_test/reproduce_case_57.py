
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
    if 'input' in {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.1280437707901001, -1.59554123878479, -0.1918199211359024, 1.9182707071304321, 0.13978038728237152, 0.40810197591781616, 0.6899324059486389, 1.8525810241699219, -1.6067850589752197, -0.5789147019386292]}, "target": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.3799492418766022, 0.7694655656814575, -0.05071712285280228, -0.3833763301372528, -0.5840621590614319, 0.3674921691417694, -0.6054382920265198, 1.916903018951416, 0.9782658219337463, -1.5031739473342896]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [1.1661255359649658, -0.6499739289283752, -2.2195775508880615, -1.1551518440246582, 0.666904628276825, 1.8700798749923706, 0.13388580083847046, 2.053828477859497, -1.0701137781143188, 0.30096688866615295]}}:
        input_info = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.1280437707901001, -1.59554123878479, -0.1918199211359024, 1.9182707071304321, 0.13978038728237152, 0.40810197591781616, 0.6899324059486389, 1.8525810241699219, -1.6067850589752197, -0.5789147019386292]}, "target": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.3799492418766022, 0.7694655656814575, -0.05071712285280228, -0.3833763301372528, -0.5840621590614319, 0.3674921691417694, -0.6054382920265198, 1.916903018951416, 0.9782658219337463, -1.5031739473342896]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [1.1661255359649658, -0.6499739289283752, -2.2195775508880615, -1.1551518440246582, 0.666904628276825, 1.8700798749923706, 0.13388580083847046, 2.053828477859497, -1.0701137781143188, 0.30096688866615295]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.1280437707901001, -1.59554123878479, -0.1918199211359024, 1.9182707071304321, 0.13978038728237152, 0.40810197591781616, 0.6899324059486389, 1.8525810241699219, -1.6067850589752197, -0.5789147019386292]}, "target": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.3799492418766022, 0.7694655656814575, -0.05071712285280228, -0.3833763301372528, -0.5840621590614319, 0.3674921691417694, -0.6054382920265198, 1.916903018951416, 0.9782658219337463, -1.5031739473342896]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [1.1661255359649658, -0.6499739289283752, -2.2195775508880615, -1.1551518440246582, 0.666904628276825, 1.8700798749923706, 0.13388580083847046, 2.053828477859497, -1.0701137781143188, 0.30096688866615295]}}
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
    api_name = "torch.nn.functional.binary_cross_entropy_with_logits"
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
    params = {"api": "paddle.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.1280437707901001, -1.59554123878479, -0.1918199211359024, 1.9182707071304321, 0.13978038728237152, 0.40810197591781616, 0.6899324059486389, 1.8525810241699219, -1.6067850589752197, -0.5789147019386292]}, "label": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.7918739318847656, -0.1359855979681015, 1.3841495513916016, 1.0965520143508911, 0.7487720847129822, 0.7139925360679626, -1.0324034690856934, -1.04307222366333, 0.1175607293844223, 0.5244409441947937]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [1.1661255359649658, -0.6499739289283752, -2.2195775508880615, -1.1551518440246582, 0.666904628276825, 1.8700798749923706, 0.13388580083847046, 2.053828477859497, -1.0701137781143188, 0.30096688866615295]}}
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
    api_name = "paddle.nn.functional.binary_cross_entropy_with_logits"
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
    print(f"Reproducing Case 57: torch.nn.functional.binary_cross_entropy_with_logits vs paddle.nn.functional.binary_cross_entropy_with_logits")
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
