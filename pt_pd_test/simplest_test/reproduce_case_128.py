
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
    if 'input' in {"api": "torch.tensordot", "a": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.6459642052650452, -0.7991920113563538, -0.4827435314655304, -0.9533286094665527, 0.12267031520605087, 1.6246784925460815, 0.32307928800582886, -0.2523534893989563, -0.2918112576007843, -1.5631908178329468]}, "b": {"shape": [4, 5, 6], "dtype": "float32", "sample_values": [-2.121854782104492, -0.6078220009803772, 1.2969945669174194, -0.022868061438202858, -0.9993022084236145, -0.5047749280929565, 0.8406200408935547, 0.546733558177948, -0.23893210291862488, -0.3668244183063507]}, "dims": 2}:
        input_info = {"api": "torch.tensordot", "a": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.6459642052650452, -0.7991920113563538, -0.4827435314655304, -0.9533286094665527, 0.12267031520605087, 1.6246784925460815, 0.32307928800582886, -0.2523534893989563, -0.2918112576007843, -1.5631908178329468]}, "b": {"shape": [4, 5, 6], "dtype": "float32", "sample_values": [-2.121854782104492, -0.6078220009803772, 1.2969945669174194, -0.022868061438202858, -0.9993022084236145, -0.5047749280929565, 0.8406200408935547, 0.546733558177948, -0.23893210291862488, -0.3668244183063507]}, "dims": 2}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.tensordot", "a": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.6459642052650452, -0.7991920113563538, -0.4827435314655304, -0.9533286094665527, 0.12267031520605087, 1.6246784925460815, 0.32307928800582886, -0.2523534893989563, -0.2918112576007843, -1.5631908178329468]}, "b": {"shape": [4, 5, 6], "dtype": "float32", "sample_values": [-2.121854782104492, -0.6078220009803772, 1.2969945669174194, -0.022868061438202858, -0.9993022084236145, -0.5047749280929565, 0.8406200408935547, 0.546733558177948, -0.23893210291862488, -0.3668244183063507]}, "dims": 2}
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
    api_name = "torch.tensordot"
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
    params = {"api": "paddle.tensordot", "x": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.3649527430534363, -0.8392096757888794, -1.0448092222213745, -1.966356635093689, 2.0562071800231934, -1.1032084226608276, -0.22125361859798431, -0.2768132984638214, 0.3074066936969757, 0.8157371878623962]}, "y": {"shape": [4, 5, 6], "dtype": "float32", "sample_values": [1.7269638776779175, -0.3996361792087555, 0.2246847301721573, 0.9325908422470093, -1.418365716934204, -1.7608088254928589, -1.5256563425064087, 1.2625840902328491, -0.551858127117157, 2.558199167251587]}, "axes": 2}
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
    api_name = "paddle.tensordot"
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
    print(f"Reproducing Case 128: torch.tensordot vs paddle.tensordot")
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
