
import torch
import mindspore as ms
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
    if 'input' in {"input": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [-2.3755927085876465, -0.8214182257652283, -0.486991822719574, 1.4049829244613647, 1.5832626819610596, 0.9192337393760681, -0.8661215901374817, -0.8675243258476257, -1.178585171699524, -0.6409078240394592]}, "other": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [0.9517129063606262, 0.25526756048202515, -1.2263190746307373, -0.5823861360549927, 0.8957569003105164, 0.13603328168392181, -2.283508777618408, -0.6570327877998352, 2.2601170539855957, 0.6589171886444092]}, "api": "torch.matmul"}:
        input_info = {"input": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [-2.3755927085876465, -0.8214182257652283, -0.486991822719574, 1.4049829244613647, 1.5832626819610596, 0.9192337393760681, -0.8661215901374817, -0.8675243258476257, -1.178585171699524, -0.6409078240394592]}, "other": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [0.9517129063606262, 0.25526756048202515, -1.2263190746307373, -0.5823861360549927, 0.8957569003105164, 0.13603328168392181, -2.283508777618408, -0.6570327877998352, 2.2601170539855957, 0.6589171886444092]}, "api": "torch.matmul"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"input": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [-2.3755927085876465, -0.8214182257652283, -0.486991822719574, 1.4049829244613647, 1.5832626819610596, 0.9192337393760681, -0.8661215901374817, -0.8675243258476257, -1.178585171699524, -0.6409078240394592]}, "other": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [0.9517129063606262, 0.25526756048202515, -1.2263190746307373, -0.5823861360549927, 0.8957569003105164, 0.13603328168392181, -2.283508777618408, -0.6570327877998352, 2.2601170539855957, 0.6589171886444092]}, "api": "torch.matmul"}
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
    api_name = "torch.matmul"
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

def test_mindspore():
    inputs = {}
    params = {"input": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [-2.3755927085876465, -0.8214182257652283, -0.486991822719574, 1.4049829244613647, 1.5832626819610596, 0.9192337393760681, -0.8661215901374817, -0.8675243258476257, -1.178585171699524, -0.6409078240394592]}, "other": {"shape": [2485, 2485], "dtype": "float32", "sample_values": [0.9517129063606262, 0.25526756048202515, -1.2263190746307373, -0.5823861360549927, 0.8957569003105164, 0.13603328168392181, -2.283508777618408, -0.6570327877998352, 2.2601170539855957, 0.6589171886444092]}, "api": "torch.matmul"}
    for k, v in params.items():
        if k == 'api':
            continue
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = ms.Tensor(np_data, dtype=getattr(ms, v['dtype']))
        else:
            inputs[k] = v
    api_name = "torch.matmul"
    try:
        parts = api_name.split('.')
        module = eval(parts[0])
        for part in parts[1:-1]:
            module = getattr(module, part)
        func = getattr(module, parts[-1])
        result = func(**inputs)
        print("MindSpore result shape:", result.shape)
        return result.asnumpy()
    except Exception as e:
        print(f"MindSpore error: {e}")
        try:
            print("Retrying with positional args...")
            result = func(*inputs.values())
            print("MindSpore result shape:", result.shape)
            return result.asnumpy()
        except Exception as e2:
            print(f"MindSpore retry error: {e2}")
            return None

if __name__ == "__main__":
    print(f"Reproducing Case 6: torch.matmul vs torch.matmul")
    torch_res = test_torch()
    ms_res = test_mindspore()
    if torch_res is not None and ms_res is not None:
        try:
            diff = np.abs(torch_res - ms_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
