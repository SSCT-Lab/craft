
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
    if 'input' in {"input": {"shape": [8, 3, 131072], "dtype": "float32", "sample_values": [-0.10616832971572876, 1.6289526224136353, 0.15156473219394684, 0.3956683576107025, -0.4576511085033417, 0.07559361308813095, -1.514945387840271, 2.4307563304901123, 0.5372056365013123, 1.8281142711639404]}, "mat2": {"shape": [8, 131072, 1], "dtype": "float32", "sample_values": [-0.10887037962675095, 0.14284007251262665, 0.4000428318977356, 1.680885910987854, 0.4826236367225647, -0.521239161491394, -1.6741193532943726, 1.6334103345870972, -0.1946883350610733, 1.4047539234161377]}, "api": "torch.bmm"}:
        input_info = {"input": {"shape": [8, 3, 131072], "dtype": "float32", "sample_values": [-0.10616832971572876, 1.6289526224136353, 0.15156473219394684, 0.3956683576107025, -0.4576511085033417, 0.07559361308813095, -1.514945387840271, 2.4307563304901123, 0.5372056365013123, 1.8281142711639404]}, "mat2": {"shape": [8, 131072, 1], "dtype": "float32", "sample_values": [-0.10887037962675095, 0.14284007251262665, 0.4000428318977356, 1.680885910987854, 0.4826236367225647, -0.521239161491394, -1.6741193532943726, 1.6334103345870972, -0.1946883350610733, 1.4047539234161377]}, "api": "torch.bmm"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"input": {"shape": [8, 3, 131072], "dtype": "float32", "sample_values": [-0.10616832971572876, 1.6289526224136353, 0.15156473219394684, 0.3956683576107025, -0.4576511085033417, 0.07559361308813095, -1.514945387840271, 2.4307563304901123, 0.5372056365013123, 1.8281142711639404]}, "mat2": {"shape": [8, 131072, 1], "dtype": "float32", "sample_values": [-0.10887037962675095, 0.14284007251262665, 0.4000428318977356, 1.680885910987854, 0.4826236367225647, -0.521239161491394, -1.6741193532943726, 1.6334103345870972, -0.1946883350610733, 1.4047539234161377]}, "api": "torch.bmm"}
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
    api_name = "torch.bmm"
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
    params = {"input": {"shape": [8, 3, 131072], "dtype": "float32", "sample_values": [-0.10616832971572876, 1.6289526224136353, 0.15156473219394684, 0.3956683576107025, -0.4576511085033417, 0.07559361308813095, -1.514945387840271, 2.4307563304901123, 0.5372056365013123, 1.8281142711639404]}, "mat2": {"shape": [8, 131072, 1], "dtype": "float32", "sample_values": [-0.10887037962675095, 0.14284007251262665, 0.4000428318977356, 1.680885910987854, 0.4826236367225647, -0.521239161491394, -1.6741193532943726, 1.6334103345870972, -0.1946883350610733, 1.4047539234161377]}, "api": "torch.bmm"}
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
    api_name = "torch.bmm"
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
    print(f"Reproducing Case 2: torch.bmm vs torch.bmm")
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
