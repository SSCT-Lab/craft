
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
    if 'input' in {"api": "torch.nn.TransformerEncoderLayer", "input": {"shape": [4, 128, 512], "dtype": "float32", "sample_values": [0.5289034843444824, 0.9619351625442505, 0.9037748575210571, 3.13706374168396, -0.4895611107349396, 0.2982839345932007, -1.6812037229537964, -1.770400047302246, -0.002571643330156803, 0.09741935133934021]}, "d_model": 512, "nhead": 8}:
        input_info = {"api": "torch.nn.TransformerEncoderLayer", "input": {"shape": [4, 128, 512], "dtype": "float32", "sample_values": [0.5289034843444824, 0.9619351625442505, 0.9037748575210571, 3.13706374168396, -0.4895611107349396, 0.2982839345932007, -1.6812037229537964, -1.770400047302246, -0.002571643330156803, 0.09741935133934021]}, "d_model": 512, "nhead": 8}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.TransformerEncoderLayer", "input": {"shape": [4, 128, 512], "dtype": "float32", "sample_values": [0.5289034843444824, 0.9619351625442505, 0.9037748575210571, 3.13706374168396, -0.4895611107349396, 0.2982839345932007, -1.6812037229537964, -1.770400047302246, -0.002571643330156803, 0.09741935133934021]}, "d_model": 512, "nhead": 8}
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
    api_name = "torch.nn.TransformerEncoderLayer"
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
    params = {"api": "mindspore.nn.TransformerEncoderLayer", "input": {"shape": [4, 128, 512], "dtype": "float32", "sample_values": [0.5289034843444824, 0.9619351625442505, 0.9037748575210571, 3.13706374168396, -0.4895611107349396, 0.2982839345932007, -1.6812037229537964, -1.770400047302246, -0.002571643330156803, 0.09741935133934021]}, "d_model": 512, "nhead": 8}
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
    api_name = "mindspore.nn.TransformerEncoderLayer"
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
    print(f"Reproducing Case 44: torch.nn.TransformerEncoderLayer vs mindspore.nn.TransformerEncoderLayer")
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
