
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
    if 'input' in {"api": "torch.lerp", "input": {"shape": [3], "dtype": "float64", "sample_values": [-0.518637997484576, 1.7183639957638757, -0.18919018127507997]}, "end": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8014839218310068, -0.3984452749893668, -0.4466430468146064, 0.6122698101053926, -1.5594357791262254, -1.0777207457916016, 0.3431585541974166, 0.6229440304133081, 0.641541644936012, -0.787036278933299]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.006381540963314977, -1.8482585300784151, -2.044629003801268, -0.5789842199399048, 1.056101251669989, 0.7882916566360036, 0.9140771642240336, -0.9870461789363153, 1.6158724901223136, -0.9648933001863843]}}:
        input_info = {"api": "torch.lerp", "input": {"shape": [3], "dtype": "float64", "sample_values": [-0.518637997484576, 1.7183639957638757, -0.18919018127507997]}, "end": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8014839218310068, -0.3984452749893668, -0.4466430468146064, 0.6122698101053926, -1.5594357791262254, -1.0777207457916016, 0.3431585541974166, 0.6229440304133081, 0.641541644936012, -0.787036278933299]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.006381540963314977, -1.8482585300784151, -2.044629003801268, -0.5789842199399048, 1.056101251669989, 0.7882916566360036, 0.9140771642240336, -0.9870461789363153, 1.6158724901223136, -0.9648933001863843]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.lerp", "input": {"shape": [3], "dtype": "float64", "sample_values": [-0.518637997484576, 1.7183639957638757, -0.18919018127507997]}, "end": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8014839218310068, -0.3984452749893668, -0.4466430468146064, 0.6122698101053926, -1.5594357791262254, -1.0777207457916016, 0.3431585541974166, 0.6229440304133081, 0.641541644936012, -0.787036278933299]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.006381540963314977, -1.8482585300784151, -2.044629003801268, -0.5789842199399048, 1.056101251669989, 0.7882916566360036, 0.9140771642240336, -0.9870461789363153, 1.6158724901223136, -0.9648933001863843]}}
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
    params = {"api": "paddle.lerp", "x": {"shape": [3], "dtype": "float64", "sample_values": [0.2290948316463794, -0.3231676657024847, 0.26881235641215423]}, "y": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.08633785824672688, -2.215822425020668, -0.19543484652276386, 0.836484137964941, -1.0677225770385452, -0.7507203344928128, 0.7927110177336647, -1.4766572235180533, 1.0578010075419184, -0.532143890173102]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.006381540963314977, -1.8482585300784151, -2.044629003801268, -0.5789842199399048, 1.056101251669989, 0.7882916566360036, 0.9140771642240336, -0.9870461789363153, 1.6158724901223136, -0.9648933001863843]}}
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
    print(f"Reproducing Case 15: torch.lerp vs paddle.lerp")
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
