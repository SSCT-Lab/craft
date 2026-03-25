
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
    if 'input' in {"api": "torch.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.7571768211020226, 0.41709532442324604, -0.9846336870179554, 0.5711412420380737, -0.7234866224198, 1.2666612531984123, -0.9710620291077331, -0.11315849445608397, -0.5552730880131715, -0.9875229624269947]}, "end": {"shape": [3], "dtype": "float64", "sample_values": [0.7321512260547355, 1.0073043574732625, 0.023385214658507663]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8644267053357196, 1.5895230361038624, -0.7504101277546562, -2.162032299927439, 0.4583890099570684, -1.1047746736415167, 0.9886087934991747, -0.38113769667341885, 0.6037132405441962, 1.401500346552158]}}:
        input_info = {"api": "torch.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.7571768211020226, 0.41709532442324604, -0.9846336870179554, 0.5711412420380737, -0.7234866224198, 1.2666612531984123, -0.9710620291077331, -0.11315849445608397, -0.5552730880131715, -0.9875229624269947]}, "end": {"shape": [3], "dtype": "float64", "sample_values": [0.7321512260547355, 1.0073043574732625, 0.023385214658507663]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8644267053357196, 1.5895230361038624, -0.7504101277546562, -2.162032299927439, 0.4583890099570684, -1.1047746736415167, 0.9886087934991747, -0.38113769667341885, 0.6037132405441962, 1.401500346552158]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.7571768211020226, 0.41709532442324604, -0.9846336870179554, 0.5711412420380737, -0.7234866224198, 1.2666612531984123, -0.9710620291077331, -0.11315849445608397, -0.5552730880131715, -0.9875229624269947]}, "end": {"shape": [3], "dtype": "float64", "sample_values": [0.7321512260547355, 1.0073043574732625, 0.023385214658507663]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8644267053357196, 1.5895230361038624, -0.7504101277546562, -2.162032299927439, 0.4583890099570684, -1.1047746736415167, 0.9886087934991747, -0.38113769667341885, 0.6037132405441962, 1.401500346552158]}}
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
    params = {"api": "paddle.lerp", "input": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [0.7571768211020226, 0.41709532442324604, -0.9846336870179554, 0.5711412420380737, -0.7234866224198, 1.2666612531984123, -0.9710620291077331, -0.11315849445608397, -0.5552730880131715, -0.9875229624269947]}, "y": {"shape": [3], "dtype": "float64", "sample_values": [0.23782954075833587, 0.47962531955157295, -0.4323011858095411]}, "weight": {"shape": [3, 3, 3], "dtype": "float64", "sample_values": [-0.8644267053357196, 1.5895230361038624, -0.7504101277546562, -2.162032299927439, 0.4583890099570684, -1.1047746736415167, 0.9886087934991747, -0.38113769667341885, 0.6037132405441962, 1.401500346552158]}}
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
    print(f"Reproducing Case 18: torch.lerp vs paddle.lerp")
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
