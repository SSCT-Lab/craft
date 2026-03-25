
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
    if 'input' in {"api": "torch.tensordot", "a": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.24538812041282654, -0.753736138343811, -0.8895144462585449, -0.8158102631568909, -0.0771017074584961, 0.34115198254585266, 0.276690810918808, 0.8271832466125488, 0.013001891784369946, 1.4535341262817383]}, "b": {"shape": [4, 3, 2], "dtype": "float32", "sample_values": [-0.7153037190437317, 0.6795977354049683, -0.7303666472434998, 0.21645858883857727, 0.045571841299533844, -0.6516003608703613, 2.143944025039673, 0.6339190006256104, -2.0251426696777344, 0.18645431101322174]}, "dims": [[1, 0], [0, 1]]}:
        input_info = {"api": "torch.tensordot", "a": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.24538812041282654, -0.753736138343811, -0.8895144462585449, -0.8158102631568909, -0.0771017074584961, 0.34115198254585266, 0.276690810918808, 0.8271832466125488, 0.013001891784369946, 1.4535341262817383]}, "b": {"shape": [4, 3, 2], "dtype": "float32", "sample_values": [-0.7153037190437317, 0.6795977354049683, -0.7303666472434998, 0.21645858883857727, 0.045571841299533844, -0.6516003608703613, 2.143944025039673, 0.6339190006256104, -2.0251426696777344, 0.18645431101322174]}, "dims": [[1, 0], [0, 1]]}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.tensordot", "a": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.24538812041282654, -0.753736138343811, -0.8895144462585449, -0.8158102631568909, -0.0771017074584961, 0.34115198254585266, 0.276690810918808, 0.8271832466125488, 0.013001891784369946, 1.4535341262817383]}, "b": {"shape": [4, 3, 2], "dtype": "float32", "sample_values": [-0.7153037190437317, 0.6795977354049683, -0.7303666472434998, 0.21645858883857727, 0.045571841299533844, -0.6516003608703613, 2.143944025039673, 0.6339190006256104, -2.0251426696777344, 0.18645431101322174]}, "dims": [[1, 0], [0, 1]]}
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
    params = {"api": "paddle.tensordot", "x": {"shape": [3, 4, 5], "dtype": "float32", "sample_values": [-0.8084936141967773, -0.501757025718689, 0.9154021143913269, 0.3287511169910431, -0.5297601819038391, 0.513267457485199, 0.09707754850387573, 0.9686449766159058, -0.7020530700683594, -0.3276621401309967]}, "y": {"shape": [4, 3, 2], "dtype": "float32", "sample_values": [0.2598828077316284, 0.7818228602409363, -1.2369507551193237, -1.320456624031067, 0.5219415426254272, 0.2969846725463867, 0.2504928410053253, 0.34644821286201477, -0.6800247430801392, 0.23225370049476624]}, "axes": [[1, 0], [0, 1]]}
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
    print(f"Reproducing Case 126: torch.tensordot vs paddle.tensordot")
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
