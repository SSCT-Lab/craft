
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
    if 'input' in {"api": "torch.nn.functional.margin_ranking_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [2.3002185821533203, -0.4676631987094879, 0.5558387041091919, 0.7420055866241455, 1.1401318311691284, 1.656137228012085]}, "input2": {"shape": [2, 3], "dtype": "float32", "sample_values": [-0.1274595558643341, 0.6196926236152649, 0.4031369090080261, 1.3202751874923706, -0.3981057107448578, -0.17421242594718933]}, "target": {"shape": [2, 3], "dtype": "float32", "sample_values": [0.9225742816925049, 0.5982088446617126, -1.0800156593322754, 0.3885430693626404, 0.4666060507297516, -0.484017550945282]}}:
        input_info = {"api": "torch.nn.functional.margin_ranking_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [2.3002185821533203, -0.4676631987094879, 0.5558387041091919, 0.7420055866241455, 1.1401318311691284, 1.656137228012085]}, "input2": {"shape": [2, 3], "dtype": "float32", "sample_values": [-0.1274595558643341, 0.6196926236152649, 0.4031369090080261, 1.3202751874923706, -0.3981057107448578, -0.17421242594718933]}, "target": {"shape": [2, 3], "dtype": "float32", "sample_values": [0.9225742816925049, 0.5982088446617126, -1.0800156593322754, 0.3885430693626404, 0.4666060507297516, -0.484017550945282]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.margin_ranking_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [2.3002185821533203, -0.4676631987094879, 0.5558387041091919, 0.7420055866241455, 1.1401318311691284, 1.656137228012085]}, "input2": {"shape": [2, 3], "dtype": "float32", "sample_values": [-0.1274595558643341, 0.6196926236152649, 0.4031369090080261, 1.3202751874923706, -0.3981057107448578, -0.17421242594718933]}, "target": {"shape": [2, 3], "dtype": "float32", "sample_values": [0.9225742816925049, 0.5982088446617126, -1.0800156593322754, 0.3885430693626404, 0.4666060507297516, -0.484017550945282]}}
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
    api_name = "torch.nn.functional.margin_ranking_loss"
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
    params = {"api": "paddle.nn.functional.margin_ranking_loss", "input": {"shape": [2, 3], "dtype": "float32", "sample_values": [2.3002185821533203, -0.4676631987094879, 0.5558387041091919, 0.7420055866241455, 1.1401318311691284, 1.656137228012085]}, "other": {"shape": [2, 3], "dtype": "float32", "sample_values": [0.5460609197616577, -1.751814603805542, -0.8083733320236206, -0.6498544216156006, -1.1042029857635498, -1.261561393737793]}, "label": {"shape": [2, 3], "dtype": "float32", "sample_values": [1.1401441097259521, -0.5246263146400452, -0.4564840793609619, -0.20302967727184296, 0.46020761132240295, 1.4377583265304565]}}
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
    api_name = "paddle.nn.functional.margin_ranking_loss"
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
    print(f"Reproducing Case 85: torch.nn.functional.margin_ranking_loss vs paddle.nn.functional.margin_ranking_loss")
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
