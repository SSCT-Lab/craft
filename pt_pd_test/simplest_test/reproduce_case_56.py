
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
    if 'input' in {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.7605412602424622, 1.9379162788391113, -1.433060884475708, 1.599412202835083, 1.0135908126831055, -0.4706394374370575, 0.9099858403205872, -1.0570529699325562, 0.7348441481590271, 0.25073710083961487]}, "target": {"shape": [10, 64], "dtype": "float32", "sample_values": [-0.5750826001167297, 0.8752562999725342, 0.8677546977996826, -1.79293954372406, -0.6063240170478821, -2.458552837371826, 0.04267207905650139, 0.12033095210790634, -0.03687884658575058, 0.7451232075691223]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [-0.399265855550766, 1.4531652927398682, 0.4394013583660126, 0.5383904576301575, 0.14643734693527222, -1.1306811571121216, -1.2638574838638306, -0.21020333468914032, -0.006295291241258383, 0.3454836308956146]}}:
        input_info = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.7605412602424622, 1.9379162788391113, -1.433060884475708, 1.599412202835083, 1.0135908126831055, -0.4706394374370575, 0.9099858403205872, -1.0570529699325562, 0.7348441481590271, 0.25073710083961487]}, "target": {"shape": [10, 64], "dtype": "float32", "sample_values": [-0.5750826001167297, 0.8752562999725342, 0.8677546977996826, -1.79293954372406, -0.6063240170478821, -2.458552837371826, 0.04267207905650139, 0.12033095210790634, -0.03687884658575058, 0.7451232075691223]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [-0.399265855550766, 1.4531652927398682, 0.4394013583660126, 0.5383904576301575, 0.14643734693527222, -1.1306811571121216, -1.2638574838638306, -0.21020333468914032, -0.006295291241258383, 0.3454836308956146]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.7605412602424622, 1.9379162788391113, -1.433060884475708, 1.599412202835083, 1.0135908126831055, -0.4706394374370575, 0.9099858403205872, -1.0570529699325562, 0.7348441481590271, 0.25073710083961487]}, "target": {"shape": [10, 64], "dtype": "float32", "sample_values": [-0.5750826001167297, 0.8752562999725342, 0.8677546977996826, -1.79293954372406, -0.6063240170478821, -2.458552837371826, 0.04267207905650139, 0.12033095210790634, -0.03687884658575058, 0.7451232075691223]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [-0.399265855550766, 1.4531652927398682, 0.4394013583660126, 0.5383904576301575, 0.14643734693527222, -1.1306811571121216, -1.2638574838638306, -0.21020333468914032, -0.006295291241258383, 0.3454836308956146]}}
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
    params = {"api": "paddle.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.7605412602424622, 1.9379162788391113, -1.433060884475708, 1.599412202835083, 1.0135908126831055, -0.4706394374370575, 0.9099858403205872, -1.0570529699325562, 0.7348441481590271, 0.25073710083961487]}, "label": {"shape": [10, 64], "dtype": "float32", "sample_values": [0.9120832085609436, -0.9792441725730896, -0.45364025235176086, 0.31886056065559387, 1.835410475730896, 0.8231834173202515, -0.5716770887374878, 0.8158895969390869, 1.8714860677719116, -0.2896633744239807]}, "weight": null, "reduction": "mean", "pos_weight": {"shape": [64], "dtype": "float32", "sample_values": [-0.399265855550766, 1.4531652927398682, 0.4394013583660126, 0.5383904576301575, 0.14643734693527222, -1.1306811571121216, -1.2638574838638306, -0.21020333468914032, -0.006295291241258383, 0.3454836308956146]}}
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
    print(f"Reproducing Case 56: torch.nn.functional.binary_cross_entropy_with_logits vs paddle.nn.functional.binary_cross_entropy_with_logits")
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
