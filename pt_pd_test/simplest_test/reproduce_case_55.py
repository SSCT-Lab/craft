
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
    if 'input' in {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.8808000599889066, -0.700598876542502, 2.258264250632768, -0.749646174023248, 0.43900424269118826, 0.6762696700934011, 1.8175223305241912, -0.016082459300647146, -1.1719405867644503, 0.34819463936440126]}, "target": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [0.9170677779610136, 1.1076214236258983, 1.488946637442427, 0.6827877665150215, -0.37292094230724987, 2.1477513782466033, -0.5997346296837879, -1.2380266846187467, 1.2414684108130947, -1.7637976011737935]}, "weight": null, "reduction": "mean", "pos_weight": null}:
        input_info = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.8808000599889066, -0.700598876542502, 2.258264250632768, -0.749646174023248, 0.43900424269118826, 0.6762696700934011, 1.8175223305241912, -0.016082459300647146, -1.1719405867644503, 0.34819463936440126]}, "target": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [0.9170677779610136, 1.1076214236258983, 1.488946637442427, 0.6827877665150215, -0.37292094230724987, 2.1477513782466033, -0.5997346296837879, -1.2380266846187467, 1.2414684108130947, -1.7637976011737935]}, "weight": null, "reduction": "mean", "pos_weight": null}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.8808000599889066, -0.700598876542502, 2.258264250632768, -0.749646174023248, 0.43900424269118826, 0.6762696700934011, 1.8175223305241912, -0.016082459300647146, -1.1719405867644503, 0.34819463936440126]}, "target": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [0.9170677779610136, 1.1076214236258983, 1.488946637442427, 0.6827877665150215, -0.37292094230724987, 2.1477513782466033, -0.5997346296837879, -1.2380266846187467, 1.2414684108130947, -1.7637976011737935]}, "weight": null, "reduction": "mean", "pos_weight": null}
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
    params = {"api": "paddle.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.8808000599889066, -0.700598876542502, 2.258264250632768, -0.749646174023248, 0.43900424269118826, 0.6762696700934011, 1.8175223305241912, -0.016082459300647146, -1.1719405867644503, 0.34819463936440126]}, "label": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [1.5191530626506138, -0.7802642228094507, -2.8092261052821903, 0.16070129924297857, 1.2297589598162457, 1.40413126290285, -0.608788767482775, -0.5280642862475076, 0.2690495507119143, 0.9536207048545463]}, "weight": null, "reduction": "mean", "pos_weight": null}
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
    print(f"Reproducing Case 55: torch.nn.functional.binary_cross_entropy_with_logits vs paddle.nn.functional.binary_cross_entropy_with_logits")
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
