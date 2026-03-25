
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
    if 'input' in {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [4, 6], "dtype": "float32", "sample_values": [0.8864266872406006, 0.0763310045003891, -0.5330225229263306, -0.2905218005180359, 0.21502722799777985, 0.1309661567211151, -0.6059187054634094, -0.18268273770809174, -1.0284048318862915, 0.18540675938129425]}, "target": {"shape": [4, 6], "dtype": "float32", "sample_values": [-1.8769018650054932, 0.23744559288024902, 0.5583072900772095, 0.03253768011927605, 1.0693914890289307, -1.8293265104293823, 0.5415839552879333, -1.7553130388259888, 0.6296471357345581, 0.6611371636390686]}, "weight": null, "reduction": "mean", "pos_weight": null}:
        input_info = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [4, 6], "dtype": "float32", "sample_values": [0.8864266872406006, 0.0763310045003891, -0.5330225229263306, -0.2905218005180359, 0.21502722799777985, 0.1309661567211151, -0.6059187054634094, -0.18268273770809174, -1.0284048318862915, 0.18540675938129425]}, "target": {"shape": [4, 6], "dtype": "float32", "sample_values": [-1.8769018650054932, 0.23744559288024902, 0.5583072900772095, 0.03253768011927605, 1.0693914890289307, -1.8293265104293823, 0.5415839552879333, -1.7553130388259888, 0.6296471357345581, 0.6611371636390686]}, "weight": null, "reduction": "mean", "pos_weight": null}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [4, 6], "dtype": "float32", "sample_values": [0.8864266872406006, 0.0763310045003891, -0.5330225229263306, -0.2905218005180359, 0.21502722799777985, 0.1309661567211151, -0.6059187054634094, -0.18268273770809174, -1.0284048318862915, 0.18540675938129425]}, "target": {"shape": [4, 6], "dtype": "float32", "sample_values": [-1.8769018650054932, 0.23744559288024902, 0.5583072900772095, 0.03253768011927605, 1.0693914890289307, -1.8293265104293823, 0.5415839552879333, -1.7553130388259888, 0.6296471357345581, 0.6611371636390686]}, "weight": null, "reduction": "mean", "pos_weight": null}
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
    params = {"api": "paddle.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [4, 6], "dtype": "float32", "sample_values": [0.8864266872406006, 0.0763310045003891, -0.5330225229263306, -0.2905218005180359, 0.21502722799777985, 0.1309661567211151, -0.6059187054634094, -0.18268273770809174, -1.0284048318862915, 0.18540675938129425]}, "label": {"shape": [4, 6], "dtype": "float32", "sample_values": [0.45131587982177734, 1.4256082773208618, 0.16892489790916443, 0.17793932557106018, 2.1208536624908447, 1.4672471284866333, -0.021635185927152634, -1.1954693794250488, -1.036331295967102, -0.39326462149620056]}, "weight": null, "reduction": "mean", "pos_weight": null}
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
    print(f"Reproducing Case 53: torch.nn.functional.binary_cross_entropy_with_logits vs paddle.nn.functional.binary_cross_entropy_with_logits")
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
