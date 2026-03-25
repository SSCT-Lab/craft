
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
    if 'input' in {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.09267059802811742, 1.1524050953322231, 0.5324437040349298, 0.6241951943706815, -0.3071987532902949, 0.1022974784940986, -0.7570799570602947, 0.622016276379285, 0.7248603734649937, 1.141497900834857]}, "target": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [1.7987042982783654, 0.48556347219281865, 2.369469626032104, -0.9114146675337352, 0.4182066342696993, -1.3909018938151176, -1.0728509932129533, -0.5105054223195851, -0.08930100576850442, -0.0957318577076777]}, "weight": null, "reduction": "sum", "pos_weight": null}:
        input_info = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.09267059802811742, 1.1524050953322231, 0.5324437040349298, 0.6241951943706815, -0.3071987532902949, 0.1022974784940986, -0.7570799570602947, 0.622016276379285, 0.7248603734649937, 1.141497900834857]}, "target": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [1.7987042982783654, 0.48556347219281865, 2.369469626032104, -0.9114146675337352, 0.4182066342696993, -1.3909018938151176, -1.0728509932129533, -0.5105054223195851, -0.08930100576850442, -0.0957318577076777]}, "weight": null, "reduction": "sum", "pos_weight": null}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.09267059802811742, 1.1524050953322231, 0.5324437040349298, 0.6241951943706815, -0.3071987532902949, 0.1022974784940986, -0.7570799570602947, 0.622016276379285, 0.7248603734649937, 1.141497900834857]}, "target": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [1.7987042982783654, 0.48556347219281865, 2.369469626032104, -0.9114146675337352, 0.4182066342696993, -1.3909018938151176, -1.0728509932129533, -0.5105054223195851, -0.08930100576850442, -0.0957318577076777]}, "weight": null, "reduction": "sum", "pos_weight": null}
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
    params = {"api": "paddle.nn.functional.binary_cross_entropy_with_logits", "input": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-0.09267059802811742, 1.1524050953322231, 0.5324437040349298, 0.6241951943706815, -0.3071987532902949, 0.1022974784940986, -0.7570799570602947, 0.622016276379285, 0.7248603734649937, 1.141497900834857]}, "label": {"shape": [2, 3, 4], "dtype": "float64", "sample_values": [-1.2938529364626659, 1.7145943246971909, 0.23708730790403842, -0.2366737374684245, -0.7556065781796925, -0.6774255397354337, 1.2332016349071966, -1.3328988757128142, 1.7410556437268034, -0.29973671028391347]}, "weight": null, "reduction": "sum", "pos_weight": null}
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
    print(f"Reproducing Case 59: torch.nn.functional.binary_cross_entropy_with_logits vs paddle.nn.functional.binary_cross_entropy_with_logits")
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
