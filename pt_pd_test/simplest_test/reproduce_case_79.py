
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
    if 'input' in {"api": "torch.nn.functional.poisson_nll_loss", "input": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.36139559745788574, 1.538036584854126, -0.03582603856921196, 1.5646436214447021, -2.6197450160980225, 0.8219025135040283, 0.08704707026481628, -0.2990073561668396, 0.0917607769370079, -1.9875688552856445]}, "target": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.32408398389816284, -0.38508227467536926, -0.6769220232963562, 0.6116762757301331, 1.0309995412826538, 0.9312801361083984, -0.8392175436019897, -0.3092123866081238, 0.3312634229660034, 0.9755451083183289]}, "log_input": true, "full": false, "eps": 1e-08, "reduction": "mean"}:
        input_info = {"api": "torch.nn.functional.poisson_nll_loss", "input": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.36139559745788574, 1.538036584854126, -0.03582603856921196, 1.5646436214447021, -2.6197450160980225, 0.8219025135040283, 0.08704707026481628, -0.2990073561668396, 0.0917607769370079, -1.9875688552856445]}, "target": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.32408398389816284, -0.38508227467536926, -0.6769220232963562, 0.6116762757301331, 1.0309995412826538, 0.9312801361083984, -0.8392175436019897, -0.3092123866081238, 0.3312634229660034, 0.9755451083183289]}, "log_input": true, "full": false, "eps": 1e-08, "reduction": "mean"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.poisson_nll_loss", "input": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.36139559745788574, 1.538036584854126, -0.03582603856921196, 1.5646436214447021, -2.6197450160980225, 0.8219025135040283, 0.08704707026481628, -0.2990073561668396, 0.0917607769370079, -1.9875688552856445]}, "target": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.32408398389816284, -0.38508227467536926, -0.6769220232963562, 0.6116762757301331, 1.0309995412826538, 0.9312801361083984, -0.8392175436019897, -0.3092123866081238, 0.3312634229660034, 0.9755451083183289]}, "log_input": true, "full": false, "eps": 1e-08, "reduction": "mean"}
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
    api_name = "torch.nn.functional.poisson_nll_loss"
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
    params = {"api": "paddle.nn.functional.poisson_nll_loss", "input": {"shape": [5, 2], "dtype": "float32", "sample_values": [0.36139559745788574, 1.538036584854126, -0.03582603856921196, 1.5646436214447021, -2.6197450160980225, 0.8219025135040283, 0.08704707026481628, -0.2990073561668396, 0.0917607769370079, -1.9875688552856445]}, "label": {"shape": [5, 2], "dtype": "float32", "sample_values": [-0.4791742265224457, -0.18565897643566132, -1.106334924697876, -1.1962065696716309, 0.8125258088111877, 1.3562400341033936, -0.07201012223958969, 1.003532886505127, 0.3616360127925873, -0.6451197266578674]}, "log_input": true, "full": false, "reduction": "mean"}
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
    api_name = "paddle.nn.functional.poisson_nll_loss"
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
    print(f"Reproducing Case 79: torch.nn.functional.poisson_nll_loss vs paddle.nn.functional.poisson_nll_loss")
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
