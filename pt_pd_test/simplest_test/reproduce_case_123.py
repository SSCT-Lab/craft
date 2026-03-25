
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
    if 'input' in {"api": "torch.split", "tensor": {"shape": [128, 232, 8, 8], "dtype": "float32", "sample_values": [1.0175172090530396, 0.7440342307090759, -1.367200255393982, -0.9632503390312195, -1.7980010509490967, 0.30057981610298157, -1.3913671970367432, -0.03689553961157799, -1.705810546875, 0.14245550334453583]}, "split_size_or_sections": 116, "dim": 1}:
        input_info = {"api": "torch.split", "tensor": {"shape": [128, 232, 8, 8], "dtype": "float32", "sample_values": [1.0175172090530396, 0.7440342307090759, -1.367200255393982, -0.9632503390312195, -1.7980010509490967, 0.30057981610298157, -1.3913671970367432, -0.03689553961157799, -1.705810546875, 0.14245550334453583]}, "split_size_or_sections": 116, "dim": 1}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.split", "tensor": {"shape": [128, 232, 8, 8], "dtype": "float32", "sample_values": [1.0175172090530396, 0.7440342307090759, -1.367200255393982, -0.9632503390312195, -1.7980010509490967, 0.30057981610298157, -1.3913671970367432, -0.03689553961157799, -1.705810546875, 0.14245550334453583]}, "split_size_or_sections": 116, "dim": 1}
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
    api_name = "torch.split"
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
    params = {"api": "paddle.split", "x": {"shape": [128, 232, 8, 8], "dtype": "float32", "sample_values": [-0.4554086923599243, 1.0777508020401, 0.033944711089134216, 0.5545748472213745, 0.39248126745224, -0.31188833713531494, 0.1853172928094864, 2.0548110008239746, 1.894527792930603, 0.45331472158432007]}, "num_or_sections": 2, "axis": 1}
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
    api_name = "paddle.split"
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
    print(f"Reproducing Case 123: torch.split vs paddle.split")
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
