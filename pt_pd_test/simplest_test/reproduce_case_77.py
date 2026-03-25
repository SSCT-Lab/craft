
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
    if 'input' in {"api": "torch.nn.functional.mse_loss", "input": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [0.20208074152469635, 1.0540552139282227, 0.11874878406524658, 1.0980745553970337, -0.06537523865699768, -0.4644322097301483, 1.8233758211135864, 0.08832000195980072, -0.4048605263233185, -0.5681345462799072]}, "target": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [-0.02461700141429901, -1.3852243423461914, -0.929452121257782, -0.2232465296983719, 0.8002648949623108, 0.2777111232280731, -1.2613375186920166, -1.5025449991226196, -0.014555256813764572, 0.1575155109167099]}, "reduction": "sum"}:
        input_info = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [0.20208074152469635, 1.0540552139282227, 0.11874878406524658, 1.0980745553970337, -0.06537523865699768, -0.4644322097301483, 1.8233758211135864, 0.08832000195980072, -0.4048605263233185, -0.5681345462799072]}, "target": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [-0.02461700141429901, -1.3852243423461914, -0.929452121257782, -0.2232465296983719, 0.8002648949623108, 0.2777111232280731, -1.2613375186920166, -1.5025449991226196, -0.014555256813764572, 0.1575155109167099]}, "reduction": "sum"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [0.20208074152469635, 1.0540552139282227, 0.11874878406524658, 1.0980745553970337, -0.06537523865699768, -0.4644322097301483, 1.8233758211135864, 0.08832000195980072, -0.4048605263233185, -0.5681345462799072]}, "target": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [-0.02461700141429901, -1.3852243423461914, -0.929452121257782, -0.2232465296983719, 0.8002648949623108, 0.2777111232280731, -1.2613375186920166, -1.5025449991226196, -0.014555256813764572, 0.1575155109167099]}, "reduction": "sum"}
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
    api_name = "torch.nn.functional.mse_loss"
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
    params = {"api": "paddle.nn.functional.mse_loss", "input": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [0.20208074152469635, 1.0540552139282227, 0.11874878406524658, 1.0980745553970337, -0.06537523865699768, -0.4644322097301483, 1.8233758211135864, 0.08832000195980072, -0.4048605263233185, -0.5681345462799072]}, "label": {"shape": [2, 3, 4], "dtype": "float32", "sample_values": [0.20517195761203766, -0.18416357040405273, 1.8471482992172241, 0.05448072776198387, -0.48335596919059753, 0.06004618480801582, 0.2066129744052887, -0.7538091540336609, 0.36744746565818787, 0.754625141620636]}, "reduction": "sum"}
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
    api_name = "paddle.nn.functional.mse_loss"
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
    print(f"Reproducing Case 77: torch.nn.functional.mse_loss vs paddle.nn.functional.mse_loss")
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
