
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
    if 'input' in {"api": "torch.nn.functional.mse_loss", "input": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8718695044517517, -1.5463645458221436, -0.8335539102554321, 0.9938636422157288, -0.13691739737987518, 0.4233677089214325, -0.21309565007686615, 2.090324640274048, 1.4017020463943481, -1.34104585647583]}, "target": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8076554536819458, -0.6421261429786682, 1.4344584941864014, 0.29548409581184387, 1.0761094093322754, -1.228389024734497, 0.9728113412857056, -1.6094528436660767, -0.9559429287910461, 0.9625091552734375]}, "reduction": "sum"}:
        input_info = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8718695044517517, -1.5463645458221436, -0.8335539102554321, 0.9938636422157288, -0.13691739737987518, 0.4233677089214325, -0.21309565007686615, 2.090324640274048, 1.4017020463943481, -1.34104585647583]}, "target": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8076554536819458, -0.6421261429786682, 1.4344584941864014, 0.29548409581184387, 1.0761094093322754, -1.228389024734497, 0.9728113412857056, -1.6094528436660767, -0.9559429287910461, 0.9625091552734375]}, "reduction": "sum"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.mse_loss", "input": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8718695044517517, -1.5463645458221436, -0.8335539102554321, 0.9938636422157288, -0.13691739737987518, 0.4233677089214325, -0.21309565007686615, 2.090324640274048, 1.4017020463943481, -1.34104585647583]}, "target": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8076554536819458, -0.6421261429786682, 1.4344584941864014, 0.29548409581184387, 1.0761094093322754, -1.228389024734497, 0.9728113412857056, -1.6094528436660767, -0.9559429287910461, 0.9625091552734375]}, "reduction": "sum"}
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
    params = {"api": "paddle.nn.functional.mse_loss", "input": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [-0.8718695044517517, -1.5463645458221436, -0.8335539102554321, 0.9938636422157288, -0.13691739737987518, 0.4233677089214325, -0.21309565007686615, 2.090324640274048, 1.4017020463943481, -1.34104585647583]}, "label": {"shape": [5, 2, 3], "dtype": "float32", "sample_values": [0.7775251269340515, 0.20714163780212402, -0.15147550404071808, 1.321049690246582, -0.253641277551651, -0.8590426445007324, -0.5917459726333618, -0.15599986910820007, -0.5567809343338013, -2.0115866661071777]}, "reduction": "sum"}
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
    print(f"Reproducing Case 74: torch.nn.functional.mse_loss vs paddle.nn.functional.mse_loss")
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
