
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
    if 'input' in {"api": "torch.tensordot", "a": {"shape": [5, 6, 7], "dtype": "float32", "sample_values": [0.24938368797302246, 1.5774532556533813, -0.09529553353786469, 0.2790215313434601, 0.6078965067863464, 0.18660911917686462, -0.4464336037635803, 0.19408999383449554, 1.073631763458252, -1.026515245437622]}, "b": {"shape": [6, 5, 4], "dtype": "float32", "sample_values": [0.6387302279472351, -1.1430048942565918, 1.6334315538406372, -1.1463453769683838, 0.30263546109199524, -0.7542758584022522, -0.06413834542036057, 0.328762412071228, 0.32135722041130066, 0.4219207465648651]}, "dims": [[1, 0], [0, 1]]}:
        input_info = {"api": "torch.tensordot", "a": {"shape": [5, 6, 7], "dtype": "float32", "sample_values": [0.24938368797302246, 1.5774532556533813, -0.09529553353786469, 0.2790215313434601, 0.6078965067863464, 0.18660911917686462, -0.4464336037635803, 0.19408999383449554, 1.073631763458252, -1.026515245437622]}, "b": {"shape": [6, 5, 4], "dtype": "float32", "sample_values": [0.6387302279472351, -1.1430048942565918, 1.6334315538406372, -1.1463453769683838, 0.30263546109199524, -0.7542758584022522, -0.06413834542036057, 0.328762412071228, 0.32135722041130066, 0.4219207465648651]}, "dims": [[1, 0], [0, 1]]}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.tensordot", "a": {"shape": [5, 6, 7], "dtype": "float32", "sample_values": [0.24938368797302246, 1.5774532556533813, -0.09529553353786469, 0.2790215313434601, 0.6078965067863464, 0.18660911917686462, -0.4464336037635803, 0.19408999383449554, 1.073631763458252, -1.026515245437622]}, "b": {"shape": [6, 5, 4], "dtype": "float32", "sample_values": [0.6387302279472351, -1.1430048942565918, 1.6334315538406372, -1.1463453769683838, 0.30263546109199524, -0.7542758584022522, -0.06413834542036057, 0.328762412071228, 0.32135722041130066, 0.4219207465648651]}, "dims": [[1, 0], [0, 1]]}
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
    params = {"api": "paddle.tensordot", "x": {"shape": [5, 6, 7], "dtype": "float32", "sample_values": [0.6327818632125854, 2.270692825317383, 0.18186625838279724, 0.24822059273719788, -0.45936089754104614, -0.8498443961143494, 0.830335795879364, -0.8560838103294373, 0.07156623899936676, -0.4776574373245239]}, "y": {"shape": [6, 5, 4], "dtype": "float32", "sample_values": [0.271578848361969, -1.276748538017273, -1.0810565948486328, 1.0531527996063232, -0.039555154740810394, 0.6815006732940674, 0.0283183753490448, 0.02975613996386528, 0.9382838010787964, -0.5160447359085083]}, "axes": [[1, 0], [0, 1]]}
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
    print(f"Reproducing Case 127: torch.tensordot vs paddle.tensordot")
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
