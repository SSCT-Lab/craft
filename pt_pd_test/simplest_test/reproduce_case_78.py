
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
    if 'input' in {"api": "torch.nn.functional.pairwise_distance", "x1": {"shape": [100, 128], "dtype": "float32", "sample_values": [0.04749922454357147, 0.23055438697338104, -0.16772545874118805, 0.42594221234321594, 1.6532641649246216, 2.1893155574798584, -0.45239782333374023, -0.6336861252784729, -1.9172505140304565, 0.2192724049091339]}, "x2": {"shape": [100, 128], "dtype": "float32", "sample_values": [1.1300963163375854, 0.231232687830925, -0.9969072937965393, 0.049373116344213486, -0.9496854543685913, 0.8359606266021729, 0.017885586246848106, 0.14250867068767548, -0.5028579831123352, -0.5396110415458679]}, "p": 2, "eps": 1e-06, "keepdim": false}:
        input_info = {"api": "torch.nn.functional.pairwise_distance", "x1": {"shape": [100, 128], "dtype": "float32", "sample_values": [0.04749922454357147, 0.23055438697338104, -0.16772545874118805, 0.42594221234321594, 1.6532641649246216, 2.1893155574798584, -0.45239782333374023, -0.6336861252784729, -1.9172505140304565, 0.2192724049091339]}, "x2": {"shape": [100, 128], "dtype": "float32", "sample_values": [1.1300963163375854, 0.231232687830925, -0.9969072937965393, 0.049373116344213486, -0.9496854543685913, 0.8359606266021729, 0.017885586246848106, 0.14250867068767548, -0.5028579831123352, -0.5396110415458679]}, "p": 2, "eps": 1e-06, "keepdim": false}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.pairwise_distance", "x1": {"shape": [100, 128], "dtype": "float32", "sample_values": [0.04749922454357147, 0.23055438697338104, -0.16772545874118805, 0.42594221234321594, 1.6532641649246216, 2.1893155574798584, -0.45239782333374023, -0.6336861252784729, -1.9172505140304565, 0.2192724049091339]}, "x2": {"shape": [100, 128], "dtype": "float32", "sample_values": [1.1300963163375854, 0.231232687830925, -0.9969072937965393, 0.049373116344213486, -0.9496854543685913, 0.8359606266021729, 0.017885586246848106, 0.14250867068767548, -0.5028579831123352, -0.5396110415458679]}, "p": 2, "eps": 1e-06, "keepdim": false}
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
    api_name = "torch.nn.functional.pairwise_distance"
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
    params = {"api": "paddle.nn.functional.pairwise_distance", "x": {"shape": [100, 128], "dtype": "float32", "sample_values": [2.258631944656372, 0.9215444922447205, -0.7211957573890686, -0.1445651650428772, -0.9068755507469177, 0.09628286957740784, 0.3334144055843353, -0.6041097044944763, 1.6200984716415405, -0.4245598018169403]}, "y": {"shape": [100, 128], "dtype": "float32", "sample_values": [1.214458703994751, 1.2612565755844116, 0.39830854535102844, 0.27539896965026855, 0.09005456417798996, 0.22470688819885254, -0.9253767728805542, 0.3833794593811035, 0.8437530398368835, 0.7041628956794739]}, "p": 2, "keepdim": false}
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
    api_name = "paddle.nn.functional.pairwise_distance"
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
    print(f"Reproducing Case 78: torch.nn.functional.pairwise_distance vs paddle.nn.functional.pairwise_distance")
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
