
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
    if 'input' in {"input": {"shape": [20, 16, 10, 50, 100], "dtype": "float32", "sample_values": [0.7752637267112732, 0.8845512866973877, -0.017568811774253845, -0.7523747086524963, -0.981509268283844, 0.2940584719181061, 0.03562941402196884, -0.7505152225494385, -0.4497143030166626, -0.49824902415275574]}, "weight": {"shape": [33, 16, 3, 5, 2], "dtype": "float32", "sample_values": [-0.09101136028766632, -0.6142028570175171, -1.0077792406082153, 0.03650888055562973, -0.33617982268333435, 0.5204266905784607, -0.8811998963356018, 1.0933336019515991, -0.1312147080898285, 0.8963345885276794]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [0.07692712545394897, 4.147119998931885, 0.5593327879905701, -0.17988839745521545, -0.8006935715675354, -1.1780678033828735, 0.4137747883796692, -0.6542391777038574, 0.21862509846687317, 0.5949686765670776]}, "stride": [2, 1, 1], "padding": [4, 2, 0], "dilation": [1, 1, 1], "groups": 1, "api": "torch.nn.functional.conv3d"}:
        input_info = {"input": {"shape": [20, 16, 10, 50, 100], "dtype": "float32", "sample_values": [0.7752637267112732, 0.8845512866973877, -0.017568811774253845, -0.7523747086524963, -0.981509268283844, 0.2940584719181061, 0.03562941402196884, -0.7505152225494385, -0.4497143030166626, -0.49824902415275574]}, "weight": {"shape": [33, 16, 3, 5, 2], "dtype": "float32", "sample_values": [-0.09101136028766632, -0.6142028570175171, -1.0077792406082153, 0.03650888055562973, -0.33617982268333435, 0.5204266905784607, -0.8811998963356018, 1.0933336019515991, -0.1312147080898285, 0.8963345885276794]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [0.07692712545394897, 4.147119998931885, 0.5593327879905701, -0.17988839745521545, -0.8006935715675354, -1.1780678033828735, 0.4137747883796692, -0.6542391777038574, 0.21862509846687317, 0.5949686765670776]}, "stride": [2, 1, 1], "padding": [4, 2, 0], "dilation": [1, 1, 1], "groups": 1, "api": "torch.nn.functional.conv3d"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"input": {"shape": [20, 16, 10, 50, 100], "dtype": "float32", "sample_values": [0.7752637267112732, 0.8845512866973877, -0.017568811774253845, -0.7523747086524963, -0.981509268283844, 0.2940584719181061, 0.03562941402196884, -0.7505152225494385, -0.4497143030166626, -0.49824902415275574]}, "weight": {"shape": [33, 16, 3, 5, 2], "dtype": "float32", "sample_values": [-0.09101136028766632, -0.6142028570175171, -1.0077792406082153, 0.03650888055562973, -0.33617982268333435, 0.5204266905784607, -0.8811998963356018, 1.0933336019515991, -0.1312147080898285, 0.8963345885276794]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [0.07692712545394897, 4.147119998931885, 0.5593327879905701, -0.17988839745521545, -0.8006935715675354, -1.1780678033828735, 0.4137747883796692, -0.6542391777038574, 0.21862509846687317, 0.5949686765670776]}, "stride": [2, 1, 1], "padding": [4, 2, 0], "dilation": [1, 1, 1], "groups": 1, "api": "torch.nn.functional.conv3d"}
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
    api_name = "torch.nn.functional.conv3d"
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
    params = {"input": {"shape": [20, 16, 10, 50, 100], "dtype": "float32", "sample_values": [0.7752637267112732, 0.8845512866973877, -0.017568811774253845, -0.7523747086524963, -0.981509268283844, 0.2940584719181061, 0.03562941402196884, -0.7505152225494385, -0.4497143030166626, -0.49824902415275574]}, "weight": {"shape": [33, 16, 3, 5, 2], "dtype": "float32", "sample_values": [-0.09101136028766632, -0.6142028570175171, -1.0077792406082153, 0.03650888055562973, -0.33617982268333435, 0.5204266905784607, -0.8811998963356018, 1.0933336019515991, -0.1312147080898285, 0.8963345885276794]}, "bias": {"shape": [33], "dtype": "float32", "sample_values": [0.07692712545394897, 4.147119998931885, 0.5593327879905701, -0.17988839745521545, -0.8006935715675354, -1.1780678033828735, 0.4137747883796692, -0.6542391777038574, 0.21862509846687317, 0.5949686765670776]}, "stride": [2, 1, 1], "padding": [4, 2, 0], "dilation": [1, 1, 1], "groups": 1, "api": "torch.nn.functional.conv3d"}
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
    api_name = "torch.nn.functional.conv3d"
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
    print(f"Reproducing Case 62: torch.nn.functional.conv3d vs torch.nn.functional.conv3d")
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
