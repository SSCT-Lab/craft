
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
    if 'input' in {"api": "torch.nn.functional.batch_norm", "input": {"shape": [2, 1024], "dtype": "float64", "sample_values": [1.3585713539054802, -1.1728909639916607, 1.2929560163489664, -0.39633158257071943, -0.575374080470187, -0.2152480525480704, 1.4177834363291346, 0.4029931578728807, 0.5587419129391814, 1.743551973631452]}, "running_mean": {"shape": [1024], "dtype": "float64", "sample_values": [0.6155262889511852, 1.0396851608563933, 1.1648652305474296, -0.09095350660762644, -2.298870666782325, 1.53795832260273, 0.6966091041813686, -1.3422826079367582, -1.8062663046209428, 0.5370511749549146]}, "running_var": {"shape": [1024], "dtype": "float64", "sample_values": [0.37870724636874054, -0.20131747628211155, 1.1252835480061165, -0.2589402492866238, -0.9383952967637732, 1.7014744311483323, 0.6895095595850868, -0.5252235905792062, -0.7012530658174593, -0.6038242200493312]}, "weight": {"shape": [1024], "dtype": "float64", "sample_values": [0.7794992122765834, 1.4001630356045387, 0.11237019472979785, 0.042050965374520424, -0.6663156699858792, 1.504140922128603, -1.0238373390213358, 1.3857534499189197, 0.6622712161394194, -0.6307442248802765]}, "bias": {"shape": [1024], "dtype": "float64", "sample_values": [-1.6425722563048788, 0.6253933698513867, -1.1446258896869395, -0.7687492148628061, 0.2448660105092128, -0.39704375100583555, 0.955875300719565, -1.0229296665788778, 1.9161781572813814, 2.0100273890533473]}, "training": true, "momentum": 0.1, "eps": 1e-07}:
        input_info = {"api": "torch.nn.functional.batch_norm", "input": {"shape": [2, 1024], "dtype": "float64", "sample_values": [1.3585713539054802, -1.1728909639916607, 1.2929560163489664, -0.39633158257071943, -0.575374080470187, -0.2152480525480704, 1.4177834363291346, 0.4029931578728807, 0.5587419129391814, 1.743551973631452]}, "running_mean": {"shape": [1024], "dtype": "float64", "sample_values": [0.6155262889511852, 1.0396851608563933, 1.1648652305474296, -0.09095350660762644, -2.298870666782325, 1.53795832260273, 0.6966091041813686, -1.3422826079367582, -1.8062663046209428, 0.5370511749549146]}, "running_var": {"shape": [1024], "dtype": "float64", "sample_values": [0.37870724636874054, -0.20131747628211155, 1.1252835480061165, -0.2589402492866238, -0.9383952967637732, 1.7014744311483323, 0.6895095595850868, -0.5252235905792062, -0.7012530658174593, -0.6038242200493312]}, "weight": {"shape": [1024], "dtype": "float64", "sample_values": [0.7794992122765834, 1.4001630356045387, 0.11237019472979785, 0.042050965374520424, -0.6663156699858792, 1.504140922128603, -1.0238373390213358, 1.3857534499189197, 0.6622712161394194, -0.6307442248802765]}, "bias": {"shape": [1024], "dtype": "float64", "sample_values": [-1.6425722563048788, 0.6253933698513867, -1.1446258896869395, -0.7687492148628061, 0.2448660105092128, -0.39704375100583555, 0.955875300719565, -1.0229296665788778, 1.9161781572813814, 2.0100273890533473]}, "training": true, "momentum": 0.1, "eps": 1e-07}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.batch_norm", "input": {"shape": [2, 1024], "dtype": "float64", "sample_values": [1.3585713539054802, -1.1728909639916607, 1.2929560163489664, -0.39633158257071943, -0.575374080470187, -0.2152480525480704, 1.4177834363291346, 0.4029931578728807, 0.5587419129391814, 1.743551973631452]}, "running_mean": {"shape": [1024], "dtype": "float64", "sample_values": [0.6155262889511852, 1.0396851608563933, 1.1648652305474296, -0.09095350660762644, -2.298870666782325, 1.53795832260273, 0.6966091041813686, -1.3422826079367582, -1.8062663046209428, 0.5370511749549146]}, "running_var": {"shape": [1024], "dtype": "float64", "sample_values": [0.37870724636874054, -0.20131747628211155, 1.1252835480061165, -0.2589402492866238, -0.9383952967637732, 1.7014744311483323, 0.6895095595850868, -0.5252235905792062, -0.7012530658174593, -0.6038242200493312]}, "weight": {"shape": [1024], "dtype": "float64", "sample_values": [0.7794992122765834, 1.4001630356045387, 0.11237019472979785, 0.042050965374520424, -0.6663156699858792, 1.504140922128603, -1.0238373390213358, 1.3857534499189197, 0.6622712161394194, -0.6307442248802765]}, "bias": {"shape": [1024], "dtype": "float64", "sample_values": [-1.6425722563048788, 0.6253933698513867, -1.1446258896869395, -0.7687492148628061, 0.2448660105092128, -0.39704375100583555, 0.955875300719565, -1.0229296665788778, 1.9161781572813814, 2.0100273890533473]}, "training": true, "momentum": 0.1, "eps": 1e-07}
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
    api_name = "torch.nn.functional.batch_norm"
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
    params = {"api": "paddle.nn.functional.batch_norm", "input": {"shape": [2, 1024], "dtype": "float64", "sample_values": [1.3585713539054802, -1.1728909639916607, 1.2929560163489664, -0.39633158257071943, -0.575374080470187, -0.2152480525480704, 1.4177834363291346, 0.4029931578728807, 0.5587419129391814, 1.743551973631452]}, "running_mean": {"shape": [1024], "dtype": "float64", "sample_values": [0.6155262889511852, 1.0396851608563933, 1.1648652305474296, -0.09095350660762644, -2.298870666782325, 1.53795832260273, 0.6966091041813686, -1.3422826079367582, -1.8062663046209428, 0.5370511749549146]}, "running_var": {"shape": [1024], "dtype": "float64", "sample_values": [0.37870724636874054, -0.20131747628211155, 1.1252835480061165, -0.2589402492866238, -0.9383952967637732, 1.7014744311483323, 0.6895095595850868, -0.5252235905792062, -0.7012530658174593, -0.6038242200493312]}, "weight": {"shape": [1024], "dtype": "float64", "sample_values": [0.7794992122765834, 1.4001630356045387, 0.11237019472979785, 0.042050965374520424, -0.6663156699858792, 1.504140922128603, -1.0238373390213358, 1.3857534499189197, 0.6622712161394194, -0.6307442248802765]}, "bias": {"shape": [1024], "dtype": "float64", "sample_values": [-1.6425722563048788, 0.6253933698513867, -1.1446258896869395, -0.7687492148628061, 0.2448660105092128, -0.39704375100583555, 0.955875300719565, -1.0229296665788778, 1.9161781572813814, 2.0100273890533473]}, "training": true, "momentum": 0.1}
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
    api_name = "paddle.nn.functional.batch_norm"
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
    print(f"Reproducing Case 51: torch.nn.functional.batch_norm vs paddle.nn.functional.batch_norm")
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
