
import torch
import tensorflow as tf
import numpy as np

def get_input_data(shape, dtype, sample_values):
    if not shape:
        return np.array(sample_values[0], dtype=dtype)
    
    total_elements = np.prod(shape)
    if len(sample_values) < total_elements:
        # Repeat values to fill the shape
        repeats = int(np.ceil(total_elements / len(sample_values)))
        full_values = (sample_values * repeats)[:total_elements]
    else:
        full_values = sample_values[:total_elements]
        
    return np.array(full_values, dtype=dtype).reshape(shape)

def test_torch():
    print("Testing torch API...")
    # Setup inputs
    inputs = {}
    
    # Handle main input
    if 'input' in {"api": "torch.sub", "input": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [-0.696514710136094, -0.1716506121535855, 0.8829329801815345, -0.4847489334692169, 0.494009368184946, -1.838059514552648, 0.4073647588561924, 0.08543516473328407, 1.5150245398760862, -1.1759042394104406]}, "other": {"shape": [5, 5], "dtype": "float64", "sample_values": [2.8532188623740184, 1.2187044079757228, 0.3573227214943599, -0.11832623833666496, -0.7326092305985069, 2.591775768391342, -0.7519662933695299, 1.53564507617768, -1.4483304870828455, 0.2230250482279526]}, "alpha": 2}:
        input_info = {"api": "torch.sub", "input": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [-0.696514710136094, -0.1716506121535855, 0.8829329801815345, -0.4847489334692169, 0.494009368184946, -1.838059514552648, 0.4073647588561924, 0.08543516473328407, 1.5150245398760862, -1.1759042394104406]}, "other": {"shape": [5, 5], "dtype": "float64", "sample_values": [2.8532188623740184, 1.2187044079757228, 0.3573227214943599, -0.11832623833666496, -0.7326092305985069, 2.591775768391342, -0.7519662933695299, 1.53564507617768, -1.4483304870828455, 0.2230250482279526]}, "alpha": 2}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
             inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
        else:
             # Handle scalar or other types if necessary
             pass
             
    # Handle other args
    params = {"api": "torch.sub", "input": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [-0.696514710136094, -0.1716506121535855, 0.8829329801815345, -0.4847489334692169, 0.494009368184946, -1.838059514552648, 0.4073647588561924, 0.08543516473328407, 1.5150245398760862, -1.1759042394104406]}, "other": {"shape": [5, 5], "dtype": "float64", "sample_values": [2.8532188623740184, 1.2187044079757228, 0.3573227214943599, -0.11832623833666496, -0.7326092305985069, 2.591775768391342, -0.7519662933695299, 1.53564507617768, -1.4483304870828455, 0.2230250482279526]}, "alpha": 2}
    for k, v in params.items():
        if k == 'api': continue
        if k == 'input': continue
        
        if isinstance(v, dict) and 'shape' in v:
            inputs[k] = torch.tensor(get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            ), dtype=getattr(torch, v['dtype']))
        else:
            inputs[k] = v

    # Call API
    # Assuming functional API for now or simple mapping
    api_name = "torch.sub"
    try:
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
            
        # Construct args list/dict
        # Heuristic: if 'input' exists, it's usually the first arg
        args = []
        kwargs = {}
        
        if 'input' in inputs:
            args.append(inputs['input'])
            
        for k, v in inputs.items():
            if k == 'input': continue
            kwargs[k] = v
            
        result = func(*args, **kwargs)
        print("Torch result shape:", result.shape)
        return result.detach().numpy()
    except Exception as e:
        print(f"Torch error: {e}")
        return None

def test_tensorflow():
    print("\nTesting TensorFlow API...")
    # Setup inputs
    inputs = {}
    
    # Handle args
    params = {"api": "tf.subtract", "x": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [-0.1625585924304854, 0.359997422286295, -1.0186830087489582, 0.2504065938317808, -0.013029900203767453, 1.4155797256506006, 0.38685684643262597, 1.115047726627907, 0.8575025112166916, 0.13798325371618467]}, "y": {"shape": [5, 5], "dtype": "float64", "sample_values": [-2.5830917749134152, 0.57653609348353, 0.9419517737369184, 0.36627025138935465, -0.21308411369589336, -1.2216802690643027, 0.8476644306026784, 0.5369323490464245, 1.359329332595588, -0.979549239305433]}}
    for k, v in params.items():
        if k == 'api': continue
        
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = tf.constant(np_data, dtype=getattr(tf, v['dtype']))
        else:
            inputs[k] = v

    # Call API
    api_name = "tf.subtract"
    try:
        if '.' in api_name:
            # Handle tf.experimental.numpy -> tf.experimental.numpy
            parts = api_name.split('.')
            module = tf
            for part in parts[1:-1]:
                module = getattr(module, part)
            func = getattr(module, parts[-1])
        else:
            func = eval(api_name)
            
        # TF usually uses named args or specific order. 
        # For simplicity in this generic script, we try to match keys to kwargs
        # But TF APIs often don't accept kwargs for all inputs (e.g. math.add(x, y))
        # We might need specific mapping logic here.
        # For now, let's try kwargs and if it fails, try positional based on common names
        
        args = []
        kwargs = {}
        
        # Common positional args mapping
        ordered_keys = ['input', 'x', 'a', 'y', 'b', 'other']
        remaining_keys = list(inputs.keys())
        
        # This is a bit hacky, but tries to guess positional args
        # Ideally we would inspect the signature or use the keys provided in the JSON if they match arg names
        
        result = func(**inputs)
        print("TensorFlow result shape:", result.shape)
        return result.numpy()
    except Exception as e:
        print(f"TensorFlow error: {e}")
        # Fallback: try positional unpacking if kwargs failed
        try:
             print("Retrying with positional args...")
             result = func(*inputs.values())
             print("TensorFlow result shape:", result.shape)
             return result.numpy()
        except Exception as e2:
             print(f"TensorFlow retry error: {e2}")
             return None

if __name__ == "__main__":
    print(f"Reproducing Case 45: torch.sub vs tf.subtract")
    torch_res = test_torch()
    tf_res = test_tensorflow()

    if torch_res is not None and tf_res is not None:
        try:
            diff = np.abs(torch_res - tf_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
