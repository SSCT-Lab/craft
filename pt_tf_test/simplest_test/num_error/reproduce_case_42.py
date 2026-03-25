
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
    if 'input' in {"api": "torch.sub", "input": {"shape": [5, 1, 5], "dtype": "float64", "sample_values": [-1.2102082921895791, -2.3099169324802693, 0.20227930016659787, 0.23822991571248145, 2.424617630616524, 0.6755595232769536, -1.4893097474842394, 0.8040364725248466, 1.997258512384492, -0.7656345220561729]}, "other": {"shape": [10, 5], "dtype": "float64", "sample_values": [0.054901104659592435, -0.3917897207037843, 0.3219211324792886, 1.6376468598790757, 0.2618892768530742, 0.7263540116767258, 0.21267772526199152, -3.0455503572792364, 1.0233223843817578, -0.7530359115517029]}, "alpha": 2}:
        input_info = {"api": "torch.sub", "input": {"shape": [5, 1, 5], "dtype": "float64", "sample_values": [-1.2102082921895791, -2.3099169324802693, 0.20227930016659787, 0.23822991571248145, 2.424617630616524, 0.6755595232769536, -1.4893097474842394, 0.8040364725248466, 1.997258512384492, -0.7656345220561729]}, "other": {"shape": [10, 5], "dtype": "float64", "sample_values": [0.054901104659592435, -0.3917897207037843, 0.3219211324792886, 1.6376468598790757, 0.2618892768530742, 0.7263540116767258, 0.21267772526199152, -3.0455503572792364, 1.0233223843817578, -0.7530359115517029]}, "alpha": 2}['input']
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
    params = {"api": "torch.sub", "input": {"shape": [5, 1, 5], "dtype": "float64", "sample_values": [-1.2102082921895791, -2.3099169324802693, 0.20227930016659787, 0.23822991571248145, 2.424617630616524, 0.6755595232769536, -1.4893097474842394, 0.8040364725248466, 1.997258512384492, -0.7656345220561729]}, "other": {"shape": [10, 5], "dtype": "float64", "sample_values": [0.054901104659592435, -0.3917897207037843, 0.3219211324792886, 1.6376468598790757, 0.2618892768530742, 0.7263540116767258, 0.21267772526199152, -3.0455503572792364, 1.0233223843817578, -0.7530359115517029]}, "alpha": 2}
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
    params = {"api": "tf.subtract", "x": {"shape": [5, 1, 5], "dtype": "float64", "sample_values": [1.6750711541136525, -0.20167222614552852, -1.7610019207101388, 0.47610923502932634, -0.7155561148970745, 0.13521861064524965, 0.6101533700409759, -0.07209147278947194, 1.2551480100163759, 1.1573951264116753]}, "y": {"shape": [10, 5], "dtype": "float64", "sample_values": [1.353868140913743, 1.4357064613462542, -1.0172602802611577, -0.21731676496495822, -0.8021556045175978, 1.7605105527656497, 0.7861241109023026, 0.08694388914581207, 1.7307177784764038, 0.22138444060599033]}}
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
    print(f"Reproducing Case 42: torch.sub vs tf.subtract")
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
