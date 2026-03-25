
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
    if 'input' in {"api": "torch.nn.functional.leaky_relu", "input": {"shape": [64, 64, 14, 14], "dtype": "float32", "sample_values": [1.0152490139007568, 0.9157294631004333, 1.366678237915039, -0.6787232160568237, -0.3300327956676483, 0.9411445260047913, 0.3657899796962738, 1.0559252500534058, -0.1308077573776245, -0.4632655084133148]}, "negative_slope": 0.2, "inplace": false}:
        input_info = {"api": "torch.nn.functional.leaky_relu", "input": {"shape": [64, 64, 14, 14], "dtype": "float32", "sample_values": [1.0152490139007568, 0.9157294631004333, 1.366678237915039, -0.6787232160568237, -0.3300327956676483, 0.9411445260047913, 0.3657899796962738, 1.0559252500534058, -0.1308077573776245, -0.4632655084133148]}, "negative_slope": 0.2, "inplace": false}['input']
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
    params = {"api": "torch.nn.functional.leaky_relu", "input": {"shape": [64, 64, 14, 14], "dtype": "float32", "sample_values": [1.0152490139007568, 0.9157294631004333, 1.366678237915039, -0.6787232160568237, -0.3300327956676483, 0.9411445260047913, 0.3657899796962738, 1.0559252500534058, -0.1308077573776245, -0.4632655084133148]}, "negative_slope": 0.2, "inplace": false}
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
    api_name = "torch.nn.functional.leaky_relu"
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
    params = {"api": "tf.nn.leaky_relu", "features": {"shape": [64, 64, 14, 14], "dtype": "float32", "sample_values": [-0.03359662741422653, -0.7332822680473328, 0.7300031781196594, -0.21513566374778748, 1.0633325576782227, 1.1205592155456543, -0.4771516025066376, 0.08987808972597122, -1.3058778047561646, 0.794809103012085]}, "alpha": 0.2}
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
    api_name = "tf.nn.leaky_relu"
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
    print(f"Reproducing Case 20: torch.nn.functional.leaky_relu vs tf.nn.leaky_relu")
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
