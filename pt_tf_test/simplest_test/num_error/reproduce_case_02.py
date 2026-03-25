
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
    if 'input' in {"api": "torch.cholesky_solve", "input": {"shape": [4, 3], "dtype": "float64", "sample_values": [-0.5552721409645636, -0.47525295226686703, 1.3817135037928914, -0.9377177102595262, 1.619509322830922, -0.6263794013946363, 0.651338871500162, 0.7426312004058384, -0.3559254861926964, 1.3253286853422923]}, "input2": {"shape": [4, 4], "dtype": "float64", "sample_values": [2.181173986736992, -0.33656665994990737, -0.07545222825245237, 1.3572993078602387, 0.010331855677279263, -0.2932572127548066, 0.9026623947298016, -0.23353015494935475, 1.1792906914610057, -0.6811202101964753]}}:
        input_info = {"api": "torch.cholesky_solve", "input": {"shape": [4, 3], "dtype": "float64", "sample_values": [-0.5552721409645636, -0.47525295226686703, 1.3817135037928914, -0.9377177102595262, 1.619509322830922, -0.6263794013946363, 0.651338871500162, 0.7426312004058384, -0.3559254861926964, 1.3253286853422923]}, "input2": {"shape": [4, 4], "dtype": "float64", "sample_values": [2.181173986736992, -0.33656665994990737, -0.07545222825245237, 1.3572993078602387, 0.010331855677279263, -0.2932572127548066, 0.9026623947298016, -0.23353015494935475, 1.1792906914610057, -0.6811202101964753]}}['input']
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
    params = {"api": "torch.cholesky_solve", "input": {"shape": [4, 3], "dtype": "float64", "sample_values": [-0.5552721409645636, -0.47525295226686703, 1.3817135037928914, -0.9377177102595262, 1.619509322830922, -0.6263794013946363, 0.651338871500162, 0.7426312004058384, -0.3559254861926964, 1.3253286853422923]}, "input2": {"shape": [4, 4], "dtype": "float64", "sample_values": [2.181173986736992, -0.33656665994990737, -0.07545222825245237, 1.3572993078602387, 0.010331855677279263, -0.2932572127548066, 0.9026623947298016, -0.23353015494935475, 1.1792906914610057, -0.6811202101964753]}}
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
    api_name = "torch.cholesky_solve"
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
    params = {"api": "tf.linalg.cholesky_solve", "rhs": {"shape": [4, 3], "dtype": "float64", "sample_values": [-0.7915610828217502, 1.2997207198272922, -0.0662788840281902, 0.33279890685757485, 0.8655976977199026, -0.42945270926623647, 0.11307986518337726, 0.18996742151162957, 0.49661258391255475, 0.2129050616965679]}, "chol": {"shape": [4, 4], "dtype": "float64", "sample_values": [-0.16436456557452947, -1.670697298377919, 0.8447017313950153, 0.15014383124452277, 0.8662834946624625, -1.0690100730794936, 0.43569785537819117, -0.7596273318310801, -0.42038554869906564, -1.3346342589314912]}}
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
    api_name = "tf.linalg.cholesky_solve"
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
    print(f"Reproducing Case 2: torch.cholesky_solve vs tf.linalg.cholesky_solve")
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
