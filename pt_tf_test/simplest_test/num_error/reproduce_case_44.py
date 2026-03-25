
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
    if 'input' in {"api": "torch.sub", "input": {"shape": [3, 1, 5], "dtype": "float32", "sample_values": [0.04091832414269447, -2.211728572845459, 0.2620096504688263, -1.379663348197937, -0.9073720574378967, 0.013192941434681416, 0.4828995168209076, -1.7690892219543457, 1.2125898599624634, 1.5131689310073853]}, "other": {"shape": [5, 5], "dtype": "float32", "sample_values": [-0.35787197947502136, -0.36822986602783203, -0.16187924146652222, -0.5564363598823547, -0.8444766402244568, -1.1995859146118164, -1.5672366619110107, -0.7143611907958984, -1.17350435256958, 0.30187439918518066]}, "alpha": 0.5}:
        input_info = {"api": "torch.sub", "input": {"shape": [3, 1, 5], "dtype": "float32", "sample_values": [0.04091832414269447, -2.211728572845459, 0.2620096504688263, -1.379663348197937, -0.9073720574378967, 0.013192941434681416, 0.4828995168209076, -1.7690892219543457, 1.2125898599624634, 1.5131689310073853]}, "other": {"shape": [5, 5], "dtype": "float32", "sample_values": [-0.35787197947502136, -0.36822986602783203, -0.16187924146652222, -0.5564363598823547, -0.8444766402244568, -1.1995859146118164, -1.5672366619110107, -0.7143611907958984, -1.17350435256958, 0.30187439918518066]}, "alpha": 0.5}['input']
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
    params = {"api": "torch.sub", "input": {"shape": [3, 1, 5], "dtype": "float32", "sample_values": [0.04091832414269447, -2.211728572845459, 0.2620096504688263, -1.379663348197937, -0.9073720574378967, 0.013192941434681416, 0.4828995168209076, -1.7690892219543457, 1.2125898599624634, 1.5131689310073853]}, "other": {"shape": [5, 5], "dtype": "float32", "sample_values": [-0.35787197947502136, -0.36822986602783203, -0.16187924146652222, -0.5564363598823547, -0.8444766402244568, -1.1995859146118164, -1.5672366619110107, -0.7143611907958984, -1.17350435256958, 0.30187439918518066]}, "alpha": 0.5}
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
    params = {"api": "tf.subtract", "x": {"shape": [3, 1, 5], "dtype": "float32", "sample_values": [-3.3239943981170654, -0.9613867402076721, 0.9706313014030457, -0.9824942946434021, 0.7481033802032471, -0.9622488617897034, 0.4300229251384735, -0.4592372179031372, 0.6938543319702148, 0.9817133545875549]}, "y": {"shape": [5, 5], "dtype": "float32", "sample_values": [-0.7107061147689819, 1.1385581493377686, 1.171203374862671, -0.31757375597953796, 1.0385226011276245, 0.028550488874316216, -0.6409881114959717, 0.9522489309310913, -0.9667481780052185, 0.18927493691444397]}}
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
    print(f"Reproducing Case 44: torch.sub vs tf.subtract")
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
