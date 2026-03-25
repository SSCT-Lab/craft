
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

def normalize_torch_value(v):
    if isinstance(v, str) and v.startswith("torch."):
        return eval(v)
    return v

def normalize_tf_value(v):
    if isinstance(v, str) and v.startswith("tf."):
        return eval(v)
    return v

def test_torch():
    print("Testing torch API...")
    # Setup inputs
    inputs = {}
    
    # Handle main input
    if 'input' in {'api': 'torch.split', 'tensor': {'shape': [128, 116, 16, 16], 'dtype': 'float32', 'sample_values': [-0.4477032423019409, -1.2523128986358643, 0.5927289128303528, 0.21673932671546936, 0.9783681035041809, -0.20300127565860748, 0.2721306085586548, 0.6485320329666138, -0.6795424222946167, -1.0966837406158447]}, 'split_size_or_sections': 58, 'dim': 1}:
        input_info = {'api': 'torch.split', 'tensor': {'shape': [128, 116, 16, 16], 'dtype': 'float32', 'sample_values': [-0.4477032423019409, -1.2523128986358643, 0.5927289128303528, 0.21673932671546936, 0.9783681035041809, -0.20300127565860748, 0.2721306085586548, 0.6485320329666138, -0.6795424222946167, -1.0966837406158447]}, 'split_size_or_sections': 58, 'dim': 1}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
             inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype'].split('.')[-1]))
        else:
             # Handle scalar or other types if necessary
             pass
             
    # Handle other args
    params = {'api': 'torch.split', 'tensor': {'shape': [128, 116, 16, 16], 'dtype': 'float32', 'sample_values': [-0.4477032423019409, -1.2523128986358643, 0.5927289128303528, 0.21673932671546936, 0.9783681035041809, -0.20300127565860748, 0.2721306085586548, 0.6485320329666138, -0.6795424222946167, -1.0966837406158447]}, 'split_size_or_sections': 58, 'dim': 1}
    for k, v in params.items():
        if k == 'api': continue
        if k == 'input': continue
        if k == 'out' and isinstance(v, dict) and 'shape' in v:
            continue
        
        if isinstance(v, dict) and 'shape' in v:
            inputs[k] = torch.tensor(get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            ), dtype=getattr(torch, v['dtype'].split('.')[-1]))
        else:
            inputs[k] = v

    # Call API
    # Assuming functional API for now or simple mapping
    api_name = "torch.split"
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
            if isinstance(k, str) and k.startswith('*'):
                if isinstance(v, (list, tuple)):
                    args.extend(list(v))
                else:
                    args.append(v)
                continue
            kwargs[k] = normalize_torch_value(v)
            
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
    params = {'api': 'tf.split', 'value': {'shape': [128, 116, 16, 16], 'dtype': 'float32', 'sample_values': [-0.55843585729599, 0.11239788681268692, -0.894186794757843, 1.073815941810608, -0.8561930060386658, 0.5991066098213196, 0.9365485310554504, -0.01413086336106062, 0.20317700505256653, -0.10214094817638397]}, 'num_or_size_splits': 58, 'axis': 1}
    for k, v in params.items():
        if k == 'api': continue
        
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = tf.constant(np_data, dtype=getattr(tf, v['dtype'].split('.')[-1]))
        else:
            inputs[k] = normalize_tf_value(v)

    # Call API
    api_name = "tf.split"
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
        
        for k, v in inputs.items():
            if isinstance(k, str) and k.startswith('*'):
                args.append(v)
                continue
            kwargs[k] = v

        result = func(*args, **kwargs)
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
    print(f"Reproducing Case 37: torch.split vs tf.split")
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
