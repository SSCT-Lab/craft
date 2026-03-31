
import torch
import tensorflow as tf
import numpy as np
import os

# Report Error: An error occurred during comparison: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (3, 7) + inhomogeneous part.

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
        try:
            return eval(v)
        except:
            return v
    return v

def normalize_tf_value(v):
    if isinstance(v, str) and v.startswith("tf."):
        try:
            return eval(v)
        except:
            return v
    return v

def test_torch():
    print("Testing torch API...")
    try:
        # Setup inputs
        inputs = {}
        
        # Handle main input
        if 'input' in {'api': 'torch.svd', 'input': {'shape': [7, 5, 3], 'dtype': 'float32', 'sample_values': [0.16273629665374756, -0.7800168991088867, 1.1304287910461426, 1.0149776935577393, -1.5480443239212036, 0.32787761092185974, -0.40950289368629456, -0.3421441316604614, -1.0289363861083984, 0.409728080034256]}, 'compute_uv': False}:
            input_info = {'api': 'torch.svd', 'input': {'shape': [7, 5, 3], 'dtype': 'float32', 'sample_values': [0.16273629665374756, -0.7800168991088867, 1.1304287910461426, 1.0149776935577393, -1.5480443239212036, 0.32787761092185974, -0.40950289368629456, -0.3421441316604614, -1.0289363861083984, 0.409728080034256]}, 'compute_uv': False}['input']
            if isinstance(input_info, dict) and 'shape' in input_info:
                 inputs['input'] = torch.tensor(get_input_data(
                    input_info['shape'], 
                    input_info['dtype'], 
                    input_info['sample_values']
                ), dtype=getattr(torch, input_info['dtype'].split('.')[-1]))
            else:
                 pass
                 
        # Handle other args
        params = {'api': 'torch.svd', 'input': {'shape': [7, 5, 3], 'dtype': 'float32', 'sample_values': [0.16273629665374756, -0.7800168991088867, 1.1304287910461426, 1.0149776935577393, -1.5480443239212036, 0.32787761092185974, -0.40950289368629456, -0.3421441316604614, -1.0289363861083984, 0.409728080034256]}, 'compute_uv': False}
        for k, v in params.items():
            if k == 'api': continue
            if k == 'input': continue
            
            if isinstance(v, dict) and 'shape' in v:
                inputs[k] = torch.tensor(get_input_data(
                    v['shape'], 
                    v['dtype'], 
                    v['sample_values']
                ), dtype=getattr(torch, v['dtype'].split('.')[-1]))
            else:
                inputs[k] = normalize_torch_value(v)

        # Call API
        api_name = "torch.svd"
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
            
        # Construct args list/dict
        args = []
        kwargs = {}
        
        # Special handling for *size or *args
        if '*size' in inputs:
            size_val = inputs.pop('*size')
            if isinstance(size_val, (list, tuple)):
                args.extend(size_val)
            else:
                args.append(size_val)
        
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
            kwargs[k] = v
            
        result = func(*args, **kwargs)
        print("Torch result shape:", result.shape)
        return result.detach().numpy()
    except Exception as e:
        print(f"Torch error: {e}")
        return None

def test_tensorflow():
    print("\nTesting TensorFlow API...")
    try:
        # Setup inputs
        inputs = {}
        
        # Handle args
        params = {'api': 'tf.linalg.svd', 'input': {'shape': [7, 5, 3], 'dtype': 'float32', 'sample_values': [0.16273629665374756, -0.7800168991088867, 1.1304287910461426, 1.0149776935577393, -1.5480443239212036, 0.32787761092185974, -0.40950289368629456, -0.3421441316604614, -1.0289363861083984, 0.409728080034256]}, 'compute_uv': False, 'full_matrices': False}
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
        api_name = "tf.linalg.svd"
        if '.' in api_name:
            parts = api_name.split('.')
            module = tf
            for part in parts[1:-1]:
                module = getattr(module, part)
            func = getattr(module, parts[-1])
        else:
            func = eval(api_name)
            
        args = []
        kwargs = {}
        
        # Special handling for *size -> shape for TF
        if '*size' in inputs:
            # TF usually expects 'shape' tuple or positional args for size
            # We try converting *size to 'shape' kwarg first if it makes sense
            # But wait, some TF APIs like tf.ones take shape as positional.
            # Let's try to map it to 'shape' in kwargs if not present
            shape_val = inputs.pop('*size')
            if 'shape' not in inputs:
                inputs['shape'] = tuple(shape_val) if isinstance(shape_val, list) else shape_val
        
        for k, v in inputs.items():
            if isinstance(k, str) and k.startswith('*'):
                if isinstance(v, (list, tuple)):
                    args.extend(list(v))
                else:
                    args.append(v)
                continue
            kwargs[k] = v

        try:
            result = func(*args, **kwargs)
            print("TensorFlow result shape:", result.shape)
            return result.numpy()
        except Exception as e_kwargs:
            print(f"TensorFlow kwargs error: {e_kwargs}")
            # Fallback: try positional unpacking
            print("Retrying with positional args...")
            
            # Prepare ordered values
            # If 'input' or 'x' is there, put it first?
            # A simple heuristic: values in order of keys? No, keys are unordered in dict.
            # We rely on the order they were inserted? Dicts are ordered in recent Python.
            # But the JSON order might not match signature order.
            
            pos_args = list(inputs.values())
            # Removing dtype from pos args if it causes issues? 
            # (Sometimes dtype is kwarg-only or last arg)
            
            result = func(*pos_args)
            print("TensorFlow result shape:", result.shape)
            return result.numpy()

    except Exception as e:
        print(f"TensorFlow error: {e}")
        return None

if __name__ == "__main__":
    print(f"Reproducing Case 18: torch.svd vs tf.linalg.svd")
    torch_res = test_torch()
    tf_res = test_tensorflow()
