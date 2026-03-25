
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
    if 'input' in {'api': 'torch.nn.functional.ctc_loss', 'log_probs': {'shape': [150, 64, 101], 'dtype': 'float64', 'sample_values': [-1.2180618146860218, 0.7837948855506784, 1.0862813904455093, -0.3280800128516353, 0.49437083571396145, 0.24156864799894312, 1.0497538446494017, -0.03217496656966901, 0.7477015293028892, 0.039290503539809396]}, 'targets': {'shape': [64, 15], 'dtype': 'int64', 'sample_values': [-5, 2, 5, 2, -6, 0, 4, -9, 1, 2]}, 'input_lengths': [88, 78, 99, 123, 121, 133, 78, 126, 118, 149, 108, 107, 111, 124, 80, 132, 93, 145, 96, 89, 149, 81, 86, 81, 118, 138, 97, 134, 77, 145, 147, 150, 132, 96, 87, 95, 108, 127, 127, 127, 83, 87, 77, 134, 132, 92, 89, 100, 76, 129, 119, 127, 139, 78, 81, 140, 115, 123, 119, 126, 143, 94, 88, 75], 'target_lengths': [13, 9, 13, 14, 8, 14, 12, 8, 8, 15, 14, 7, 15, 14, 15, 12, 10, 13, 14, 12, 13, 15, 13, 10, 12, 10, 9, 10, 15, 10, 15, 15, 11, 15, 8, 12, 15, 8, 15, 11, 13, 11, 13, 12, 9, 10, 15, 11, 14, 13, 8, 7, 12, 9, 10, 15, 11, 8, 13, 14, 15, 13, 11, 13], 'blank': 0, 'reduction': 'mean', 'zero_infinity': False}:
        input_info = {'api': 'torch.nn.functional.ctc_loss', 'log_probs': {'shape': [150, 64, 101], 'dtype': 'float64', 'sample_values': [-1.2180618146860218, 0.7837948855506784, 1.0862813904455093, -0.3280800128516353, 0.49437083571396145, 0.24156864799894312, 1.0497538446494017, -0.03217496656966901, 0.7477015293028892, 0.039290503539809396]}, 'targets': {'shape': [64, 15], 'dtype': 'int64', 'sample_values': [-5, 2, 5, 2, -6, 0, 4, -9, 1, 2]}, 'input_lengths': [88, 78, 99, 123, 121, 133, 78, 126, 118, 149, 108, 107, 111, 124, 80, 132, 93, 145, 96, 89, 149, 81, 86, 81, 118, 138, 97, 134, 77, 145, 147, 150, 132, 96, 87, 95, 108, 127, 127, 127, 83, 87, 77, 134, 132, 92, 89, 100, 76, 129, 119, 127, 139, 78, 81, 140, 115, 123, 119, 126, 143, 94, 88, 75], 'target_lengths': [13, 9, 13, 14, 8, 14, 12, 8, 8, 15, 14, 7, 15, 14, 15, 12, 10, 13, 14, 12, 13, 15, 13, 10, 12, 10, 9, 10, 15, 10, 15, 15, 11, 15, 8, 12, 15, 8, 15, 11, 13, 11, 13, 12, 9, 10, 15, 11, 14, 13, 8, 7, 12, 9, 10, 15, 11, 8, 13, 14, 15, 13, 11, 13], 'blank': 0, 'reduction': 'mean', 'zero_infinity': False}['input']
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
    params = {'api': 'torch.nn.functional.ctc_loss', 'log_probs': {'shape': [150, 64, 101], 'dtype': 'float64', 'sample_values': [-1.2180618146860218, 0.7837948855506784, 1.0862813904455093, -0.3280800128516353, 0.49437083571396145, 0.24156864799894312, 1.0497538446494017, -0.03217496656966901, 0.7477015293028892, 0.039290503539809396]}, 'targets': {'shape': [64, 15], 'dtype': 'int64', 'sample_values': [-5, 2, 5, 2, -6, 0, 4, -9, 1, 2]}, 'input_lengths': [88, 78, 99, 123, 121, 133, 78, 126, 118, 149, 108, 107, 111, 124, 80, 132, 93, 145, 96, 89, 149, 81, 86, 81, 118, 138, 97, 134, 77, 145, 147, 150, 132, 96, 87, 95, 108, 127, 127, 127, 83, 87, 77, 134, 132, 92, 89, 100, 76, 129, 119, 127, 139, 78, 81, 140, 115, 123, 119, 126, 143, 94, 88, 75], 'target_lengths': [13, 9, 13, 14, 8, 14, 12, 8, 8, 15, 14, 7, 15, 14, 15, 12, 10, 13, 14, 12, 13, 15, 13, 10, 12, 10, 9, 10, 15, 10, 15, 15, 11, 15, 8, 12, 15, 8, 15, 11, 13, 11, 13, 12, 9, 10, 15, 11, 14, 13, 8, 7, 12, 9, 10, 15, 11, 8, 13, 14, 15, 13, 11, 13], 'blank': 0, 'reduction': 'mean', 'zero_infinity': False}
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
    api_name = "torch.nn.functional.ctc_loss"
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
    params = {'api': 'tf.nn.ctc_loss', 'logits': {'shape': [150, 64, 101], 'dtype': 'float32', 'sample_values': [-0.06217528507113457, -0.5689347982406616, -1.6094557046890259, 0.3481694459915161, 1.067386507987976, -0.11036987602710724, 0.47980883717536926, 0.376119464635849, 1.3479396104812622, 2.478111982345581]}, 'labels': {'shape': [64, 15], 'dtype': 'int64', 'sample_values': [-9, -4, -8, -6, -8, -9, -3, 5, -7, 7]}, 'label_length': [13, 9, 13, 14, 8, 14, 12, 8, 8, 15, 14, 7, 15, 14, 15, 12, 10, 13, 14, 12, 13, 15, 13, 10, 12, 10, 9, 10, 15, 10, 15, 15, 11, 15, 8, 12, 15, 8, 15, 11, 13, 11, 13, 12, 9, 10, 15, 11, 14, 13, 8, 7, 12, 9, 10, 15, 11, 8, 13, 14, 15, 13, 11, 13], 'logit_length': [88, 78, 99, 123, 121, 133, 78, 126, 118, 149, 108, 107, 111, 124, 80, 132, 93, 145, 96, 89, 149, 81, 86, 81, 118, 138, 97, 134, 77, 145, 147, 150, 132, 96, 87, 95, 108, 127, 127, 127, 83, 87, 77, 134, 132, 92, 89, 100, 76, 129, 119, 127, 139, 78, 81, 140, 115, 123, 119, 126, 143, 94, 88, 75], 'blank_index': 0}
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
    api_name = "tf.nn.ctc_loss"
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
    print(f"Reproducing Case 20: torch.nn.functional.ctc_loss vs tf.nn.ctc_loss")
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
