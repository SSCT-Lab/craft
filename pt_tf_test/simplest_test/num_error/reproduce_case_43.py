
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
    if 'input' in {"api": "torch.sub", "input": {"shape": [5, 5], "dtype": "float64", "sample_values": [-0.42966933834029775, 0.3024523956789702, -1.5627218816292792, -0.06085498124058987, -0.8630630152108206, -0.39700673315930357, 1.3723193600255912, -0.9916339565507732, -0.17925173372276385, -0.4478564431501653]}, "other": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [1.6519272014848345, 0.8422467705385851, -0.869634631172171, -0.8997003268982221, 1.582907435207458, -0.00016095311265082907, -0.4446985375388623, -0.7407410550500334, -0.7033974970890295, -0.049159702012363096]}, "alpha": 2}:
        input_info = {"api": "torch.sub", "input": {"shape": [5, 5], "dtype": "float64", "sample_values": [-0.42966933834029775, 0.3024523956789702, -1.5627218816292792, -0.06085498124058987, -0.8630630152108206, -0.39700673315930357, 1.3723193600255912, -0.9916339565507732, -0.17925173372276385, -0.4478564431501653]}, "other": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [1.6519272014848345, 0.8422467705385851, -0.869634631172171, -0.8997003268982221, 1.582907435207458, -0.00016095311265082907, -0.4446985375388623, -0.7407410550500334, -0.7033974970890295, -0.049159702012363096]}, "alpha": 2}['input']
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
    params = {"api": "torch.sub", "input": {"shape": [5, 5], "dtype": "float64", "sample_values": [-0.42966933834029775, 0.3024523956789702, -1.5627218816292792, -0.06085498124058987, -0.8630630152108206, -0.39700673315930357, 1.3723193600255912, -0.9916339565507732, -0.17925173372276385, -0.4478564431501653]}, "other": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [1.6519272014848345, 0.8422467705385851, -0.869634631172171, -0.8997003268982221, 1.582907435207458, -0.00016095311265082907, -0.4446985375388623, -0.7407410550500334, -0.7033974970890295, -0.049159702012363096]}, "alpha": 2}
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
    params = {"api": "tf.subtract", "x": {"shape": [5, 5], "dtype": "float64", "sample_values": [-0.374444550639041, 0.11858592824710654, 0.28508772315009884, 0.8697280265250569, 0.6779598993679209, 0.8177729281457274, -2.30491461279716, 0.6636293920552045, 1.4051984270901545, 0.45386818656079053]}, "y": {"shape": [5, 5, 5], "dtype": "float64", "sample_values": [-0.31442637402621954, -1.560926293942269, -1.3402236181480776, -0.49408627661350374, 0.4234137304157923, -0.3868681193965746, 0.19405932037902424, 1.1859585634396548, -0.8245065939705212, 0.47552037990080054]}}
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
    print(f"Reproducing Case 43: torch.sub vs tf.subtract")
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
