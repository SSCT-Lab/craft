# This code file is used to reproduce and compare the behavioral differences between PyTorch and TensorFlow on numerical APIs (here cholesky_solve）。
# Process overview：
# 1) as given shape/dtype/sample_values Construct input numpy data；
# 2) Convert numpy data to torch.tensor and tf.constant；
# 3) call corresponding API（torch.cholesky_solve and tf.linalg.cholesky_solve）；
# 4) Print the result shape and calculate the maximum value of the difference between the two values ​​when both are returned successfully.。

import torch
import tensorflow as tf
import numpy as np

def get_input_data(shape, dtype, sample_values):
    # Generate a numpy array based on shape and sample_values：
    # - If shape is empty, it is considered a scalar and is used directly. sample_values[0]
    # - Otherwise sample_values ​​will be repeated/Truncate to fill the specified shape, then reshape
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
    # Construct a dictionary of input parameters on the torch side
    inputs = {}
    
    # Processing the main input: here by detecting whether the fixed dictionary contains 'input'（always True），
    # extract input from shape/dtype/sample_values to construct torch.tensor
    if 'input' in {"api": "torch.cholesky_solve", "input": {"shape": [3, 2], "dtype": "float32", "sample_values": [-0.33505409955978394, 0.5129841566085815, 0.33704325556755066, 0.7546667456626892, -0.8685919642448425, -2.908078908920288]}, "input2": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.30667033791542053, -1.3762863874435425, 0.06743878871202469, 0.3087283968925476, 1.9169071912765503, 0.17984506487846375, -0.5016515254974365, -0.6793695688247681, 0.34692952036857605]}}:
        input_info = {"api": "torch.cholesky_solve", "input": {"shape": [3, 2], "dtype": "float32", "sample_values": [-0.33505409955978394, 0.5129841566085815, 0.33704325556755066, 0.7546667456626892, -0.8685919642448425, -2.908078908920288]}, "input2": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.30667033791542053, -1.3762863874435425, 0.06743878871202469, 0.3087283968925476, 1.9169071912765503, 0.17984506487846375, -0.5016515254974365, -0.6793695688247681, 0.34692952036857605]}}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
             inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
        else:
             # Handle scalar or other types if necessary
             pass
             
    # Process other parameters (such as input2, etc.). If it is a dictionary with shape, it will also be converted to torch.tensor
    params = {"api": "torch.cholesky_solve", "input": {"shape": [3, 2], "dtype": "float32", "sample_values": [-0.33505409955978394, 0.5129841566085815, 0.33704325556755066, 0.7546667456626892, -0.8685919642448425, -2.908078908920288]}, "input2": {"shape": [3, 3], "dtype": "float32", "sample_values": [-0.30667033791542053, -1.3762863874435425, 0.06743878871202469, 0.3087283968925476, 1.9169071912765503, 0.17984506487846375, -0.5016515254974365, -0.6793695688247681, 0.34692952036857605]}}
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

    # Call the torch API (here using string reflection to get the function object）
    # Agreement: If exists 'input' key, usually as the first positional argument and the rest as kwargs
    api_name = "torch.cholesky_solve"
    try:
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
            
        # Assembly parameters: will inputs['input'] into positional parameters, and the rest as keyword parameters
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
    # Construct a dictionary of input parameters on the TensorFlow side
    inputs = {}
    
    # Handle all arguments: dictionary with shape converted to tf.constant
    params = {"api": "tf.linalg.cholesky_solve", "rhs": {"shape": [3, 2], "dtype": "float32", "sample_values": [-1.0670522451400757, -0.6584956645965576, -0.8646241426467896, 0.727580189704895, 0.7642821073532104, -1.552671194076538]}, "chol": {"shape": [3, 3], "dtype": "float32", "sample_values": [0.20300863683223724, -1.8740577697753906, -0.4296968877315521, 0.5470719933509827, -0.9220492839813232, -0.9780227541923523, 0.15461786091327667, 0.41276776790618896, -2.111048698425293]}}
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

    # Call TF API: Get the function object layer by layer through string path getattr
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
            
        # Note: TF usually uses named parameters, here it is called directly with kwargs.；
        # If it fails, it will try again with positional parameters.
        
        args = []
        kwargs = {}
        
        result = func(**inputs)
        print("TensorFlow result shape:", result.shape)
        return result.numpy()
    except Exception as e:
        print(f"TensorFlow error: {e}")
        # Bottom line: If the named parameter method fails, try calling by positional parameters
        try:
             print("Retrying with positional args...")
             result = func(*inputs.values())
             print("TensorFlow result shape:", result.shape)
             return result.numpy()
        except Exception as e2:
             print(f"TensorFlow retry error: {e2}")
             return None

if __name__ == "__main__":
    # Main entry: run the tests of torch and TF separately, and then calculate the difference when both return successfully
    print(f"Reproducing Case 1: torch.cholesky_solve vs tf.linalg.cholesky_solve")
    torch_res = test_torch()
    tf_res = test_tensorflow()

    if torch_res is not None and tf_res is not None:
        try:
            # Numerical difference: do the absolute value of the element-wise difference between two numpy arrays and print the maximum difference
            diff = np.abs(torch_res - tf_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
