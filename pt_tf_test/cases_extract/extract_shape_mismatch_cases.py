import json
import re
import os
import numpy as np

def generate_script(case_idx, case_data, output_dir):
    torch_case = case_data['torch_case']
    tf_case = case_data['tf_case']
    
    script_content = f"""
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
    inputs = {{}}
    
    # Handle main input
    if 'input' in {repr(torch_case)}:
        input_info = {repr(torch_case)}['input']
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
    params = {repr(torch_case)}
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
    api_name = "{torch_case['api']}"
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
        kwargs = {{}}
        
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
        print(f"Torch error: {{e}}")
        return None

def test_tensorflow():
    print("\\nTesting TensorFlow API...")
    # Setup inputs
    inputs = {{}}
    
    # Handle args
    params = {repr(tf_case)}
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
    api_name = "{tf_case['api']}"
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
        kwargs = {{}}
        
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
        print(f"TensorFlow error: {{e}}")
        # Fallback: try positional unpacking if kwargs failed
        try:
             print("Retrying with positional args...")
             result = func(*inputs.values())
             print("TensorFlow result shape:", result.shape)
             return result.numpy()
        except Exception as e2:
             print(f"TensorFlow retry error: {{e2}}")
             return None

if __name__ == "__main__":
    print(f"Reproducing Case {case_idx}: {torch_case['api']} vs {tf_case['api']}")
    torch_res = test_torch()
    tf_res = test_tensorflow()

    if torch_res is not None and tf_res is not None:
        try:
            diff = np.abs(torch_res - tf_res)
            max_diff = np.max(diff)
            print(f"\\nMax difference: {{max_diff}}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {{e}}")
"""
    
    file_path = os.path.join(output_dir, f"reproduce_shape_case_{case_idx:02d}.py")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"Generated {file_path}")

def parse_report(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by file sections
    file_sections = content.split('================================================================================')
    
    valid_cases = []
    
    for section in file_sections:
        if not section.strip():
            continue
            
        # Split by cases
        cases = re.split(r'样例 \d+:', section)
        
        for case in cases:
            if 'comparison_error: 形状不匹配' not in case:
                continue
                
            # Extract JSONs
            try:
                torch_match = re.search(r'torch_test_case:\s*({.*?})\s*tensorflow_test_case:', case, re.DOTALL)
                tf_match = re.search(r'tensorflow_test_case:\s*({.*})', case, re.DOTALL)
                
                if not torch_match or not tf_match:
                    continue
                    
                torch_json = json.loads(torch_match.group(1))
                tf_json = json.loads(tf_match.group(1))
                
                torch_api = torch_json.get('api', '')
                tf_api = tf_json.get('api', '')
                
                # Skip if TF API looks like a Torch API (bad data)
                if tf_api.startswith('torch.'):
                    continue
                    
                # Check for semantic equivalence
                # Heuristic: base name match
                pt_base = torch_api.split('.')[-1].lower()
                tf_base = tf_api.split('.')[-1].lower()
                
                # Manual whitelist for known equivalents that might have different names
                # or strict check: base name must be same
                is_equivalent = False
                
                if pt_base == tf_base:
                    is_equivalent = True
                # Handle cases like torch.add vs tf.math.add (handled by base name)
                # Handle cases like torch.sub vs tf.math.subtract
                elif (pt_base == 'sub' and tf_base == 'subtract') or \
                     (pt_base == 'mul' and tf_base == 'multiply') or \
                     (pt_base == 'div' and tf_base == 'divide') or \
                     (pt_base == 'true_divide' and tf_base == 'truediv'):
                     is_equivalent = True
                     
                # Skip addmm vs matmul as user requested "same semantic"
                if pt_base == 'addmm' and tf_base == 'matmul':
                    is_equivalent = False
                    
                if is_equivalent:
                    valid_cases.append({
                        'torch_case': torch_json,
                        'tf_case': tf_json,
                        'error_msg': case.split('\n')[1].strip()
                    })
                        
            except Exception as e:
                # print(f"Error parsing case: {e}")
                continue
                
    return valid_cases

if __name__ == "__main__":
    cases = parse_report(r'd:\graduate\DFrameworkTest\pt_tf_test\analysis\comparison_error_samples_report.txt')
    print(f"Found {len(cases)} cases.")
    output_dir = r'd:\graduate\DFrameworkTest\pt_tf_test\simplest_test\shape_error'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, case in enumerate(cases):
        print(f"--- Case {i+1} ---")
        print(f"Torch API: {case['torch_case']['api']}")
        print(f"TF API: {case['tf_case']['api']}")
        generate_script(i+1, case, output_dir)
