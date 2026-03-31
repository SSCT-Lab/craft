import json
import re
import os
import numpy as np

def generate_script(case_idx, case_data, output_dir):
    torch_case = case_data['torch_case']
    tf_case = case_data['tf_case']
    error_msg = case_data.get('error_msg', 'Unknown error')
    
    script_content = f"""
import torch
import tensorflow as tf
import numpy as np
import os

# Report Error: {error_msg}

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
                 pass
                 
        # Handle other args
        params = {repr(torch_case)}
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
        api_name = "{torch_case['api']}"
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
            
        # Construct args list/dict
        args = []
        kwargs = {{}}
        
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
        print(f"Torch error: {{e}}")
        return None

def test_tensorflow():
    print("\\nTesting TensorFlow API...")
    try:
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
        if '.' in api_name:
            parts = api_name.split('.')
            module = tf
            for part in parts[1:-1]:
                module = getattr(module, part)
            func = getattr(module, parts[-1])
        else:
            func = eval(api_name)
            
        args = []
        kwargs = {{}}
        
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
            print(f"TensorFlow kwargs error: {{e_kwargs}}")
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
        print(f"TensorFlow error: {{e}}")
        return None

if __name__ == "__main__":
    print(f"Reproducing Case {case_idx}: {torch_case['api']} vs {tf_case['api']}")
    torch_res = test_torch()
    tf_res = test_tensorflow()
"""
    
    file_path = os.path.join(output_dir, f"reproduce_other_error_case_{case_idx:02d}.py")
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
        cases = re.split(r'Sample \d+:', section)
        
        for case in cases:
            if not case.strip(): continue
            
            # Extract Error
            error_match = re.search(r'comparison_error: (.*)', case)
            if not error_match:
                continue
            error_msg = error_match.group(1).strip()
            
            # Filter: exclude Shape Mismatch and Value Mismatch
            if "Shape mismatch" in error_msg or "Values ​​do not match" in error_msg:
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
                pt_base = torch_api.split('.')[-1].lower()
                tf_base = tf_api.split('.')[-1].lower()
                
                # Remove common prefixes like 'reduce_' from TF
                tf_base_clean = tf_base.replace('reduce_', '')
                
                is_equivalent = False
                
                if pt_base == tf_base:
                    is_equivalent = True
                elif pt_base == tf_base_clean:
                    is_equivalent = True
                # Manual mappings
                elif (pt_base == 'sub' and tf_base == 'subtract') or \
                     (pt_base == 'mul' and tf_base == 'multiply') or \
                     (pt_base == 'div' and tf_base == 'divide') or \
                     (pt_base == 'true_divide' and tf_base == 'truediv') or \
                     (pt_base == 'absolute' and tf_base == 'abs') or \
                     (pt_base == 'abs' and tf_base == 'absolute') or \
                     (pt_base == 'pow' and tf_base == 'pow') or \
                     (pt_base == 'matmul' and tf_base == 'matmul'):
                     is_equivalent = True
                     
                # Skip addmm vs matmul
                if pt_base == 'addmm' and tf_base == 'matmul':
                    is_equivalent = False
                    
                if is_equivalent:
                    valid_cases.append({
                        'torch_case': torch_json,
                        'tf_case': tf_json,
                        'error_msg': error_msg
                    })
                        
            except Exception as e:
                # print(f"Error parsing case: {e}")
                continue
                
    return valid_cases

if __name__ == "__main__":
    cases = parse_report(r'd:\graduate\DFrameworkTest\pt_tf_test\analysis\comparison_error_samples_report.txt')
    print(f"Found {len(cases)} cases.")
    output_dir = r'd:\graduate\DFrameworkTest\pt_tf_test\simplest_test\other_error'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, case in enumerate(cases):
        print(f"--- Case {i+1} ---")
        print(f"Torch API: {case['torch_case']['api']}")
        print(f"TF API: {case['tf_case']['api']}")
        generate_script(i+1, case, output_dir)
