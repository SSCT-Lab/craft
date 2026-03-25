import json
import re
import os
import numpy as np

def generate_script(case_idx, case_data, output_dir):
    torch_case = case_data['torch_case']
    ms_case = case_data['ms_case']
    script_content = f"""
import torch
import mindspore as ms
import numpy as np

def get_input_data(shape, dtype, sample_values):
    if not shape:
        return np.array(sample_values[0], dtype=dtype)
    total_elements = np.prod(shape)
    if len(sample_values) < total_elements:
        repeats = int(np.ceil(total_elements / len(sample_values)))
        full_values = (sample_values * repeats)[:total_elements]
    else:
        full_values = sample_values[:total_elements]
    return np.array(full_values, dtype=dtype).reshape(shape)

def test_torch():
    inputs = {{}}
    if 'input' in {json.dumps(torch_case)}:
        input_info = {json.dumps(torch_case)}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {json.dumps(torch_case)}
    for k, v in params.items():
        if k == 'api' or k == 'input':
            continue
        if isinstance(v, dict) and 'shape' in v:
            inputs[k] = torch.tensor(get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            ), dtype=getattr(torch, v['dtype']))
        else:
            inputs[k] = v
    api_name = "{torch_case['api']}"
    try:
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
        args = []
        kwargs = {{}}
        if 'input' in inputs:
            args.append(inputs['input'])
        for k, v in inputs.items():
            if k == 'input':
                continue
            kwargs[k] = v
        result = func(*args, **kwargs)
        print("Torch result shape:", result.shape)
        return result.detach().numpy()
    except Exception as e:
        print(f"Torch error: {{e}}")
        return None

def test_mindspore():
    inputs = {{}}
    params = {json.dumps(ms_case)}
    for k, v in params.items():
        if k == 'api':
            continue
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = ms.Tensor(np_data, dtype=getattr(ms, v['dtype']))
        else:
            inputs[k] = v
    api_name = "{ms_case['api']}"
    try:
        parts = api_name.split('.')
        module = eval(parts[0])
        for part in parts[1:-1]:
            module = getattr(module, part)
        func = getattr(module, parts[-1])
        result = func(**inputs)
        print("MindSpore result shape:", result.shape)
        return result.asnumpy()
    except Exception as e:
        print(f"MindSpore error: {{e}}")
        try:
            print("Retrying with positional args...")
            result = func(*inputs.values())
            print("MindSpore result shape:", result.shape)
            return result.asnumpy()
        except Exception as e2:
            print(f"MindSpore retry error: {{e2}}")
            return None

if __name__ == "__main__":
    print(f"Reproducing Case {case_idx}: {torch_case['api']} vs {ms_case['api']}")
    torch_res = test_torch()
    ms_res = test_mindspore()
    if torch_res is not None and ms_res is not None:
        try:
            diff = np.abs(torch_res - ms_res)
            max_diff = np.max(diff)
            print(f"\\nMax difference: {{max_diff}}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {{e}}")
"""
    file_path = os.path.join(output_dir, f"reproduce_case_{case_idx:02d}.py")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"Generated {file_path}")

def parse_report(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    file_sections = content.split('================================================================================')
    valid_cases = []
    for section in file_sections:
        if not section.strip():
            continue
        cases = re.split(r'样例 \d+:', section)
        for case in cases:
            if 'comparison_error: 数值不匹配' not in case:
                continue
            try:
                torch_match = re.search(r'torch_test_case:\s*({.*?})\s*mindspore_test_case:', case, re.DOTALL)
                ms_match = re.search(r'mindspore_test_case:\s*({.*})', case, re.DOTALL)
                if not torch_match or not ms_match:
                    continue
                torch_json = json.loads(torch_match.group(1))
                ms_json = json.loads(ms_match.group(1))
                torch_api = torch_json.get('api', '')
                ms_api = ms_json.get('api', '')
                pt_base = torch_api.split('.')[-1].lower()
                ms_base = ms_api.split('.')[-1].lower()
                is_equivalent = False
                if pt_base == ms_base:
                    is_equivalent = True
                elif (pt_base == 'sub' and ms_base == 'subtract') or \
                     (pt_base == 'mul' and ms_base == 'multiply') or \
                     (pt_base == 'div' and ms_base == 'divide') or \
                     (pt_base == 'true_divide' and ms_base == 'truediv'):
                    is_equivalent = True
                if is_equivalent:
                    valid_cases.append({
                        'torch_case': torch_json,
                        'ms_case': ms_json,
                        'error_msg': case.split('\n')[1].strip()
                    })
            except Exception:
                continue
    return valid_cases

if __name__ == "__main__":
    cases = parse_report(r'd:\graduate\DFrameworkTest\pt_ms_test\analysis\comparison_error_samples_report.txt')
    print(f"Found {len(cases)} cases.")
    output_dir = r'd:\graduate\DFrameworkTest\pt_ms_test\simplest_test'
    os.makedirs(output_dir, exist_ok=True)
    for i, case in enumerate(cases):
        print(f"--- Case {i+1} ---")
        print(f"Torch API: {case['torch_case']['api']}")
        print(f"MindSpore API: {case['ms_case']['api']}")
        generate_script(i+1, case, output_dir)

