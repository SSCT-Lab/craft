# 说明：
# 本代码用于从对比错误报告（comparison_error_samples_report.txt）中解析
# 具备“数值不匹配”类型、且 PyTorch 与 TensorFlow API 语义等价的样例，
# 并为每个样例自动生成一个最小可运行的复现脚本（reproduce_case_XX.py）。
#
# 核心流程：
# 1) parse_report：读取并解析报告文件，筛选出数值不匹配且 API 语义等价的案例；
# 2) generate_script：按解析出的 torch/tf 输入参数生成复现脚本内容并写入文件；
# 3) main：执行解析，打印统计信息，并批量生成复现脚本。

import json
import re
import os
import numpy as np

def generate_script(case_idx, case_data, output_dir):
    # 根据单个样例的 torch_case / tf_case 构造一份独立的复现脚本
    # 复现脚本中包含：
    # - 输入数据构造：根据 shape/dtype/sample_values 生成 numpy，再转 torch/tf 张量
    # - API 调用：通过字符串反射方式获取函数并调用
    # - 结果对比：打印形状，并计算 torch 与 tf 结果的最大差异
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

def test_torch():
    print("Testing torch API...")
    # Setup inputs
    inputs = {{}}
    
    # Handle main input
    if 'input' in {json.dumps(torch_case)}:
        input_info = {json.dumps(torch_case)}['input']
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
    params = {json.dumps(torch_case)}
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
            kwargs[k] = v
            
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
    params = {json.dumps(tf_case)}
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
        
        result = func(**inputs)
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
    
    file_path = os.path.join(output_dir, f"reproduce_case_{case_idx:02d}.py")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"Generated {file_path}")

def parse_report(file_path):
    # 解析报告文件，提取“数值不匹配”的案例，并判断 torch/tf API 是否语义等价
    # 返回结构：[{ 'torch_case': <dict>, 'tf_case': <dict>, 'error_msg': <str> }, ...]
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分隔线拆分各文件段
    file_sections = content.split('================================================================================')
    
    valid_cases = []
    
    for section in file_sections:
        if not section.strip():
            continue
            
        # 将当前文件段按“样例 N:”拆分
        cases = re.split(r'样例 \d+:', section)
        
        for case in cases:
            # 仅保留“数值不匹配”的对比错误样例
            if 'comparison_error: 数值不匹配' not in case:
                continue
                
            # 提取 torch 与 tensorflow 测试样例的 JSON 段
            try:
                torch_match = re.search(r'torch_test_case:\s*({.*?})\s*tensorflow_test_case:', case, re.DOTALL)
                tf_match = re.search(r'tensorflow_test_case:\s*({.*})', case, re.DOTALL)
                
                if not torch_match or not tf_match:
                    continue
                    
                torch_json = json.loads(torch_match.group(1))
                tf_json = json.loads(tf_match.group(1))
                
                torch_api = torch_json.get('api', '')
                tf_api = tf_json.get('api', '')
                
                # 清洗异常数据：若 TF API 以 torch. 开头，跳过
                if tf_api.startswith('torch.'):
                    continue
                    
                # 判断语义等价：采用启发式-基于函数名的“基名”匹配
                pt_base = torch_api.split('.')[-1].lower()
                tf_base = tf_api.split('.')[-1].lower()
                
                # 白名单：处理名称不同但语义相同的常见对，如 sub<->subtract、mul<->multiply 等
                is_equivalent = False
                
                if pt_base == tf_base:
                    is_equivalent = True
                elif (pt_base == 'sub' and tf_base == 'subtract') or \
                     (pt_base == 'mul' and tf_base == 'multiply') or \
                     (pt_base == 'div' and tf_base == 'divide') or \
                     (pt_base == 'true_divide' and tf_base == 'truediv'):
                     is_equivalent = True
                     
                # 特例：用户要求严格语义相同，跳过 addmm 与 matmul
                if pt_base == 'addmm' and tf_base == 'matmul':
                    is_equivalent = False
                    
                if is_equivalent:
                    # 记录有效样例，包括两侧的 JSON 及错误消息
                    valid_cases.append({
                        'torch_case': torch_json,
                        'tf_case': tf_json,
                        'error_msg': case.split('\n')[1].strip()
                    })
                    
                    # Process all cases, so no early return here
                    # if len(valid_cases) >= 10:
                    #     return valid_cases
                        
            except Exception as e:
                # print(f"Error parsing case: {e}")
                continue
                
    return valid_cases

if __name__ == "__main__":
    # 主入口：解析报告并批量生成复现脚本
    cases = parse_report(r'd:\graduate\DFrameworkTest\pt_tf_test\analysis\comparison_error_samples_report.txt')
    print(f"Found {len(cases)} cases.")
    output_dir = r'd:\graduate\DFrameworkTest\pt_tf_test\simplest_test'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, case in enumerate(cases):
        print(f"--- Case {i+1} ---")
        print(f"Torch API: {case['torch_case']['api']}")
        print(f"TF API: {case['tf_case']['api']}")
        generate_script(i+1, case, output_dir)
