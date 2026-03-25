"""对 TF seed 进行 fuzzing，生成变体测试

基于合规的 TF seed 测试，通过变异输入参数（tensor shapes, scalar values）生成 fuzzing 变体。
"""
import json
import ast
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from component.migration.dev_extract_tf_core import TF_HEADER, safe_name


def extract_tf_input_patterns(tf_code: str) -> List[Dict]:
    """从 TF 测试代码中提取输入模式
    
    返回:
        List[Dict]: 每个元素包含 {'type': 'tensor|scalar|list', 'shape': ..., 'dtype': ..., 'value': ...}
    """
    patterns = []
    
    try:
        tree = ast.parse(tf_code)
        
        for node in ast.walk(tree):
            # 查找 tf.constant, tf.zeros, tf.ones 等调用
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'tf':
                        func_name = node.func.attr
                
                if func_name in ('constant', 'zeros', 'ones', 'fill', 'random_normal', 'random_uniform'):
                    # 提取 shape/value 参数
                    shape = None
                    value = None
                    dtype = None
                    
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, (ast.List, ast.Tuple)):
                            # 可能是 shape 或 value
                            try:
                                # 尝试解析为 shape（整数列表）
                                shape_vals = []
                                for elt in first_arg.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                                        shape_vals.append(int(elt.value))
                                    elif isinstance(elt, ast.Constant) and isinstance(elt.value, list):
                                        # 嵌套列表，可能是 value
                                        value = _extract_literal_value(first_arg)
                                        break
                                if not value:
                                    shape = tuple(shape_vals) if shape_vals else None
                            except:
                                value = _extract_literal_value(first_arg)
                        elif isinstance(first_arg, ast.Constant):
                            value = first_arg.value
                    
                    # 查找 dtype 参数
                    for kw in node.keywords:
                        if kw.arg == 'dtype':
                            if isinstance(kw.value, ast.Attribute):
                                dtype = kw.value.attr if hasattr(kw.value, 'attr') else None
                    
                    patterns.append({
                        'type': 'tensor',
                        'generator': f'tf.{func_name}',
                        'shape': shape,
                        'value': value,
                        'dtype': dtype,
                        'node': node,
                        'line': node.lineno
                    })
            
            # 查找标量常量赋值
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant):
                            value = node.value.value
                            if isinstance(value, (int, float)):
                                patterns.append({
                                    'type': 'scalar',
                                    'name': target.id,
                                    'value': value,
                                    'node': node,
                                    'line': node.lineno
                                })
                        elif isinstance(node.value, (ast.List, ast.Tuple)):
                            # 列表/元组常量
                            list_value = _extract_literal_value(node.value)
                            if list_value:
                                patterns.append({
                                    'type': 'list',
                                    'name': target.id,
                                    'value': list_value,
                                    'node': node,
                                    'line': node.lineno
                                })
    
    except Exception as e:
        print(f"[WARN] 解析代码失败: {e}")
    
    return patterns


def _extract_literal_value(node) -> Optional:
    """提取字面量值"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, (ast.List, ast.Tuple)):
        return [_extract_literal_value(x) for x in node.elts]
    return None


def mutate_shape(shape: Tuple) -> Tuple:
    """变异 shape，生成相似的随机 shape"""
    if not shape:
        return shape
    
    mutated = []
    for dim in shape:
        if dim is None:
            mutated.append(None)
        elif isinstance(dim, int) and dim > 0:
            # 在 ±30% 范围内变异，但至少为 1
            delta = max(1, dim // 3)
            mutated.append(max(1, dim + random.randint(-delta, delta)))
        else:
            mutated.append(dim)
    
    return tuple(mutated)


def mutate_scalar(value: float) -> float:
    """变异标量值"""
    if isinstance(value, int):
        delta = max(1, abs(value) // 3) if abs(value) > 0 else 1
        return value + random.randint(-delta, delta)
    elif isinstance(value, float):
        return value * (1 + random.uniform(-0.3, 0.3))
    return value


def mutate_list(value: List) -> List:
    """变异列表值（对每个元素进行标量变异）"""
    if not isinstance(value, list):
        return value
    
    mutated = []
    for item in value:
        if isinstance(item, (int, float)):
            mutated.append(mutate_scalar(item))
        elif isinstance(item, list):
            mutated.append(mutate_list(item))
        else:
            mutated.append(item)
    
    return mutated


def generate_fuzz_variant(tf_code: str, patterns: List[Dict], variant_idx: int) -> str:
    """基于 TF 代码和输入模式生成一个 fuzzing 变体"""
    lines = tf_code.split('\n')
    mutated_lines = lines.copy()
    
    # 按行号倒序处理，避免行号变化影响
    patterns_by_line = sorted(patterns, key=lambda p: p.get('line', 0), reverse=True)
    
    for pattern in patterns_by_line:
        line_idx = pattern.get('line', 0) - 1
        if line_idx < 0 or line_idx >= len(mutated_lines):
            continue
        
        original_line = mutated_lines[line_idx]
        mutated_line = original_line
        
        if pattern['type'] == 'tensor' and pattern.get('shape'):
            # 变异 shape
            old_shape = pattern['shape']
            new_shape = mutate_shape(old_shape)
            # 替换 shape（简单字符串替换）
            old_shape_str = str(old_shape).replace(' ', '')
            new_shape_str = str(new_shape).replace(' ', '')
            mutated_line = mutated_line.replace(old_shape_str, new_shape_str)
        
        elif pattern['type'] == 'scalar':
            # 变异标量值
            old_value = pattern['value']
            new_value = mutate_scalar(old_value)
            # 只替换完整的值（避免替换部分数字）
            old_str = str(old_value)
            new_str = str(new_value)
            # 使用单词边界确保完整替换
            mutated_line = re.sub(rf'\b{re.escape(old_str)}\b', new_str, mutated_line)
        
        elif pattern['type'] == 'list':
            # 变异列表值
            old_value = pattern['value']
            new_value = mutate_list(old_value)
            old_str = str(old_value).replace(' ', '')
            new_str = str(new_value).replace(' ', '')
            mutated_line = mutated_line.replace(old_str, new_str)
        
        mutated_lines[line_idx] = mutated_line
    
    return '\n'.join(mutated_lines)


def generate_fuzz_variant_file(seed_file: Path, variant_idx: int, tf_func_name: str, tf_func_code: str, patterns: List[Dict]) -> Optional[str]:
    """为单个变体生成独立的文件内容"""
    try:
        # 生成变体代码
        mutated_code = generate_fuzz_variant(tf_func_code, patterns, variant_idx)
        # 修改函数名
        fuzz_func_name = f"{tf_func_name}_fuzz_{variant_idx}"
        mutated_code = re.sub(rf'def\s+{tf_func_name}\s*\(', f'def {fuzz_func_name}(', mutated_code, count=1)
        
        # 生成单个变体的文件内容
        fuzz_code = f"""{TF_HEADER}
# Fuzzing test variant {variant_idx} generated from {seed_file.name}
# Original test function: {tf_func_name}
# Variant index: {variant_idx}

{mutated_code}

def main():
    \"\"\"运行 fuzzing 变体 {variant_idx}\"\"\"
    from pprint import pprint
    out = capture_output({fuzz_func_name})
    print('\\n' + '=' * 80)
    print(f'Fuzzing 变体 {variant_idx} 执行结果')
    print('=' * 80)
    pprint(out)
    if out['success']:
        print(f"\\n✓ {fuzz_func_name}: PASS")
    else:
        print(f"\\n✗ {fuzz_func_name}: FAIL")
        if out.get('exception'):
            print(f"错误: {{out['exception']}}")

if __name__ == '__main__':
    main()
"""
        
        return fuzz_code
    
    except Exception as e:
        print(f"[ERROR] 生成变体 {variant_idx} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_tf_function_from_seed(seed_file: Path) -> Optional[tuple]:
    """从 seed 文件中提取 TF 函数信息"""
    try:
        tf_code = seed_file.read_text(encoding='utf-8')
        
        # 提取 TF 函数代码（找到 tf_xxx 函数）
        tf_func_match = re.search(r'def\s+(tf_\w+)\s*\([^)]*\):.*?(?=def\s+main|if __name__)', tf_code, re.DOTALL)
        if not tf_func_match:
            return None
        
        tf_func_name = tf_func_match.group(1)
        tf_func_code = tf_func_match.group(0)
        
        # 提取输入模式
        patterns = extract_tf_input_patterns(tf_func_code)
        
        return tf_func_name, tf_func_code, patterns
    
    except Exception as e:
        print(f"[ERROR] 提取 {seed_file.name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="对 TF seed 进行 fuzzing，生成变体测试")
    parser.add_argument("--seeds-file", default="dev/tf_seeds.jsonl", help="Seed 列表文件")
    parser.add_argument("--output-dir", default="dev/tf_fuzz", help="Fuzzing 测试输出目录")
    parser.add_argument("--num-variants", type=int, default=5, help="每个 seed 生成的变体数量")
    parser.add_argument("--limit", type=int, default=-1, help="限制处理的 seed 数量，-1 表示全部")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    args = parser.parse_args()
    
    seeds_file = Path(args.seeds_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not seeds_file.exists():
        print(f"[ERROR] Seed 文件不存在: {seeds_file}")
        return
    
    # 加载 seeds
    seeds = []
    for line in seeds_file.open():
        seeds.append(json.loads(line))
    
    if args.limit > 0:
        seeds = seeds[:args.limit]
    
    print(f"[INFO] 找到 {len(seeds)} 个 seed")
    print(f"[INFO] 每个 seed 生成 {args.num_variants} 个变体")
    
    success_count = 0
    failed_count = 0
    
    # 生成 fuzzing 索引
    fuzz_index = []
    
    for seed in seeds:
        core_file = Path(seed["core_file"])
        if not core_file.exists():
            print(f"[WARN] Seed 文件不存在: {core_file}")
            failed_count += 1
            continue
        
        try:
            # 提取 TF 函数信息
            func_info = extract_tf_function_from_seed(core_file)
            if not func_info:
                print(f"[WARN] {core_file.name}: 无法提取 TF 函数")
                failed_count += 1
                continue
            
            tf_func_name, tf_func_code, patterns = func_info
            
            if not patterns:
                print(f"[WARN] {core_file.name}: 未找到输入模式")
                failed_count += 1
                continue
            
            # 生成输出文件名前缀
            seed_name = seed.get("core_name", core_file.stem)
            
            # 为每个变体生成单独的文件
            seed_success = 0
            for variant_idx in range(args.num_variants):
                # 生成单个变体的文件
                fuzz_code = generate_fuzz_variant_file(
                    core_file, variant_idx, tf_func_name, tf_func_code, patterns
                )
                
                if not fuzz_code:
                    continue
                
                # 生成输出文件名（每个变体一个文件）
                output_file = output_dir / f"{seed_name}_fuzz_{variant_idx}.py"
                
                # 检查是否已存在
                if output_file.exists() and not args.force:
                    print(f"[SKIP] {output_file.name} 已存在，跳过")
                    continue
                
                # 写入文件
                output_file.write_text(fuzz_code, encoding='utf-8')
                seed_success += 1
                
                # 记录到索引
                fuzz_index.append({
                    "seed_file": str(core_file),
                    "seed_name": seed_name,
                    "fuzz_file": str(output_file),
                    "variant_idx": variant_idx,
                    "source_file": seed.get("source_file", ""),
                    "test_name": seed.get("test_name", ""),
                })
            
            if seed_success > 0:
                print(f"[OK] {core_file.name}: 生成 {seed_success}/{args.num_variants} 个变体")
                success_count += seed_success
            else:
                failed_count += 1
        
        except Exception as e:
            print(f"[ERROR] 处理 {core_file.name} 失败: {e}")
            failed_count += 1
            import traceback
            traceback.print_exc()
    
    # 保存 fuzzing 索引
    fuzz_index_file = output_dir / "tf_fuzz_index.jsonl"
    with open(fuzz_index_file, "w") as f:
        for item in fuzz_index:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n[DONE] 成功: {success_count}, 失败: {failed_count}")
    print(f"[INFO] Fuzzing 测试已保存到: {output_dir}")
    print(f"[INFO] Fuzzing 索引已保存到: {fuzz_index_file}")


if __name__ == "__main__":
    main()

