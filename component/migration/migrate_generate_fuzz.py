"""基于已迁移的测试生成 fuzzing 测试

从已迁移的 PyTorch 测试中提取输入模式，生成随机输入的 fuzzing 测试。
"""
import ast
import re
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from component.migration.migrate_generate_tests import safe_filename


def extract_input_patterns(pt_code: str) -> List[Dict]:
    """从 PyTorch 测试代码中提取输入模式
    
    返回:
        List[Dict]: 每个元素包含 {'type': 'tensor|scalar|list', 'shape': ..., 'dtype': ..., 'value': ...}
    """
    patterns = []
    
    try:
        tree = ast.parse(pt_code)
        
        for node in ast.walk(tree):
            # 查找 torch.randn, torch.rand, torch.zeros, torch.ones 等调用
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch':
                        func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                
                if func_name in ('randn', 'rand', 'zeros', 'ones', 'randint', 'full'):
                    # 提取 shape 参数
                    shape = None
                    dtype = None
                    
                    if node.args:
                        # 第一个参数通常是 shape
                        if isinstance(node.args[0], (ast.Tuple, ast.List)):
                            shape = tuple(
                                x.value if isinstance(x, ast.Constant) else None
                                for x in node.args[0].elts
                            )
                        elif isinstance(node.args[0], ast.Constant):
                            shape = (node.args[0].value,)
                    
                    # 查找 dtype 参数
                    for kw in node.keywords:
                        if kw.arg == 'dtype':
                            if isinstance(kw.value, ast.Attribute):
                                dtype = kw.value.attr if hasattr(kw.value, 'attr') else None
                    
                    patterns.append({
                        'type': 'tensor',
                        'generator': f'torch.{func_name}',
                        'shape': shape,
                        'dtype': dtype,
                        'node': node
                    })
                
                # 查找常量张量（如 torch.tensor([...])）
                elif func_name == 'tensor':
                    if node.args:
                        patterns.append({
                            'type': 'tensor_literal',
                            'generator': 'torch.tensor',
                            'value': _extract_literal_value(node.args[0]),
                            'node': node
                        })
            
            # 查找标量常量
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
                                    'node': node
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
            # 在 ±20% 范围内变异
            delta = max(1, dim // 5)
            mutated.append(max(1, dim + random.randint(-delta, delta)))
        else:
            mutated.append(dim)
    
    return tuple(mutated)


def mutate_scalar(value: float) -> float:
    """变异标量值"""
    if isinstance(value, int):
        delta = max(1, abs(value) // 5)
        return value + random.randint(-delta, delta)
    elif isinstance(value, float):
        return value * (1 + random.uniform(-0.2, 0.2))
    return value


def generate_fuzz_test(pt_test_file: Path, pt_test_func: str, patterns: List[Dict], 
                       num_variants: int = 5) -> str:
    """基于已迁移的 PyTorch 测试生成 fuzzing 测试"""
    
    # 读取原始测试代码
    source = pt_test_file.read_text(encoding='utf-8')
    
    # 提取 PyTorch 测试函数
    pt_func_match = re.search(rf'def\s+({pt_test_func})\s*\([^)]*\):.*?(?=def\s+|# =====|if __name__)', 
                             source, re.DOTALL)
    if not pt_func_match:
        return None
    
    pt_func_code = pt_func_match.group(0)
    
    fuzz_tests = []
    
    for variant_idx in range(num_variants):
        # 变异输入模式
        mutated_code = pt_func_code
        
        # 替换 tensor 生成调用
        for pattern in patterns:
            if pattern['type'] == 'tensor' and pattern.get('shape'):
                old_shape = str(pattern['shape'])
                new_shape = mutate_shape(pattern['shape'])
                
                # 替换 shape
                mutated_code = re.sub(
                    rf'torch\.{pattern["generator"].split(".")[-1]}\s*\(\s*{re.escape(old_shape)}\s*\)',
                    f'torch.{pattern["generator"].split(".")[-1]}({new_shape})',
                    mutated_code
                )
            
            elif pattern['type'] == 'scalar':
                old_value = pattern['value']
                new_value = mutate_scalar(old_value)
                mutated_code = mutated_code.replace(str(old_value), str(new_value))
        
        # 生成 fuzz 测试函数名
        fuzz_func_name = f"{pt_test_func}_fuzz_{variant_idx}"
        mutated_code = re.sub(rf'def\s+{pt_test_func}\s*\(', f'def {fuzz_func_name}(', mutated_code, count=1)
        
        fuzz_tests.append(mutated_code)
    
    # 组合所有 fuzz 测试
    fuzz_code = f"""# Fuzzing tests generated from {pt_test_file.name}
# Original test function: {pt_test_func}
# Generated {num_variants} variants with mutated inputs

import torch
import numpy as np

{chr(10).join(fuzz_tests)}

# Run all fuzz variants
if __name__ == "__main__":
    variants = [{', '.join([f"{pt_test_func}_fuzz_{i}" for i in range(num_variants)])}]
    
    for variant_func in variants:
        try:
            print(f"Running {{variant_func.__name__}}...")
            variant_func()
            print(f"  ✓ {{variant_func.__name__}} PASSED")
        except Exception as e:
            print(f"  ✗ {{variant_func.__name__}} FAILED: {{e}}")
            import traceback
            traceback.print_exc()
"""
    
    return fuzz_code


def process_migrated_test(pt_test_file: Path, num_variants: int = 5) -> Optional[str]:
    """处理一个已迁移的测试文件，生成 fuzzing 测试"""
    
    source = pt_test_file.read_text(encoding='utf-8')
    
    # 查找 PyTorch 测试函数（以 _pt 结尾）
    pt_func_match = re.search(r'def\s+(test_\w+_pt)\s*\([^)]*\):', source)
    if not pt_func_match:
        return None
    
    pt_test_func = pt_func_match.group(1)
    
    # 提取输入模式
    patterns = extract_input_patterns(source)
    
    if not patterns:
        print(f"[WARN] {pt_test_file.name}: 未找到输入模式")
        return None
    
    # 生成 fuzzing 测试
    fuzz_code = generate_fuzz_test(pt_test_file, pt_test_func, patterns, num_variants)
    
    return fuzz_code


def main():
    parser = argparse.ArgumentParser(description="基于已迁移测试生成 fuzzing 测试")
    parser.add_argument("--migrated-dir", default="migrated_tests", help="已迁移测试目录")
    parser.add_argument("--output-dir", default="migrated_tests_fuzz", help="Fuzzing 测试输出目录")
    parser.add_argument("--num-variants", type=int, default=5, help="每个测试生成的变体数量")
    parser.add_argument("--limit", type=int, default=-1, help="限制处理的测试数量，-1 表示全部")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    
    args = parser.parse_args()
    
    migrated_dir = Path(args.migrated_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not migrated_dir.exists():
        print(f"[ERROR] 已迁移测试目录不存在: {migrated_dir}")
        return
    
    # 获取所有已迁移的测试文件
    test_files = sorted(migrated_dir.glob("*.py"))
    
    if args.limit > 0:
        test_files = test_files[:args.limit]
    
    print(f"[INFO] 找到 {len(test_files)} 个已迁移的测试文件")
    
    success_count = 0
    failed_count = 0
    
    for pt_file in test_files:
        try:
            fuzz_code = process_migrated_test(pt_file, args.num_variants)
            
            if not fuzz_code:
                failed_count += 1
                continue
            
            # 生成输出文件名
            output_file = output_dir / f"{pt_file.stem}_fuzz.py"
            
            # 检查是否已存在
            if output_file.exists() and not args.force:
                print(f"[SKIP] {output_file.name} 已存在，跳过")
                continue
            
            # 写入文件
            output_file.write_text(fuzz_code, encoding='utf-8')
            print(f"[OK] 生成: {output_file.name}")
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] 处理 {pt_file.name} 失败: {e}")
            failed_count += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n[DONE] 成功: {success_count}, 失败: {failed_count}")
    print(f"[INFO] Fuzzing 测试已保存到: {output_dir}")


if __name__ == "__main__":
    main()

