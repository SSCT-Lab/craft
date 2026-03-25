import re
import sys
from collections import Counter, defaultdict

def categorize(error: str) -> str:
    e = error.lower()
    if 'invalid combination of arguments' in e or ('got' in e and 'expected' in e and 'argument' in e):
        return '参数组合无效'
    if 'unexpected keyword argument' in e:
        return '意外关键字参数'
    if 'multiple values for argument' in e:
        return '参数重复赋值'
    if 'dimensions must be equal' in e or 'shapes' in e and ('must be' in e or 'incompatible' in e):
        return '形状不匹配'
    if 'size 0' in e or 'non-zero size' in e or 'empty tensor' in e:
        return '零维/空张量错误'
    if 'cannot convert' in e or 'expected dtype' in e or ('dtype' in e and ('invalid' in e or 'not' in e)):
        return 'dtype转换/类型错误'
    if 'not implemented' in e or 'attributeerror' in e:
        return '接口缺失/未实现'
    if 'invalidargumenterror' in e or 'failed' in e and 'convert' in e:
        return '运行时参数非法'
    return '其他'

def parse_errors(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    errors = []
    for line in text.splitlines():
        m = re.search(r'^\s*tensorflow_error:\s*(.+)$', line)
        if m:
            errors.append(m.group(1).strip())
    return errors

def main():
    path = r'd:\graduate\DFrameworkTest\pt_tf_test\analysis\tensorflow_error_samples_report.txt'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    errors = parse_errors(path)
    cat_counter = Counter()
    sample_messages = defaultdict(int)
    for e in errors:
        cat = categorize(e)
        cat_counter[cat] += 1
        sample_messages[e] += 1
    print(f'样例总数: {len(errors)}')
    print('按类别统计（降序）：')
    for cat, cnt in cat_counter.most_common():
        print(f'- {cat}: {cnt}')
    print('代表性错误消息（Top 10）：')
    for msg, cnt in sorted(sample_messages.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'- [{cnt}] {msg}')

if __name__ == '__main__':
    main()
