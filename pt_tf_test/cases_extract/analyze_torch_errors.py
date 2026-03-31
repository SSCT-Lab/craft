import re
import sys
from collections import Counter, defaultdict

def categorize(error: str) -> str:
    e = error.lower()
    if 'invalid combination of arguments' in e:
        return 'Invalid parameter combination'
    if 'unexpected keyword argument' in e:
        return 'Unexpected keyword argument'
    if 'multiple values for argument' in e:
        return 'Repeated assignment of parameters'
    if 'must match the size of tensor' in e or 'size of tensor' in e or 'sizes of tensors' in e:
        return 'Shape mismatch'
    if 'non-zero size' in e or 'size 0' in e:
        return 'Zero-dimensional reduction error'
    if 'expected scalar type' in e or 'cannot convert' in e or 'dtype' in e and 'expected' in e:
        return 'dtypeConvert/type error'
    if 'received an invalid' in e or 'got' in e and 'expected' in e:
        return 'Signature does not match/Parameter error'
    return 'other'

def parse_errors(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    errors = []
    api_files = []
    for line in text.splitlines():
        m = re.match(r'^\s*document:\s*llm_enhanced_(.+?)_(\d{8})_(\d{6})\.json', line)
        if m:
            api_files.append(m.group(1))
            continue
        m2 = re.search(r'^\s*torch_error:\s*(.+)$', line)
        if m2:
            errors.append(m2.group(1).strip())
    return errors, api_files

def main():
    path = r'd:\graduate\DFrameworkTest\pt_tf_test\analysis\torch_error_samples_report.txt'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    errors, api_files = parse_errors(path)
    cat_counter = Counter()
    sample_messages = defaultdict(int)
    for e in errors:
        cat = categorize(e)
        cat_counter[cat] += 1
        sample_messages[e] += 1
    print(f'Analyze files: {path}')
    print(f'Total number of samples: {len(errors)}')
    print('\nStatistics by category (descending order）：')
    for cat, cnt in cat_counter.most_common():
        print(f'- {cat}: {cnt}')
    print('\nRepresentative error message（Top 10）：')
    for msg, cnt in sorted(sample_messages.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'- [{cnt}] {msg}')
        
if __name__ == '__main__':
    main()
