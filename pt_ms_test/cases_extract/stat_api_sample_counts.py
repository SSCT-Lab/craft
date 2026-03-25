import re
import sys

def parse_counts(text):
    api_counts = {}
    current_api = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^文件:\s*llm_enhanced_(.+?)_(\d{8})_(\d{6})\.json$', line)
        if m:
            current_api = m.group(1)
            if current_api not in api_counts:
                api_counts[current_api] = 0
            continue
        if current_api and re.match(r'^样例\s+\d+:', line):
            api_counts[current_api] += 1
    return api_counts

def main():
    # path = r'd:\graduate\DFrameworkTest\pt_ms_test\analysis\comparison_error_samples_report.txt'
    # path = r'd:\graduate\DFrameworkTest\pt_ms_test\analysis\torch_error_samples_report.txt'
    path = r'd:\graduate\DFrameworkTest\pt_ms_test\analysis\mindspore_error_samples_report.txt'
    if len(sys.argv) > 1:
        path = sys.argv[1]

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    counts = parse_counts(text)
    print(f'统计文件: {path}')
    print(f'API总数: {len(counts)}')
    print('API\t样例总数')
    for api, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f'{api}\t{cnt}')

if __name__ == '__main__':
    main()
