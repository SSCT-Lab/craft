# ./component/doc_analyzer.py
"""Analyze test issues with an LLM and official documentation."""
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from component.doc.doc_crawler_factory import get_doc_content, detect_framework
from component.migration.migrate_generate_tests import get_qwen_client, load_api_key

DEFAULT_MODEL = "qwen-flash"
DEFAULT_KEY_PATH = "aliyun.key"


def build_analysis_prompt(
    error_message: str,
    tf_code: str,
    pt_code: str,
    tf_docs: List[str],
    pt_docs: List[str],
    tf_output: Optional[str] = None,
    pt_output: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
        """Build the analysis prompt (task / TF info / PT info / evaluation)."""
    
        tf_docs_text = "\n\n".join(tf_docs) if tf_docs else "No relevant TensorFlow docs found"
        pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "No relevant PyTorch docs found"
    
        prompt = f"""You are a senior deep learning framework expert who analyzes differences between TensorFlow and PyTorch.

[Task]
We have a set of unit tests migrated from TensorFlow to PyTorch. Based on the code, input/output info,
and official docs, determine whether the observed exception or behavior difference is "expected"
(e.g., framework semantic differences, numerical precision, boundary behaviors), or due to migration bugs.

[Overall Error / Meta]
Below is a compact summary of the overall status and comparison meta for this case:
{error_message}

----------------------------------------
[TensorFlow Test Info]

1. Code (TF-side test logic):
```python
{tf_code}
```

2. Typical inputs / usage (if inferable from code or docs, explain clearly in analysis):
- Based on the TF code and docs above, summarize likely inputs (shape, dtype, value range, etc.).

3. Official docs (TensorFlow):
{tf_docs_text}

4. Actual output / behavior (TensorFlow):
```text
{tf_output or "(TF execution log not provided; infer expected behavior from code and docs)"}
```

----------------------------------------
[PyTorch Test Info]

1. Code (migrated PT test logic):
```python
{pt_code}
```

2. Typical inputs / usage:
- Based on the PyTorch code and docs, summarize the inputs used (shape, dtype, value range, etc.),
    and whether they align with the TensorFlow side.

3. Official docs (PyTorch):
{pt_docs_text}

4. Actual output / behavior (PyTorch):
```text
{pt_output or "(PT execution log not provided; infer behavior from code and docs)"}
```
"""
    
    if context:
        prompt += f"""
----------------------------------------
[Additional Context]
{context}
"""
    
    prompt += """
----------------------------------------
[Evaluation Requirements]
Based on the above, provide a rigorous technical assessment and answer:

1. Is the behavior expected?
     - Can the TF/PT difference (or PT exception) be considered an expected framework difference
         (e.g., API semantics, default parameters, numeric behavior)?
     - If it is expected, clearly describe the difference, cite relevant doc sections (natural language
         summary is fine), and whether tests should explicitly tolerate the difference.

2. Is there a migration error?
     - If you believe the migration implementation is wrong (e.g., input mismatch, shape/dtype issues,
         incorrect assertions), point to the specific location (key code snippet is enough) and explain
         the correct migration approach.

3. Conclusion label (explicitly include in your answer):
     - Provide a one-sentence overall judgment, e.g.:
         - "Conclusion: expected difference; no code changes needed; explain in report"
         - "Conclusion: migration error; PyTorch-side code needs changes"
         - "Conclusion: insufficient information; need more logs or docs"
"""
    
    return prompt


def analyze_with_llm(
    client,
    error_message: str,
    tf_code: str,
    pt_code: str,
    tf_apis: List[str],
    pt_apis: List[str],
    tf_output: Optional[str] = None,
    pt_output: Optional[str] = None,
    context: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Analyze the issue using an LLM."""
    
    # Crawl related docs
    print("[INFO] Crawling docs...")
    tf_docs = []
    pt_docs = []
    
    for api in tf_apis[:5]:  # Crawl up to 5 related APIs
        try:
            doc_content = get_doc_content(api, "tensorflow")
            if doc_content and "Unable to fetch" not in doc_content:
                tf_docs.append(doc_content)
        except Exception as e:
            print(f"[WARN] Failed to crawl TF docs {api}: {e}")
    
    for api in pt_apis[:5]:
        try:
            doc_content = get_doc_content(api, "pytorch")
            if doc_content and "Unable to fetch" not in doc_content:
                pt_docs.append(doc_content)
        except Exception as e:
            print(f"[WARN] Failed to crawl PT docs {api}: {e}")
    
    print(f"[INFO] Retrieved {len(tf_docs)} TF docs, {len(pt_docs)} PT docs")
    
    # Build prompt (separate TF/PT outputs)
    prompt = build_analysis_prompt(
        error_message=error_message,
        tf_code=tf_code,
        pt_code=pt_code,
        tf_docs=tf_docs,
        pt_docs=pt_docs,
        tf_output=tf_output,
        pt_output=pt_output,
        context=context,
    )
    
    # Call LLM
    try:
        if hasattr(client, 'chat'):
            # New version
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            analysis = resp.choices[0].message.content.strip()
        else:
            # Legacy version
            resp = client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            analysis = resp.choices[0].message.content.strip()
        
        return analysis
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return None


def analyze_test_error(
    error_message: str,
    test_file: str,
    tf_apis: Optional[List[str]] = None,
    pt_apis: Optional[List[str]] = None,
    tf_output: Optional[str] = None,
    pt_output: Optional[str] = None,
    context: Optional[str] = None,
) -> Optional[str]:
    """Analyze test error."""
    
    # Read test file
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"[ERROR] Test file does not exist: {test_file}")
        return None
    
    test_content = test_path.read_text(encoding='utf-8')
    
    # Extract TF and PT code (simple approach, can be improved)
    import re
    
    # Extract TF test function
    tf_match = re.search(r'def\s+(test\w+)\(\):.*?(?=def\s+test.*?_pt|# ===== PyTorch|if __name__)', 
                         test_content, re.DOTALL)
    tf_code = tf_match.group(0) if tf_match else ""
    
    # Extract PT test function
    pt_match = re.search(r'def\s+(test.*?_pt)\(\):.*?(?=def\s+test.*?|# ===== Main|if __name__)', 
                         test_content, re.DOTALL)
    pt_code = pt_match.group(0) if pt_match else ""
    
    # If API lists are not provided, try extracting from code
    if not tf_apis:
        tf_apis = re.findall(r'tf\.\w+(?:\.\w+)*', tf_code)
    if not pt_apis:
        pt_apis = re.findall(r'torch\.\w+(?:\.\w+)*', pt_code)
    
    # Initialize LLM client
    try:
        client = get_qwen_client(DEFAULT_KEY_PATH)
    except Exception as e:
        print(f"[ERROR] Unable to initialize LLM client: {e}")
        return None
    
    # Run analysis (include TF/PT output snippets)
    return analyze_with_llm(
        client=client,
        error_message=error_message,
        tf_code=tf_code,
        pt_code=pt_code,
        tf_apis=tf_apis,
        pt_apis=pt_apis,
        tf_output=tf_output,
        pt_output=pt_output,
        context=context,
    )


def main():
    """CLI tool."""
    parser = argparse.ArgumentParser(description="Analyze migration test issues")
    parser.add_argument("error", help="Error message")
    parser.add_argument("--test-file", "-t", required=True, help="Test file path")
    parser.add_argument("--tf-apis", nargs="+", help="TensorFlow API list")
    parser.add_argument("--pt-apis", nargs="+", help="PyTorch API list")
    parser.add_argument("--context", "-c", help="Additional context")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH, help="API key path")
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    # Analyze
    analysis = analyze_test_error(
        args.error,
        args.test_file,
        args.tf_apis,
        args.pt_apis,
        args.context
    )
    
    if analysis:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"[SUCCESS] Analysis saved to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("Analysis Result")
            print("=" * 80)
            print(analysis)
    else:
        print("[ERROR] Analysis failed")


if __name__ == "__main__":
    main()

