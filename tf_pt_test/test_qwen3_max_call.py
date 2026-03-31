import os
from pathlib import Path

from openai import OpenAI


def load_api_key() -> tuple[str | None, str]:
    """Load API key with priority: project aliyun.key > env var DASHSCOPE_API_KEY."""
    project_root = Path(__file__).resolve().parents[1]
    key_file = project_root / "aliyun.key"

    if key_file.exists():
        key_value = key_file.read_text(encoding="utf-8").strip()
        if key_value:
            return key_value, f"file:{key_file}"

    env_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if env_key:
        return env_key, "env:DASHSCOPE_API_KEY"

    return None, "none"


def test_qwen3_max_call() -> int:
    api_key, key_source = load_api_key()

    if not api_key:
        print("[ERROR] No available API key found.")
        print("[HINT] Create aliyun.key in the project root or set DASHSCOPE_API_KEY.")
        return 1

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please tell me whether you are the qwen3-max model or the qwen3.5-plus model."},
            ],
            timeout=60,
        )

        content = completion.choices[0].message.content if completion.choices else ""
        print("[OK] qwen3-max call succeeded")
        print(f"[INFO] Key source: {key_source}")
        print("[RESPONSE]", content)
        return 0

    except Exception as error:
        print("[ERROR] qwen3-max call failed")
        print(f"[INFO] Key source: {key_source}")
        print(f"Error: {error}")
        print("See docs: https://help.aliyun.com/model-studio/developer-reference/error-code")
        return 2


if __name__ == "__main__":
    raise SystemExit(test_qwen3_max_call())
