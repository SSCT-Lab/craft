import traceback

import numpy as np
import torch
import paddle

PD_TEST_CASE = {'api': 'paddle.hypot',
 'x': {'dtype': 'float16', 'shape': [0, 3, 4]},
 'y': {'dtype': 'float16', 'shape': [1, 3, 1]}}
OTHER_TEST_CASE = {'api': 'torch.hypot',
 'x': {'dtype': 'float16', 'shape': [0, 3, 4]},
 'y': {'dtype': 'float16', 'shape': [1, 3, 1]}}


def _np_from_spec(spec):
    shape = tuple(spec.get("shape", []))
    dtype = spec.get("dtype", "float32")

    if dtype in {"bool", "bool_"}:
        return np.zeros(shape, dtype=np.bool_)
    if "int" in dtype or "uint" in dtype:
        return np.ones(shape, dtype=np.dtype(dtype))
    if "complex" in dtype:
        real = np.ones(shape, dtype=np.float32)
        imag = np.ones(shape, dtype=np.float32) * 0.5
        return (real + 1j * imag).astype(np.dtype(dtype))
    return np.ones(shape, dtype=np.dtype(dtype))


def _is_tensor_spec(value):
    return isinstance(value, dict) and "shape" in value and "dtype" in value


def _to_paddle(value):
    if _is_tensor_spec(value):
        arr = _np_from_spec(value)
        return paddle.to_tensor(arr, dtype=value["dtype"])
    if isinstance(value, list):
        return [_to_paddle(item) for item in value]
    return value


def _other_dtype(dtype_name: str):
    name = dtype_name.replace("torch.", "")
    return getattr(torch, name)


def _to_other(value):
    if _is_tensor_spec(value):
        arr = _np_from_spec(value)
        return torch.tensor(arr, dtype=_other_dtype(value["dtype"]))
    if isinstance(value, list):
        return [_to_other(item) for item in value]
    if isinstance(value, str) and (value.startswith("torch.") or value.startswith("mindspore.") or value.startswith("tf.") or value.startswith("tensorflow.")):
        return _other_dtype(value)
    return value


def _materialize(value):
    if isinstance(value, paddle.Tensor):
        _ = value.numpy()
        return
    if isinstance(value, torch.Tensor):
        _ = value.detach().cpu().numpy()
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _materialize(item)


def _resolve_callable(api_path: str):
    normalized = api_path
    if normalized.startswith("tf."):
        normalized = "tensorflow." + normalized[3:]
    parts = normalized.split(".")
    obj = __import__(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def _run_case(test_case, converter):
    api_name = test_case["api"]
    fn = _resolve_callable(api_name)

    kwargs = {}
    try:
        for key, value in test_case.items():
            if key == "api":
                continue
            kwargs[key] = converter(value)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    errors = []

    def _try(*args, **inner_kwargs):
        try:
            out = fn(*args, **inner_kwargs)
            _materialize(out)
            return True, ""
        except Exception as exc:
            errors.append(exc)
            return False, ""

    ok, _ = _try(**kwargs)
    if ok:
        return True, ""


    if api_name.startswith("torch."):
        alias = dict(kwargs)
        if "axis" in alias and "dim" not in alias:
            alias["dim"] = alias.pop("axis")
        if "x" in alias and "input" not in alias:
            alias["input"] = alias["x"]
        if "y" in alias and "other" not in alias:
            alias["other"] = alias["y"]
        ok, _ = _try(**alias)
        if ok:
            return True, ""

    if "input" in kwargs and "x" in kwargs and "y" in kwargs:
        rest = {k: v for k, v in kwargs.items() if k not in {"input", "x", "y"}}
        ok, _ = _try(kwargs["input"], kwargs["x"], kwargs["y"], **rest)
        if ok:
            return True, ""

    if "x" in kwargs and "y" in kwargs and "weight" in kwargs:
        rest = {k: v for k, v in kwargs.items() if k not in {"x", "y", "weight"}}
        ok, _ = _try(kwargs["x"], kwargs["y"], kwargs["weight"], **rest)
        if ok:
            return True, ""

    if "x" in kwargs and "y" in kwargs:
        rest = {k: v for k, v in kwargs.items() if k not in {"x", "y"}}
        ok, _ = _try(kwargs["x"], kwargs["y"], **rest)
        if ok:
            return True, ""

    if "x" in kwargs:
        rest = {k: v for k, v in kwargs.items() if k != "x"}
        ok, _ = _try(kwargs["x"], **rest)
        if ok:
            return True, ""

    if "input" in kwargs:
        rest = {k: v for k, v in kwargs.items() if k != "input"}
        ok, _ = _try(kwargs["input"], **rest)
        if ok:
            return True, ""

    if errors:
        last = errors[-1]
        return False, f"{type(last).__name__}: {last}"
    return False, "UnknownError: no invocation path matched"


def main():
    print("Candidate 133: paddle.hypot vs torch.hypot")
    print(f"Paddle version: {paddle.__version__}")
    print(f"Other framework version (torch): {torch.__version__}")

    pd_ok, pd_err = _run_case(PD_TEST_CASE, _to_paddle)
    other_ok, other_err = _run_case(OTHER_TEST_CASE, _to_other)

    if pd_ok:
        print("[PD] success")
    else:
        print("[PD] error")
        print(pd_err)

    if other_ok:
        print("[OTHER] success")
    else:
        print("[OTHER] error")
        print(other_err)

    reproduced = (not pd_ok) and other_ok
    print(f"REPRODUCED_PADDLE_ERROR_ONLY={reproduced}")


if __name__ == "__main__":
    main()
