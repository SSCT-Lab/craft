
import torch
import paddle
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
    inputs = {}
    if 'input' in {"api": "torch.nn.functional.triplet_margin_loss", "anchor": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48905149102211, 1.2753400802612305, -1.2169709205627441, -0.35232681035995483, 0.24103747308254242, 0.6245335936546326, -0.3956297039985657, -0.22465163469314575, 2.5373735427856445, -0.5747144818305969]}, "positive": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [1.717810034751892, -0.1272154450416565, -1.3185163736343384, -0.45896482467651367, 1.1904960870742798, -0.4216008484363556, 0.24884384870529175, -2.2942800521850586, -1.3392122983932495, -0.6697834730148315]}, "negative": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48965904116630554, 1.3440203666687012, -0.3458004891872406, 0.9071406722068787, 0.2800755798816681, -1.6802493333816528, -1.1936050653457642, -0.07833614945411682, -0.6536622047424316, 1.656090497970581]}, "margin": 0.5, "p": 1, "eps": 1e-07, "swap": true, "reduction": "sum"}:
        input_info = {"api": "torch.nn.functional.triplet_margin_loss", "anchor": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48905149102211, 1.2753400802612305, -1.2169709205627441, -0.35232681035995483, 0.24103747308254242, 0.6245335936546326, -0.3956297039985657, -0.22465163469314575, 2.5373735427856445, -0.5747144818305969]}, "positive": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [1.717810034751892, -0.1272154450416565, -1.3185163736343384, -0.45896482467651367, 1.1904960870742798, -0.4216008484363556, 0.24884384870529175, -2.2942800521850586, -1.3392122983932495, -0.6697834730148315]}, "negative": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48965904116630554, 1.3440203666687012, -0.3458004891872406, 0.9071406722068787, 0.2800755798816681, -1.6802493333816528, -1.1936050653457642, -0.07833614945411682, -0.6536622047424316, 1.656090497970581]}, "margin": 0.5, "p": 1, "eps": 1e-07, "swap": true, "reduction": "sum"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.triplet_margin_loss", "anchor": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48905149102211, 1.2753400802612305, -1.2169709205627441, -0.35232681035995483, 0.24103747308254242, 0.6245335936546326, -0.3956297039985657, -0.22465163469314575, 2.5373735427856445, -0.5747144818305969]}, "positive": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [1.717810034751892, -0.1272154450416565, -1.3185163736343384, -0.45896482467651367, 1.1904960870742798, -0.4216008484363556, 0.24884384870529175, -2.2942800521850586, -1.3392122983932495, -0.6697834730148315]}, "negative": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48965904116630554, 1.3440203666687012, -0.3458004891872406, 0.9071406722068787, 0.2800755798816681, -1.6802493333816528, -1.1936050653457642, -0.07833614945411682, -0.6536622047424316, 1.656090497970581]}, "margin": 0.5, "p": 1, "eps": 1e-07, "swap": true, "reduction": "sum"}
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
    api_name = "torch.nn.functional.triplet_margin_loss"
    try:
        if '.' in api_name:
            module_name, func_name = api_name.rsplit('.', 1)
            module = eval(module_name)
            func = getattr(module, func_name)
        else:
            func = eval(api_name)
        args = []
        kwargs = {}
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
        print(f"Torch error: {e}")
        return None

def test_paddle():
    inputs = {}
    params = {"api": "paddle.nn.functional.triplet_margin_loss", "input": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [1.8458017110824585, 1.28496253490448, -0.4424093961715698, 0.10090163350105286, -0.21294881403446198, 0.48764175176620483, 0.7369362115859985, 0.6837621331214905, 0.1372884064912796, -0.21449889242649078]}, "positive": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [1.717810034751892, -0.1272154450416565, -1.3185163736343384, -0.45896482467651367, 1.1904960870742798, -0.4216008484363556, 0.24884384870529175, -2.2942800521850586, -1.3392122983932495, -0.6697834730148315]}, "negative": {"shape": [50, 64, 2], "dtype": "float32", "sample_values": [-0.48965904116630554, 1.3440203666687012, -0.3458004891872406, 0.9071406722068787, 0.2800755798816681, -1.6802493333816528, -1.1936050653457642, -0.07833614945411682, -0.6536622047424316, 1.656090497970581]}, "margin": 0.5, "p": 1, "swap": true, "reduction": "sum"}
    for k, v in params.items():
        if k == 'api':
            continue
        if isinstance(v, dict) and 'shape' in v:
            np_data = get_input_data(
                v['shape'], 
                v['dtype'], 
                v['sample_values']
            )
            inputs[k] = paddle.to_tensor(np_data, dtype=getattr(paddle, v['dtype']))
        else:
            inputs[k] = v
    api_name = "paddle.nn.functional.triplet_margin_loss"
    try:
        parts = api_name.split('.')
        module = eval(parts[0])
        for part in parts[1:-1]:
            module = getattr(module, part)
        func = getattr(module, parts[-1])
        result = func(**inputs)
        print("Paddle result shape:", result.shape)
        return result.numpy()
    except Exception as e:
        print(f"Paddle error: {e}")
        try:
            print("Retrying with positional args...")
            result = func(*inputs.values())
            print("Paddle result shape:", result.shape)
            return result.numpy()
        except Exception as e2:
            print(f"Paddle retry error: {e2}")
            return None

if __name__ == "__main__":
    print(f"Reproducing Case 83: torch.nn.functional.triplet_margin_loss vs paddle.nn.functional.triplet_margin_loss")
    torch_res = test_torch()
    pd_res = test_paddle()
    if torch_res is not None and pd_res is not None:
        try:
            diff = np.abs(torch_res - pd_res)
            max_diff = np.max(diff)
            print(f"\nMax difference: {max_diff}")
            if np.isnan(max_diff):
                print("Difference contains NaN")
        except Exception as e:
            print(f"Error computing difference: {e}")
