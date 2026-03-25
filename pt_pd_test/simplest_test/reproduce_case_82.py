
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
    if 'input' in {"api": "torch.nn.functional.triplet_margin_loss", "anchor": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.47857484221458435, -1.3141095638275146, -1.2893248796463013, 1.2618346214294434, 0.6209537386894226, 0.9855097532272339, 0.7829182744026184, -0.5071091055870056, 0.8487926125526428, 1.8724522590637207]}, "positive": {"shape": [100, 128], "dtype": "float32", "sample_values": [-1.3363287448883057, 0.8080044388771057, 0.36612260341644287, -0.7082329988479614, -0.2670901119709015, -1.370373249053955, 0.9377487301826477, 1.354630470275879, -0.5360546112060547, 1.0976194143295288]}, "negative": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.0416579395532608, 0.01745523512363434, 1.4835410118103027, -0.25789907574653625, 1.0094608068466187, -1.2948304414749146, -0.12396742403507233, 0.6575034260749817, 0.7811842560768127, -0.16308282315731049]}, "margin": 1.0, "p": 2, "eps": 1e-06, "swap": false, "reduction": "mean"}:
        input_info = {"api": "torch.nn.functional.triplet_margin_loss", "anchor": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.47857484221458435, -1.3141095638275146, -1.2893248796463013, 1.2618346214294434, 0.6209537386894226, 0.9855097532272339, 0.7829182744026184, -0.5071091055870056, 0.8487926125526428, 1.8724522590637207]}, "positive": {"shape": [100, 128], "dtype": "float32", "sample_values": [-1.3363287448883057, 0.8080044388771057, 0.36612260341644287, -0.7082329988479614, -0.2670901119709015, -1.370373249053955, 0.9377487301826477, 1.354630470275879, -0.5360546112060547, 1.0976194143295288]}, "negative": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.0416579395532608, 0.01745523512363434, 1.4835410118103027, -0.25789907574653625, 1.0094608068466187, -1.2948304414749146, -0.12396742403507233, 0.6575034260749817, 0.7811842560768127, -0.16308282315731049]}, "margin": 1.0, "p": 2, "eps": 1e-06, "swap": false, "reduction": "mean"}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.triplet_margin_loss", "anchor": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.47857484221458435, -1.3141095638275146, -1.2893248796463013, 1.2618346214294434, 0.6209537386894226, 0.9855097532272339, 0.7829182744026184, -0.5071091055870056, 0.8487926125526428, 1.8724522590637207]}, "positive": {"shape": [100, 128], "dtype": "float32", "sample_values": [-1.3363287448883057, 0.8080044388771057, 0.36612260341644287, -0.7082329988479614, -0.2670901119709015, -1.370373249053955, 0.9377487301826477, 1.354630470275879, -0.5360546112060547, 1.0976194143295288]}, "negative": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.0416579395532608, 0.01745523512363434, 1.4835410118103027, -0.25789907574653625, 1.0094608068466187, -1.2948304414749146, -0.12396742403507233, 0.6575034260749817, 0.7811842560768127, -0.16308282315731049]}, "margin": 1.0, "p": 2, "eps": 1e-06, "swap": false, "reduction": "mean"}
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
    params = {"api": "paddle.nn.functional.triplet_margin_loss", "input": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.09370189160108566, -1.7807759046554565, -1.6368248462677002, 0.05324530974030495, -2.473527431488037, 0.44235050678253174, -0.4106084704399109, -0.19523930549621582, 0.7010370492935181, -0.5690581798553467]}, "positive": {"shape": [100, 128], "dtype": "float32", "sample_values": [-1.3363287448883057, 0.8080044388771057, 0.36612260341644287, -0.7082329988479614, -0.2670901119709015, -1.370373249053955, 0.9377487301826477, 1.354630470275879, -0.5360546112060547, 1.0976194143295288]}, "negative": {"shape": [100, 128], "dtype": "float32", "sample_values": [-0.0416579395532608, 0.01745523512363434, 1.4835410118103027, -0.25789907574653625, 1.0094608068466187, -1.2948304414749146, -0.12396742403507233, 0.6575034260749817, 0.7811842560768127, -0.16308282315731049]}, "margin": 1.0, "p": 2, "swap": false, "reduction": "mean"}
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
    print(f"Reproducing Case 82: torch.nn.functional.triplet_margin_loss vs paddle.nn.functional.triplet_margin_loss")
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
