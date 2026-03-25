
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
    if 'input' in {"api": "torch.nn.functional.batch_norm", "input": {"shape": [1, 512, 4, 4], "dtype": "float32", "sample_values": [-0.6440367102622986, -1.0677266120910645, -0.90932297706604, 0.7177445292472839, 0.6274353265762329, -1.230667233467102, -0.25043633580207825, 2.703683614730835, -1.3681317567825317, -0.7131175994873047]}, "running_mean": {"shape": [512], "dtype": "float32", "sample_values": [0.7941598296165466, 1.2431628704071045, 0.6886394619941711, -0.7664967179298401, 0.7036862969398499, -0.7485829591751099, 0.22695103287696838, -0.6623467803001404, 0.43439191579818726, 0.2609267830848694]}, "running_var": {"shape": [512], "dtype": "float32", "sample_values": [1.707387089729309, 0.1351487636566162, -1.6744136810302734, 0.08067172020673752, -0.837338387966156, 1.0101394653320312, 0.5301785469055176, 0.3865327835083008, -0.2953154742717743, 0.8787783980369568]}, "weight": {"shape": [512], "dtype": "float32", "sample_values": [1.2831664085388184, 0.6224907040596008, 0.5146893262863159, -0.7443417906761169, -1.172934889793396, -0.5626687407493591, 0.13605733215808868, -0.22751083970069885, 0.4043237566947937, -0.7544737458229065]}, "bias": {"shape": [512], "dtype": "float32", "sample_values": [1.4871186017990112, 0.6284974217414856, -0.7776644229888916, -0.20116494596004486, -1.5331151485443115, 0.9596675038337708, -0.8350678086280823, 0.12558215856552124, -1.1213479042053223, 0.7716975808143616]}, "training": true, "momentum": 0.001, "eps": 0.001}:
        input_info = {"api": "torch.nn.functional.batch_norm", "input": {"shape": [1, 512, 4, 4], "dtype": "float32", "sample_values": [-0.6440367102622986, -1.0677266120910645, -0.90932297706604, 0.7177445292472839, 0.6274353265762329, -1.230667233467102, -0.25043633580207825, 2.703683614730835, -1.3681317567825317, -0.7131175994873047]}, "running_mean": {"shape": [512], "dtype": "float32", "sample_values": [0.7941598296165466, 1.2431628704071045, 0.6886394619941711, -0.7664967179298401, 0.7036862969398499, -0.7485829591751099, 0.22695103287696838, -0.6623467803001404, 0.43439191579818726, 0.2609267830848694]}, "running_var": {"shape": [512], "dtype": "float32", "sample_values": [1.707387089729309, 0.1351487636566162, -1.6744136810302734, 0.08067172020673752, -0.837338387966156, 1.0101394653320312, 0.5301785469055176, 0.3865327835083008, -0.2953154742717743, 0.8787783980369568]}, "weight": {"shape": [512], "dtype": "float32", "sample_values": [1.2831664085388184, 0.6224907040596008, 0.5146893262863159, -0.7443417906761169, -1.172934889793396, -0.5626687407493591, 0.13605733215808868, -0.22751083970069885, 0.4043237566947937, -0.7544737458229065]}, "bias": {"shape": [512], "dtype": "float32", "sample_values": [1.4871186017990112, 0.6284974217414856, -0.7776644229888916, -0.20116494596004486, -1.5331151485443115, 0.9596675038337708, -0.8350678086280823, 0.12558215856552124, -1.1213479042053223, 0.7716975808143616]}, "training": true, "momentum": 0.001, "eps": 0.001}['input']
        if isinstance(input_info, dict) and 'shape' in input_info:
            inputs['input'] = torch.tensor(get_input_data(
                input_info['shape'], 
                input_info['dtype'], 
                input_info['sample_values']
            ), dtype=getattr(torch, input_info['dtype']))
    params = {"api": "torch.nn.functional.batch_norm", "input": {"shape": [1, 512, 4, 4], "dtype": "float32", "sample_values": [-0.6440367102622986, -1.0677266120910645, -0.90932297706604, 0.7177445292472839, 0.6274353265762329, -1.230667233467102, -0.25043633580207825, 2.703683614730835, -1.3681317567825317, -0.7131175994873047]}, "running_mean": {"shape": [512], "dtype": "float32", "sample_values": [0.7941598296165466, 1.2431628704071045, 0.6886394619941711, -0.7664967179298401, 0.7036862969398499, -0.7485829591751099, 0.22695103287696838, -0.6623467803001404, 0.43439191579818726, 0.2609267830848694]}, "running_var": {"shape": [512], "dtype": "float32", "sample_values": [1.707387089729309, 0.1351487636566162, -1.6744136810302734, 0.08067172020673752, -0.837338387966156, 1.0101394653320312, 0.5301785469055176, 0.3865327835083008, -0.2953154742717743, 0.8787783980369568]}, "weight": {"shape": [512], "dtype": "float32", "sample_values": [1.2831664085388184, 0.6224907040596008, 0.5146893262863159, -0.7443417906761169, -1.172934889793396, -0.5626687407493591, 0.13605733215808868, -0.22751083970069885, 0.4043237566947937, -0.7544737458229065]}, "bias": {"shape": [512], "dtype": "float32", "sample_values": [1.4871186017990112, 0.6284974217414856, -0.7776644229888916, -0.20116494596004486, -1.5331151485443115, 0.9596675038337708, -0.8350678086280823, 0.12558215856552124, -1.1213479042053223, 0.7716975808143616]}, "training": true, "momentum": 0.001, "eps": 0.001}
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
    api_name = "torch.nn.functional.batch_norm"
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
    params = {"api": "paddle.nn.functional.batch_norm", "input": {"shape": [1, 512, 4, 4], "dtype": "float32", "sample_values": [-0.6440367102622986, -1.0677266120910645, -0.90932297706604, 0.7177445292472839, 0.6274353265762329, -1.230667233467102, -0.25043633580207825, 2.703683614730835, -1.3681317567825317, -0.7131175994873047]}, "running_mean": {"shape": [512], "dtype": "float32", "sample_values": [0.7941598296165466, 1.2431628704071045, 0.6886394619941711, -0.7664967179298401, 0.7036862969398499, -0.7485829591751099, 0.22695103287696838, -0.6623467803001404, 0.43439191579818726, 0.2609267830848694]}, "running_var": {"shape": [512], "dtype": "float32", "sample_values": [1.707387089729309, 0.1351487636566162, -1.6744136810302734, 0.08067172020673752, -0.837338387966156, 1.0101394653320312, 0.5301785469055176, 0.3865327835083008, -0.2953154742717743, 0.8787783980369568]}, "weight": {"shape": [512], "dtype": "float32", "sample_values": [1.2831664085388184, 0.6224907040596008, 0.5146893262863159, -0.7443417906761169, -1.172934889793396, -0.5626687407493591, 0.13605733215808868, -0.22751083970069885, 0.4043237566947937, -0.7544737458229065]}, "bias": {"shape": [512], "dtype": "float32", "sample_values": [1.4871186017990112, 0.6284974217414856, -0.7776644229888916, -0.20116494596004486, -1.5331151485443115, 0.9596675038337708, -0.8350678086280823, 0.12558215856552124, -1.1213479042053223, 0.7716975808143616]}, "training": true, "momentum": 0.001}
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
    api_name = "paddle.nn.functional.batch_norm"
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
    print(f"Reproducing Case 52: torch.nn.functional.batch_norm vs paddle.nn.functional.batch_norm")
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
