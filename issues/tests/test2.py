# import numpy as np
# import torch
# import paddle

# input_data = [-0.877115400943947, -0.17205383221963766, 0.14039304124475271,
#               0.9606750841273599, 1.4295494246249612, -0.45974536025361523,
#               -1.4130413570449318, 1.8013200780067657, 1.0087162320309664,
#               -0.02971757573285859]
# input_np = np.array(input_data, dtype=np.float64)

# input_pt = torch.tensor(input_np)
# out_pt = torch.erfinv(input_pt)

# input_pd = paddle.to_tensor(input_np)
# out_pd = paddle.erfinv(input_pd)

# pt_np = out_pt.numpy()
# pd_np = out_pd.numpy()
# oob_mask = np.abs(input_np) >= 1
# print(f"PyTorch erfinv output: {pt_np}")
# print(f"PaddlePaddle erfinv output: {pd_np}")
# max_diff = np.max(np.abs(pt_np - pd_np))
# print(f"Maximum difference: {max_diff}")

import numpy as np
import torch
import paddle

input_data = [[0.2596988081932068], [-0.9840201735496521], [-0.2839934229850769],
              [0.931747317314148], [-1.0186134576797485], [-1.6229819059371948],
              [-0.2458077371120453], [-1.4968321323394775], [-0.9785251617431641],
              [0.8734411001205444]]
input_np = np.array(input_data, dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.erfinv(input_pd)

pt_np = out_pt.numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch erfinv output:\n{out_pt.numpy()}")
print(f"PaddlePaddle erfinv output:\n{out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")