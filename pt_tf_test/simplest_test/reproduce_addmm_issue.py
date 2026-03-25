import torch
import tensorflow as tf
import numpy as np

def test_torch():
    print("Testing torch.addmm...")
    input_data = [
       0.9778106212615967, 
       1.1486157178878784, 
       1.6314395666122437, 
       0.31658536195755005, 
       1.2712117433547974, 
       0.9300848841667175 
    ]
    mat1_data = [
       -0.0341772697865963, 
       0.7339072227478027, 
       -0.01572246290743351, 
       0.16780751943588257, 
       -0.8884471654891968, 
       1.350381851196289 
    ]
    mat2_data = [
       -0.8923236131668091, 
       -0.7148515582084656, 
       0.6624495387077332, 
       -0.4650103747844696, 
       0.8016815781593323, 
       -2.323143482208252, 
       0.3188072443008423, 
       0.36480751633644104, 
       -1.4174838066101074 
    ]

    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(2, 3)
    mat1_tensor = torch.tensor(mat1_data, dtype=torch.float32).reshape(2, 3)
    mat2_tensor = torch.tensor(mat2_data, dtype=torch.float32).reshape(3, 3)

    result = torch.addmm(input_tensor, mat1_tensor, mat2_tensor)
    print("Torch result:")
    print(result)
    return result.numpy()

def test_tensorflow():
    print("\nTesting tf.linalg.matmul...")
    a_data = [
       -0.3076030910015106, 
       -0.06987522542476654, 
       0.7971499562263489, 
       0.2920219600200653, 
       -0.02548869326710701, 
       0.21697759628295898 
    ]
    b_data = [
       0.3589474558830261, 
       0.6074601411819458, 
       0.013397765345871449, 
       0.29318681359291077, 
       0.7770669460296631, 
       1.3498646020889282, 
       0.8732182383537292, 
       1.187211513519287, 
       0.36229848861694336 
    ]

    a_tensor = tf.constant(a_data, dtype=tf.float32, shape=[2, 3])
    b_tensor = tf.constant(b_data, dtype=tf.float32, shape=[3, 3])

    result = tf.linalg.matmul(a_tensor, b_tensor)
    print("TensorFlow result:")
    print(result)
    return result.numpy()

if __name__ == "__main__":
    torch_res = test_torch()
    tf_res = test_tensorflow()

    diff = np.abs(torch_res - tf_res)
    max_diff = np.max(diff)
    print(f"\nMax difference: {max_diff}")
