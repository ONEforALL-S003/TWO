import subprocess
import tensorflow as tf
import numpy as np


for i in range(44):
    temp = np.load(f'./out/Net_Conv2d/{i}.npy')
    
    print(f'{i}번 npy : size=={temp.size} dtype=={temp.dtype} type=={temp.shape}')




# subprocess.run(['./tflite2circle', './out/conv2d_original.tflite', './out/conv2d_original.circle'])

# subprocess.run(['./circledump', './out/conv2d_original.circle'])

subprocess.run(['./q-implant', './out/conv2d_original.circle', './out/Net_Conv2d/qparam.json', './out/conv2d_quant.circle'])