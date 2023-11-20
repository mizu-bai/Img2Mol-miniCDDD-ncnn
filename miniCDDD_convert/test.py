import ncnn
import numpy as np

dummy_input = np.random.random(size=(1, 512)).astype(np.float32)
print(dummy_input.dtype)
print(ncnn.Mat(dummy_input))
