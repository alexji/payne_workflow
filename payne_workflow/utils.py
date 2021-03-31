import time, os, glob, sys
import numpy as np

def load_nn(path):
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"].astype(np.float16)
    w_array_1 = tmp["w_array_1"].astype(np.float16)
    w_array_2 = tmp["w_array_2"].astype(np.float16)
    b_array_0 = tmp["b_array_0"].astype(np.float16)
    b_array_1 = tmp["b_array_1"].astype(np.float16)
    b_array_2 = tmp["b_array_2"].astype(np.float16)
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    x_min[0] /= 1000.
    x_max[0] /= 1000.
    x_min, x_max = x_min.astype(np.float16), x_max.astype(np.float16)
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()

    ## Hardcoded for now
    wavelength_payne = np.arange(3500, 7000.01, 0.03)
    
    return NN_coeffs, wavelength_payne

