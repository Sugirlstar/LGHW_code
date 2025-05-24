import numpy as np
import os

filename = "/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy"

with open(filename, 'rb') as f:
    magic = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    print("Magic:", magic)
    print("Shape:", shape)
    print("Fortran order:", fortran_order)
    print("Dtype:", dtype)
    