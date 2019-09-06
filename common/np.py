# coding: utf-8
from common.config import GPU
import plaidml.keras
plaidml.keras.install_backend()
import numpy as np
from keras import backend as K
import time
isnot_affine=True
import importlib
A=False

if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    np.add.at = np.scatter_add

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')

else:
    def dots(a,b):
        f=K.dot(K.variable(np.asarray(a)),K.variable(np.asarray(b)))
        return np.array(K.get_value(f),dtype='float32')
    def sums(a,*,axis=None):
        f=K.sum(K.variable(np.asarray(a)),axis)
        return np.array(K.get_value(f),dtype='float32')
    
    def npdot_split_time(arrs, arr):
        start_time = time.time()
        dot = [np.dot(s, arr) for s in arrs]
        dot_matrix = np.array(dot,dtype='float32')
        print(time.time()-start_time)
        return dot_matrix
    np.sum=sums
    
    if(isnot_affine):
        np.dot=dots
    else:
        importlib.reload(np)





