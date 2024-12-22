import numpy as np


def Gini(vectors):
    return np.sum(np.square(vectors), axis=1)


def Margin(vectors):
    sorted_arr = np.sort(vectors, axis=1)[:, ::-1]
    diff = sorted_arr[:, 0] - sorted_arr[:, 1]
    return np.array(diff)
    
    
def MaxP(vectors):
    return np.max(vectors, axis=1)


