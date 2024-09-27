import numpy as np
import scipy.signal

def combined_shape(buffer_size, shape=None):
    """
    buffer를 위한 데이터 크기 반환.
    length: 버퍼의 크기
    shape: 개별 데이터의 shape
    """
    if shape is None:
        return (buffer_size,)
    return (buffer_size, shape) if np.isscalar(shape) else (buffer_size, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]