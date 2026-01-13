import numpy as np
import torch
import math
import time
import logging
import os

def shuffle(*arrays, **kwargs):
        require_indices = kwargs.get('indices', False)

        if len(set(len(x) for x in arrays)) != 1:
            raise ValueError('All inputs to shuffle must have '
                            'the same length.')

        shuffle_indices = np.arange(len(arrays[0]))
        np.random.shuffle(shuffle_indices)

        if len(arrays) == 1:
            result = arrays[0][shuffle_indices]
        else:
            result = tuple(x[shuffle_indices] for x in arrays)

        if require_indices:
            return result, shuffle_indices
        else:
            return result