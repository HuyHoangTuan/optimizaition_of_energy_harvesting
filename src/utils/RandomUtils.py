import numpy as np
import random

seed = 2
random.seed(2)
RandomUtils = np.random.default_rng(seed = seed)

def custom_random():
    return RandomUtils.random()

def uniform(low = 0.0, height = 1.0, size = None):
    return RandomUtils.uniform(low = low, high = height, size = size)

def exponential(scale = 1.0, size = None):
    return RandomUtils.exponential(scale = scale, size = size)

def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    return RandomUtils.choice(a = a, size = size, replace = replace, p = p, axis = axis, shuffle = shuffle)

def sample(population, k, *, counts=None):
    return random.sample(population, k, counts = counts)

__all__ = ['custom_random', 'uniform', 'exponential', 'choice', 'sample']
