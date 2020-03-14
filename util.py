import numpy as np
import math

def close(a, b, tol=1e-8):
    return abs(a - b) < tol

def minput(prompt=''):
    lst = []
    print(prompt, end='')
    while True:
        i = input()
        if len(i) is 0:
            break
        lst.append(i)
    return lst

def set_zero(x, tol=1e-8):
    idx = np.bitwise_and(x < tol, x > -tol)
    y = x.copy()
    y[idx] = 0
    return y

def get_k(n):
    return (1 << n) - 1

def get_m(n):
    return int((1 << (n - 2)) * n * (n - 1) / 2 + n)

def b2n(b):
    return int(math.log(b.size + 1, 2))
