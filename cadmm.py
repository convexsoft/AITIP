import math
import ctypes
import numpy as np
from util import *
import os
import tempfile

cdp = ctypes.POINTER(ctypes.c_double)
cip = ctypes.POINTER(ctypes.c_int)

class Cadmm:
    def __init__(self, use_gpu):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        lib_subpath = '{}/cadmm.so'.format('cuADMM' if use_gpu else 'mklADMM')
        lib_path = os.path.join(root_dir, lib_subpath)
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

    def random_b(self, n):
        k = get_k(n)
        b = np.zeros(k, dtype=np.double)
        b_p = b.ctypes.data_as(cdp)
        self.lib.get_random_b(n, b_p)
        return b

    def build_common_ary(self, n, klst):
        k = get_k(n)
        klst = klst.astype(np.int32)
        klst_p = klst.ctypes.data_as(cip)
        count = np.zeros(1, dtype=np.int32)
        count_p = count.ctypes.data_as(cip)
        common = np.zeros(k, dtype=np.int32)
        common_p = common.ctypes.data_as(cip)

        self.lib.buildCommonAry(ctypes.c_int(n), klst_p, count_p, common_p)
        return common[:count[0]]

    def get_elements(self, i):
        n = math.ceil(math.log(i, 2)) + 1
        count = np.zeros(1, dtype=np.int32)
        elements = np.zeros(n, dtype=np.int32)
        count_p = count.ctypes.data_as(cip)
        element_p = elements.ctypes.data_as(cip)

        self.lib.getElements(ctypes.c_int(i), count_p, element_p)
        return elements[:count[0]]

    def get_subsets(self, i):
        count = np.zeros(1, dtype=np.int32)
        subsets = np.zeros(i, dtype=np.int32)
        count_p = count.ctypes.data_as(cip)
        subsets_p = subsets.ctypes.data_as(cip)

        self.lib.getSubsets(ctypes.c_int(i), count_p, subsets_p)
        return subsets[:count[0]]

    def Arow(self, n, row):
        k = get_k(n)
        out = np.zeros(k, dtype=np.double)
        out_p = out.ctypes.data_as(cdp)

        self.lib.Arow(ctypes.c_int(n), ctypes.c_int(row), out_p)

        return out

    def solve(self, b, E=None, crossover=1, maxTime=1024, accuracy=1e-8, threads=0, dev=False):
        print('max time:', maxTime)
        print('accuracy:', accuracy)

        n = b2n(b)
        m = get_m(n)
        k = get_k(n)
        l = 0

        b = b.astype(np.double)

        E_p = None
        if E is not None:
            E = np.asfortranarray(E.astype(np.double))
            l = E.shape[0]
            E_p = E.ctypes.data_as(cdp)

        outObj = np.zeros(1, dtype=np.double)
        outLmb = np.zeros(2 * (m + k) + l, dtype=np.double)
        outObj_p = outObj.ctypes.data_as(cdp)
        outLmb_p = outLmb.ctypes.data_as(cdp)

        fp = tempfile.NamedTemporaryFile()

        self.lib.admm.restype = ctypes.c_int
        rc = self.lib.admm(ctypes.c_int(n), ctypes.c_int(l), b.ctypes.data, E_p, outObj_p, outLmb_p, ctypes.c_int(crossover), ctypes.c_double(maxTime), ctypes.c_double(accuracy), ctypes.c_int(threads), None if dev else ctypes.c_char_p(fp.name.encode('utf-8')))

        output = fp.read().decode('utf-8')
        fp.close()

        print('return code:', rc)

        outLmb = outLmb.reshape((2 * (m + k) + l, 1))

        l1 = outLmb[:m+k,:]
        l2 = outLmb[m+k:2*(m+k),:]
        l3 = outLmb[2*(m+k):,:]

        return (rc, outObj[0], None, None, l1, l2, l3, output)
