import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm

def L(n0, n1, k0, k1, alpha, delta):
    ans = 0
    N = 10000
    for i in range(N):
        qq = np.random.beta(k1, n1 - k1 + 1)
        pp = np.random.beta(k0, n0 - k0 + 1)
        if(qq - pp > alpha or pp - qq > alpha):
            ans += 1
        if(ans > delta * N):
            return 1
    return ans * 1.0 / N

def L_(n0, n1, k0, k1, alpha, delta):
    ans = 0
    N = 10000
    for i in range(N):
        qq = np.random.beta(k1, n1 - k1 + 1)
        pp = np.random.beta(k0, n0 - k0 + 1)
        qq_ = np.random.beta(k1 + 1, n1 - k1)
        pp_ = np.random.beta(k0 + 1, n0 - k0)
        if(qq - pp_ > alpha or pp - qq_ > alpha):
            ans += 1
        if(ans > delta * N):
            return 1
    return ans * 1.0 / N

def M(k_00, k_01, k_10, k_11, n_00, n_01, n_10, n_11):
    return k_10 / (n_10 + 1) * (1 - p1) * p_y0 + k_11 / (n_11 + 1) * p1 * p_y1 + (n_00 + 0.5 - k_00) / (n_00 + 1) * (1 - p1) * (1 - p_y0) + (n_01 + 0.5 - k_01) / (n_01 + 1) * p1 * (1 - p_y1)

def DEOO(th_0, th_1):
    c10 = 0
    c11 = 0
    C10 = 0
    C11 = 0
    for i in range(len(te_10)):
        if(te_10[i] > 0.5):
            c10 += 1
        if(te_10[i] > th_0):
            C10 += 1
    for i in range(len(te_11)):
        if(te_11[i] > 0.5):
            c11 += 1
        if(te_11[i] > th_1):
            C11 += 1
    DEOO = c10 / len(te_10) - c11 / len(te_11)
    DEOO_ = C10 / len(te_10) - C11 / len(te_11)
    ans = np.zeros(2)
    ans[0] = DEOO
    ans[1] = DEOO_
    return ans

def DPE(th_0, th_1):
    c00 = 0
    c01 = 0
    C00 = 0
    C01 = 0
    for i in range(len(te_00)):
        if(te_00[i] > 0.5):
            c00 += 1
        if(te_00[i] > th_0):
            C00 += 1
    for i in range(len(te_01)):
        if(te_01[i] > 0.5):
            c01 += 1
        if(te_01[i] > th_1):
            C01 += 1
    DPE = c00 / len(te_00) - c01 / len(te_01)
    DPE_ = C00 / len(te_00) - C01 / len(te_01)
    ans = np.zeros(2)
    ans[0] = DPE
    ans[1] = DPE_
    return ans

def mis(th_0, th_1):
    c10_ = 0
    c11_ = 0
    C10_ = 0
    C11_ = 0
    for i in range(len(te_10)):
        if(te_10[i] < 0.5):
            c10_ += 1
        if(te_10[i] < th_0):
            C10_ += 1
    for i in range(len(te_11)):
        if(te_11[i] < 0.5):
            c11_ += 1
        if(te_11[i] < th_1):
            C11_ += 1
    for i in range(len(te_00)):
        if(te_00[i] > 0.5):
            c10_ += 1
        if(te_00[i] > th_0):
            C10_ += 1
    for i in range(len(te_01)):
        if(te_01[i] > 0.5):
            c11_ += 1
        if(te_01[i] > th_1):
            C11_ += 1
    mis = (c10_ + c11_) / (len(te_10) + len(te_11) + len(te_00) + len(te_01))
    mis_ = (C10_ + C11_) / (len(te_10) + len(te_11) + len(te_00) + len(te_01))
    ans = np.zeros(2)
    ans[0] = mis
    ans[1] = mis_
    return ans