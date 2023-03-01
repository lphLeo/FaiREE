# preparation: generating a list

import numpy as np
from tqdm import tqdm
L_new = np.ones([65, 280, 20])
n0 = 36
n1 = 139
for k0 in range(1, n0 + 1):
    for k1 in tqdm(range(1, n1 + 1)):
        for alpha in np.arange(0.04, 0.24, 0.01):
            ans = 0
            for i in range(10000):
                pp = np.random.beta(k0, n0 - k0 + 1)
                qq = np.random.beta(k1, n1 - k1 + 1)
                if(abs(pp - qq) > alpha):
                    ans += 1
            L_new[k0][k1][int(100 * alpha - 4)] = ans / 10000
np.save("L_36_139.npy", L_new)