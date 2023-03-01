import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
import Functions

# loading list
L_new = np.load("L_new_62_274.npy")

NN = 10
DEOO_fin_ours = [[], [], [], [], [], []]
mis_fin_ours = [[], [], [], [], [], []]
DPE_fin_ours = [[], [], [], [], [], []]
p1 = 0.7
p_y1 = 0.7
p_y0 = 0.4
p = 57
N = 1000
N_1 = int(N * p1)
N_0 = N - N_1
N_11 = int(N_1 * p_y1)
N_01 = N_1 - N_11
N_10 = int(N_0 * p_y0)
N_00 = N_0 - N_10
sigma = 1
delta = 0.1
cnt = np.zeros(6)
mean = []
for enum in tqdm(range(NN)):

    # data generation
    mu_00 = 3 * np.random.rand(1)
    mu_01 = np.random.rand(1)
    mu_10 = np.random.rand(1)
    mu_11 = np.random.rand(1)
    #x_00 = np.random.normal(mu_00, sigma, size = (N_00, p))
    # x_00 = np.random.gamma(mu_00, sigma, size = (N_00, p))
    x_00 = np.random.standard_t(4, size = (N_00, p))
    # x_00 = np.random.exponential(3, size = (N_00, p))
    #x_00 = np.random.laplace(mu_00, 1, size = (N_00, p))
    y_00 = np.zeros(N_00, dtype = int)
    #x_01 = np.random.normal(mu_01, sigma, size = (N_01, p))
    # x_01 = np.random.gamma(mu_01, 1, size = (N_01, p))
    #x_01 = np.random.standard_t(3, size = (N_01, p))
    x_01 = np.random.chisquare(2, size = (N_01, p))
    y_01 = np.zeros(N_01, dtype = int)
    #x_10 = np.random.normal(mu_10, sigma, size = (N_10, p))
    x_10 = np.random.chisquare(4, size = (N_10, p))
    # x_10 = np.random.gamma(mu_10, 1, size = (N_10, p))
    # x_10 = np.random.standard_t(4, size = (N_10, p))
    y_10 = np.ones(N_10, dtype = int)
    # x_11 = np.random.normal(mu_11, sigma, size = (N_11, p))
    x_11 = np.random.laplace(mu_11, 1, size = (N_11, p))
    y_11 = np.ones(N_11, dtype = int)
    X_0 = np.append(x_00, x_10, axis = 0)
    Y_0 = np.append(y_00, y_10)
    X_1 = np.append(x_01, x_11, axis = 0)
    Y_1 = np.append(y_01, y_11)
    X_tr_00, X_te_00, Y_tr_00, Y_te_00 = train_test_split(x_00, y_00, test_size = 0.2)
    X_tr_01, X_te_01, Y_tr_01, Y_te_01 = train_test_split(x_01, y_01, test_size = 0.2)
    X_tr_10, X_te_10, Y_tr_10, Y_te_10 = train_test_split(x_10, y_10, test_size = 0.2)
    X_tr_11, X_te_11, Y_tr_11, Y_te_11 = train_test_split(x_11, y_11, test_size = 0.2)
    X_tr_00r, X_tr_00c, Y_tr_00r, Y_tr_00c = train_test_split(X_tr_00, Y_tr_00, test_size = 0.7)
    X_tr_01r, X_tr_01c, Y_tr_01r, Y_tr_01c = train_test_split(X_tr_01, Y_tr_01, test_size = 0.7)
    X_tr_10r, X_tr_10c, Y_tr_10r, Y_tr_10c = train_test_split(X_tr_10, Y_tr_10, test_size = 62)
    X_tr_11r, X_tr_11c, Y_tr_11r, Y_tr_11c = train_test_split(X_tr_11, Y_tr_11, test_size = 274)
    X_tr_0 = np.append(X_tr_00, X_tr_10, axis = 0)
    Y_tr_0 = np.append(Y_tr_00, Y_tr_10)
    X_tr_1 = np.append(X_tr_01, X_tr_11, axis = 0)
    Y_tr_1 = np.append(Y_tr_01, Y_tr_11)
    X_tr_0r = np.append(X_tr_00r, X_tr_10r, axis = 0)
    Y_tr_0r = np.append(Y_tr_00r, Y_tr_10r)
    X_tr_1r = np.append(X_tr_01r, X_tr_11r, axis = 0)
    Y_tr_1r = np.append(Y_tr_01r, Y_tr_11r)
    X_tr_0c = np.append(X_tr_00c, X_tr_10c, axis = 0)
    Y_tr_0c = np.append(Y_tr_00c, Y_tr_10c)
    X_tr_1c = np.append(X_tr_01c, X_tr_11c, axis = 0)
    Y_tr_1c = np.append(Y_tr_01c, Y_tr_11c)
    X_tr_r = np.append(X_tr_0r, X_tr_1r, axis = 0)
    Y_tr_r = np.append(Y_tr_0r, Y_tr_1r)
    X_tr_c = np.append(X_tr_0c, X_tr_1c, axis = 0)
    Y_tr_c = np.append(Y_tr_0c, Y_tr_1c)
    X_te_0 = np.append(X_te_00, X_te_10, axis = 0)
    Y_te_0 = np.append(Y_te_00, Y_te_10)
    X_te_1 = np.append(X_te_01, X_te_11, axis = 0)
    Y_te_1 = np.append(Y_te_01, Y_te_11)
    X_te = np.append(X_te_0, X_te_1, axis = 0)
    Y_te = np.append(Y_te_0, Y_te_1)
    X_tr = np.append(X_tr_0, X_tr_1, axis = 0)
    Y_tr = np.append(Y_tr_0, Y_tr_1)

    log_reg = LogisticRegression(max_iter = 10000)
    log_reg_ = LogisticRegression(max_iter = 10000)
    clm = log_reg.fit(X_tr_r, Y_tr_r)
    clm_ = log_reg_.fit(X_tr, Y_tr)
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0
    t_00 = []
    t_01 = []
    t_10 = []
    t_11 = []
    prob_0 = clm.predict_proba(X_tr_0c)
    prob_1 = clm.predict_proba(X_tr_1c)
    for i in range(len(X_tr_0c)):
        if(Y_tr_0c[i] == 1):
            n_10 += 1
            t_10.append(prob_0[i, 1])
        else:
            n_00 += 1
            t_00.append(prob_0[i, 1])
    for i in range(len(X_tr_1c)):
        if(Y_tr_1c[i] == 1):
            n_11 += 1
            t_11.append(prob_1[i, 1])
        else:
            n_01 += 1
            t_01.append(prob_1[i, 1])
    t_00 = np.array(t_00)
    t_01 = np.array(t_01)
    t_10 = np.array(t_10)
    t_11 = np.array(t_11)
    t_00 = np.sort(t_00)
    t_01 = np.sort(t_01)
    t_10 = np.sort(t_10)
    t_11 = np.sort(t_11)
    min = 10000 * np.ones(6)
    Th_0 = -1 * np.ones(6)
    Th_1 = -1 * np.ones(6)
    for k_10 in range(1, n_10 + 1):
        for k_11 in range(1, n_11 + 1):
            k_00 = n_00
            if(t_00[0] > t_10[k_10 - 1]):
                k_00 = 0
            for i in range(n_00 - 1):
                if(t_00[i] <= t_10[k_10 - 1] and t_00[i + 1] > t_10[k_10 - 1]):
                    k_00 = i + 1
                    break
            k_01 = n_01
            if(t_01[0] > t_11[k_11 - 1]):
                k_01 = 0
            for i in range(n_01 - 1):
                if(t_01[i] <= t_11[k_11 - 1] and t_01[i + 1] > t_11[k_11 - 1]):
                    k_01 = i + 1
                    break
            if(k_01 != 0 and k_01 != n_01 and k_00 != 0 and k_00 != n_00):
                for aa in range(6):
                    alpha = 0.02 * aa + 0.08
                    if(L_new[k_10][k_11][int(100 * alpha - 4)] <= delta and Functions.L_(n_00, n_01, k_00, k_01, alpha, delta) < 1):
                        if(Functions.M(k_00, k_01, k_10, k_11, n_00, n_01, n_10, n_11) < min[aa]):
                            cnt[aa] += 1
                            min[aa] = Functions.M(k_00, k_01, k_10, k_11, n_00, n_01, n_10, n_11)
                            Th_0[aa] = k_10 - 1
                            Th_1[aa] = k_11 - 1
    te_00 = []
    te_01 = []
    te_10 = []
    te_11 = []
    for i in range(len(X_te_0)):
        if(Y_te_0[i] == 1):
            te_10.append(clm.predict_proba(X_te_0)[i, 1])
        else:
            te_00.append(clm.predict_proba(X_te_0)[i, 1])
    for i in range(len(X_te_1)):
        if(Y_te_1[i] == 1):
            te_11.append(clm.predict_proba(X_te_1)[i, 1])
        else:
            te_01.append(clm.predict_proba(X_te_1)[i, 1])
    te_00 = np.array(te_00)
    te_01 = np.array(te_01)
    te_10 = np.array(te_10)
    te_11 = np.array(te_11)
    for aa in range(6):
        if(Th_0[aa] > -1):
            DEOO_fin_ours[aa].append(abs(Functions.DEOO(t_10[int(Th_0[aa])], t_11[int(Th_1[aa])])[1]))
            DPE_fin_ours[aa].append(abs(Functions.DPE(t_10[int(Th_0[aa])], t_11[int(Th_1[aa])])[1]))
            mis_fin_ours[aa].append(Functions.mis(t_10[int(Th_0[aa])], t_11[int(Th_1[aa])])[1])
            
DEOO_fin_ours = np.array(DEOO_fin_ours)
mis_fin_ours = np.array(mis_fin_ours)
DPE_fin_ours = np.array(DPE_fin_ours)

for i in range(6):
    DEOO_fin_ours[i] = np.array(DEOO_fin_ours[i])
    mis_fin_ours[i] = np.array(mis_fin_ours[i])
    DPE_fin_ours[i] = np.array(DPE_fin_ours[i])

for i in range(6):
    print("alpha={}".format(0.02 * i + 0.08))
    print("DEOO_mean:{}".format(np.mean(DEOO_fin_ours[i])))
    print("DEOO_95quantile:{}".format(np.percentile(DEOO_fin_ours[i],95)))
    print("DPE_mean:{}".format(np.mean(DPE_fin_ours[i])))
    print("DPE_95quantile:{}".format(np.percentile(DPE_fin_ours[i],95)))
    print("mis_error:{}".format(1 - np.mean(mis_fin_ours[i])))