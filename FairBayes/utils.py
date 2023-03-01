import numpy as np
import pandas as pd

import numpy as np

###calculate thresholds to satisfies DDP constraint######







def threshold_DemPa(eta1, eta0, delta=0, pre_level=1e-5):
    pa_hat = len(eta1) / (len(eta1) + len(eta0))
    p1 = np.mean(eta1 > 1 / 2)
    p0 = np.mean(eta0 > 1 / 2)



    if abs(p1 - p0) <= delta:
        tstar = 0

    elif p1 - p0 > delta:
        tmin = 0
        tmax = min(pa_hat, (1 - pa_hat))

        while tmax - tmin > pre_level:
            tmid = (tmin + tmax)/2
            t1 = 0.5 + tmid / 2 / pa_hat
            t0 = 0.5 - tmid / 2 / (1 - pa_hat)
            DDP  = np.mean(eta1 > t1) - np.mean(eta0 > t0)
            if DDP > delta:
                tmin = tmid
            else:
                tmax = tmid
        tstar = (tmin + tmax) / 2

    else:
        tmin = -1*min(pa_hat, (1 - pa_hat))
        tmax = 0
        while tmax - tmin > pre_level:
            tmid = (tmin + tmax) / 2
            t1 = 0.5 + tmid / 2 / pa_hat
            t0 = 0.5 - tmid / 2 / (1 - pa_hat)
            DDP = np.mean(eta1 > t1) - np.mean(eta0 > t0)
            if DDP > -delta:
                tmin = tmid
            else:
                tmax = tmid
        tstar = (tmin + tmax) / 2

    t1star = 0.5 + tstar / 2 / pa_hat
    t0star = 0.5 - tstar / 2 / (1 - pa_hat)
    return t1star, t0star




def threshold_EqqOp(eta1, eta0, Y1, Y0, delta=0, pre_level=1e-5):
    pa_hat = len(eta1) / (len(eta1) + len(eta0))
    py1_hat = np.mean(Y1)
    py0_hat = np.mean(Y0)

    p1 = np.mean(eta1[Y1==1] > 1 / 2)
    p0 = np.mean(eta0[Y0==1] > 1 / 2)



    if abs(p1 - p0) <= delta:
        tstar = 0

    elif p1 - p0 > delta:
        tmin = 0
        tmax = pa_hat * py1_hat

        while tmax - tmin > pre_level:
            tmid = (tmin + tmax)/2
            t1 = pa_hat * py1_hat / (2 * pa_hat * py1_hat - tmid)
            t0 = (1- pa_hat) * py0_hat / (2 * (1- pa_hat) * py0_hat + tmid)
            DEO = np.mean(eta1[Y1==1] > t1) - np.mean(eta0[Y0==1] > t0)
            if DEO > delta:
                tmin = tmid
            else:
                tmax = tmid
        tstar = (tmin + tmax) / 2

    else:
        tmin = -1 * (1 - pa_hat) * py0_hat
        tmax = 0
        while tmax - tmin > pre_level:
            tmid = (tmin + tmax) / 2
            t1 = pa_hat * py1_hat / (2 * pa_hat * py1_hat -tmid)
            t0 = (1- pa_hat) * py0_hat / (2 * (1- pa_hat) * py0_hat + tmid)
            DEO = np.mean(eta1[Y1==1] > t1) - np.mean(eta0[Y0==1] > t0)
            if DEO > -delta:
                tmin = tmid
            else:
                tmax = tmid
        tstar = (tmin + tmax) / 2

    t1star = pa_hat * py1_hat / (2 * pa_hat * py1_hat - tstar)
    t0star = (1 - pa_hat) * py0_hat / (2 * (1 - pa_hat) * py0_hat + tstar)

    return t1star, t0star




def balance_DDP(eta_base, t_base, eta, tmin, tmax, pre_level=1e-5):
    pbase = np.mean(eta_base > t_base)
    while tmax-tmin > pre_level:
        tmid = (tmin + tmax)/2
        pmid =  np.mean(eta > tmid)
        if pmid > pbase:
            tmin = tmid
        else:
            tmax = tmid
    tstar = (tmin + tmax)/2
    return tstar


def threshold_DemPa_multi(eta,Z, pre_level=1e-5):
    Z_list = sorted(list(set(Z)))
    lenzlist = len(Z_list)
    eta_first =  eta[Z==0]
    tmin1 = 0
    tmax1 = np.max(eta_first)
    pa1 = len(eta_first)/len(eta)
    while tmax1-tmin1 > pre_level:
        tmid1 = (tmin1 + tmax1)/2
        tstar = [tmid1]
        s = [(tmid1 - 0.5) * pa1 ]
        for z in range(1,lenzlist):
            eta_z = eta[Z==z]
            tzmin=0
            tzmax = np.max(eta_z)
            t_z = balance_DDP(eta_first,tmid1,eta_z,tzmin,tzmax)
            paz = len(eta_z)/len(eta)
            s_z = (t_z - 0.5) * paz
            s.append(s_z)
            tstar.append(t_z)
        s = np.array(s)
        ssum = sum(s)

        if ssum>0:
            tmax1=tmid1
        else:
            tmin1=tmid1
    tstar = np.array(tstar)
    return tstar

def measures_from_Yhat_DemPa(Y1hat, Y0hat, Y1, Y0):
    assert isinstance(Y1hat, np.ndarray)
    assert isinstance(Y0hat, np.ndarray)
    assert isinstance(Y1, np.ndarray)
    assert isinstance(Y0, np.ndarray)

    datasize = len(Y1) + len(Y0)
    # Accuracy
    acc = ((Y1hat == Y1).sum() + (Y0hat == Y0).sum()) / datasize
    # DDP
    DDP = abs(np.mean(Y1hat) - np.mean(Y0hat))

    data = [acc, DDP]
    columns = ['acc', 'DDP']
    return pd.DataFrame([data], columns=columns)





def measures_from_Yhat_EqqOp(Y1hat, Y0hat, Y1, Y0):
    assert isinstance(Y1hat, np.ndarray)
    assert isinstance(Y0hat, np.ndarray)
    assert isinstance(Y1, np.ndarray)
    assert isinstance(Y0, np.ndarray)

    datasize = len(Y1) + len(Y0)
    # Accuracy
    acc = ((Y1hat == Y1).sum() + (Y0hat == Y0).sum()) / datasize
    # DEO
    DEO = abs(np.mean(Y1hat[Y1==1]) - np.mean(Y0hat[Y0==1]))

    data = [acc, DEO]
    columns = ['acc', 'DEO']
    return pd.DataFrame([data], columns=columns)







def measures_from_Yhat_DemPa_multi(eta, Z, Y, t):
    datasize= len(Y)
    Z_list = sorted(list(set(Z)))
    lenzlist = len(Z_list)
    accsum=0
    dppsum=0
    ddpset=[]
    for z in range(lenzlist):
        Yz = Y[Z==z]
        Yz_hat = (eta[Z==z]>t[z])
        accsum += (Yz==Yz_hat).sum()
        dppsum += Yz_hat.sum()
        dppz = np.mean(Yz_hat)
        ddpset.append(dppz)
    # Accuracy
    acc = accsum/datasize

    # DDP
    ddp_mean = dppsum/datasize
    ddpset = np.array(ddpset)

    DDP = np.sum(np.abs(ddpset - ddp_mean))

    data = [acc, DDP]
    columns = ['acc', 'DDP']
    return pd.DataFrame([data], columns=columns)










