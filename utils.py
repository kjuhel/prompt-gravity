# coding: utf-8

import numpy as np
from scipy import signal

def compute_theory(stala, stalo, evla, evlo, gamma_r, gamma_s, gamma_d, iiM, tt, t1, t2, channel):
    G = 6.674e-11
    R = 6370000.

    # station coordinates
    stala *= np.pi / 180
    stalo *= np.pi / 180

    # event coordinates
    evla *= np.pi / 180
    evlo *= np.pi /180

    rake   = np.array([[np.cos(gamma_r), -np.sin(gamma_r), 0], [np.sin(gamma_r), np.cos(gamma_r), 0], [0, 0, 1]])
    strike = np.array([[np.cos(gamma_s), np.sin(gamma_s), 0], [-np.sin(gamma_s), np.cos(gamma_s), 0], [0, 0, 1]])
    dip    = np.array([[np.cos(gamma_d), 0, -np.sin(gamma_d)], [0, 1, 0], [np.sin(gamma_d), 0, np.cos(gamma_d)]])

    ex = strike.dot(dip).dot(np.array([0, 0, 1]))
    ez = strike.dot(dip).dot(rake).dot(np.array([0, 1, 0]))

    rst = R * np.array([np.cos(stala)*np.sin(stalo), np.cos(stala)*np.cos(stalo), np.sin(stala)])
    rev = R * np.array([np.cos(evla)*np.sin(evlo), np.cos(evla)*np.cos(evlo), np.sin(evla)])
    dr  = rst - rev

    er  = rst / np.linalg.norm(rst)
    er0 =  dr / np.linalg.norm(dr)

    eNS = np.array([np.sin(stala)*np.sin(stalo), np.sin(stala)*np.cos(stalo), -np.cos(stala)])
    eNS = eNS / np.linalg.norm(eNS)

    eWE = np.array([np.cos(stalo), -np.sin(stalo), 0.])
    eWE = eWE / np.linalg.norm(eWE)

    a = ex.T.dot(er0)*ez
    b = ez.T.dot(er0)*ex
    c = ex.T.dot(er0)*ez.T.dot(er0)*er0

    vec = - 6*G / np.linalg.norm(dr)**4 * ( a + b - 5*c )
    th_UD = er.T.dot(vec) * iiM
    th_WE = eWE.T.dot(vec) * iiM
    th_NS = eNS.T.dot(vec) * iiM

    if channel == 'LHZ':
        th = th_UD
    elif channel == 'LHN':
        th = th_NS
    else:
        th = th_WE

    # filter band applied to the seismic data during preprocessing
    fcut1 = 0.001 / (0.5 * 1.0)
    [b, a] = signal.iirfilter(4, fcut1, btype='high', ftype='bessel', output='ba')
    th = signal.lfilter(b, a, th)

    fcut2 = 2*0.059904 / (0.5 * 1.0)
    [b, a] = signal.iirfilter(8, fcut2, btype='low', ftype='bessel', output='ba')
    th = signal.lfilter(b, a, th)

    mask = np.nonzero( (tt >= t1*np.ones(len(tt))) * (tt <= t2*np.ones(len(tt))) )
    return np.mean(th[mask]*1.0E8)


def apprentissage(y, npts, deg, n_extr, idx, noise):

    p = np.polyfit(range(npts), y[idx : idx + npts, :], deg)
    f = np.array([np.polyval(coeff, range(npts + n_extr)) for coeff in p.T]).T

    res = y[idx : idx + npts + n_extr, :] - f
    res_std1 = np.std(res[:npts, :], axis=0)
    res_std2 = np.std(res[-n_extr:, :], axis=0)

    # data cuts
    out = [np.mean(res[-30:,r]) for r in range(y.shape[1]) \
           if res_std1[r] < 1.25*noise and res_std1[r] > 0.75*noise and res_std2[r] < 10*noise]

    return out

