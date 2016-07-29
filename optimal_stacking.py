# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates
from utils import *
from obspy.core import read, UTCDateTime
import time

t0 = time.time()

###########################################
#           set some parameters           #
###########################################

n_vect = [1900] # np.arange(1000, 2000, 100) # duration T of the interval
d_vect = [1]    # range(1,3)                 # degree of fitting polynomial
n_extr = 65       # length of the extrapolation window
n_scan = 10       # repetition of fitting procedure every n seconds

t1 = 35
t2 = 65

stanames = ['KAM', 'WJM', 'NAA', 'KNY', 'KZS', 'TGA']

# input parameters for theoretical estimation : EOST W-phase solution
evla  = 37.92
evlo  = 143.11
evdep = 19500.
gamma_s = 196.* np.pi / 180
gamma_d = 12. * np.pi / 180
gamma_r = 85. * np.pi / 180

with open('./tohoku_wei2012epsl_momrate.txt', 'r') as f:
    lines = f.readlines()
    tt   = [float(l.split()[0]) for l in lines]
    Mdot = [1.0E-7*float(l.split()[1]) for l in lines]

dt  = tt[1] - tt[0]
MM  = np.cumsum(Mdot)*dt
iiM = np.cumsum(np.cumsum(MM))*dt**2
tt  = tt[::10]
iiM = iiM[::10]


###########################################
# estimation of the weighting coefficient #
###########################################

coeff = np.zeros((len(stanames), 3)) # 3 channels : LHE, LHN, LNZ
stack_thk = np.zeros(10*24*3600)

for ista, staname in enumerate(stanames):

    if staname == 'KAM':
        # read GWR data file
        filestr = './kam/highpassed_data/KAM.03.06.15.txt'
        with open(filestr, 'r') as f:
            lines = f.readlines()
            gr_thk = np.array([float(l.split()[1]) for l in lines])
            t = np.array([float(l.split()[0]) for l in lines])

        tEQ = dates.datestr2num('11 March 2011 05:46:21')
        idx_thk = int(np.where(t == tEQ)[0])

        # noise estimation before Tohoku
        noise = np.std(gr_thk[idx_thk-1800:idx_thk])

        # analytical signal
        stala = 36.42
        stalo = 137.31
        th = compute_theory(stala, stalo, evla, evlo, gamma_r, gamma_s, gamma_d, iiM, tt, t1, t2, 'LHZ')
        coeff[ista, 2] = th / noise**2
        stack_thk += gr_thk * coeff[ista, 2]

    else:
        # read seismic data file
        filestr = './fnet/' + staname + '.LHZ.03.06.15.sac'
        st = read(filestr)

        for tr in st:
            # convert m/s/s --> ugal
            gr_thk = tr.data * 1.0E8

            # noise estimation before Tohoku
            timeEQ = UTCDateTime(2011, 3, 11, 5, 46, 21)
            tmp = tr.copy()
            tmp.trim(timeEQ - 1800, timeEQ)
            tmp.data = tmp.data * 1.0E8
            noise = np.std(tmp.data)

            # analytical signal
            stala = tr.stats.sac.stla
            stalo = tr.stats.sac.stlo
            th = compute_theory(stala, stalo, evla, evlo, gamma_r, gamma_s, gamma_d, iiM, tt, t1, t2, tr.stats.channel)

            if tr.stats.channel == 'LHE':
                coeff[ista, 0] = th / noise**2
                stack_thk += gr_thk * coeff[ista, 0]

            elif tr.stats.channel == 'LHN':
                coeff[ista, 1] = th / noise**2
                stack_thk += gr_thk * coeff[ista, 1]

            else:
                coeff[ista, 2] = th / noise**2
                stack_thk += gr_thk * coeff[ista, 2]


# noise estimation before Tohoku for the stacked waveform
noise = np.std(stack_thk[idx_thk-1800:idx_thk])


###########################################
#          load background data           #
###########################################

stack = np.zeros((10*24*3600, 6)) # 6 = len(filedates)

for ista, staname in enumerate(stanames):

    if staname == 'KAM':

        filedates = ['.03.01.11.txt', '.03.12.21.txt', '.03.22.31.txt', \
                     '.04.01.10.txt', '.04.11.20.txt', '.04.21.30.txt']

        # read gravimetric background data files
        for idate, filedate in enumerate(filedates):

            filestr = './kam/highpassed_data/' + staname + filedate
            with open(filestr, 'r') as f:
                lines = f.readlines()
                gr = np.array([float(l.split()[1]) for l in lines])

            stack[:, idate] += gr * coeff[ista, 2]

    else:

        filedates = ['.LHZ.03.01.11.sac', '.LHZ.03.12.21.sac', '.LHZ.03.22.31.sac', \
                     '.LHZ.04.01.10.sac', '.LHZ.04.11.20.sac', '.LHZ.04.21.30.sac']

        # read seismic background data files
        for idate, filedate in enumerate(filedates):

            filestr = './fnet/' + staname + filedate
            st = read(filestr)

            for tr in st:
				if tr.stats.channel == 'LHZ':
					stack[:, idate] += tr.data * 1.0E8 * coeff[ista, 2]
				elif tr.stats.channel == 'LHN':
					stack[:, idate] += tr.data * 1.0E8 * coeff[ista, 1]
				else:
					stack[:, idate] += tr.data * 1.0E8 * coeff[ista, 0]


###########################################
#          statistical analysis           #
#  apply the least-squares polynomial fit #
#   to the stacked background waveforms   #
###########################################

distmin = 9999.0

for npt in n_vect:
    # compute beginning of every time intervals
    idx_init = np.arange(0, len(stack), n_scan)
    while idx_init[-1] + npt + n_extr > len(stack):
        idx_init = idx_init[:-1]

    for deg in d_vect:
        l = [apprentissage(stack, npt, deg, n_extr, idx, noise) for idx in idx_init]
        out = [item for sublist in l for item in sublist]

        if np.std(out) < distmin:
            distmin = np.std(out)
            opt = out
            npt_opt = npt
            deg_opt = deg

# npt_opt and deg_opt are the optimal set of parameters [T, d]
# opt contains the reduced gravity signals for this set

###########################################
#     compute false alarm probability     #
###########################################

p = np.polyfit(range(npt_opt), stack_thk[idx_thk - npt_opt + 1 : idx_thk + 1], deg_opt)
f = np.polyval(p, range(npt_opt + n_extr))

res = stack_thk[idx_thk - npt_opt + 1 : idx_thk + 1 + n_extr] - f

a_thk = np.mean(res[-30:])
print 'reduced gravity signal A for the event interval =', a_thk, 'ugal'

pr = sum(abs(a) > abs(a_thk) for a in opt)
print 'false alarm probability = ' , pr, '/', len(opt), '=', 1.0 * pr / len(opt)


# save reduced gravity signals A from optimal parameters and event interval into file
print 'optimal set of parameters : T = ', npt_opt, ', d = ', deg_opt
outsave = opt + [a_thk]
np.savetxt('./opt.txt', outsave)


print 'ellapsed time to run = ', time.time() - t0, 'seconds'
