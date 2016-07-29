# coding : utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates
from utils import apprentissage
import time


def estim_noise(y, npts, deg, idx):
    y = y[idx-npts+1:idx+1]
    p = np.polyfit(range(npts), y, deg)
    f = np.polyval(p, range(npts))
    res = y - f
    return np.std(res)


t0 = time.time()

###########################################
#           set some parameters           #
###########################################

n_vect = np.arange(300, 3000, 200)  # duration T of the interval
d_vect = range(1,4)                 # degree of fitting polynomial
n_extr = 65       # length of the extrapolation window
n_scan = 10       # repetition of fitting procedure every n seconds

t1 = 35
t2 = 65


###########################################
#     noise level before Tohoku event     #
###########################################

# read GWR data file
filestr = './kam/raw_data/KAM.03.06.15.txt'
with open(filestr, 'r') as f:
    lines = f.readlines()
    gr_thk = [float(l.split()[1]) for l in lines]
    t = np.array([float(l.split()[0]) for l in lines])

tEQ = dates.datestr2num('11 March 2011 05:46:21')
idx_thk = int(np.where(t == tEQ)[0])

# compt = 0
# moy   = 0
# for npt in n_vect:
#     for deg in d_vect:
#         l = estim_noise(gr_thk, npt, deg, idx_thk)
#         moy += l
#         compt +=1

# noise = moy / compt
noise = 0.2
print 'averaged noise before Tohoku event = ', noise, 'ugal'


###########################################
#          load background data           #
###########################################

filenames = ['.03.01.11.txt', '.03.12.21.txt', '.03.22.31.txt', '.04.01.10.txt', '.04.11.20.txt', '.04.21.30.txt']
gr = np.zeros( (10*24*3600, len(filenames)) )

for ifile, filename in enumerate(filenames):
    filestr = './kam/raw_data/KAM' + filename

    with open(filestr, 'r') as f:
        lines = f.readlines()
        gr[:, ifile] = [float(l.split()[1]) for l in lines]


###########################################
#          statistical analysis           #
#  apply the least-squares polynomial fit #
#       to the background waveforms       #
###########################################

distmin = 9999.0

for npt in n_vect:
    # compute beginning of every time intervals
    idx_init = np.arange(0, len(gr), n_scan)
    while idx_init[-1] + npt + n_extr > len(gr):
        idx_init = idx_init[:-1]

    for deg in d_vect:
        l = np.array([apprentissage(gr, npt, deg, n_extr, idx, noise) for idx in idx_init])
        out = [item for sublist in l for item in sublist if not np.isnan(item)]

        if np.std(out) < distmin:
            distmin = np.std(out)
            opt = out
            opt_full = l
            npt_opt = npt
            deg_opt = deg

# npt_opt and deg_opt are the optimal set of parameters [T, d]
# opt contains the reduced gravity signals for this set


###########################################
#     compute false alarm probability     #
###########################################

p = np.polyfit(range(npt_opt), gr_thk[idx_thk - npt_opt + 1 : idx_thk + 1], deg_opt)
f = np.polyval(p, range(npt_opt + n_extr))

res = gr_thk[idx_thk - npt_opt + 1 : idx_thk + 1 + n_extr] - f

a_thk = np.mean(res[-30:])
print 'reduced gravity signal A for the event interval =', a_thk, 'ugal'

pr = sum(abs(res) > abs(a_thk) for res in opt)
print 'false alarm probability = ' , pr, '/', len(opt), '=', 1.0 * pr / len(opt)


# save reduced gravity signals A from optimal parameters and event interval into file
print 'optimal set of parameters : T = ', npt_opt, ', d = ', deg_opt
outsave = opt + [a_thk]
np.savetxt('./opt_kam.txt', outsave)


print 'ellapsed time to run = ', time.time() - t0, 'seconds'

