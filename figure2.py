# coding : utf-8

from matplotlib import pyplot as plt
from matplotlib import dates
import numpy as np
import subprocess
import time


def apprentissage(y, npts, deg, n_extr, idx, noise):
    p = np.polyfit(range(npts), y[idx : idx + npts, :], deg)
    f = np.array([np.polyval(coeff, range(npts + n_extr)) for coeff in p.T]).T

    res = y[idx : idx + npts + n_extr, :] - f
    res_std1 = np.std(res[:npts, :], axis=0)
    res_std2 = np.std(res[-n_extr:, :], axis=0)

    out = np.zeros(y.shape[1])
    for r in range(y.shape[1]):
        if res_std1[r] < 1.25*noise and res_std1[r] > 0.75*noise and res_std2[r] < 10*noise:
            out[r] = np.mean(res[-30:,r])
        else:
            out[r] = None

    return out


t0 = time.time()

###########################################
#           set some parameters           #
###########################################

t_opt  = 690    # optimal duration T of the interval
d_opt  = 2      # optimal degree of fitting polynomial
n_extr = 65     # length of the extrapolation window
n_scan = 10     # repetition of fitting procedure every n seconds

t1 = 35
t2 = 65

noise = 0.2

###########################################
#               load data                 #
###########################################

filenames = ['.03.01.11.txt', '.03.06.15.txt']
gr = np.zeros( (10*24*3600, len(filenames)) )
t  = np.zeros( (10*24*3600, len(filenames)) )

for ifile, filename in enumerate(filenames):
    filestr = './kam/raw_data/KAM' + filename

    with open(filestr, 'r') as f:
        lines = f.readlines()
        gr[:, ifile] = [float(l.split()[1]) for l in lines]
        t[:, ifile]  = np.array([float(l.split()[0]) for l in lines])

tEQ = dates.datestr2num('11 March 2011 05:46:21')
idx_thk = int(np.where(t[:,1] == tEQ)[0])


###########################################
#          statistical analysis :         #
#      apply the optimal least-squares    #
#      polynomial fit to the waveforms    #
###########################################


# compute beginning of every time intervals
idx_init = np.arange(2, len(gr), n_scan)
while idx_init[-1] + t_opt + n_extr > len(gr):
    idx_init = idx_init[:-1]

opt_full = np.array([apprentissage(gr, t_opt, d_opt, n_extr, idx, noise) for idx in idx_init])

# t_opt and d_opt are the optimal set of parameters [T, d]
# opt contains the reduced gravity signals for this set



###########################################
#     compute false alarm probability     #
###########################################

p = np.polyfit(range(t_opt), gr[idx_thk - t_opt + 1 : idx_thk + 1, 1], d_opt)
f = np.polyval(p, range(t_opt + n_extr))

res = gr[idx_thk - t_opt + 1 : idx_thk + 1 + n_extr, 1] - f

a_thk = np.mean(res[-30:])
print 'reduced gravity signal A for the event interval =', a_thk, 'ugal'



###########################################
# graphics : plot figure 2 of the article #
###########################################

idx_init = np.arange(2, len(gr), n_scan)
while idx_init[-1] + t_opt + n_extr > len(gr):
	idx_init = idx_init[:-1]
idx_stop = idx_init + t_opt + n_extr

dd0 = ['01 march 2011, 05:46:00', '06 march 2011']

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_axes([0.09, 0.13, 0.9, 0.74])
ax1.grid(ls='--', lw=0.5)
ax1.set_axisbelow(True)

ax1.plot(24*3600*(t[idx_thk - t_opt + 1: idx_thk + 1 + n_extr] - tEQ), res, 'gold')

ax1.set_ylim(-0.55, 0.55)
ax1.set_xlim(24*3600*(dates.datestr2num('11 march 2011, 5:35:00')-tEQ), 24*3600*(dates.datestr2num('11 march 2011, 5:47:26')-tEQ))

ax1.set_xlabel('Time (in seconds from main shock)', fontsize=12, labelpad=5)
ax1.set_ylabel(r'Amplitude ($\mu$gal)', fontsize=12)

ax1.tick_params(axis='both', labelsize=10)


ax2 = ax1.twiny()
ax2.set_axisbelow(True)

ax2.plot(t[idx_thk - t_opt + 1: idx_thk + 1 + n_extr,1], res, color='lightgray', label='Fit residuals')

for iw, wd0 in enumerate(dd0):
    tt = np.array([dates.seconds(item) for item in idx_stop])
    tt += dates.datestr2num(wd0)

    if iw == 0:
        ax2.plot(tt, opt_full[:,iw], 'cornflowerblue', lw=2, label=r'Reduced gravity signals $\mathcal{A}$')
    else:
        ax2.plot(tt, opt_full[:,iw], 'cornflowerblue', lw=2)

ax2.legend(loc='upper left', fontsize=10)

ax2.axvline(x=tEQ, c='k', ymin=0.1, ymax=0.9, lw=2)
ax2.axhline(y=a_thk, c='darkorange', lw=1.5, ls='--')
ax2.axhline(y=-a_thk, c='darkorange', lw=1.5, ls='--')

locator = dates.AutoDateLocator()
formatter = dates.AutoDateFormatter(locator)
formatter.scaled = {1. : '%d/%m', 1./24. : '%H:%M:%S', 1./(24.*60.) : '%H:%M:%S'}
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

ax2.set_xlabel('Time (UTC), march 11th 2011', fontsize=12, labelpad=5)

ax2.set_xlim(dates.datestr2num('11 march 2011, 5:35:00'), dates.datestr2num('11 march 2011, 5:47:26'))

ax2.set_ylim(-0.55, 0.55)

ax2.tick_params(axis='both', labelsize=10)

file = './graphics/figure2.pdf'
fig.savefig(file, dpi=300)
subprocess.call(["open", "-a", "/Applications/Skim.app", file])

print 'ellapsed time to run = ', time.time() - t0, 'seconds'
