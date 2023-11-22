# coding: utf-8

from matplotlib import pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from matplotlib.pylab import normpdf

import subprocess

from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


def get_axis_limits(ax, scalex, scaley):
    return ax.get_xlim()[1]*scalex, ax.get_ylim()[1]*scaley

# define model function to be used to fit the data
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


#  !-------------------------------------------------------!
#  !   load A distributions for optimal set of paramters   !
#  !   (for Kamioka alone, and for the stacked waveform)   !
#  !-------------------------------------------------------!

with open('./opt_kam.txt', 'r') as f:
    lines = f.readlines()
    opt_kam   = [float(l) for l in lines[:-1]]
    a_thk_kam = float(lines[-1])

with open('./opt.txt', 'r') as f:
    lines = f.readlines()
    opt   = [float(l) for l in lines[:-1]]
    a_thk = float(lines[-1])


mu_kam = np.mean(opt_kam)
mu     = np.mean(opt)

sigma_kam = np.std(opt_kam)
sigma     = np.std(opt)


# define x-axises
n_bins = 301

xmax1 = np.abs(opt_kam).max()
bin_edges1 = np.linspace(-xmax1, xmax1, n_bins)
bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:])/2
b1 = np.linspace(0, xmax1, n_bins)

xmax2 = np.abs(opt).max()
bin_edges2 = np.linspace(-xmax2, xmax2, n_bins)
bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:])/2
b2 = np.linspace(0, xmax2, n_bins)


# p0 is the initial guess for the fitting
# coefficients (A, mu and sigma above)
p1 = [10000, mu_kam, sigma_kam]
p2 = [10000, mu, sigma]


# plot histogram for reduced gravity signals
f, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 8))
plt.subplots_adjust(left=0.11, bottom=0.06, right=0.98, top=0.99, wspace=0.35, hspace=0.2)

ax.flatten()[0].grid()
ax.flatten()[1].grid()
ax.flatten()[2].grid()
ax.flatten()[3].grid()

ax.flatten()[0].set_axisbelow(True)
ax.flatten()[1].set_axisbelow(True)
ax.flatten()[2].set_axisbelow(True)
ax.flatten()[3].set_axisbelow(True)

# subplot 1
n1, bins, patches = ax.flatten()[0].hist(opt_kam, bins=bin_edges1, histtype='stepfilled', color='cornflowerblue')

coeff, var_matrix = curve_fit(gauss, bin_centers1, n1, p0=p1)
hist_fit1 = gauss(bin_centers1, *coeff)
ax.flatten()[0].plot(bin_centers1, hist_fit1, 'k', linewidth=1.5)


# subplot 2
n2, bins, patches = ax.flatten()[2].hist(opt, bins=bin_edges2, histtype='stepfilled', color='cornflowerblue')

coeff, var_matrix = curve_fit(gauss, bin_centers2, n2, p0=p2)
hist_fit2 = gauss(bin_centers2, *coeff)
ax.flatten()[2].plot(bin_centers2, hist_fit2, 'k', linewidth=1.5)


# subplots 3 & 4 :
n, bins, patches = ax.flatten()[1].hist(np.abs(opt_kam), bins=b1, normed=True,
                                        cumulative=-1, histtype='step',
                                        log=True, lw=1.5, color='cornflowerblue')

n, bins, patches = ax.flatten()[3].hist(np.abs(opt), bins=b2, normed=True,
                                        cumulative=-1, histtype='step',
                                        log=True, lw=1.5, color='cornflowerblue')

n = len(hist_fit1)
a = hist_fit1[:n/2]
b = hist_fit1[n/2:]
c = a[::-1] + b
y1 = c[::-1].cumsum()
y1 = y1/y1.max()
ax.flatten()[1].semilogy(bin_centers1[n/2:], y1[::-1], 'k', lw=1.5)

n = len(hist_fit2)
a = hist_fit2[:n/2]
b = hist_fit2[n/2:]
c = a[::-1] + b
y2 = c[::-1].cumsum()
y2 = y2/y2.max()
ax.flatten()[3].semilogy(bin_centers2[n/2:], y2[::-1], 'k', lw=1.5)

ax.flatten()[0].axvline(x=a_thk_kam, c='darkorange', ls='--', lw=1.)
ax.flatten()[0].axvline(x=-a_thk_kam, c='darkorange', ls='--', lw=1.)
ax.flatten()[2].axvline(x=a_thk, c='darkorange', ls='--', lw=1.)
ax.flatten()[2].axvline(x=-a_thk, c='darkorange', ls='--', lw=1.)
ax.flatten()[1].axvline(x=np.abs(a_thk_kam), c='darkorange', ls='--', lw=1.)
ax.flatten()[3].axvline(x=np.abs(a_thk), c='darkorange', ls='--', lw=1.)

x_label = r'Reduced gravity signal $\mathcal{A}$ ($\mu$gal)'
ax.flatten()[0].set_xlabel(x_label, fontsize=10)
ax.flatten()[1].set_xlabel(x_label, fontsize=10)

x_label = r'Reduced gravity signal $\mathcal{A}$ (dimensionless)'
ax.flatten()[2].set_xlabel(x_label, fontsize=10)
ax.flatten()[3].set_xlabel(x_label, fontsize=10)

ax.flatten()[0].set_ylabel('Number of events', fontsize=10)
ax.flatten()[2].set_ylabel('Number of events', fontsize=10)
ax.flatten()[1].set_ylabel('Statistical significance p', fontsize=10)
ax.flatten()[3].set_ylabel('Statitiscal significance p', fontsize=10)

ax.flatten()[0].set_xlim([-0.18, 0.18])
ax.flatten()[2].set_xlim([-0.8, 0.8])
ax.flatten()[1].set_xlim([0, 0.75])
ax.flatten()[3].set_xlim([0, xmax2])

ax.flatten()[2].set_ylim([0, 5500.0])
ax.flatten()[1].set_ylim([1.0E-6, 1.0])
ax.flatten()[3].set_ylim([1.0E-6, 1.0])

ax.flatten()[0].tick_params(axis='both', labelsize=10)
ax.flatten()[1].tick_params(axis='both', labelsize=10)
ax.flatten()[2].tick_params(axis='both', labelsize=10)
ax.flatten()[3].tick_params(axis='both', labelsize=10)


# annotate station name
string1 = '(a) KA station'
string2 = '(c) KA station'
string3 = '(b) KA + F-net'
string4 = '(d) KA + F-net'

at1 = AnchoredText(string1, prop=dict(size=8), frameon=True, loc=1,)
at2 = AnchoredText(string2, prop=dict(size=8), frameon=True, loc=1,)
at3 = AnchoredText(string3, prop=dict(size=8), frameon=True, loc=1,)
at4 = AnchoredText(string4, prop=dict(size=8), frameon=True, loc=1,)

at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at4.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

ax.flatten()[0].add_artist(at1)
ax.flatten()[1].add_artist(at2)
ax.flatten()[2].add_artist(at3)
ax.flatten()[3].add_artist(at4)

#ax.flatten()[0].annotate('a : KAM', xy=get_axis_limits(ax.flatten()[0], 0.70, 0.94), fontsize=8)
#ax.flatten()[1].annotate('c : KAM', xy=get_axis_limits(ax.flatten()[1], 0.85, 0.40), fontsize=8)
#ax.flatten()[2].annotate('b : KAM + F-net', xy=get_axis_limits(ax.flatten()[2], 0.45, 0.94), fontsize=8)
#ax.flatten()[3].annotate('d : KAM + F-net', xy=get_axis_limits(ax.flatten()[3], 0.70, 0.40), fontsize=8)


file = './graphics/figure3.pdf'
f.savefig(file, dpi=300)
subprocess.call(["open", "-a", "/Applications/Skim.app", file])
