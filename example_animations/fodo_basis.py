"""For the sake of cleanliness, use the basic FODO and other common functions (used for animations) in this file"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from tracking.linear_tracking import LinearLattice

# Generic graph setup
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['text.usetex'] = True
plt.rcParams['lines.markersize'] = 2.5

plt.rcParams["text.latex.preamble"] = plt.rcParams["text.latex.preamble"].join([r"\usepackage{marvosym}", ])  # More could be added to this preamble

# Lattice parameters
l_mag = 0.08
l_drift = 0.125
k0 = 50.0

fodo = LinearLattice([dict(length=l_mag, strength=+k0), dict(length=l_drift, strength=0.0),
                      dict(length=l_mag, strength=-k0), dict(length=l_drift, strength=0.0)])
fodoy = LinearLattice([dict(length=l_mag, strength=-k0), dict(length=l_drift, strength=0.0),
                      dict(length=l_mag, strength=+k0), dict(length=l_drift, strength=0.0)])
lattice_length = 2*(l_mag+l_drift)

def reset_figax(fig, ax, tight_layout=True, **kwargs):
    vals = dict(xlabel="Position [mm]", ylabel="Angle [mrad]",
                xlim=(-2.0, 2.0), ylim=(-4.0, 4.0))
    vals.update(kwargs)
    ax.cla()
    ax.set_xlabel(vals['xlabel'], size=24)
    ax.set_ylabel(vals['ylabel'], size=24)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlim(*vals['xlim'])
    ax.set_ylim(*vals['ylim'])
    if tight_layout:
        fig.tight_layout()

def circle_to(r, theta_0, theta_1, nvals=250):
    """Returns linearly spaced x and y corresponding to a circle from theta_0 to theta_1 with radius r"""
    tvals = np.linspace(theta_0, theta_1, nvals)
    xvals = r*np.cos(tvals)
    yvals = r*np.sin(tvals)

    return xvals, yvals
