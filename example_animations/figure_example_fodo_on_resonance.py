"""Showing that, as delta is varied, the phase space of a triplet lattice varies wildly if the magnet strength is fixed"""

from fodo_basis import *
import matplotlib.animation as animation
from matplotlib import cm

def get_approx_tune(k0, l_mag=l_mag, l_drift=l_drift):
    rtk = np.sqrt(k0)
    phi = rtk*l_mag
    fmag = np.array([[np.cos(phi), np.sin(phi)/rtk], [-rtk*np.sin(phi), np.cos(phi)]])
    dmag = np.array([[np.cosh(phi), np.sinh(phi)/rtk], [rtk*np.sinh(phi), np.cosh(phi)]])
    drift = np.array([[1.0, l_drift], [0.0, 1.0]])

    tmat = drift @ dmag @ drift @ fmag
    trace = tmat[0, 0] + tmat[1, 1]
    tune = np.arccos(trace/2)/(2*np.pi)
    return tune

from tracking.nonlinear_tracking import DriftStraight, QuadrupoleStraight, NonlinearLattice, MultipoleStraight

if __name__ == '__main__':
    k0 = 119.22  # 123.22 to generate plot on resonance
    add_sextupole = True
    constant_emittance = False
    fmag  = QuadrupoleStraight(l_mag, k0)
    dmag  = QuadrupoleStraight(l_mag, -k0)
    drift = DriftStraight(l_drift)
    if add_sextupole:
        sextupole = MultipoleStraight(1E-5, [0, k0, k0*10000])
        fodolist = [fmag, drift, dmag, drift, sextupole]
    else:
        fodolist = [fmag, drift, dmag, drift]
    fodo = NonlinearLattice(fodolist)

    x0_vals = np.linspace(1.0, 6.5, 5)*1E-4
    if constant_emittance:
        pass
    else:
        X0_vals = [np.array([x0, 0.0, 0.0, 0.0]) for x0 in x0_vals]

    #delta_vals = np.linspace(-0.4, 1.0, 100)
    delta_vals = np.array([0.0])
    beta_0 = 1.0
    count_turns = 750
    turns_arr = np.arange(1, count_turns+1+0.001, 1)

    print("Doing tracking")
    X_many_turns = np.array([fodo.track_many_turns(X0, (0.0, beta_0), count_turns, 0.0) for X0 in X0_vals])
    print("Tracking completed")
    x =  X_many_turns[:, 0, :]
    px = X_many_turns[:, 1, :]
    y =  X_many_turns[:, 2, :]
    py = X_many_turns[:, 3, :]
    xprime = px/np.sqrt(1-px**2-py**2)
    x *= 1000
    xprime *= 1000


    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7),constrained_layout=True)

    colornorm = cm.colors.PowerNorm(1.0, vmin=x0_vals[0], vmax=x0_vals[-1])
    scalarMap = cm.ScalarMappable(norm=colornorm, cmap=cm.coolwarm)

    reset_figax(fig, ax, tight_layout=False, xlim=(-3.0, 3.0), ylim=(-12.5, 12.5))
    #cax.text(0.5, -0.5, r'$\frac{\Delta P}{P}$', ha='center', va='baseline', size=14)


    for i, x0 in enumerate((x0_vals)):
        xvals = x[i]
        xprimevals = xprime[i]
        c = x0*np.ones_like(xvals)

        ax.scatter(xvals, xprimevals, c=c, s=plt.rcParams['lines.markersize']*10, cmap='copper', edgecolor=scalarMap.to_rgba(x0), linewidth=0.0,
                   vmin=x0_vals[0], vmax=x0_vals[-1])

    fig.show()
    #fig.savefig('results/figs/off_resonance.png', dpi=150)
