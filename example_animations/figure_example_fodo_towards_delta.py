"""Showing that, as delta is varied, the phase space of a triplet lattice varies wildly if the magnet strength is fixed"""

from fodo_basis import *
import matplotlib.animation as animation
from matplotlib import cm


from tracking.nonlinear_tracking import DriftStraight, QuadrupoleStraight, NonlinearLattice, MultipoleStraight

if __name__ == '__main__':
    like_synchrotron = False
    add_sextupole = False
    constant_emittance = False
    fmag  = QuadrupoleStraight(l_mag, k0)
    dmag  = QuadrupoleStraight(l_mag, -k0)
    drift = DriftStraight(l_drift)
    if add_sextupole:
        sextupole = MultipoleStraight(0.005, [0, k0, k0*1000])
        fodolist = [fmag, drift, dmag, drift, sextupole]
    else:
        fodolist = [fmag, drift, dmag, drift]
    fodo = NonlinearLattice(fodolist)

    x0 = 0.0007606693378842227
    if constant_emittance:
        pass
    else:
        X0 = np.array([x0, 0.0, 0.0, 0.0])

    #delta_vals = np.linspace(-0.4, 1.0, 100)
    delta_min = -0.525
    delta_max = 0.525
    delta_vals = np.linspace(delta_min, delta_max, 7)
    if like_synchrotron:
        delta_vals *= 0
    beta_0 = 1.0
    count_turns = 500
    turns_arr = np.arange(1, count_turns+1+0.001, 1)

    print("Doing tracking")
    X_many_turns = np.array([fodo.track_many_turns(X0, (delta, beta_0), count_turns, 0.0) for delta in delta_vals])
    print("Tracking completed")
    x =  X_many_turns[:, 0, :]
    px = X_many_turns[:, 1, :]
    y =  X_many_turns[:, 2, :]
    py = X_many_turns[:, 3, :]
    xprime = px/np.sqrt(1+2*delta_vals[:, np.newaxis]*beta_0 + delta_vals[:, np.newaxis]**2-px**2-py**2)
    x *= 1000
    xprime *= 1000


    # Set up the figure
    gs_kw = dict(width_ratios=[7, 1])
    fig, axd = plt.subplot_mosaic([['left', 'right'],
                                   ['left', 'right']],
                                  gridspec_kw=gs_kw, figsize=(8, 7),
                                  constrained_layout=True)
    ax, cax = [axd['left'], axd['right']]

    colornorm = cm.colors.TwoSlopeNorm(0.0, vmin=delta_min, vmax=delta_max)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm.coolwarm, norm=colornorm, orientation='vertical')
    scalarMap = cm.ScalarMappable(norm=colornorm, cmap=cm.coolwarm)

    reset_figax(fig, ax, tight_layout=False, xlim=(-2.5, 2.5), ylim=(-10.0, 10.0))
    #cax.text(0.5, -0.5, r'$\frac{\Delta P}{P}$', ha='center', va='baseline', size=14)
    cax.annotate(r'$\frac{\Delta P}{P}$', (0.5, -0.0), ha='center', va='center', size=24)


    for i, delta in enumerate((delta_vals)):
        xvals = x[i]
        xprimevals = xprime[i]
        if like_synchrotron:
            c = delta_max*np.ones_like(xvals)
        else:
            c = delta*np.ones_like(xvals)

        ax.scatter(xvals, xprimevals, c=c, s=plt.rcParams['lines.markersize']*25, cmap='coolwarm', edgecolor=scalarMap.to_rgba(delta), linewidth=0.0,
                   vmin=delta_min, vmax=delta_max)


    fig.show()
