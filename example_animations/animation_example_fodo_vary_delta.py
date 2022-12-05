"""Showing that, as delta is varied, the tune of a triplet lattice varies wildly if the magnet strength is fixed"""

from fodo_basis import *
import matplotlib.animation as animation
from matplotlib import cm


from tracking.nonlinear_tracking import DriftStraight, QuadrupoleStraight, NonlinearLattice, MultipoleStraight

if __name__ == '__main__':
    add_sextupole = True
    fmag  = QuadrupoleStraight(l_mag, k0)
    dmag  = QuadrupoleStraight(l_mag, -k0)
    drift = DriftStraight(l_drift)
    if add_sextupole:
        sextupole = MultipoleStraight(0.005, [0, k0, k0*1500])
        fodolist = [fmag, drift, dmag, drift, sextupole]
    else:
        fodolist = [fmag, drift, dmag, drift]
    fodo = NonlinearLattice(fodolist)

    x0 = 0.0007606693378842227
    X0 = np.array([x0, 0.0, 0.0, 0.0])

    #delta_vals = np.linspace(-0.4, 1.0, 100)
    delta_min = -0.525
    delta_max = 0.525
    delta_vals = np.linspace(delta_min, delta_max, 1500)
    beta_0 = 1.0
    count_turns = 24
    turns_arr = np.arange(1, count_turns+1+0.001, 1)

    print("Doing tracking")
    X_many_turns = np.array([fodo.track_many_turns(X0, (delta, beta_0), count_turns, 0.0) for delta in delta_vals])
    print("Tracking completed")
    x =  np.concatenate((X_many_turns[:, 0, :], X_many_turns[:, 0, :][::-1]))
    px = np.concatenate((X_many_turns[:, 1, :], X_many_turns[:, 1, :][::-1]))
    y =  np.concatenate((X_many_turns[:, 2, :], X_many_turns[:, 2, :][::-1]))
    py = np.concatenate((X_many_turns[:, 3, :], X_many_turns[:, 3, :][::-1]))
    delta_vals = np.concatenate((delta_vals, delta_vals[::-1]))
    xprime = px/np.sqrt(1+2*delta_vals[:, np.newaxis]*beta_0 + delta_vals[:, np.newaxis]**2-px**2-py**2)
    x *= 1000
    xprime *= 1000

    # Calculate tunes using an equivalent linear lattice
    muvals = []
    for delta in delta_vals:
        keq = k0/(1+delta)
        fodo = LinearLattice([dict(length=l_mag, strength=+keq), dict(length=l_drift, strength=0.0),
                              dict(length=l_mag, strength=-keq), dict(length=l_drift, strength=0.0)])
        _, _, _, mu = fodo.calculate_periodic_lattice_params(return_mu_too=True)
        muvals.append(mu)
    nuvals = np.array(muvals)/(2*np.pi)

    # Set up the figure
    gs_kw = dict(width_ratios=[7, 1], height_ratios=[7, 2])
    fig, axd = plt.subplot_mosaic([['upper', 'right'],
                                   ['lower', 'right']],
                                  gridspec_kw=gs_kw, figsize=(8, 9),
                                  constrained_layout=True)
    ax, cax, caxturn = [axd['upper'], axd['lower'],  axd['right']]
    caxplot = cax.twinx()
    caxplot.yaxis.tick_left()
    caxplot.yaxis.set_label_position("left")

    colornorm = cm.colors.TwoSlopeNorm(0.0, vmin=delta_min, vmax=delta_max)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm.coolwarm, norm=colornorm, orientation='horizontal')
    scalarMap = cm.ScalarMappable(norm=colornorm, cmap=cm.coolwarm)
    cax.set_xlabel(r'$\frac{P-P_0}{P_0}$')

    colornorm2 = mpl.colors.Normalize(vmin=0, vmax=count_turns)
    cb2 = mpl.colorbar.ColorbarBase(caxturn, cmap=mpl.cm.viridis, norm=colornorm2, orientation='vertical')
    caxturn.set_xlabel('Turn Number')

    for i in range(3):
        caxplot.axvline(0, color='k', ls='--', lw=0.5)

    numin = 0.0  #np.min(nuvals)
    numax = 0.5  #np.max(nuvals)

    def update(frame_number):
        print(frame_number)
        reset_figax(fig, ax, tight_layout=False, xlim=(-2.5, 2.5), ylim=(-10.0, 10.0))
        caxplot.set_ylim(numin, numax)
        caxplot.set_ylabel("Tune")
        for i in range(3):
            try:
                caxplot.lines.pop(0)
            except IndexError:
                pass


        xvals = x[frame_number]
        xprimevals = xprime[frame_number]

        delta = delta_vals[frame_number]
        nu = nuvals[frame_number]



        ax.scatter(xvals, xprimevals, c=turns_arr, s=plt.rcParams['lines.markersize']*25, cmap='viridis', edgecolor=scalarMap.to_rgba(delta), linewidth=1.5)
        caxplot.plot(delta_vals, nuvals, color='k', zorder=10)
        caxplot.axvline(delta, color='k', ls='--', lw=0.5, zorder=10)
        caxplot.axhline(nu, color='k', ls='--', lw=0.5, zorder=10)

    ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    if add_sextupole:
        title = 'results/multi_turn_fodo_fixed_fields_sext.mp4'
    else:
        title = 'results/multi_turn_fodo_fixed_fields.mp4'
    ani.save(title, writer=writer_video)
