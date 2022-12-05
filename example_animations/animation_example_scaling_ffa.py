"""Show how a scaling FFA has many orbits, spatially separated, with constant tune"""

from fodo_basis import *
import matplotlib.animation as animation
from matplotlib import cm

from tracking.nonlinear_tracking import ScalingFFACurved, NonlinearLattice


if __name__ == '__main__':
    l_mag = 0.1
    k_val = 27.5
    m_val = 0.01
    r0_val = 2.0
    beta_0 = 1.0
    ffa = ScalingFFACurved(l_mag, k_val, r0_val, m_val)
    ffa_lattice = NonlinearLattice([ffa])

    count_turns = 24
    count_delta = 5
    delta_max = 0.25

    delta_vals = np.linspace(-delta_max, delta_max, count_delta)
    turns_vals = np.arange(1, count_turns + 1 + 0.001, 1)

    X0 = np.array([0.00275, 0., 0.0000, 0.0])

    x_centre = lambda delta_val : np.array([r0_val*(np.power(1+2*beta_0*delta_val+delta_val**2, 1/(2*k_val+2.0))-1), 0, 0, 0])  # Analytically find centre of orbit where there's only one magnet

    print("Doing tracking")
    X_many_turns = np.array([ffa_lattice.track_many_turns(X0 + x_centre(delta), (delta, beta_0), count_turns, 0.0) for delta in delta_vals])
    print("Tracking completed")
    x =  np.concatenate((X_many_turns[:, 0, :], X_many_turns[:, 0, :][::-1]))
    px = np.concatenate((X_many_turns[:, 1, :], X_many_turns[:, 1, :][::-1]))
    y =  np.concatenate((X_many_turns[:, 2, :], X_many_turns[:, 2, :][::-1]))
    py = np.concatenate((X_many_turns[:, 3, :], X_many_turns[:, 3, :][::-1]))
    delta_vals = np.concatenate((delta_vals, delta_vals[::-1]))
    xprime = px/np.sqrt(1+2*delta_vals[:, np.newaxis]*beta_0 + delta_vals[:, np.newaxis]**2-px**2-py**2)
    x *= 1000
    xprime *= 1000

    # Set up the figure
    gs_kw = dict(width_ratios=[7, 1], height_ratios=[7, 0])
    fig, axd = plt.subplot_mosaic([['left', 'right'],
                                   ['left', 'right']],
                                  gridspec_kw=gs_kw, figsize=(7, 8),
                                  constrained_layout=True)
    ax, caxturn,  = [axd['left'], axd['right']]

    colornorm = cm.colors.TwoSlopeNorm(0.0, vmin=-delta_max, vmax=delta_max)
    #cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm.coolwarm, norm=colornorm, orientation='horizontal')
    scalarMap = cm.ScalarMappable(norm=colornorm, cmap=cm.coolwarm)
    #cax.set_xlabel(r'$\frac{P-P_0}{P_0}$')

    colornorm2 = mpl.colors.Normalize(vmin=0, vmax=count_turns)
    cb2 = mpl.colorbar.ColorbarBase(caxturn, cmap=mpl.cm.viridis, norm=colornorm2, orientation='vertical')
    caxturn.set_xlabel('Turn Number')

    numin = 0.0  #np.min(nuvals)
    numax = 0.5  #np.max(nuvals)

    def update(frame_number):
        print(frame_number)
        ax.cla()
        vals = dict(xlabel="Position (mm)", ylabel="Angle (mrad)")
        #            xlim=(-2.0, 2.0), ylim=(-4.0, 4.0))
        ax.set_xlabel(vals['xlabel'])
        ax.set_ylabel(vals['ylabel'])
        #ax.set_xlim(*vals['xlim'])
        #ax.set_ylim(*vals['ylim'])

        for i in range(count_delta):
            delta = delta_vals[i]
            xvals = x[i]
            xprimevals = xprime[i]
            ax.scatter(xvals, xprimevals, c=turns_vals, s=plt.rcParams['lines.markersize'] * 25, cmap='viridis', edgecolor=scalarMap.to_rgba(delta), linewidth=1.5)
            ax.plot(xvals, xprimevals, color=scalarMap.to_rgba(delta), linewidth=1.5, zorder=-10)

    update(0)
    #ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(10))
    #writer_video = animation.FFMpegWriter(fps=30)
    #title = 'results/multi_turn_ffa.mp4'
    #ani.save(title, writer=writer_video)
