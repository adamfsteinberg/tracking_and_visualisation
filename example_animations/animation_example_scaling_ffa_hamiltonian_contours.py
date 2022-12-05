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

    count_delta = 500
    delta_max = 0.25

    delta_vals = np.linspace(-delta_max, delta_max, count_delta)

    x_centre = lambda delta_val : np.array([r0_val*(np.power(1+2*beta_0*delta_val+delta_val**2, 1/(2*k_val+2.0))-1), 0, 0, 0])  # Analytically find centre of orbit where there's only one magnet

    x_ham = np.linspace(-30, 30, 150)*1E-3
    px_ham = np.linspace(-25, 25, 150)*1E-3
    X_ham, Px_ham = np.meshgrid(x_ham, px_ham)

    print("Getting Hamiltonian values")
    hamiltonian_values = [ffa.hamiltonian([X_ham, Px_ham, 0, 0], delta, beta_0) for delta in delta_vals]
    print("Tracking completed")
    delta_vals = np.concatenate((delta_vals, delta_vals[::-1]))
    xprime_ham = px_ham / np.sqrt(1 + 2 * delta_vals[:, np.newaxis] * beta_0 + delta_vals[:, np.newaxis] ** 2 - px_ham ** 2)
    x_ham *= 1000
    px_ham *= 1000
    xprime_ham *= 1000

    # Set up the figure
    gs_kw = dict(width_ratios=[14, 1], height_ratios=[7, 0])
    fig, axd = plt.subplot_mosaic([['left', 'right'],
                                   ['left', 'right']],
                                  gridspec_kw=gs_kw, figsize=(15, 7),
                                  constrained_layout=True)
    ax, caxdelta,  = [axd['left'], axd['right']]

    colornorm = cm.colors.TwoSlopeNorm(0.0, vmin=-delta_max, vmax=delta_max)
    cb1 = mpl.colorbar.ColorbarBase(caxdelta, cmap=cm.coolwarm, norm=colornorm, orientation='vertical')
    scalarMap = cm.ScalarMappable(norm=colornorm, cmap=cm.coolwarm)
    caxdelta.set_xlabel(r'$\frac{P-P_0}{P_0}$')

    def update(frame_number):
        print(frame_number)
        ax.cla()
        vals = dict(xlabel="Position (mm)", ylabel="Angle (mrad)",  xlim=(-30.0, 30.0), ylim=(-15.0, 15.0))
        reset_figax(fig, ax, False, **vals)
        h = np.abs(hamiltonian_values[frame_number])
        spacing = np.linspace(0, 1, 50)**(1/6)
        levels = (h.max()-h.min())*spacing+h.min()
        #levels = np.geomspace(h.min(), h.max(), 50)
        delta = delta_vals[frame_number]
        ax.contour(x_ham, xprime_ham[frame_number], h, linewidths=0.8, levels=levels, colors=[scalarMap.to_rgba(delta)])
        #ax.scatter(xvals, xprimevals, c=delta, s=plt.rcParams['lines.markersize'] * 25, cmap='viridis', edgecolor=scalarMap.to_rgba(delta), linewidth=1.5)
        #ax.plot(xvals, xprimevals, color=scalarMap.to_rgba(delta), linewidth=1.5, zorder=-10)

    #update(0)
    ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    title = 'results/ffa_hamiltonian_contours_basic_px.mp4'
    ani.save(title, writer=writer_video)
