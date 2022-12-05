"""Plot the periodic Twiss parameters in the linear case"""

from fodo_basis import *
import matplotlib.animation as animation
import matplotlib.patches as patches


if __name__ == '__main__':

    # Tracking parameters
    epsilon = 2.5E-6
    count_turns = 1001
    s0 = 0.0

    beta0, alpha0, gamma0, mu = fodo.calculate_periodic_lattice_params(s0, return_mu_too=True)
    x0 = np.sqrt(epsilon / gamma0)
    px0 = 0.000

    # First of all, get the full phase space ellipse for the initial coordinates by iterating over many turns

    xvals_initial = [x0]
    pxvals_initial = [px0]
    for i in range(count_turns-1):
        newvals = fodo.transfer_function(xvals_initial[-1], pxvals_initial[-1], 2*l_mag+2*l_drift, 0.0)
        xvals_initial.append(newvals[0])
        pxvals_initial.append(newvals[1])
    xvals_initial = np.array(xvals_initial)
    pxvals_initial = np.array(pxvals_initial)


    phase_space_ellipse_at_s = lambda s: fodo.transfer_function(xvals_initial[np.newaxis, :], pxvals_initial[np.newaxis, :], s[:, np.newaxis], 0.0)

    # Set up the figure: ax1 for real space, ax2 for action-angle
    gs_kw = dict(width_ratios=[7, 0], height_ratios=[7, 2])
    fig, axd = plt.subplot_mosaic([['upper', 'upper'],
                                   ['lower', 'lower']],
                                  gridspec_kw=gs_kw, figsize=(7, 9),
                                  constrained_layout=True)

    ax, ax2 = [axd['upper'], axd['lower']]


    s_values = np.linspace(0, lattice_length, 501)[:-1]
    x, px = phase_space_ellipse_at_s(s_values)
    beta, alpha, gamma = fodo.calculate_periodic_lattice_params(s_values)
    x *= 1000
    px *= 1000

    xtext = -1.85
    ytext = 3.65
    textsize = 25
    max2 = np.max(beta)*1.1
    def update(frame_number):
        print(frame_number)
        ax2.cla()
        ax2.set_xlim(s_values[0]*100, s_values[-1]*100)
        ax2.set_ylim(0, max2)
        ax2.set_xlabel("Longitudinal Position (cm)")
        ax2.set_ylabel("Beta Function (m)")
        reset_figax(fig, ax, tight_layout=False)

        xvals = x[frame_number]
        pxvals = px[frame_number]

        sval = s_values[frame_number]

        #x_highlight = x[frame_number, turn_number]
        #px_highlight = px[frame_number, turn_number]
        beta_highlight = beta[frame_number]
        gamma_highlight = gamma[frame_number]

        ax.scatter(xvals, pxvals, color='grey', s=plt.rcParams['lines.markersize']*5)
        ax2.plot(s_values*100, beta, color='grey', lw=1.75, zorder=-5)

        #ax.scatter(x_highlight, px_highlight, color='darkred', s=plt.rcParams['lines.markersize']*20)
        ax2.scatter(sval*100, beta_highlight, color='darkred', s=plt.rcParams['lines.markersize']*20, zorder=5)

        horiz_arrow = patches.FancyArrowPatch((0, 0), (np.sqrt(epsilon/gamma_highlight)*1000, 0), arrowstyle='<->', mutation_scale=20, color='#c66b3d')
        verti_arrow = patches.FancyArrowPatch((0, 0), (0, np.sqrt(epsilon/beta_highlight)*1000), arrowstyle='<->', mutation_scale=20, color='#1e2761')
        ax.add_patch(horiz_arrow)
        ax.add_patch(verti_arrow)
        ax.text(np.sqrt(epsilon/gamma_highlight)*250, -0.75, r"${\sqrt{\frac{\varepsilon}{\gamma}}}$", color='#c66b3d', fontsize=25,
                horizontalalignment="center", verticalalignment="center")
        ax.text(-0.25, np.sqrt(epsilon/beta_highlight)*250, r"$\sqrt{\frac{\varepsilon}{\beta}}$", color='#1e2761', fontsize=25,
                horizontalalignment="center", verticalalignment="center")

        if 0 <= sval and sval < l_mag:
            ax.text(xtext, ytext, "Focusing", color='firebrick', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        elif l_mag + l_drift <= sval and sval <= 2*l_mag+l_drift:
            ax.text(xtext, ytext, "Defocusing", color='royalblue', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        else:
            ax.text(xtext, ytext, "Drift", color='grey', fontsize=textsize, horizontalalignment="left", verticalalignment="center")


    ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    title = 'results/twiss_demo.mp4'
    ani.save(title, writer=writer_video)
