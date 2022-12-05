"""A quick check that all the basic ideas for animations are working, by converting a lattice from the usual
coordinates to action angle variables.
"""
from fodo_basis import *

import matplotlib.animation as animation

if __name__ == '__main__':
    # Tracking parameters
    epsilon = 2.5E-6
    count_turns = 1001
    s0 = 0.0

    beta, alpha, gamma, mu = fodo.calculate_periodic_lattice_params(s0, return_mu_too=True)
    x0 = np.sqrt(epsilon/gamma)
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

    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7))

    s_values = np.linspace(0, lattice_length, 361)[:-1]  # Last point = first point
    count_turns_for_animation = 10
    x, px = phase_space_ellipse_at_s(s_values)
    angle, action = fodo.transform_to_action_angle(x, px, s_values[:, np.newaxis])
    normalised_angle = (angle - np.min(angle))/(np.max(angle)-np.min(angle))  # Between 0 and 1
    x *= 1000
    px *= 1000

    xtext = -1.85
    ytext = 3.65
    textsize = 25

    with_color = False
    def update_first_section(frame_number):
        reset_figax(fig, ax)
        xvals = x[frame_number]
        pxvals = px[frame_number]
        if with_color:
            colors = normalised_angle[frame_number]
        else:
            colors = 'k'
        sval = s_values[frame_number]
        #ax.scatter(xvals, pxvals, color='k')
        ax.scatter(xvals, pxvals, c=colors, cmap='hsv')
        if 0 <= sval and sval < l_mag:
            ax.text(xtext, ytext, "Focusing", color='firebrick', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        elif l_mag + l_drift <= sval and sval <= 2*l_mag+l_drift:
            ax.text(xtext, ytext, "Defocusing", color='royalblue', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        else:
            ax.text(xtext, ytext, "Drift", color='grey', fontsize=textsize, horizontalalignment="left", verticalalignment="center")

    ani = animation.FuncAnimation(fig, update_first_section, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    if with_color:
        title = "results/straightforward_fodo_color.mp4"
    else:
        title = 'results/straightforward_fodo.mp4'
    ani.save(title, writer=writer_video)
