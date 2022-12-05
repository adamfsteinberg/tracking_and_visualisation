"""An animation of a single particle going through many turns"""

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

    base = 51
    count_turns_for_ani = 40
    s_section_bounds = [0, lattice_length * 5, lattice_length * 10, lattice_length * 20, lattice_length * 40]
    s_values_a = np.linspace(s_section_bounds[0], s_section_bounds[1], base * 10)[:-1]
    s_values_b = np.linspace(s_section_bounds[1], s_section_bounds[2], base * 5)[:-1]
    s_values_c = np.linspace(s_section_bounds[2], s_section_bounds[3], base * 5)[:-1]
    s_values_d = np.linspace(s_section_bounds[3], s_section_bounds[4], base * 5)[:-1]
    s_values = np.concatenate((s_values_a, s_values_b, s_values_c, s_values_d))
    x, px = phase_space_ellipse_at_s(s_values%lattice_length)
    x *= 1000
    px *= 1000

    xtext = -1.85
    ytext = 3.65
    textsize = 25

    plot_ellipse_from_turn = 150

    def update(frame_number):
        reset_figax(fig, ax)
        xvals = x[frame_number]
        pxvals = px[frame_number]

        sval = s_values[frame_number]
        turn_number = int(np.floor(sval/lattice_length))
        sfrac = sval - turn_number*lattice_length

        x_highlight = x[frame_number, turn_number]
        x_highlighted = x[0, 0:turn_number+1]
        px_highlight = px[frame_number, turn_number]
        px_highlighted = px[0, 0:turn_number+1]

        if turn_number+1 >=plot_ellipse_from_turn:
            ax.scatter(xvals, pxvals, color='grey', s=plt.rcParams['lines.markersize']*5)
        else:
            ax.scatter(x_highlighted, px_highlighted, color='grey', s=plt.rcParams['lines.markersize']*5)
        ax.scatter(x_highlight, px_highlight, color='darkred', s=plt.rcParams['lines.markersize']*20)
        if 0 <= sfrac and sfrac < l_mag:
            ax.text(xtext, ytext, "Focusing", color='firebrick', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        elif l_mag + l_drift <= sfrac and sfrac <= 2*l_mag+l_drift:
            ax.text(xtext, ytext, "Defocusing", color='royalblue', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        else:
            ax.text(xtext, ytext, "Drift", color='grey', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        ax.text(0.2, ytext, f"Turn Number: {turn_number+1}", color='k', fontsize=textsize, horizontalalignment="left", verticalalignment="center")

        play_print = r'\Forward'
        count_play_print = np.sum([1 if slim < sval else 0 for slim in s_section_bounds])
        ax.text(1.0, -ytext, f'{play_print*count_play_print}', color='k', fontsize=textsize, horizontalalignment="left", verticalalignment="center")

    ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    title = 'results/follow_one_particle_fodo.mp4'
    ani.save(title, writer=writer_video)
