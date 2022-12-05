"""Show the evolution of an ensemble of particles
"""
from fodo_basis import *

from numpy.random import default_rng
from scipy.stats import truncnorm, rv_continuous

seed = 12345
rng1 = default_rng(seed)
rng2 = default_rng(seed+1)
rng3 = default_rng(seed+2)
rng4 = default_rng(seed+3)
import matplotlib.animation as animation

rng_phase_space = default_rng(0)

def populate_phase_space(beta, alpha, gamma, epsilon, num_particles, means=None):
    if means is None:
        means = [0, 0]  # Centred at origin by default
    sigma11 = epsilon*beta
    sigma12 = -epsilon*alpha
    sigma22 = epsilon*gamma

    cov = np.array([[sigma11, sigma12],
                    [sigma12, sigma22]])

    samples = rng_phase_space.multivariate_normal(means, cov, num_particles)
    q1 = samples[:, 0]
    q2 = samples[:, 1]
    return q1, q2


if __name__ == '__main__':
    # Tracking parameters
    epsilon_max_x = 2.5E-6*0.0625
    epsilon_max_y = 3.5E-6*0.0625

    count_turns = 20
    count_frames_per_turn = 100
    num_particles = 1000
    filename = 'results/phase_space_tracking/matched_20_turn.mp4'


    s0 = 0.0
    beta, alpha, gamma, mu = fodo.calculate_periodic_lattice_params(s0, return_mu_too=True)
    betay, alphay, gammay, muy = fodoy.calculate_periodic_lattice_params(s0, return_mu_too=True)  # Easy case as equivalent

    beta *= 1.5
    alpha *= 1.25
    gamma = (1+ alpha**2)/beta

    betay *= 0.5
    alphay *= 0.75
    gammay = (1+ alphay**2)/betay

    x0, px0 = populate_phase_space(beta, alpha, gamma, epsilon_max_x, num_particles)
    y0, py0 = populate_phase_space(betay, alphay, gammay, epsilon_max_y, num_particles)


    colors_val = np.sqrt(x0**2 + y0**2)[:, np.newaxis]

    # First of all, get the full phase space ellipse for the initial coordinates by iterating over many turns

    xvals_initial = [x0]
    yvals_initial = [y0]
    pxvals_initial = [px0]
    pyvals_initial = [py0]
    for i in range(count_turns-1):
        newvals = fodo.transfer_function(xvals_initial[-1], pxvals_initial[-1], 2*l_mag+2*l_drift, 0.0)
        newyvals = fodoy.transfer_function(yvals_initial[-1], pyvals_initial[-1], 2*l_mag+2*l_drift, 0.0)
        xvals_initial.append(newvals[0])
        yvals_initial.append(newyvals[0])
        pxvals_initial.append(newvals[1])
        pyvals_initial.append(newyvals[1])
    xvals_initial = np.array(xvals_initial)
    yvals_initial = np.array(yvals_initial)
    pxvals_initial = np.array(pxvals_initial)
    pyvals_initial = np.array(pyvals_initial)

    phase_space_ellipse_at_s = lambda s: fodo.transfer_function(xvals_initial[np.newaxis, ...], pxvals_initial[np.newaxis, ...], s[..., np.newaxis, np.newaxis], 0.0)
    phase_space_ellipse_at_s_y = lambda s: fodoy.transfer_function(yvals_initial[np.newaxis, ...], pyvals_initial[np.newaxis, ...], s[..., np.newaxis, np.newaxis], 0.0)

    # Set up the figure
    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    spec = fig.add_gridspec(ncols=2, nrows=2)
    ax_up = fig.add_subplot(spec[0, 0])
    ax_mi = fig.add_subplot(spec[1, 0])
    ax_ri = fig.add_subplot(spec[1, 1])
    #fig.subplots_adjust(wspace=0, hspace=0)
    #ax_up.set_xticklabels([])
    #ax_ri.set_yticklabels([])

    s_values = np.linspace(0, lattice_length, count_frames_per_turn, endpoint=False)
    x, px = phase_space_ellipse_at_s(s_values)
    y, py = phase_space_ellipse_at_s_y(s_values)
    x *= 1000
    y *= 1000
    px *= 1000
    py *= 1000

    xtext = -1.85
    ytext = 3.65
    textsize = 25

    x_vals_for_magnet = np.linspace(0.1, 2.5, 200)
    inf_like = np.ones_like(x_vals_for_magnet)*10000
    y_vals_for_magnet = 2/x_vals_for_magnet

    total_frame_number = count_frames_per_turn * count_turns

    def update_first_section(frame_number):
        print(f'{frame_number+1}/{count_frames_per_turn*count_turns}')
        reset_figax(fig, ax_up, tight_layout=False, xlabel="", ylabel="Angle ($x$) [mrad]", xlim=(-2.0, 2.0), ylim=(-4.0, 4.0))
        reset_figax(fig, ax_mi, tight_layout=False, xlabel="Position ($x$) [mm]", ylabel="Position ($y$) [mm]", xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
        reset_figax(fig, ax_ri, tight_layout=True,  xlabel="Angle ($y$) [mrad]", ylabel="", xlim=(-4.0, 4.0), ylim=(-2.0, 2.0))
        #fig.subplots_adjust(wspace=0, hspace=0)
        leng_num = frame_number % count_frames_per_turn
        turn_num = frame_number*count_turns//total_frame_number
        xvals = x[leng_num, turn_num]
        yvals = y[leng_num, turn_num]
        pxvals = px[leng_num, turn_num]
        pyvals = py[leng_num, turn_num]
        colors = colors_val  # Matches each particle
        sval = s_values[leng_num]
        #ax.scatter(xvals, pxvals, color='k')
        ax_up.scatter(xvals, pxvals, c=colors, cmap='hsv')
        ax_mi.scatter(xvals, yvals, c=colors, cmap='hsv')
        ax_ri.scatter(pyvals, yvals, c=colors, cmap='hsv')
        if 0 <= sval and sval < l_mag:
            c1 = 'royalblue'
            c2 = 'firebrick'
        elif l_mag + l_drift <= sval and sval <= 2*l_mag+l_drift:
            c1 = 'firebrick'
            c2 = 'royalblue'
        else:
            c1 = 'white'
            c2 = 'white'
        if count_frames_per_turn > 25:
            ax_mi.fill_between(x_vals_for_magnet, y_vals_for_magnet, inf_like, lw=2, zorder=-10, color=c1, alpha=0.5)
            ax_mi.fill_between(-x_vals_for_magnet, -y_vals_for_magnet, -inf_like, lw=2, zorder=-10, color=c1, alpha=0.5)
            ax_mi.fill_between(x_vals_for_magnet, -y_vals_for_magnet, -inf_like, lw=2, zorder=-10, color=c2, alpha=0.5)
            ax_mi.fill_between(-x_vals_for_magnet, y_vals_for_magnet, inf_like, lw=2, zorder=-10, color=c2, alpha=0.5)


    ani = animation.FuncAnimation(fig, update_first_section, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    ani.save(filename, writer=writer_video)
