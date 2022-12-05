"""An animation of a single particle in phase space (x, px) and action-angle (phi, J) variables"""

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

    # Set up the figure: ax1 for real space, ax2 for action-angle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    turns_back_to_start = 2*np.pi/mu

    s_values = np.linspace(0, lattice_length*turns_back_to_start, 751)[:-1]
    x, px = phase_space_ellipse_at_s(s_values%lattice_length)
    angle, action = fodo.transform_to_action_angle(x, px, s_values[:, np.newaxis]%lattice_length)
    x *= 1000
    px *= 1000
    action *= 1E6

    xtext = -1.85
    ytext = 3.65
    textsize = 25

    epsilon *= 1E6

    # Equation to describe the definition of J and phi
    action_eq = (r"\begin{eqnarray*}"
                 r"2J &=& \gamma x^2 + 2 \alpha x p_x + \beta p_x^2\\"
                 r"\end{eqnarray*}")
    angle_eq  = (r"\begin{eqnarray*}"
                 r"\tan(\phi) &=& -\alpha -\beta \frac{p_x}{x} "
                 r"\end{eqnarray*}")

    def update(frame_number):
        ax2.cla()
        ax2.set_xlim(-epsilon, epsilon)
        ax2.set_ylim(-epsilon, epsilon)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        reset_figax(fig, ax1)

        xvals = x[frame_number]
        pxvals = px[frame_number]
        actionvals = action[frame_number]
        anglevals = angle[frame_number]

        sval = s_values[frame_number]
        turn_number = int(np.floor(sval/lattice_length))
        sfrac = sval - turn_number*lattice_length

        x_highlight = x[frame_number, turn_number]
        px_highlight = px[frame_number, turn_number]
        action_highlight = action[frame_number, turn_number]
        angle_highlight = angle[frame_number, turn_number]

        ax1.scatter(xvals, pxvals, color='grey', s=plt.rcParams['lines.markersize']*5)
        ax2.scatter(actionvals*np.cos(anglevals), actionvals*np.sin(anglevals), color='grey', s=plt.rcParams['lines.markersize']*5)

        ax1.scatter(x_highlight, px_highlight, color='darkred', s=plt.rcParams['lines.markersize']*20)
        ax2.scatter(action_highlight*np.cos(angle_highlight), action_highlight*np.sin(angle_highlight), color='darkred', s=plt.rcParams['lines.markersize']*20)

        # Lines for labelling the action and the angle
        ax2.plot([0, action_highlight*np.cos(angle_highlight)], [0, action_highlight*np.sin(angle_highlight)], color='#c66b3d', zorder=-10, lw=1.5)  # From centre to highlighted point
        ax2.plot([0, epsilon/2], [0, 0], color='k', zorder=-10, lw=1.5)  # From centre to edge along x-axis, for angle
        x_for_angle, y_for_angle = circle_to(epsilon/15, 0, angle_highlight%(2*np.pi))
        ax2.plot(x_for_angle, y_for_angle, color="#1e2761", lw=1.25, zorder=-15)

        # Display the values of the action and angle
        ax2.text(-1.6, -1.7, f'$J = {action_highlight:.2f}$ mm mrad', color="#c66b3d", fontsize=27, horizontalalignment="left", verticalalignment="center")
        ax2.text(-1.6, -2.25, fr'$\phi = {angle_highlight%(2*np.pi)*180/np.pi:.0f} ^\circ$', color="#1e2761", fontsize=27, horizontalalignment="left", verticalalignment="center")


        # Display the definition of the action and the angle
        ax2.text(-1.2, 2.25, action_eq, color="#c66b3d", fontsize=27, horizontalalignment="left", verticalalignment="center")
        ax2.text(-1.65, 1.7, angle_eq , color="#1e2761", fontsize=27, horizontalalignment="left", verticalalignment="center")
        """
        if 0 <= sfrac and sfrac < l_mag:
            ax1.text(xtext, ytext, "Focusing", color='firebrick', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        elif l_mag + l_drift <= sfrac and sfrac <= 2*l_mag+l_drift:
            ax1.text(xtext, ytext, "Defocusing", color='royalblue', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        else:
            ax1.text(xtext, ytext, "Drift", color='grey', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        ax1.text(0.2, ytext, f"Turn Number: {turn_number+1}", color='k', fontsize=textsize, horizontalalignment="left", verticalalignment="center")
        """

    ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    title = 'results/action_angle_demo.mp4'
    ani.save(title, writer=writer_video)
