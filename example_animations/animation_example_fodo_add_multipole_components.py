"""Showing how a plot of the phase space varies as the multipole components are varied"""

from fodo_basis import *
import matplotlib.animation as animation
from matplotlib import cm


from tracking.nonlinear_tracking import DriftStraight, NonlinearLattice, MultipoleStraight, QuadrupoleStraight

if __name__ == '__main__':
    k0 = 125
    def make_lattice(strengths):
        fmag = QuadrupoleStraight(l_mag, k0)
        dmag = QuadrupoleStraight(l_mag, -k0)
        sextupole = MultipoleStraight(0.005, strengths)
        drift = DriftStraight(l_drift)

        fodolist = [fmag, drift, sextupole, dmag, drift]
        fodo = NonlinearLattice(fodolist)
        return fodo

    n_orbits = 5
    n_sext_strengths = 101
    sext_strengths = np.linspace(0, k0*500, n_sext_strengths)
    lattices = [make_lattice([0, 0, sext_component]) for sext_component in sext_strengths]

    x0_original = 0.0007606693378842227
    X0_vals = [np.array([x0, 0, 0, 0]) for x0 in np.linspace(0, x0_original*10, n_orbits)]

    delta = 0.0
    beta_0 = 1.0
    count_turns = 150
    turns_arr = np.arange(1, count_turns+1+0.001, 1)

    print("Doing tracking")
    X_many_turns = np.array([[lattice.track_many_turns(X0, (delta, beta_0), count_turns, 0.0) for lattice in lattices] for X0 in X0_vals])
    print("Tracking completed")
    x =  X_many_turns[:, :, 0, :]
    px = X_many_turns[:, :, 1, :]
    y =  X_many_turns[:, :, 2, :]
    py = X_many_turns[:, :, 3, :]
    xprime = px/np.sqrt(1+2*delta*beta_0 + delta**2-px**2-py**2)
    x *= 1000
    xprime *= 1000

    # Set up the figure
    fig, ax = plt.subplots(figsize=(7, 7))


    def update(frame_number):
        print(frame_number)
        reset_figax(fig, ax, xlim=(-30, 30), ylim=(-125, 125))
        #ax.cla()  # Temporary

        xvals = x[:, frame_number]
        xprimevals = xprime[:, frame_number]

        for i in range(n_orbits):  # Could colour the different orbits
            ax.scatter(xvals[i], xprimevals[i], c=turns_arr, s=plt.rcParams['lines.markersize']*2.5, cmap='viridis')

    ani = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=int(1E7))
    writer_video = animation.FFMpegWriter(fps=30)
    title = 'results/multi_turn_fodo_adding_sextupole3.mp4'
    ani.save(title, writer=writer_video)
