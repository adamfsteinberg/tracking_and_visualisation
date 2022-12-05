# Functions here allow for linear tracking in a cartesian coordinate system

import numpy as np

from tracking.tracking_utility import LatticeBase

def transfer_map(x0, px0, l, k0):
    """Manually transfer coordinates (x0, px0) -> (x1, px1).
    Assumes a quadrupole of length l [m], and normalised strength k0 [m^-2]
    """
    sqrtk = np.sqrt(np.abs(k0))
    if k0 > 0.0:
        v=dict(c = np.cos(sqrtk*l),  s = np.sin(sqrtk*l)/sqrtk,  cp=-np.sin(sqrtk*l)*sqrtk, sp=np.cos(sqrtk*l))
    elif k0 < 0.0:
        v=dict(c = np.cosh(sqrtk*l),  s = np.sinh(sqrtk*l)/sqrtk,  cp=np.sinh(sqrtk*l)*sqrtk, sp=np.cosh(sqrtk*l))
    else:
        v=dict(c = 1.0, s = l, cp = 0.0, sp = 1.0)

    x  = x0*v['c']  + px0*v['s']
    px = x0*v['cp'] + px0*v['sp']

    return x, px

class LinearLattice(LatticeBase):
    """A class to contain a simple lattice, for basic tracking.
    Only allows for tracking in one plane (eg x, px)
    """

    def __init__(self, elements_list):
        """Each entry in elements_list should be a dict, containing {length: < >, strength: < >}"""
        super().__init__()
        self.elements_list = elements_list

        try:
            self.elements_len  = [element['length']    for element in elements_list]
            self.elements_stren = [element['strength'] for element in elements_list]
        except KeyError:
            print("WARNING: at least one entry in elements_list does not contain ['length'] or ['strength']")
            self.elements_len   = []
            self.elements_stren = []

        self.previous_distances = [0.0]
        for element_length in self.elements_len[:-1]:
            self.previous_distances.append(element_length + self.previous_distances[-1])

        self.elements_number = len(self.elements_len)
        self.total_length = np.sum(self.elements_len)

    def __add__(self, other):
        if not isinstance(other, dict):
            raise TypeError(f"Attempting to add {type(other)} to a LinearLattice, currently only addition of dicts allowed")
        new_elements_list = self.elements_list.append(other)
        return LinearLattice(new_elements_list)

    def transfer_function(self, x0, px0, s0=0.0, delta_s=0.0, periodic=True):
        """The transfer function for the given lattice.
        x0, px0 are the initial position and normalised momentum. x0 measured in m, px0 is dimensionless.
        s0 is the initial longitudinal position along the beamline. Measured in m. Default value 0.0
        delta_s is the distance travelled along the beamline, relative to s0. Measured in m. Default value 0.0
        If periodic, particle tracking at the end of the lattice will wrap back to the beginning.
        x0, px0, s0, delta_s can be floats, or numpy arrays. If numpy arrays, shapes must be mutually broadcastable.

        Returns x1, px1

        Note that no checking is performed for 'sensible' values (eg delta_s > 0.0).
        """
        x1, px1 = x0, px0

        # Calculate the distance that will be travelled through each of the elements: min is 0, max is element length
        s_elements = self._basic_travel_distance_calculation(delta_s)
        s_removed = s0

        for i in range(self.elements_number):
            x1, px1 = transfer_map(x1, px1, np.maximum(s_elements[i] - s_removed, 0.0), self.elements_stren[i])
            s_removed = np.maximum(s_removed-self.elements_len[i], 0.0)

        # If there is still more s to travel and it's periodic, loop back to start of cell
        s_remaining = np.maximum(self._basic_travel_distance_calculation(s0), 0.0)
        if periodic:
            for i in range(self.elements_number):
                x1, px1 = transfer_map(x1, px1, s_remaining[i], self.elements_stren[i])
        return x1, px1

    def calculate_periodic_lattice_params(self, s=0.0, return_mu_too=False):
        """Calculate the periodic values of the lattice functions at some point s"""
        # Calculate the linear transfer matrix
        m11, m21 = self.transfer_function(1, 0, s, self.total_length)
        m12, m22 = self.transfer_function(0, 1, s, self.total_length)
        # Use the transfer matrix to produce the lattice parameters (following their well-known definitions)
        cos_mu = (m11+m22)/2
        sin_mu = np.sqrt(1-cos_mu**2)
        beta = m12/sin_mu
        alpha = (m11-cos_mu)/sin_mu
        gamma = -m21/sin_mu

        if return_mu_too:
            mu = np.arccos(cos_mu)
            return beta, alpha, gamma, mu
        return beta, alpha, gamma

    def transform_to_action_angle(self, x, px, s):
        """Convert from (x, px) to (phi, J) using the usual transformation"""
        beta, alpha, gamma = self.calculate_periodic_lattice_params(s)
        action = (gamma*x**2 + 2*alpha*x*px + beta*px**2)/2
        angle = np.arctan2(-alpha*x -beta*px, x)

        return angle, action

if __name__ == '__main__':
    # By way of a test,follow a single initial coordinate over many turns through a FODO lattice
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['text.usetex'] = True


    # Tracking parameters
    epsilon = 2.5E-6
    count_turns = 1001
    s0 = 0.0

    # Lattice parameters
    l_mag = 0.05
    l_drift = 0.1
    k0 = 50.0

    fodo = LinearLattice([dict(length=l_mag, strength=+k0), dict(length=l_drift, strength=0.0),
                          dict(length=l_mag, strength=-k0), dict(length=l_drift, strength=0.0)])
    lattice_length = 2*(l_mag+l_drift)

    beta, alpha, gamma, mu = fodo.calculate_periodic_lattice_params(s0, return_mu_too=True)
    x0 = np.sqrt(epsilon/gamma)
    px0 = 0.000


    # Perform tracking over many turns
    xvals = [x0]
    pxvals = [px0]
    for i in range(count_turns):
        newvals = fodo.transfer_function(xvals[-1], pxvals[-1], s0, lattice_length)
        xvals.append(newvals[0])
        pxvals.append(newvals[1])
    xvals = np.array(xvals)
    pxvals = np.array(pxvals)

    fig, ax = plt.subplots()

    # Plot tracking, and how Twiss parameters are related to the phase space ellipse
    ax.scatter(xvals*1000, pxvals*1000, marker='x', color='grey', s=25)
    horiz_arrow = patches.FancyArrowPatch((0, 0), (np.sqrt(epsilon/gamma)*1000, 0), arrowstyle='<->', mutation_scale=20, color='firebrick')
    verti_arrow = patches.FancyArrowPatch((0, 0), (0, np.sqrt(epsilon/beta)*1000), arrowstyle='<->', mutation_scale=20, color='royalblue')
    ax.add_patch(horiz_arrow)
    ax.add_patch(verti_arrow)
    ax.text(np.sqrt(epsilon/gamma)*1250, -0.75, r"${\sqrt{\frac{\varepsilon}{\gamma}}}$", color='firebrick', fontsize=25,
            horizontalalignment="center", verticalalignment="center")
    ax.text(-0.25, np.sqrt(epsilon/beta)*1250, r"$\sqrt{\frac{\varepsilon}{\beta}}$", color='royalblue', fontsize=25,
            horizontalalignment="center", verticalalignment="center")

    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Angle (mrad)")
    ax.set_title("Example FODO Phase Space Plot")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-4.0, 4.0)
    fig.tight_layout()

    # Could add to this example in the future: action-angle variables
    #angle, action = fodo.transform_to_action_angle(xvals, pxvals, s0)
    #action *= 1E6  # Convert to mm mrad
    #plt.figure()
    #plt.plot(action*np.cos(angle), action*np.sin(angle))
