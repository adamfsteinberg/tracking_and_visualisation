import numpy as np
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

# Still to actually be used
#import warnings
#np.seterr(all='warn')
#warnings.filterwarnings('error')

from tracking.tracking_utility import LatticeBase

class NonlinearElement(ABC):
    """A class for general 4D tracking through a general (nonlinear) element.
    Note that the element can be linear, however it may be easier to use the dedicated LinearLattice class in this case.
    """

    def __init__(self, element_length):
        """A parent class for a general nonlinear element
        """
        self.element_length = element_length

        self.extra_parameters: dict = NotImplemented  # Requires that all 'extra parameters' are listed in this dict

    @abstractmethod
    def dX_ds(self, s, X, *args) -> np.ndarray:
        """The differential equations to describe tracking a particle through the element.
        IMPORTANT: the user must ensure that the arguments passed match the functions given to the class
        """
        pass
        #return np.array()
        #return np.array([self.dx_ds(s, X, *args) ,
        #                 self.dpx_ds(s, X, *args),
        #                 self.dy_ds(s, X, *args) ,
        #                 self.dpy_ds(s, X, *args)
        #                 ])

    @abstractmethod
    def hamiltonian(self, *args):
        """Useful where the Hamiltonian is to be accessed directly, such as for plotting Hamiltonian contours
        pass"""

    def track(self, X0, args, delta_s=None, s0=0.0, s_eval=None, **kwargs):
        """Quick utility method to do the particle tracking, by integrating the differential equations directly"""
        if delta_s is None:
            delta_s = self.element_length
        s_span = (s0, s0+delta_s)
        res = solve_ivp(self.dX_ds, s_span, X0, 'DOP853', s_eval, args=args, **kwargs)

        if s_eval is not None:
            return res.t, res.y  # (s, X)  # Coordinates for all the specified s
        else:
            return res.y[:, -1]  # X[-1], final coordinates

class NonlinearLattice(LatticeBase):
    """A container for NonlinearElements, for tracking purposes"""

    def __init__(self, list_of_elements: list):
        super().__init__()

        if False in [isinstance(element, NonlinearElement) for element in list_of_elements]:
            raise TypeError(f"Only instances on NonlinearElement can be added to a NonlinearLattice")
        self.elements_list = list_of_elements
        self.elements_len = [element.element_length for element in list_of_elements]

        self.previous_distances = [0.0]
        for element_length in self.elements_len[:-1]:
            self.previous_distances.append(element_length + self.previous_distances[-1])

        self.elements_number = len(self.elements_len)
        self.len_total = np.sum(self.elements_len)

    def track(self, X0, args, s0=0.0, delta_s=None, periodic=True, **kwargs):
        """Todo: write docstring
        args corresponds to the arguments that must be sent for tracking, likely delta and beta_0
        Defaults to one full turn
        """
        if delta_s is None:
            delta_s = self.len_total

        X1 = X0
        # Calculate the distance that will be travelled through each of the elements: min is 0, max is element length
        s_elements = self._basic_travel_distance_calculation(delta_s)
        s_removed = s0

        for i in range(self.elements_number):
            X1 = self.elements_list[i].track(X1, args, np.maximum(s_elements[i] - s_removed, 0.0), **kwargs)
            s_removed = np.maximum(s_removed-self.elements_len[i], 0.0)


        # If there is still more s to travel and it's periodic, loop back to start of cell
        if periodic:
            s_remaining = self._basic_travel_distance_calculation(s0)
            for i in range(self.elements_number):
                X1 = self.elements_list[i].track(X1, args, s_remaining[i], **kwargs)
                #print(self.elements_list[i])
                #print(X1)

        return X1

    def track_many_turns(self, X0, args, count_turns, s0=0.0, q_max=10.0, p_max=0.3):  # Fairly conservatively low p_max
        """Utility function to quickly track through many turns.
        Returns the coordinates at the end of each turn, including the starting point
        """
        X_lists = ([X0[0]], [X0[1]], [X0[2]], [X0[3]])

        for i in range(count_turns):
            if not np.isnan(X0).any():
                X0 = self.track(X0, args, s0)
            else:
                X0 = [np.nan, np.nan, np.nan, np.nan]
            X0abs = np.abs(X0)
            if X0abs[0] > q_max or X0abs[2] > q_max or X0abs[1] > p_max or X0abs[3] > p_max or np.isnan(X0).any():
                # Break if it's diverged
                print("Tracking is diverging!")
            for j in range(4):
                X_lists[j].append(X0[j])
        x, px, y, py = [np.array(v) for v in X_lists]
        return x, px, y, py


# <editor-fold desc="Default implemented elements for nonlinear tracking">

class ScalingFFACurved(NonlinearElement):
    """Implements the equations of motion for a scaling FFA in curved coordinates, where r0 = rho = 1/h"""

    def __init__(self, element_length, k, r0, m, sigma=+1):
        super().__init__(element_length)

        self.extra_parameters = dict(k=k, r0=r0, m=m, sigma=sigma)

    def dX_ds(self, s, X, *args) -> np.ndarray:
        """
        args are delta (Delta P / P_0) and beta_0
        """
        x, px, y, py = X
        delta, beta = args
        P = np.sqrt(1+2*delta*beta + delta**2-px**2-py**2)

        dx_ds = px*(1+x/self.extra_parameters['r0'])/P
        dpx_ds = (P-self.extra_parameters['sigma']*(1+x/self.extra_parameters['r0'])**(self.extra_parameters['k']+1)*np.cos(self.extra_parameters['m']*y))/self.extra_parameters['r0']
        dy_ds = py*(1+x/self.extra_parameters['r0'])/P
        dpy_ds = self.extra_parameters['sigma']*self.extra_parameters['m']*(1+x/self.extra_parameters['r0'])**(self.extra_parameters['k']+2)*np.sin(self.extra_parameters['m']*y)/(self.extra_parameters['k']+2)
        return np.array([dx_ds, dpx_ds, dy_ds, dpy_ds])

    def hamiltonian(self, X, *args):
        """Where X = [x, px, y, py]"""
        bracket = 1+ X[0]/self.extra_parameters['r0']
        momentum_term = -bracket*np.sqrt(1+2*args[0]*args[1]+args[0]**2 -X[1]**2 -X[3]**3)
        position_term = np.power(bracket, self.extra_parameters['k']+2)*np.cos(self.extra_parameters['m']*X[2])/(self.extra_parameters['k']+2)*self.extra_parameters['sigma']
        return momentum_term + position_term + args[0]/args[1]

class ScalingFFAStraight(NonlinearElement):
    """Implements the equations of motion for a straight-scaling FFA"""

    def __init__(self, element_length, ref_rig, m, B0, x0):
        super().__init__(element_length)
        self.extra_parameters = dict(ref_rig=ref_rig, m=m, B0=B0, x0=x0)

    def dX_ds(self, s, X, *args) -> np.ndarray:
        """
        args are the actual rigidity
        """
        x, px, y, py = X
        rig = args[0]  # Should only be the one argument
        a = (rig/self.extra_parameters['ref_rig'])**2 - px**2 - py**2
        # Stop-gap solution to prevent diverging solutions not working (TODO: Find more general way to solve in parent class)
        if a <= 0:
            return np.array([0, 0, 0, 0])
        P = np.sqrt(a)

        dx_ds = px/P
        dpx_ds = -self.extra_parameters['B0']/self.extra_parameters['ref_rig']*np.exp(self.extra_parameters['m']*(x-self.extra_parameters['x0']))*np.cos(self.extra_parameters['m']*y)
        dy_ds = py/P
        dpy_ds = self.extra_parameters['B0']/self.extra_parameters['ref_rig']*np.exp(self.extra_parameters['m']*(x-self.extra_parameters['x0']))*np.sin(self.extra_parameters['m']*y)
        return np.array([dx_ds, dpx_ds, dy_ds, dpy_ds])

    def hamiltonian(self, *args):
        return NotImplemented


class DriftStraight(NonlinearElement):
    """Implements the equations of motion for a drift in cartesian coordinates"""

    def __init__(self, element_length, ref_rig=None):
        super().__init__(element_length)
        self.extra_parameters = dict(ref_rig=ref_rig)

    def dX_ds(self, s, X, *args) -> np.ndarray:
        """
        args are delta (Delta P / P_0) and beta_0
        """
        x, px, y, py = X
        try:
            delta, beta = args
            a = 1+2*delta*beta + delta**2-px**2-py**2
        except ValueError:  # If passing rigidity instead
            rig = args[0]
            a = (rig/self.extra_parameters['ref_rig'])**2 - px**2 -py**2
        # Stop-gap solution to prevent diverging solutions not working (TODO: Find more general way to solve in parent class)
        if a <= 0:
            return np.array([0, 0, 0, 0])
        P = np.sqrt(a)

        dx_ds = px/P
        dpx_ds = 0.0
        dy_ds = py/P
        dpy_ds = 0.0
        return np.array([dx_ds, dpx_ds, dy_ds, dpy_ds])

    def hamiltonian(self, *args):
        return NotImplemented


class QuadrupoleStraight(NonlinearElement):
    """Implements the equations of motion for a quadrupole in cartesian coordinates"""
    def __init__(self, element_length, K):
        """Quadrupole strength K, in m^-2.
        Strength is normalised to the 'reference momentum'"""
        super().__init__(element_length)
        self.extra_parameters = dict(K=K)

    def dX_ds(self, s, X, *args) -> np.ndarray:
        """
        args are delta (Delta P / P_0) and beta_0
        """
        x, px, y, py = X
        delta, beta = args
        a = 1+2*delta*beta + delta**2-px**2-py**2
        # Stop-gap solution to prevent diverging solutions not working (TODO: Find more general way to solve in parent class)
        if a <= 0:
            return np.array([0, 0, 0, 0])
        P = np.sqrt(a)

        dx_ds = px/P
        dpx_ds = -self.extra_parameters['K']*x
        dy_ds = py/P
        dpy_ds = self.extra_parameters['K']*y
        return np.array([dx_ds, dpx_ds, dy_ds, dpy_ds])

    def hamiltonian(self, *args):
        return NotImplemented


class MultipoleStraight(NonlinearElement):
    """Implements the equations of motion for a multipole in cartesian coordinates"""
    def __init__(self, element_length, strengths):
        """Strengths should be formatted as a list, [dipole, quadrupole, sextupole, octupole, ...]
        There is no limit to the maximum pole number, and low-order poles that are not included should have 0 strength
        Note: Assumes that there are no skew field components
        Strength is normalised to the 'reference momentum'"""
        super().__init__(element_length)
        self.extra_parameters = dict(strengths=strengths)
        self.highest_order = len(strengths)

    def dX_ds(self, s, X, *args) -> np.ndarray:
        """
        args are delta (Delta P / P_0) and beta_0
        """
        x, px, y, py = X
        delta, beta = args
        a = 1+2*delta*beta + delta**2-px**2-py**2
        # Stop-gap solution to prevent diverging solutions not working (TODO: Find more general way to solve in parent class)
        if a <= 0:
            return np.array([0, 0, 0, 0])
        P = np.sqrt(a)

        x_iy_n_sum = np.sum([self.extra_parameters['strengths'][n]*(x+1j*y)**n for n in range(self.highest_order)])

        dx_ds = px/P
        dpx_ds = -np.real(x_iy_n_sum)
        dy_ds = py/P
        dpy_ds = -np.imag(x_iy_n_sum)  # = + Re(j*x_iy_n_sum)
        return np.array([dx_ds, dpx_ds, dy_ds, dpy_ds])

    def hamiltonian(self, *args):
        return NotImplemented


# </editor-fold>

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    """Straight scaling FFA - using The KURRI example"""
    m = 11.0
    l_fmag = 0.15
    l_dmag = 0.30
    l0_bet = 0.5
    l0_out = 1.5
    x0 = 0.0
    B0_f = 0.136
    B0_d = -0.1438

    rig_ref = 0.383  # 7MeV H
    rig_min = 0.250  # 3MeV
    rig_max = 0.481  # 11MeV

    fmag = ScalingFFAStraight(l_fmag, rig_ref, m, B0_f, x0)
    dmag = ScalingFFAStraight(l_dmag, rig_ref, m, B0_d, x0)
    dri0 = DriftStraight(l0_out, rig_ref)
    dri1 = DriftStraight(l0_bet, rig_ref)
    lattice = NonlinearLattice([dri0, fmag, dri1, dmag, dri1, fmag, dri0])

    X0 = [0.08, 0.0, 0.0, 0.0]
    rig = rig_ref
    count_turns = 50

    x, px, y, py = lattice.track_many_turns(X0, [rig], count_turns, 0.0)

    fig, ax = plt.subplots()
    # Note: this arrangement could well be VERTICALLY unstable, even if horizontally stable
    ax.scatter(x, px, marker='x', c=np.linspace(0, 1, len(x)))

    fig.show()

if __name__ == '__main__0':
    # By way of a test, track through a single (shoddy) scaling FFA magnet (as no D-magnet, vertically unstable)
    import matplotlib.pyplot as plt

    # Lattice parameters
    l_mag = 0.1
    k = 27.5
    m_val = 0.01
    r0 = 2.0

    # Tracking parameters
    s0 = 0.0
    delta = 0.0
    beta_0 = 1.0
    count_turns = 23
    X0 = np.array([0.001, 0., 0.0000, 0.0])
    track_centre = True

    # For a pure scaling FFA magnet, the centre of the orbit can be determined algebraically
    x_centre = r0*(np.power(1+2*beta_0*delta+delta**2, 1/(2*k+2.0))-1)

    if track_centre:
        X0 += x_centre

    ffa_magnet = ScalingFFACurved(l_mag, k, r0, m_val)
    lattice = NonlinearLattice([ffa_magnet])
    x, px, y, py = lattice.track_many_turns(X0, (delta, beta_0), count_turns, s0)

    fig, ax = plt.subplots()
    # Note: this arrangement could well be VERTICALLY unstable, even if horizontally stable
    ax.scatter(x, px, marker='x', c=np.linspace(0, 1, len(x)))
    #plt.scatter(y, py, marker='x', color='k')
    #plt.axhline(0, color='r')

    # For interest, plot Hamiltonian contours
    x_ham = np.linspace(-20, 20, 100)*1E-3
    px_ham = np.linspace(-10, 10, 100)*1E-3
    X_ham, Px_ham = np.meshgrid(x_ham, px_ham)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    hamiltonian_values = ffa_magnet.hamiltonian([X_ham, Px_ham, 0, 0], delta, beta_0)
    ax1.contour(x_ham, px_ham, np.abs(hamiltonian_values), colors='k', linewidths=0.5, levels=50)
