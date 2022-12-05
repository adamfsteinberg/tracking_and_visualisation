"""A set of utility functions for animations etc"""
import numpy as np

def interpolate_between(xi, xf, count_steps):
    """Interpolates between initial and final positions
    Returns an array with one more axis than the originally passed values, where the new axis is the interpolation

    TODO Implement other kinds of steps (currently only linear option)
    """
    if isinstance(xi, float):
        xi = [xi]
        xf = [xf]
    xi = np.array(xi)
    xf = np.array(xf)
    if xi.shape != xf.shape:
        raise IndexError("The arrays passed for interpolation must have the same shape")

    interpolated_steps = xi[..., np.newaxis]*(1 + np.linspace(0, 1, count_steps)[(np.newaxis,)*xi.ndim])

    return np.moveaxis(interpolated_steps, -1, 0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x_points = [0, 1, 2, 3, 4]
    yi = [1, 1, 1, 1, 1]
    yf = [2, 2, 2, 2, 2]

    count_steps = 100

    y_interpolated = interpolate_between(yi, yf, count_steps)

    #plt.figure()
    for i in range(count_steps):
        plt.plot(x_points, y_interpolated[i])