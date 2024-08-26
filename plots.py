from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import griddata


def plot_2D_irregular_heatmap(x, y, z, xi=None, yi=None, ax=None, log_z=False,**kwargs):
    if ax is None:
        ax = plt.gca()
    if xi is None:
        xi = np.linspace(x.min(), x.max(), 1024)
    if yi is None:
        yi = np.linspace(y.min(), y.max(), 1024)
    if log_z:
        norm = matplotlib.colors.LogNorm()
    else:
        norm = None
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="linear")

    im = ax.pcolormesh(xi, yi, zi, shading="auto",
                       norm=norm, **kwargs)
    plt.colorbar(im, ax=ax)
    return im


def plot_scatter_value(x, y, values, resolution=1024, ax=None,
                       norm=matplotlib.colors.LogNorm(),
                       default_value=0.0):
    if ax is None:
        ax = plt.gca()
    # Convert JAX arrays to NumPy for operations not supported in JAX
    x_np, y_np, values_np = (np.array(x.flatten()), np.array(y.flatten()),
                             np.array(values.flatten()))

    # Identify and remove NaNs and Infs from the data
    valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np) & ~np.isnan(values_np) & \
                 ~np.isinf(x_np) & ~np.isinf(y_np) & ~np.isinf(values_np)
    x_filtered, y_filtered, values_filtered = x_np[valid_mask], y_np[valid_mask], values_np[valid_mask]

    # Binning the coordinates
    xmin, xmax = x_filtered.min(), x_filtered.max()
    ymin, ymax = y_filtered.min(), y_filtered.max()
    xbins = np.linspace(xmin, xmax, resolution)
    ybins = np.linspace(ymin, ymax, resolution)
    hist, xedges, yedges = np.histogram2d(x_filtered, y_filtered, bins=(xbins, ybins), weights=values_filtered)
    counts, _, _ = np.histogram2d(x_filtered, y_filtered, bins=(xbins, ybins))

    # Calculate mean values for bins with more than one point
    mean_values = np.divide(hist, counts,
                            out=default_value*np.ones_like(hist),
                            where=counts>0)

    # Plotting
    im = ax.imshow(mean_values.T, extent=[xmin, xmax, ymin, ymax],
               origin='lower', cmap='viridis', interpolation='nearest',
               aspect='auto', norm=norm)
    plt.colorbar(im, ax=ax)
    return im


