import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import h5py as h5
import pathlib

# Figure settings
### FIGURE SIZES
fig_width_pt = 2*246.0  # Get this from LaTeX using \the\columnwidth (this is for prd)
inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean      # height in inches
square_size = [fig_width, fig_width]
rect_size = [fig_width, fig_height]
long_size = [1.5 * fig_width, fig_height]
longest_size = [2 * fig_width, fig_height]
vert_rect_size = [fig_width, 2*fig_height]
vert_square_size = [fig_width, 2*fig_width]
vert_long_size = [1.5 * fig_width, 2*fig_height]



def get_color_name(rgb):
    """
    Return the closest named color in Matplotlib's color dictionary for a given RGB(A) tuple.

    Parameters:
    - rgb (tuple): RGB or RGBA tuple with values between 0 and 1.

    Returns:
    - str: Name of the closest color.
    """
    # Convert to RGB if RGBA is given
    if len(rgb) == 4:
        rgb = rgb[:3]  # Ignore alpha channel
    
    # Convert RGB to np array for distance calculation
    rgb = np.array(rgb)

    # Get list of named colors in Matplotlib
    color_names = mcolors.CSS4_COLORS  # More extensive than BASE_COLORS

    # Convert named colors to RGB values
    color_rgb = np.array([mcolors.to_rgb(color) for color in color_names.values()])

    # Compute Euclidean distance between input RGB and named colors
    distances = np.linalg.norm(color_rgb - rgb, axis=1)

    # Find the closest color
    closest_color = list(color_names.keys())[np.argmin(distances)]
    
    return closest_color

def get_colormap_colors(n, cmap_name="viridis"):
    """
    Return an array of n colors sampled from the specified colormap.
    
    Parameters:
    - n (int): Number of colors to return.
    - cmap_name (str): Name of the colormap (default: 'viridis').
    
    Returns:
    - List of RGB tuples.
    """
    cmap = plt.get_cmap(cmap_name)
    
    colors = [cmap(i / (n - 1)) for i in range(n)]  # Normalize indices
    name = []
    for i in range(n):
        name.append(get_color_name(colors[i]))
    return name
    

def galaxy_spectrum(alpha, amp, a1, ak, b1, bk, logf2, Tobs, f = 1e-3):
    logf1 = a1 * np.log10(Tobs) + b1
    logfkn = ak * np.log10(Tobs) + bk

    Sh = 10**amp*np.exp(-(f/10**logf1)**alpha) *\
        (f**(-7./3.))*0.5*(1.0 + np.tanh(-(f-10**logfkn)/10**logf2))
    
    L = 8.3
    fstar = 1 / L
    x = 2*np.pi * (f / fstar) * np.sin(2*np.pi * (f / fstar))

    return Sh*x**2


def galaxy_spectrum_short(alpha, amp, logf1, logf2, logfkn, f = 1e-3):


    Sh = 10**amp*np.exp(-(f/10**logf1)**alpha) *\
        (f**(-7./3.))*0.5*(1.0 + np.tanh(-(f-10**logfkn)/10**logf2))
    
    L = 8.3
    fstar = 1 / L
    x = 2*np.pi * (f / fstar) * np.sin(2*np.pi * (f / fstar))

    return Sh*x**2
    
    

def kl_divergence_1d(p_samples, q_samples, grid=None, bins=1000, epsilon=1e-10):
    """
    Compute KL divergence between two 1D sample distributions using KDE.
    
    Parameters:
    -----------
    p_samples : array-like
        Samples from distribution P
    q_samples : array-like  
        Samples from distribution Q
    grid : array-like, optional
        Evaluation grid. If None, created automatically
    bins : int
        Number of grid points if grid is None
    epsilon : float
        Small value to avoid log(0)
    
    Returns:
    --------
    float : KL divergence D(P||Q)
    """
    # Handle edge cases
    if len(p_samples) == 0 or len(q_samples) == 0:
        return np.inf
    
    # KDE
    kde_p = gaussian_kde(p_samples)
    kde_q = gaussian_kde(q_samples)
    
    # Grid
    if grid is None:
        xmin = min(np.min(p_samples), np.min(q_samples))
        xmax = max(np.max(p_samples), np.max(q_samples))
        # Add small buffer to avoid edge effects
        buffer = 0.1 * (xmax - xmin)
        grid = np.linspace(xmin - buffer, xmax + buffer, bins)
    
    dx = grid[1] - grid[0]
    
    # Densities
    p = kde_p(grid)
    q = kde_q(grid)
    
    # Add epsilon to avoid numerical issues
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    
    # Normalize to ensure they integrate to 1
    p = p / (np.sum(p) * dx)
    q = q / (np.sum(q) * dx)
    
    # KL divergence
    kl = np.sum(p * np.log(p / q)) * dx
    
    return kl

def jensen_shannon_divergence_1d(p_samples, q_samples, grid=None, bins=1000, epsilon=1e-10):
    """
    Compute Jensen-Shannon divergence between two 1D sample distributions using KDE.
    
    Parameters:
    -----------
    p_samples : array-like
        Samples from distribution P
    q_samples : array-like  
        Samples from distribution Q
    grid : array-like, optional
        Evaluation grid. If None, created automatically
    bins : int
        Number of grid points if grid is None
    epsilon : float
        Small value to avoid log(0)
    
    Returns:
    --------
    float : Jensen-Shannon divergence
    """
    # Handle edge cases
    if len(p_samples) == 0 or len(q_samples) == 0:
        return np.inf
    
    # KDE for p and q
    kde_p = gaussian_kde(p_samples)
    kde_q = gaussian_kde(q_samples)
    
    # Grid
    if grid is None:
        xmin = min(np.min(p_samples), np.min(q_samples))
        xmax = max(np.max(p_samples), np.max(q_samples))
        # Add small buffer to avoid edge effects
        buffer = 0.1 * (xmax - xmin)
        grid = np.linspace(xmin - buffer, xmax + buffer, bins)
    
    dx = grid[1] - grid[0]
    
    # Densities
    p = kde_p(grid)
    q = kde_q(grid)
    
    # Add epsilon to avoid numerical issues
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    
    # Normalize to ensure they integrate to 1
    p = p / (np.sum(p) * dx)
    q = q / (np.sum(q) * dx)
    
    # Average distribution (M)
    m = 0.5 * (p + q)
    
    # Compute KL divergences to M with safe division
    kl_p_m = np.sum(p * np.log(p / m)) * dx
    kl_q_m = np.sum(q * np.log(q / m)) * dx
    
    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd

def jensen_shannon_per_parameter(samples_p, samples_q, parameter_names=None):
    """
    Compute Jensen-Shannon divergence for each parameter dimension.
    
    Parameters:
    -----------
    samples_p : array-like, shape (n_samples, n_params)
        Samples from distribution P
    samples_q : array-like, shape (n_samples, n_params)  
        Samples from distribution Q
    parameter_names : list, optional
        Names of parameters for reporting
    
    Returns:
    --------
    list : Jensen-Shannon divergences for each parameter
    dict : If parameter_names provided, returns dict with named results
    """
    # Ensure samples are 2D arrays
    samples_p = np.atleast_2d(samples_p)
    samples_q = np.atleast_2d(samples_q)
    
    # If 1D arrays were passed, transpose to get correct shape
    if samples_p.shape[0] == 1:
        samples_p = samples_p.T
    if samples_q.shape[0] == 1:
        samples_q = samples_q.T
    
    n_params = samples_p.shape[1]
    
    # Check dimensions match
    if samples_q.shape[1] != n_params:
        raise ValueError("samples_p and samples_q must have same number of parameters")
    
    print(f"Computing JS divergence for {n_params} parameters")
    
    jsd_values = []
    for i in range(n_params):
        try:
            jsd = jensen_shannon_divergence_1d(samples_p[:, i], samples_q[:, i])
            jsd_values.append(jsd)
        except Exception as e:
            print(f"Error computing JSD for parameter {i}: {e}")
            jsd_values.append(np.nan)
    
    # Return as dict if parameter names provided
    if parameter_names is not None:
        if len(parameter_names) != n_params:
            print(f"Warning: {len(parameter_names)} names provided for {n_params} parameters")
        return dict(zip(parameter_names[:n_params], jsd_values))
    
    return jsd_values


