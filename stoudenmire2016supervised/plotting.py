import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

import numpy as np

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
HUGE_SIZE = 20

plt.style.use('bmh')

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=BIG_SIZE)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
plt.rc('figure', figsize=(8,6))          # figure size
plt.rc('lines', linewidth=3)             # controls line width


def use_svg_display():
    """
    Use the svg format to display a plot in Jupyter.
    """
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5,2.5)):
    """
    Set the figure size for matplotlib.
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    Set the axes for matplotlib.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(b=True, which='both', color='k', linestyle=(0, (1, 10)))


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(8,6), axes=None,
         title=None):
    """
    Plot data points.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    axes.set_title(title)


def show_img(img: np.ndarray, axes=None, title=None, figsize=(1.5,1.5)) -> None:
    """
    Shows a single image.
    """
    axes = axes if axes else plt.gca()
    set_figsize(figsize)
    axes.imshow(img, cmap='Greys')
    axes.set_xticks([])
    axes.set_yticks([])
    if title:
        axes.set_title(title, fontsize=9)


def show_img_grid(imgs, titles) -> None:
    """
    Shows a grid of images.
    """
    n = int(np.ceil(len(imgs)**0.5))
    _, axes = plt.subplots(n,n)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        show_img(img, axes[i // n][i % n], title)