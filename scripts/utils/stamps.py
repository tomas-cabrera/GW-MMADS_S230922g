import matplotlib.pyplot as plt

###############################################################################


def plot_stamp(stamp, ax=None, **kwargs):
    # Initialize figure and axis if needed
    if ax is None:
        fig, ax = plt.subplots()

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot stamp
    ax.imshow(stamp[:, ::-1], cmap="Greys_r", **kwargs)

    return ax
