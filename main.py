import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


def _validate_input(data: NDArray[np.complex128], time: NDArray) -> None:
    """Validate input data for complex matrix animation.

    Args:
        data: Input array to validate

    Raises:
        ValueError: If data is not complex128 or not a square matrix
    """
    if not isinstance(data, np.ndarray) or data.dtype != np.complex128:
        raise ValueError("Input must be a complex128 numpy array")

    if not isinstance(time, np.ndarray):
        raise ValueError("Time must be a numpy array")

    if len(data.shape) != 3:
        raise ValueError("Input must have shape (time, row, column)")

    if len(time.shape) != 1:
        raise ValueError("Time must be a 1D array")

    if data.shape[0] != time.shape[0]:
        raise ValueError("Time steps must match the first dimension of the data")

    _, rows, cols = data.shape
    if rows != cols:
        raise ValueError(
            f"Each frame must be a square matrix, got shape ({rows}, {cols})"
        )


def setup_figure(
    shape: tuple[int, int],
    title: str = "Complex Matrix Hinton Plot",
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Set up the figure and axes for the Hinton plot.

    Args:
        shape: Shape of the matrix (rows, columns)
        title: Title of the plot

    Returns:
        Figure and Axes objects
    """
    plt.ioff()
    fig = plt.figure(figsize=(14, 10))  # Increased figure size for colorbar

    # Create main axis for the plot
    ax = plt.axes((0.1, 0.1, 0.7, 0.8))  # [left, bottom, width, height]
    rows, cols = shape
    set_ax(ax, cols, rows, title, row_names=row_names, col_names=col_names)
    # Create polar axis for the colorbar
    cax = plt.axes((0.75, 0.4, 0.2, 0.2), projection="polar")

    return fig, ax, cax


def set_ax(
    ax: plt.Axes,
    cols: int,
    rows: int,
    title: str,
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
) -> None:
    ax.set_title(title, fontsize=24)
    ax.set_xlim(-1, cols)
    ax.set_ylim(-1, rows)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.invert_yaxis()
    if row_names:
        # set fontsize=14
        ax.set_yticklabels(row_names, fontsize=16)
    if col_names:
        ax.set_xticklabels(col_names, fontsize=16)


def update(
    frame_num: int,
    data: np.ndarray,
    time: np.ndarray,
    ax: plt.Axes,
    cmap: str = "hsv",
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
    time_unit: str = "",
) -> None:
    """Update function for animation.

    Args:
        frame_num: Frame number
        data: Complex array of shape (time, row, column)
        ax: Matplotlib axes object
        cmap: Colormap name for phase visualization
    """
    # Get current frame data
    frame_data = data[frame_num]
    rows, cols = frame_data.shape
    ax.clear()
    set_ax(
        ax,
        cols,
        rows,
        f"Time {time[frame_num]: .2f} {time_unit}",
        row_names=row_names,
        col_names=col_names,
    )
    # Calculate magnitudes and phases
    magnitudes = np.abs(frame_data)
    phases = np.angle(frame_data)  # Returns phases in range [-π, π]

    # Normalize magnitudes
    max_magnitude = magnitudes.max()
    if max_magnitude == 0:
        max_magnitude = 1

    # Get colormap
    _cmap = plt.get_cmap(cmap)

    # Plot each element
    for i in range(rows):
        for j in range(cols):
            magnitude = magnitudes[i, j]
            phase = phases[i, j]
            value = frame_data[i, j]

            if magnitude > 0:
                # Size based on normalized magnitude
                size = (magnitude / max_magnitude) * 0.95

                # Color based on phase (normalize from [-π, π] to [0, 1])
                color = _cmap((phase + np.pi) / (2 * np.pi))

                # Create and add rectangle
                rect = Rectangle(
                    (j - size / 2, i - size / 2),
                    size,
                    size,
                    facecolor=color,
                    edgecolor="gray",
                )
                ax.add_patch(rect)

                # Add text annotation
                text = f"{value: .2f}"
                ax.text(
                    j,
                    i,
                    text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
                )


def set_cyclic_colorbar(ax: plt.Axes) -> mcolors.Colormap:
    theta = np.linspace(0.0, 2 * np.pi, 100)
    r = np.linspace(0, 1, 100)

    Theta, R = np.meshgrid(theta, r)

    cmap = plt.get_cmap("hsv")
    norm = mcolors.Normalize(vmin=0.0, vmax=2 * np.pi)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"], fontsize=14
    )
    ax.set_yticks([])
    ax.pcolormesh(
        Theta, R, Theta, cmap=cmap, norm=norm
    )  # , shading="auto", alpha=R / R.max())
    return cmap


def create_animation(
    data: np.ndarray,
    time: np.ndarray,
    title: str = "Complex Matrix Hinton Plot",
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
    time_unit: str = "",
    interval: int = 200,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Create an animation of complex matrix Hinton plots.

    Args:
        data: Complex array of shape (time, row, column)
        interval: Time interval between frames in milliseconds

    Returns:
        Figure and Animation objects
    """

    time_steps, rows, cols = data.shape
    fig, ax, cax = setup_figure(
        (rows, cols),
        title=title,
        row_names=row_names,
        col_names=col_names,
    )

    set_cyclic_colorbar(cax)

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,  # type: ignore
        frames=time_steps,
        fargs=(data, time, ax, "hsv", row_names, col_names, time_unit),
        interval=interval,
        blit=False,
    )

    return fig, anim


def save_animation(
    anim: animation.FuncAnimation,
    filename: str = "animation.gif",
    fps: int = 5,
    dpi: int = 100,
) -> None:
    """Save animation as a GIF file.

    Args:
        anim: Animation object
        filename: Output filename
        fps: Frames per second
        dpi: Dots per inch for the output
    """
    print(f"Saving animation to {filename}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    print("Animation saved successfully!")


def main(
    data: NDArray[np.complex128],
    time: NDArray[np.float64] | None = None,
    title: str = "Complex Matrix Hinton Plot",
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
    time_unit: str = "",
    save_gif: bool = False,
    gif_filename: str = "animation.gif",
    fps: int = 5,
    dpi: int = 100,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Main function to create Hinton plot animation from complex matrix data.

    Args:
        data: Complex array of shape (time, row, column)
        save_gif: Whether to save the animation as a GIF
        gif_filename: Output filename for GIF
        fps: Frames per second for GIF
        dpi: Dots per inch for the output GIF

    Returns:
        Figure and Animation objects

    Example:
        >>> # Create a 3x3 complex matrix that evolves over 10 time steps
        >>> t = np.linspace(0, 2*np.pi, 10)
        >>> data = np.zeros((10, 3, 3), dtype=np.complex128)
        >>> for i in range(10):
        ...     data[i] = np.exp(1j * t[i]) * np.random.random((3, 3))
        >>> fig, anim = main(data, save_gif=True)
        >>> plt.show()
    """
    if time is None:
        time = np.arange(data.shape[0], dtype=np.float64)

    _validate_input(data, time)

    fig, anim = create_animation(
        data,
        time,
        title=title,
        row_names=row_names,
        col_names=col_names,
        time_unit=time_unit,
    )

    if save_gif:
        save_animation(anim, gif_filename, fps, dpi)

    return fig, anim


if __name__ == "__main__":
    # Create example data: rotating complex numbers
    time_steps = 20
    size = 3  # 3x3 matrix
    t = np.linspace(0, 2 * np.pi, time_steps)

    # Initialize complex matrix
    test_data = np.zeros((time_steps, size, size), dtype=np.complex128)

    # Create rotating complex numbers with varying magnitudes
    for i in range(time_steps):
        magnitude = np.random.random((size, size)) + 0.5  # Random magnitudes > 0.5
        phase = (
            t[i] + np.random.random((size, size)) * np.pi / 4
        )  # Base rotation + noise
        test_data[i] = magnitude * np.exp(1j * phase)

    # Create animation and save as GIF
    print("Creating animation...")
    fig, anim = main(
        test_data,
        time=t,
        title="Complex Matrix Hinton Plot",
        save_gif=True,
        gif_filename="complex_matrix.gif",
        time_unit="fs",
        row_names=[r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$"],
        col_names=[r"$\langle 0|$", r"$\langle 1|$", r"$\langle 2|$"],
        fps=5,
        dpi=100,
    )
    plt.show()
