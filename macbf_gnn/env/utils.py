import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.linalg import inv, solve_discrete_are
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from matplotlib.pyplot import Axes
from typing import Optional


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the Riccati equation
    X = solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K


def plot_graph(
        ax: Axes,
        data: Data,
        radius: float,
        color: str,
        with_label: bool = True,
        plot_edge: bool = False,
        alpha: float = 1.0,
        danger_radius: Optional[float] = None,
        safe_radius: Optional[float] = None
) -> Axes:
    pos = data.pos.cpu().detach().numpy()
    for i in range(pos.shape[0]):
        ax.add_patch(plt.Circle((pos[i, 0], pos[i, 1]), radius=radius, color=color, clip_on=False, alpha=alpha))
        if with_label:
            ax.text(pos[i, 0], pos[i, 1], f'{i}', size=12, color="k", family="sans-serif", weight="normal",
                    horizontalalignment="center", verticalalignment="center", transform=ax.transData, clip_on=True)
        if danger_radius is not None:
            ax.add_patch(
                plt.Circle((pos[i, 0], pos[i, 1]),
                           radius=danger_radius, color='red', clip_on=False, alpha=alpha, fill=False))
        if safe_radius is not None:
            ax.add_patch(
                plt.Circle((pos[i, 0], pos[i, 1]),
                           radius=safe_radius, color='green', clip_on=False, alpha=alpha, fill=False))
    if plot_edge:
        graph = to_networkx(data)
        nx.draw_networkx_edges(graph, pos)
    return ax


def plot_node_3d(ax, pos: np.ndarray, r: float, color: str, alpha: float, grid: int = 10) -> Axes:
    u = np.linspace(0, 2 * np.pi, grid)
    v = np.linspace(0, np.pi, grid)
    x = r * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    return ax


def plot_graph_3d(
        ax,
        data: Data,
        radius: float,
        color: str,
        with_label: bool = True,
        plot_edge: bool = False,
        alpha: float = 1.0,
):
    pos = data.pos.cpu().detach().numpy()
    for i in range(pos.shape[0]):
        plot_node_3d(ax, pos[i], radius, color, alpha)
        if with_label:
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], f'{i}', size=12, color="k", family="sans-serif", weight="normal",
                    horizontalalignment="center", verticalalignment="center")
    if plot_edge:
        for i in data.edge_index[0]:
            for j in data.edge_index[1]:
                vec = pos[i, :] - pos[j, :]
                x = [pos[i, 0] - 2 * radius * vec[0], pos[j, 0] + 2 * radius * vec[0]]
                y = [pos[i, 1] - 2 * radius * vec[1], pos[j, 1] + 2 * radius * vec[1]]
                z = [pos[i, 2] - 2 * radius * vec[2], pos[j, 2] + 2 * radius * vec[2]]
                ax.plot(x, y, z, linewidth=1.0, color="k")
    return ax
