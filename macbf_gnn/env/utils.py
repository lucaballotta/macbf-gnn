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
