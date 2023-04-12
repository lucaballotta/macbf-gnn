import torch
import numpy as np
import os
import datetime
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy

from typing import Tuple, Callable, Optional, Union
from torch_geometric.data import Data, Batch
from torch import Tensor
from tqdm import tqdm

from macbf_gnn.env import MultiAgentEnv
from macbf_gnn.algo.cbf_gnn import CBFGNN


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logger(
        log_path: str,
        env_name: str,
        algo_name: str,
        seed: int,
        args: dict = None,
        hyper_params: dict = None,
) -> str:
    """
    Initialize the logger. The logger dir should include the following path:
        - <log folder>
            - <env name>
                - <algo name>
                    - seed<seed>_<experiment time>
                        - settings.yaml: the experiment setting

    Parameters
    ----------
    log_path: str,
        name of the log folder
    env_name: str,
        name of the training environment
    algo_name: str,
        name of the algorithm
    seed: int,
        random seed used
    args: dict,
        arguments to be written down: {argument name: value}
    hyper_params: dict
        hyper-parameters for training

    Returns
    -------
    log_path: str,
        path of the log
    """
    # make log path
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # make path with specific env
    if not os.path.exists(os.path.join(log_path, env_name)):
        os.mkdir(os.path.join(log_path, env_name))

    # make path with specific algorithm
    if not os.path.exists(os.path.join(log_path, env_name, algo_name)):
        os.mkdir(os.path.join(log_path, env_name, algo_name))

    # record the experiment time
    start_time = datetime.datetime.now()
    start_time = start_time.strftime('%Y%m%d%H%M%S')
    if not os.path.exists(os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}')):
        os.mkdir(os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}'))

    # set up log, summary writer
    log_path = os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}')

    # write args
    log = open(os.path.join(log_path, 'settings.yaml'), 'w')
    if args is not None:
        for key in args.keys():
            log.write(f'{key}: {args[key]}\n')
    if 'algo' not in args.keys():
        log.write(f'algo: {algo_name}\n')
    if hyper_params is not None:
        log.write('hyper_params:\n')
        for key1 in hyper_params.keys():
            if type(hyper_params[key1]) == dict:
                log.write(f'  {key1}: \n')
                for key2 in hyper_params[key1].keys():
                    log.write(f'    {key2}: {hyper_params[key1][key2]}\n')
            else:
                log.write(f'  {key1}: {hyper_params[key1]}\n')
    else:
        log.write('hyper_params: using default hyper-parameters')
    log.close()

    return log_path


def read_settings(path: str) -> dict:
    """
    Read the training settings.

    Parameters
    ----------
    path: str,
        path to the training log

    Returns
    -------
    settings: dict,
        a dict of training settings
    """
    with open(os.path.join(path, 'settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings


def eval_ctrl_epi(
        controller: Callable, env: MultiAgentEnv, seed: int = 0, make_video: bool = True, plot_edge: bool = True, verbose: bool = True,
) -> Tuple[float, float, Tuple[Union[Tuple[np.array, ...], np.array]], dict]:
    """
    Evaluate the controller for one episode.

    Parameters
    ----------
    controller: Callable,
        controller that gives action given a graph
    env: MultiAgentEnv,
        test environment
    seed: int,
        random seed
    make_video: bool,
        if true, return the video (a tuple of numpy arrays)
    plot_edge: bool,
        if true, plot the edge of the agent graph
    verbose: bool,
        if true, print the evaluation information

    Returns
    -------
    epi_reward: float,
        episode reward
    epi_length: float,
        episode length
    video: Optional[Tuple[np.array]],
        a tuple of numpy arrays
    info: dict,
        a dictionary of other information, including safe or not
    """
    set_seed(seed)
    epi_length = 0.
    epi_reward = 0.
    video = []
    data = env.reset()
    t = 0
    safe = True
    pbar = tqdm()
    states = []
    while True:
        action = controller(data)
        states.append(data.states)
        next_data, reward, done, info = env.step(action)
        epi_length += 1
        epi_reward += reward
        t += 1
        pbar.update(1)
        if 'safe' in info.keys():
            safe = safe and info['safe']

        if make_video:
            video.append(env.render(plot_edge=plot_edge))

        data = next_data

        if done:
            if verbose:
                message = f'reward: {epi_reward:.2f}, length: {epi_length}'
                if 'safe' in info.keys():
                    message += f', safe: {safe}'
                print(message)
            break
    return epi_reward, epi_length, tuple(video), {'safe': safe}


def plot_cbf_contour(
        cbf_fun: CBFGNN,
        data: Data,
        env: MultiAgentEnv,
        agent_id: int,
        x_dim: int,
        y_dim: int,
        action: Optional[Tensor] = None,
        attention: bool = True
):
    """
    Plot the contour of the learned CBF.

    Parameters
    ----------
    cbf_fun: Callable,
        function for the learned CBF
    data: Data,
        current graph
    env: MultiAgentEnv,
        current environment
    agent_id: int,
        the CBF of this agent is plotted
    x_dim: int,
        the x dimension for the plot
    y_dim: int,
        the y dimension for the plot
    action: Tensor,
        the current action taken

    Returns
    -------
    ax: the plot
    """
    n_mesh = 30
    low_lim, high_lim = env.state_lim
    x, y = np.meshgrid(
        np.linspace(low_lim[x_dim].cpu(), high_lim[x_dim].cpu(), n_mesh),
        np.linspace(low_lim[y_dim].cpu(), high_lim[y_dim].cpu(), n_mesh)
    )
    plot_data = []
    for i in range(n_mesh):
        for j in range(n_mesh):
            state = copy.deepcopy(data.states)
            state[agent_id, x_dim] = x[i, j]
            state[agent_id, y_dim] = y[i, j]
            new_data = Data(x=data.x, edge_index=data.edge_index, pos=state[:, :2],
                            edge_attr=state[data.edge_index[0]]-state[data.edge_index[1]])
            plot_data.append(new_data)
    plot_data = Batch.from_data_list(plot_data)
    cbf = cbf_fun(plot_data).view(n_mesh, n_mesh, env.num_agents)[:, :, agent_id].detach().cpu()
    ax = env.render(return_ax=True)
    if attention:
        ax = plot_attention(ax, cbf_fun.attention, data, agent_id)
    plt.contourf(x, y, cbf, cmap=sns.color_palette("rocket", as_cmap=True), levels=15, alpha=0.5)
    plt.colorbar()
    plt.contour(x, y, cbf, levels=[0.0], colors='blue')
    plt.xlabel(f'dim: {x_dim}')
    plt.ylabel(f'dim: {y_dim}')
    if action is not None:
        ax.text(0., 0.89, f'action: {action.cpu().detach().numpy()}', transform=ax.transAxes, fontsize=14)

    return ax


def plot_attention(ax, attention_fun: Callable, data: Data, agent_id: int):
    attention = attention_fun(data).cpu().detach().numpy()
    pos = data.pos.cpu().detach().numpy()
    edge_index = data.edge_index.cpu().detach().numpy()
    edge_centers = (pos[edge_index[0], :] + pos[edge_index[1], :]) / 2
    for i, text_point in enumerate(edge_centers):
        if edge_index[1, i] == agent_id:
            ax.text(text_point[0], text_point[1], f'{attention[i, 0]:.2f}', size=12, color="k", family="sans-serif",
                    weight="normal", horizontalalignment="center", verticalalignment="center", clip_on=True)
    return ax


def read_params(env: str) -> Optional[dict]:
    """
    Read the pre-defined training hyper-parameters.

    Parameters
    ----------
    env: str,
        name of the environment

    Returns
    -------
    params: Optional[dict],
        the training hyper-parameters if the environment is found, or None
    """
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, 'hyperparams.yaml')
    with open(path) as f:
        params = yaml.safe_load(f)
    if env in params.keys():
        return params[env]
    else:
        return None
