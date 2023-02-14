import torch
import numpy as np
import os
import datetime
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Callable, Optional, List, Union, Any
from torch_geometric.data import Data, Batch

from macbf_gnn.env import MultiAgentEnv


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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


def read_settings(path: str):
    with open(os.path.join(path, 'settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings


def eval_ctrl_epi(
        controller: Callable, env: MultiAgentEnv, seed: int = 0, make_video: bool = True, verbose: bool = True,
) -> tuple[float, float, Tuple[Union[tuple[np.array, ...], np.array]]]:
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
    """
    set_seed(seed)
    epi_length = 0.
    epi_reward = 0.
    video = []
    data = env.reset()
    t = 0
    while True:
        action = controller(data)
        next_data, reward, done, _ = env.step(action)
        epi_length += 1
        epi_reward += reward
        t += 1

        if make_video:
            video.append(env.render())

        data = next_data

        if done:
            if verbose:
                print(f'reward: {epi_reward:.2f}, length: {epi_length}')
            break
    return epi_reward, epi_length, tuple(video)


def plot_cbf_contour(cbf_fun: Callable, data: Data, env: MultiAgentEnv, agent_id: int, x_dim: int, y_dim: int):
    n_mesh = 30
    low_lim, high_lim = env.state_lim
    x, y = np.meshgrid(
        np.linspace(low_lim[x_dim].cpu(), high_lim[x_dim].cpu(), n_mesh),
        np.linspace(low_lim[y_dim].cpu(), high_lim[y_dim].cpu(), n_mesh)
    )
    plot_data = []
    # all_cbf = []
    for i in range(n_mesh):
        cbf_row = []
        for j in range(n_mesh):
            state = data.states
            state[agent_id, x_dim] = x[i, j]
            state[agent_id, y_dim] = y[i, j]
            # new_data = data
            # new_data.states = state
            # new_data.pos = state[:, :2]
            new_data = Data(x=data.x, edge_index=data.edge_index, pos=state[:, :2],
                            edge_attr=state[data.edge_index[0]]-state[data.edge_index[1]])
            # new_data = Data(x=torch.zeros_like(state), pos=state[:, :2], states=state)
            # new_data = env.add_communication_links(new_data)
            plot_data.append(new_data)
    #         cbf = cbf_fun(new_data).view(1, 1, env.num_agents)[:, :, agent_id].detach().cpu()
    #         cbf_row.append(cbf)
    #     all_cbf.append(torch.cat(cbf_row, dim=1))
    # cbf = torch.cat(all_cbf, dim=0)
    plot_data = Batch.from_data_list(plot_data)
    cbf = cbf_fun(plot_data).view(n_mesh, n_mesh, env.num_agents)[:, :, agent_id].detach().cpu()
    ax = env.render(return_ax=True)
    # ax = plt.imshow(fig)
    plt.contourf(x, y, cbf, cmap=sns.color_palette("rocket", as_cmap=True), levels=15, alpha=0.5)
    plt.colorbar()
    plt.contour(x, y, cbf, levels=[0.0], colors='blue')
    # plt.scatter(env.goal_point[0, x_dim].cpu(), env.goal_point[0, y_dim].cpu(), s=10, alpha=1, c='black')
    # plt.xlim((lower_limit[x_dim].cpu(), upper_limit[x_dim].cpu()))
    # plt.ylim((lower_limit[y_dim].cpu(), upper_limit[y_dim].cpu()))
    plt.xlabel(f'dim: {x_dim}')
    plt.ylabel(f'dim: {y_dim}')

    return ax
