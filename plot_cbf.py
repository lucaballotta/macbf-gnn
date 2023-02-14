import torch
import os
import cv2
import numpy as np
import argparse
import time
import multiprocess
import matplotlib.pyplot as plt
import shutil

from macbf_gnn.trainer.utils import set_seed, read_settings, plot_cbf_contour
from macbf_gnn.env import make_env
from macbf_gnn.algo import make_algo


def plot_cbf(args):
    # set random seed
    set_seed(args.seed)

    # testing will be done on cpu
    device = torch.device('cpu')

    # load training settings
    try:
        settings = read_settings(args.path)
    except TypeError:
        raise TypeError('Cannot find configuration file in the path')

    # make environment
    env = make_env(
        env=settings['env'] if args.env is None else args.env,
        num_agents=settings['num_agents'],
        device=device
    )
    env.test()

    # build algorithm
    algo = make_algo(
        algo=settings['algo'],
        env=env,
        num_agents=settings['num_agents'],
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=env.action_dim,
        device=device
    )
    model_path = os.path.join(args.path, 'models')
    if args.iter is not None:
        # load the controller at given iteration
        algo.load(os.path.join(model_path, f'step_{args.iter}'))
    else:
        # load the last controller
        controller_name = os.listdir(model_path)
        controller_name = [i for i in controller_name if 'step' in i]
        controller_id = sorted([int(i.split('step_')[1].split('.')[0]) for i in controller_name])
        algo.load(os.path.join(model_path, f'step_{controller_id[-1]}'))
    fig_path = os.path.join(args.path, 'figs')

    # mkdir for the video and the figures
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig_path = os.path.join(fig_path, f'agent_{args.agent}')
    if os.path.exists(fig_path):
        shutil.rmtree(fig_path)
    os.mkdir(fig_path)

    # simulate the environment and plot the CBFs
    data = env.reset()
    t = 0
    while True:
        action = algo.apply(data)
        next_data, reward, done, _ = env.step(action)
        if hasattr(algo, 'cbf'):
            ax = plot_cbf_contour(algo.cbf, data, env, args.agent, args.x_dim, args.y_dim)
            h_prev = algo.cbf(data)[args.agent]
            h_post = algo.cbf(next_data)[args.agent]
            h_deriv = (h_post - h_prev) / env.dt + algo.params['alpha'] * h_prev
            if h_deriv < 0:
                ax.text(0., 0.93, f"violate derivative term: {h_deriv.item():.2f}", transform=ax.transAxes, fontsize=14)
            plt.savefig(os.path.join(fig_path, f'{t}.png'))
            plt.close()
        else:
            raise KeyError('The algorithm must has a CBF function')
        data = next_data
        t += 1
        if done:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--agent', type=int, default=0)
    parser.add_argument('--x-dim', type=int, default=0)
    parser.add_argument('--y-dim', type=int, default=1)

    # default
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    plot_cbf(args)
