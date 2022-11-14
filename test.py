import torch
import os
import cv2
import numpy as np
import argparse
import time
import multiprocessing

from macbf_gnn.trainer.utils import set_seed, read_settings, eval_ctrl_epi
from macbf_gnn.env import make_env
from macbf_gnn.algo import make_algo


def test(args):
    # set random seed
    set_seed(args.seed)

    # testing will be done on cpu
    device = torch.device('cpu')

    # load training settings
    try:
        settings = read_settings(args.path)
    except TypeError:
        settings = None

    # make environment
    env = make_env(
        env=settings['env'] if args.env is None else args.env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        device=device
    )

    # build algorithm
    if args.path is None:
        # evaluate the nominal controller
        def nominal(x):
            return torch.zeros(x.num_nodes, env.action_dim, device=device)
        controller = nominal
        args.path = f'./logs/{args.env}'
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        if not os.path.exists(args.path):
            os.mkdir(args.path)
        if not os.path.exists(os.path.join(args.path, 'nominal')):
            os.mkdir(os.path.join(args.path, 'nominal'))
        video_path = os.path.join(args.path, 'nominal', 'videos')
    else:
        # evaluate the learned controller
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
        controller = algo.act
        video_path = os.path.join(args.path, 'videos')

    # mkdir for the video and the figures
    if not args.no_video and not os.path.exists(video_path):
        os.mkdir(video_path)

    # evaluate policy
    pool = multiprocessing.Pool()
    arguments = [(controller, env, np.random.randint(100000), not args.no_video) for _ in range(args.epi)]
    print('> Processing...')
    start_time = time.time()
    results = pool.starmap(eval_ctrl_epi, arguments)
    rewards, lengths, video = zip(*results)
    video = sum(video, ())

    # make video
    print(f'> Making video...')
    if not args.no_video:
        out = cv2.VideoWriter(
            os.path.join(video_path, f'reward{np.mean(rewards):.2f}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            25,
            (video[-1].shape[1], video[-1].shape[0])
        )

        # release the video
        for fig in video:
            out.write(fig)
        out.release()

    # print evaluation results
    print(f'average reward: {np.mean(rewards):.2f}, average length: {np.mean(lengths):.2f}')
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('-n', '--num-agents', type=int, default=None)
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--epi', type=int, default=5)
    parser.add_argument('--no-video', action='store_true', default=False)

    # default
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    test(args)
