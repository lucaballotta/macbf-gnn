import argparse
import torch
import os

from macbf_gnn.env import make_env
from macbf_gnn.algo import make_algo
from macbf_gnn.trainer import Trainer
from macbf_gnn.trainer.utils import set_seed, init_logger, read_params


def train(args):
    # set random seed
    set_seed(args.seed)

    # set up training device
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'> Training with {device}')

    # make environment
    delay_aware = True
    env = make_env(args.env, args.num_agents, device, delay_aware=delay_aware)
    env_test = make_env(args.env, args.num_agents, device, delay_aware=delay_aware)

    # set training params
    params = read_params(args.env)
    if params is None:
        params = {  # set up custom hyper-parameters
            'alpha': 1.,
            'eps': 0.02,
            'inner_iter': 10,
            'loss_action_coef': 0.01,
            'loss_pred_coef': 1.,
            'loss_unsafe_coef': 1.,
            'loss_safe_coef': 1.,
            'loss_h_dot_coef': 0.5
        }
        print('> Using custom hyper-parameters')
    else:
        print('> Using pre-defined hyper-parameters')
        
    # set up logger
    log_path = init_logger(
        args.log_path, args.env, args.algo, args.seed, vars(args), hyper_params=params
    )

    # build algorithm
    algo = make_algo(
        args.algo, env, args.num_agents, env.node_dim, env.edge_dim, env.action_dim, device, hyperparams=params
    )

    # set up trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_path
    )

    # start training
    trainer.train(args.steps, eval_interval=args.steps // 20, eval_epi=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('-n', '--num-agents', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--algo', type=str, default='macbfgnn')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # default
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--log-path', type=str, default='./logs')

    args = parser.parse_args()
    train(args)
