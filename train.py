from ast import Num
import numpy as np
import torch
import torch.optim as optim
import argparse
import os
import datetime

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import controller_gnn
import cbf_gnn

from utils import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # setup logs
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    start_time = datetime.datetime.now()
    start_time = start_time.strftime('%Y%m%d%H%M%S')
    if not os.path.exists(os.path.join('logs', start_time)):
        os.mkdir(os.path.join('logs', f'seed{args.seed}_{start_time}'))

    log_dir = os.path.join('logs', f'seed{args.seed}_{start_time}')
    if not os.path.exists(os.path.join(log_dir, 'models')):
        os.mkdir(os.path.join(log_dir, 'models'))

    model_path = os.path.join(log_dir, 'models')

    # create controller and CBF
    num_agents = args.num_agents
    feat_dim = 32
    cbf_controller = controller_gnn.Controller(in_dim=4).to(device)
    # cbf_certificate = cbf_gnn.GNN(feat_dim=feat_dim, phi_dim=feat_dim, aggr_dim=feat_dim,
    #                               num_agents=num_agents, state_dim=4).to(device)
    cbf_certificate = cbf_gnn.CBFGNN(state_dim=4, phi_dim=feat_dim, num_agents=num_agents).to(device)

    # create optimizers
    optim_controller = optim.Adam(cbf_controller.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optim_cbf = optim.Adam(cbf_certificate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # networks weights regularization
    # weight_loss = [WEIGHT_DECAY * torch.square(torch.norm(v)) for v in cbf_controller.parameters()]
    # weight_loss.append([WEIGHT_DECAY * torch.square(torch.norm(v)) for v in cbf_certificate.parameters()])

    loss_lists_np = []
    acc_lists_np = []
    safety_rates = []

    torch.autograd.set_detect_anomaly(True)

    # jointly train controller and CBF
    for i_step in tqdm(range(1, TRAIN_STEPS + 1)):

        # generate initial states and goals
        states_curr, goals_curr = generate_data(args.num_agents)
        states_curr = torch.from_numpy(states_curr).to(device)
        goals_curr = torch.from_numpy(goals_curr).to(device)
        states_trajectory = []
        goals_trajectory = []
        actions_trajectory = []
        data_traj = []

        # run system for INNER_LOOPS steps to generate consistent trajectory
        for _ in range(INNER_LOOPS):
            states_trajectory.append(states_curr)
            goals_trajectory.append(goals_curr)

            # compute the control input using the trained controller
            actions_curr = cbf_controller(states_curr, goals_curr)
            if np.random.uniform() < ADD_NOISE_PROB:
                noise = torch.randn(actions_curr.shape) * NOISE_SCALE
                actions_curr = actions_curr + noise

            # store the communication graph
            edge_index, edge_attr = build_comm_links(states_curr, num_agents)
            data_traj.append(Data(x=torch.ones_like(states_curr), edge_index=edge_index, edge_attr=edge_attr))

            # simulate the system for one step
            states_curr = states_curr + dynamics(states_curr, actions_curr) * TIME_STEP
            actions_trajectory.append(actions_curr)

            # states_trajectory = states_curr
            # goals_trajectory = goals_curr
            # actions_trajectory = actions_curr

            if torch.mean(
                torch.norm(states_curr[:, :2] - goals_curr, dim=1)
            ) < DIST_MIN_CHECK:
                break

        states_trajectory = torch.cat(states_trajectory, dim=0)
        goals_trajectory = torch.cat(goals_trajectory, dim=0)
        actions_trajectory = torch.cat(actions_trajectory, dim=0)

        # compute loss for batch of trajectory states
        data_traj = Batch.from_data_list(data_traj)
        h_trajectory = cbf_certificate(data_traj)
        loss_dang, loss_safe, acc_dang, acc_safe = loss_barrier(h_trajectory, states_trajectory)
        loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = \
            loss_derivatives(states_trajectory, actions_trajectory, h_trajectory, cbf_certificate)
        loss_action_iter = loss_actions(states_trajectory, goals_trajectory, actions_trajectory)
        loss_list_iter = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action_iter]
        acc_list_iter = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
        loss_lists_np.append(loss_list_iter)
        acc_lists_np.append(acc_list_iter)
        # total_loss_iter = 10 * torch.add(loss_list_iter + weight_loss)
        # total_loss_iter = 10 * torch.add(loss_list_iter)
        # total_loss_iter = torch.Tensor(10 * loss_list_iter)
        # print(loss_list_iter)
        total_loss_iter = 10 * torch.stack(loss_list_iter).sum()

        # apply optimization step
        optim_controller.zero_grad()
        optim_cbf.zero_grad()
        total_loss_iter.backward()
        optim_controller.step()
        optim_cbf.step()

        # compute the safety rate
        # safety_rate = 1 - np.mean(ttc_dangerous_mask_np(states_curr), axis=1)
        # safety_rate = np.mean(safety_rate == 1)
        # safety_rates.append(safety_rate)

        # save trained weights
        if i_step % SAVE_STEPS == 0:
            torch.save(cbf_certificate, os.path.join(model_path, f'step{i_step}_certificate.pkl'))
            torch.save(cbf_controller, os.path.join(model_path, f'step{i_step}_controller.pkl'))


if __name__ == '__main__':
    main()