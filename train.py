from logging import PlaceHolder
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import itertools

from core import *
from config import *
import controller
import cbf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create controller and CBF
    cbf_controller = controller.Controller(in_dim=4).to(device)
    cbf_certificate = cbf.CBF(in_dim=4).to(device)

    # create optimizers
    optim_controller = optim.Adam(cbf_controller.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optim_cbf = optim.Adam(cbf_certificate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # networks weights regularization
    # weight_loss = [WEIGHT_DECAY * torch.square(torch.norm(v)) for v in cbf_controller.parameters()]
    # weight_loss.append([WEIGHT_DECAY * torch.square(torch.norm(v)) for v in cbf_certificate.parameters()])

    loss_lists_np = []
    acc_lists_np = []
    safety_rates = []

    # jointly train controller and CBF
    for _ in tqdm(range(TRAIN_STEPS)):

        # generate initial states and goals
        states_curr, goals_curr = generate_data(args.num_agents, DIST_MIN_THRES)
        states_curr = torch.from_numpy(states_curr).to(device)
        goals_curr = torch.from_numpy(goals_curr).to(device)
        states_trajectory = []
        goals_trajectory = []
        actions_trajectory = []

        # run system for INNER_LOOPS steps to generate consistent trajectory
        for _ in range(INNER_LOOPS):
            states_trajectory.append(states_curr)
            goals_trajectory.append(goals_curr)

            # compute the control input using the trained controller
            actions_curr = cbf_controller(states_curr, goals_curr)
            if np.random.uniform() < ADD_NOISE_PROB:
                noise = torch.randn(actions_curr.shape) * NOISE_SCALE
                actions_curr = actions_curr + noise

            # simulate the system for one step
            states_curr = states_curr + dynamics(states_curr, actions_curr) * TIME_STEP
            actions_trajectory.append(actions_curr)

            # compute the safety rate
            # safety_rate = 1 - np.mean(ttc_dangerous_mask_np(states_curr), axis=1)
            # safety_rate = np.mean(safety_rate == 1)
            # safety_rates.append(safety_rate)
            
            if torch.mean(
                torch.norm(states_curr[:, :2] - goals_curr, dim=1)
            ) < DIST_MIN_CHECK:
                break

        states_trajectory = torch.cat(states_trajectory, dim=0)
        goals_trajectory = torch.cat(goals_trajectory, dim=0)
        actions_trajectory = torch.cat(actions_trajectory, dim=0)

        # compute loss for batch of trajectory states
        h_trajectory = cbf_certificate(states_trajectory)
        (loss_dang, loss_safe, acc_dang, acc_safe) = loss_barrier(h_trajectory, states_trajectory)
        (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
        ) = loss_derivatives(states_trajectory, actions_trajectory, h_trajectory, cbf_certificate)
        loss_action_iter = loss_actions(states_trajectory, goals_trajectory, actions_trajectory)
        loss_list_iter = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action_iter]
        acc_list_iter = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
        loss_lists_np.append(loss_list_iter)
        acc_lists_np.append(acc_list_iter)
        # total_loss_iter = 10 * torch.add(loss_list_iter + weight_loss)
        # total_loss_iter = 10 * torch.add(loss_list_iter)
        total_loss_iter = torch.Tensor(10 * loss_list_iter)

        # apply optimization step
        optim_controller.zero_grad()
        optim_cbf.zero_grad()
        total_loss_iter.backward()
        optim_controller.step()
        optim_cbf.step()


if __name__ == '__main__':
    main()
