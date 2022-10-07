import numpy as np
import torch
import torch.optim as optim
import argparse
import os
import datetime

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import Batch

from controller_gnn import Controller
from cbf_gnn import CBF

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

    # create CBF and controller
    NUM_AGENTS = args.num_agents
    cbf_certificate = CBF(
        node_dim=STATE_DIM, edge_dim=STATE_DIM, phi_dim=FEAT_DIM, num_agents=NUM_AGENTS).to(device)
    cbf_controller = Controller(
        node_dim=STATE_DIM, edge_dim=STATE_DIM, phi_dim=FEAT_DIM, num_agents=NUM_AGENTS, action_dim=ACTION_DIM).to(device)
    FEEDBACK_GAIN = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
    FEEDBACK_GAIN = torch.from_numpy(FEEDBACK_GAIN).to(device)
    FEEDBACK_GAIN = FEEDBACK_GAIN.type(torch.float32)

    # create optimizers
    optim_controller = optim.Adam(cbf_controller.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optim_cbf = optim.Adam(cbf_certificate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # networks weights regularization
    # weight_loss = [WEIGHT_DECAY * torch.square(torch.norm(v)) for v in cbf_controller.parameters()]
    # weight_loss.append([WEIGHT_DECAY * torch.square(torch.norm(v)) for v in cbf_certificate.parameters()])

    loss_lists = []
    acc_lists_np = []
    safety_rates = []

    torch.autograd.set_detect_anomaly(True)

    # jointly train controller and CBF
    for i_train in tqdm(range(1, TRAIN_STEPS + 1)):

        # generate initial states and goals
        states_curr, goals_curr = generate_data(args.num_agents)
        states_curr = torch.from_numpy(states_curr).to(device)
        goals_curr = torch.from_numpy(goals_curr).to(device)
        states_trajectory = []
        goals_trajectory = []
        actions_diff_trajectory = []
        data_trajectory = []

        # run system for INNER_LOOPS steps to generate consistent trajectory
        skipped_steps = 0
        for _ in range(BATCH_SIZE_MAX):
            feedback_curr = torch.concat([states_curr[:, :2] - goals_curr, states_curr[:, 2:]], dim=1)
            actions_ref_curr = torch.matmul(feedback_curr, torch.t(FEEDBACK_GAIN))

            # build communication graph
            edge_index, edge_attr = communication_links(states_curr, NUM_AGENTS, device)
            if edge_index.numel():
                states_trajectory.append(states_curr)
                goals_trajectory.append(goals_curr)
                data = Data(x=torch.ones_like(states_curr).to(device), edge_index=edge_index, edge_attr=edge_attr)
                data_trajectory.append(data)

                # compute the control input using the trained controller
                actions_diff_curr = cbf_controller(data)
                if np.random.uniform() < ADD_NOISE_PROB:
                    noise = torch.randn(actions_diff_curr.shape) * NOISE_SCALE
                    actions_diff_curr = actions_diff_curr + noise
                    
                actions_diff_trajectory.append(actions_diff_curr)
                actions_curr = actions_ref_curr + actions_diff_curr
                
            else:
                
                # if agents cannot communicate,
                # skip step in training and apply reference control actions
                skipped_steps += 1
                actions_curr = actions_ref_curr
                
            # simulate the system for one step
            states_curr = states_curr + dynamics(states_curr, actions_curr) * TIME_STEP
            
            # check if agents have reached goals
            if torch.max(
                torch.norm(states_curr[:, :2] - goals_curr, dim=1)
            ) < DIST_GOAL_TOL:
                break
        
        batch_size = len(states_trajectory)
        states_trajectory = torch.cat(states_trajectory, dim=0)
        goals_trajectory = torch.cat(goals_trajectory, dim=0)
        actions_diff_trajectory = torch.cat(actions_diff_trajectory, dim=0)
        
        # compute loss for batch of trajectory states
        data_trajectory = Batch.from_data_list(data_trajectory)
        h_trajectory = cbf_certificate(data_trajectory)
        loss_dang_traj, loss_safe_traj, loss_safe_deriv_traj, _, _, _ = loss_cbf(h_trajectory, states_trajectory, batch_size, NUM_AGENTS, device)
        # loss_safe_deriv, _ = loss_cbf_deriv(h_trajectory, states_trajectory)
        loss_action_traj = loss_actions(actions_diff_trajectory)
        loss_list_traj = [2 * loss_dang_traj, loss_safe_traj, loss_safe_deriv_traj, 0.01 * loss_action_traj]
        # loss_list_iter = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action_iter]
        # acc_list_iter = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
        loss_lists.append(loss_list_traj)
        # acc_lists_np.append(acc_list_iter)
        # total_loss_iter = 10 * torch.add(loss_list_iter + weight_loss)
        # total_loss_iter = 10 * torch.add(loss_list_iter)
        # total_loss_iter = torch.Tensor(10 * loss_list_iter)
        # print(loss_list_iter)
        total_loss_traj = 10 * torch.stack(loss_list_traj).sum()

        # apply optimization step
        optim_controller.zero_grad()
        optim_cbf.zero_grad()
        total_loss_traj.backward()
        optim_controller.step()
        optim_cbf.step()

        # compute the safety rate
        # safety_rate = 1 - np.mean(ttc_dangerous_mask_np(states_curr), axis=1)
        # safety_rate = np.mean(safety_rate == 1)
        # safety_rates.append(safety_rate)

        # save trained weights
        if i_train % SAVE_STEPS == 0:
            torch.save(cbf_certificate, os.path.join(model_path, f'step{i_train}_certificate.pkl'))
            torch.save(cbf_controller, os.path.join(model_path, f'step{i_train}_controller.pkl'))


if __name__ == '__main__':
    main()