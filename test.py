import sys
sys.dont_write_bytecode = True
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.nn import ReLU

import controller
import cbf

from core import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

    
def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))


def render_init():
    fig = plt.figure(figsize=(9, 4))
    return fig


def refine_actions(states, actions, cbf_certificate):
    
    # Refine control inputs so that next states comply with safety requirements
    
    h, mask = cbf_certificate(states)
    deriv_gap = -h + TIME_STEP * ALPHA_CBF * h
    actions_gap = torch.zeros_like(actions)
    
    for _ in range(REFINE_LOOPS):
        states_next = states + dynamics(states, actions + actions_gap) * TIME_STEP
        h_next, mask_next = cbf_certificate(states_next)
        deriv = h_next + deriv_gap
        deriv = deriv * mask * mask_next
        error = torch.sum(ReLU(-deriv), dim=1)
        error_gradient = torch.gradient(error, spacing=actions_gap)[0]
        actions_gap -= REFINE_LEARNING_RATE * error_gradient
        
    return actions + actions_gap


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create controller and CBF
    cbf_controller = controller.Controller(in_dim=4).to(device)
    cbf_certificate = cbf.CBF(in_dim=4).to(device)
    
    # restore saved weights
    model_path = os.path.join(args.path, 'models')
    if args.step is not None:
        cbf_certificate.load_state_dict(torch.load(os.path.join(model_path, f'step{args.step}_certificate.pkl')))
        cbf_controller.load_state_dict(torch.load(os.path.join(model_path, f'step{args.step}_controller.pkl')))
    else:
        # load the last saved controller
        files = os.listdir(model_path)
        steps = [int(i.split('step')[1].split('_')[0]) for i in files if 'step' in i]
        step = sorted(steps)[-1]
        cbf_certificate.load_state_dict(torch.load(os.path.join(model_path, f'step{step}_certificate.pkl')))
        cbf_controller.load_state_dict(torch.load(os.path.join(model_path, f'step{step}_controller.pkl')))

    safety_ratios_epoch = []
    safety_ratios_epoch_lqr = []

    dist_errors = []
    init_dist_errors = []
    accuracy_lists = []

    safety_reward = []
    dist_reward = []
    safety_reward_baseline = []
    dist_reward_baseline = []

    if args.vis:
        plt.ion()
        plt.close()
        fig = render_init()
        
    K = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)

    for istep in range(EVALUATE_STEPS):
        start_time = time.time()

        safety_info = []
        safety_info_baseline = []
        
        # randomly generate the initial conditions states_init and the goal states goals
        states_init, goals = generate_data(args.num_agents, DIST_MIN_THRES * 1.5)
        states_curr, goals_curr = np.copy(states_init), np.copy(goals)
        init_dist_errors.append(np.mean(np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1)))
        
        # store the trajectory for visualization
        state_trajectory_nn = []
        state_trajectory_lq = []

        safety_nn = []
        safety_lq = []
        
        # run INNER_LOOPS steps to reach the current goals
        for _ in range(INNER_LOOPS):
            
            # compute the control input
            actions_curr = cbf_controller(states_curr, goals_curr)
            actions_curr = refine_actions(states_curr, actions_curr, cbf_certificate)
            
            # simulate the system for one step
            states_curr += dynamics(states_curr, actions_curr) * TIME_STEP
            state_trajectory_nn.append(states_curr)
            
            # compute safety rate
            safety_ratio = 1 - np.mean(ttc_dangerous_mask_np(states_curr), axis=1)
            safety_nn.append(safety_ratio)
            safety_info.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio = np.mean(safety_ratio == 1)
            safety_ratios_epoch.append(safety_ratio)
            
            # compute accuracies
            h_curr, _ = cbf_certificate(states_curr)
            (_, _, acc_dang, acc_safe) = loss_barrier(h_curr, states_curr)
            (_, _, acc_dang_deriv, acc_safe_deriv) = loss_derivatives(
                states_curr, actions_curr, h_curr, cbf_certificate)
            acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
            accuracy_lists.append(acc_list)

            if args.vis:
                # break if the agents are already very close to their goals
                if np.amax(
                    np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1)
                    ) < DIST_MIN_CHECK / 3:
                    time.sleep(1)
                    break
                # if the agents are very close to their goals, safely switch to LQR
                if np.mean(
                    np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1)
                    ) < DIST_MIN_CHECK / 2:
                    K = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
                    s_ref = np.concatenate([states_curr[:, :2] - goals_curr, states_curr[:, 2:]], axis=1)
                    actions_lq = -s_ref.dot(K.T)
                    states_curr = states_curr + np.concatenate(
                        [states_curr[:, 2:], actions_lq], axis=1) * TIME_STEP                
            else:
                if np.mean(
                    np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1)
                    ) < DIST_MIN_CHECK:
                    break

        dist_errors.append(np.mean(np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1)))
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean(
            (np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1) < 0.2).astype(np.float32) * 10))

        # run the simulation using LQR controller without considering collision
        states_curr, goals_curr = np.copy(states_init), np.copy(goals)
        for _ in range(INNER_LOOPS):
            s_ref = np.concatenate([states_curr[:, :2] - goals_curr, states_curr[:, 2:]], axis=1)
            actions_lq = -s_ref.dot(K.T)
            states_curr += dynamics(states_curr, actions_lq) * TIME_STEP
            state_trajectory_lq.append(states_curr)
            safety_ratio = 1 - np.mean(ttc_dangerous_mask_np(states_curr), axis=1)
            safety_lq.append(safety_ratio)
            safety_info_baseline.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio = np.mean(safety_ratio == 1)
            safety_ratios_epoch_lqr.append(safety_ratio)
            if np.mean(
                    np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1)
                    ) < DIST_MIN_CHECK / 3:
                    break

        safety_reward_baseline.append(np.mean(
            np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean(
            (np.linalg.norm(states_curr[:, :2] - goals_curr, axis=1) < 0.2).astype(np.float32) * 10))

        if args.vis:
            # visualize the trajectories
            vis_range = max(1, np.amax(np.abs(states_init[:, :2])))
            agent_size = 100 / vis_range ** 2
            goals_curr = goals_curr / vis_range
            for j in range(max(len(state_trajectory_nn), len(state_trajectory_lq))):
                plt.clf()
                plt.subplot(121)
                j_ours = min(j, len(state_trajectory_nn)-1)
                states_curr = state_trajectory_nn[j_ours] / vis_range
                plt.scatter(states_curr[:, 0], states_curr[:, 1], 
                            color='darkorange', 
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(goals_curr[:, 0], goals_curr[:, 1], 
                            color='deepskyblue', 
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_nn[j_ours])
                plt.scatter(states_curr[safety<1, 0], states_curr[safety<1, 1], 
                            color='red', 
                            s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('Ours: Safety Rate = {:.3f}'.format(
                    np.mean(safety_ratios_epoch)), fontsize=14)

                plt.subplot(122)
                j_lqr = min(j, len(state_trajectory_lq)-1)
                states_curr = state_trajectory_lq[j_lqr] / vis_range
                plt.scatter(states_curr[:, 0], states_curr[:, 1], 
                            color='darkorange', 
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(goals_curr[:, 0], goals_curr[:, 1], 
                            color='deepskyblue', 
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_lq[j_lqr])
                plt.scatter(states_curr[safety<1, 0], states_curr[safety<1, 1], 
                            color='red', 
                            s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('LQR: Safety Rate = {:.3f}'.format(
                    np.mean(safety_ratios_epoch_lqr)), fontsize=14)

                fig.canvas.draw()
                plt.pause(0.01)
            plt.clf()

        end_time = time.time()
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(
            istep + 1, EVALUATE_STEPS, end_time - start_time))

    print_accuracy(accuracy_lists)
    print('Distance Error (Final | Inititial): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(init_dist_errors)))
    print('Mean Safety Ratio (Learning | LQR): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_lqr)))
    print('Reward Safety (Learning | LQR): {:.4f} | {:.4f}, Reward Distance: {:.4f} | {:.4f}'.format(
        np.mean(safety_reward), np.mean(safety_reward_baseline), 
        np.mean(dist_reward), np.mean(dist_reward_baseline)))
    

if __name__ == '__main__':
    main()