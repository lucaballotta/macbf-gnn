import numpy as np
import torch
from torch_geometric.nn import TransformerConv
from config import *
from torch import Tensor


'''class AttentionNet(Aggregation):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.aggregator = TransformerConv(in_channels=in_dim, out_channels=in_dim)


    def forward(self, x: torch.Tensor, 
                edge_index: torch.Union[torch.Tensor, torch.SparseTensor]) -> torch.Tensor:
        return self.aggregator(x, edge_index)'''


def communication_links(states, num_agents):

    # compute inter-agent distances
    rel_position = torch.unsqueeze(states[:, :2], dim=1) - torch.unsqueeze(states[:, :2], dim=0)
    dist = torch.norm(rel_position, dim=2, keepdim=True)
    
    # build communication mask
    mask = torch.squeeze((torch.less_equal(dist, COMM_RADIUS)) & (torch.greater(dist, 0)))     

    # build communication graph
    edge_sources = torch.tensor([], dtype=torch.int64)
    edge_sinks = torch.tensor([], dtype=torch.int64)
    edge_attrs = torch.tensor([]).type_as(states)
    
    for agent in range(num_agents):
        neighs = torch.cat(torch.where(mask[agent]), dim=0)
        edge_sources = torch.cat([edge_sources, torch.ones_like(neighs) * agent])
        edge_sinks = torch.cat([edge_sinks, neighs])
        for neigh in range(neighs.shape[0]):
            edge_attrs = torch.cat([edge_attrs, (states[neigh, :] - states[agent, :]).unsqueeze(0)], dim=0)
        
    edge_index = torch.tensor([edge_sources.tolist(), edge_sinks.tolist()])
    
    return edge_index, edge_attrs


def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, np.pi*2, num=num, endpoint=False).reshape(-1, 1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    
    return circle


def generate_obstacle_rectangle(center, sides, num=12):
    # calculate the number of points on each side of the rectangle
    a, b = sides  # side lengths
    n_side_1 = int(num // 2 * a / (a+b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3
    # top
    side_1 = np.concatenate([
        np.linspace(-a/2, a/2, n_side_1, endpoint=False).reshape(-1, 1), 
        b/2 * np.ones(n_side_1).reshape(-1, 1)], axis=1)
    # right
    side_2 = np.concatenate([
        a/2 * np.ones(n_side_2).reshape(-1, 1),
        np.linspace(b/2, -b/2, n_side_2, endpoint=False).reshape(-1, 1)], axis=1)
    # bottom
    side_3 = np.concatenate([
        np.linspace(a/2, -a/2, n_side_3, endpoint=False).reshape(-1, 1), 
        -b/2 * np.ones(n_side_3).reshape(-1, 1)], axis=1)
    # left
    side_4 = np.concatenate([
        -a/2 * np.ones(n_side_4).reshape(-1, 1),
        np.linspace(-b/2, b/2, n_side_4, endpoint=False).reshape(-1, 1)], axis=1)

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)

    return rectangle


def generate_data(num_agents):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros(shape=(num_agents, 2), dtype=np.float32)
    goals = np.zeros(shape=(num_agents, 2), dtype=np.float32)

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(size=(2,)) * side_length
        dist_min = np.min(np.linalg.norm(states - candidate, axis=1))
        if dist_min <= DIST_MIN_THRES:
            continue
        states[i] = candidate
        i = i + 1

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i]
        dist_min = np.min(np.linalg.norm(goals - candidate, axis=1))
        if dist_min <= DIST_MIN_THRES:
            continue
        goals[i] = candidate
        i = i + 1

    states = np.concatenate(
        [states, np.zeros(shape=(num_agents, 2), dtype=np.float32)], axis=1)

    return states, goals


def dynamics(states, actions):
    """ The ground robot dynamics (double integrator).
    
    Args:
        states (N, 4): The current state.
        actions (N, 2): The acceleration taken by each agent.
    Returns:
        dsdt (N, 4): The time derivative of s.
    """

    return torch.cat([states[:, 2:], actions], dim=1)


def loss_cbf(h_trajectory, states_trajectory, batch_size, num_agents):
    """ Build the loss function for the control barrier functions.

    Args:
        h_trajectory (bs, n): The control barrier function of n agents for bs data-points.
        states_trajectory (bs x n, s): The s-dim states of n agents for bs data-points.
    """

    dang_mask_traj = torch.empty((0,), dtype=torch.bool)
    for i in range(batch_size):
        states = states_trajectory[i*num_agents:(i+1)*num_agents,:]
        dang_mask = ttc_dangerous_mask(states)
        dang_mask = torch.reshape(dang_mask, (1, num_agents))
        dang_mask_traj = torch.cat([dang_mask_traj, dang_mask], dim=0)

    h_dang = torch.masked_select(h_trajectory, dang_mask_traj)
    if h_dang.numel():
        max_val_dang = torch.maximum(h_dang + GAMMA, torch.zeros_like(h_dang))
        loss_dang = torch.mean(max_val_dang)
        acc_dang = torch.mean(torch.less(h_dang, 0).type_as(h_dang))
        
    else:
        loss_dang = torch.tensor(0.0)
        acc_dang = torch.tensor(-1.0)
        
    h_safe = torch.masked_select(h_trajectory, torch.logical_not(dang_mask_traj))
    if h_safe.numel():
        max_val_safe = torch.maximum(-h_safe + GAMMA, torch.zeros_like(h_safe))
        loss_safe = torch.mean(max_val_safe)
        acc_safe = torch.mean(torch.greater_equal(h_safe, 0).type_as(h_safe))
        
    else:
        loss_safe = torch.tensor(0.0)
        acc_safe = torch.tensor(-1.0)
    
    h_deriv = h_trajectory[1:,:] - h_trajectory[:-1,:]
    h_deriv_safe = torch.masked_select(h_deriv, torch.logical_not(dang_mask_traj[:-1,:]))
    if h_deriv_safe.numel():
        h_safe_first = torch.masked_select(h_trajectory[:-1,:], torch.logical_not(dang_mask_traj[:-1,:]))
        max_val_safe_deriv = torch.maximum(-h_deriv_safe - ALPHA_CBF * h_safe_first + GAMMA, torch.zeros_like(h_deriv_safe))
        loss_safe_deriv = torch.mean(max_val_safe_deriv)
        acc_safe_deriv = torch.mean(torch.greater_equal(h_deriv_safe, 0).type_as(h_deriv_safe))
        
    else:
        loss_safe_deriv = torch.tensor(0.0)
        acc_safe_deriv = torch.tensor(-1.0)

    return loss_dang, loss_safe, loss_safe_deriv, acc_dang, acc_safe, acc_safe_deriv


def loss_actions(states_trajectory, goals_trajectory, actions_trajectory):
    feedback = torch.concat([states_trajectory[:, :2] - goals_trajectory, states_trajectory[:, 2:]], dim=1)
    actions_ref = torch.matmul(feedback, torch.transpose(FEEDBACK_GAIN, 0, 1))
    actions_diff = actions_trajectory - actions_ref
    actions_diff_norm = torch.norm(actions_diff, dim=1)
    loss_actions_traj = torch.mean(actions_diff_norm)
    # action_ref_norm = torch.sum(torch.square(actions_ref), dim=1)
    # action_net_norm = torch.sum(torch.square(actions_trajectory), dim=1)
    # norm_diff = torch.abs(action_net_norm - action_ref_norm)
    # loss_actions_traj = torch.mean(norm_diff)

    return loss_actions_traj


def ttc_dangerous_mask(states):
    states_diff = torch.unsqueeze(states, dim=1) - torch.unsqueeze(states, dim=0)
    states_diff = torch.concat(
        [states_diff, torch.unsqueeze(torch.eye(states.size(dim=0)).type_as(states), dim=2)], dim=2)
    x, y, vx, vy, eye = torch.split(states_diff, 1, dim=2)
    x = x + eye
    y = y + eye
    a = vx ** 2 + vy ** 2
    b = 2 * (x * vx + y * vy)
    c = x ** 2 + y ** 2 - DIST_MIN_THRES ** 2
    dist_dangerous = torch.less(c, 0)
    has_two_positive_roots = torch.logical_and(
        torch.greater(b ** 2 - 4 * a * c, 0),
        torch.logical_and(torch.greater(c, 0), torch.less(b, 0)))
    root_less_than_ttc = torch.logical_or(
        torch.less(-b - 2 * a * TIME_TO_COLLISION, 0),
        torch.less((b + 2 * a * TIME_TO_COLLISION) ** 2, b ** 2 - 4 * a * c))
    has_root_less_than_ttc = torch.logical_and(has_two_positive_roots, root_less_than_ttc)
    mask = torch.logical_or(dist_dangerous, has_root_less_than_ttc)
    (mask, _) = torch.max(mask, dim=1)
    
    return mask


def ttc_dangerous_mask_np(s):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    x += np.expand_dims(np.eye(np.shape(s)[0]), 2)
    y += np.expand_dims(np.eye(np.shape(s)[0]), 2)
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - DIST_GOAL_TOL ** 2
    dist_dangerous = np.less(gamma, 0)

    has_two_positive_roots = np.logical_and(
        np.greater(beta ** 2 - 4 * alpha * gamma, 0),
        np.logical_and(np.greater(gamma, 0), np.less(beta, 0)))
    root_less_than_ttc = np.logical_or(
        np.less(-beta - 2 * alpha * TIME_TO_COLLISION_CHECK, 0),
        np.less((beta + 2 * alpha * TIME_TO_COLLISION_CHECK) ** 2, beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = np.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerous = np.logical_or(dist_dangerous, has_root_less_than_ttc)

    return ttc_dangerous