import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from config import *
from torch import Tensor


class MLP(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None):
        super().__init__()

        layers = []
        units = in_channels
        for next_units in hidden_layers:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
            
        layers.append(nn.Linear(units, out_channels))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


'''class AttentionNet(Aggregation):

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.aggregator = TransformerConv(in_channels=in_dim, out_channels=in_dim)


    def forward(self, x: torch.Tensor, 
                edge_index: torch.Union[torch.Tensor, torch.SparseTensor]) -> torch.Tensor:
        return self.aggregator(x, edge_index)'''


def build_comm_links(states, num_agents):

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
        for i_neigh in range(neighs.shape[0]):
            edge_attrs = torch.cat([edge_attrs, (states[i_neigh, :] - states[agent, :]).unsqueeze(0)], dim=0)
        
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


def loss_barrier(h, states):
    """ Build the loss function for the control barrier functions.

    Args:
        h (N, 1): The control barrier function.
        states (N, 4): The current state of N agents.
    """

    h_reshape = torch.reshape(h, (-1,))
    dang_mask = ttc_dangerous_mask(states)
    dang_mask_reshape = torch.reshape(dang_mask, (-1,))

    dang_h = torch.masked_select(h_reshape, dang_mask_reshape)
    safe_h = torch.masked_select(h_reshape, torch.logical_not(dang_mask_reshape))

    num_dang = dang_h.size(dim=0)
    num_safe = safe_h.size(dim=0)
    
    if num_dang:
        dang_max_val = torch.maximum(dang_h + GAMMA, torch.zeros_like(dang_h))
        loss_dang = torch.sum(dang_max_val) / num_dang
        acc_dang = torch.less_equal(dang_h, 0)
        acc_dang = torch.sum(acc_dang.type(torch.float32)) / num_dang

    else:
        loss_dang = torch.Tensor([0.0])
        acc_dang = torch.Tensor([-1.0])

    if num_safe:
        safe_max_val = torch.maximum(-safe_h + GAMMA, torch.zeros_like(safe_h))
        loss_safe = torch.sum(safe_max_val) / num_safe
        acc_safe = torch.greater(safe_h, 0)
        acc_safe = torch.sum(safe_h.type(torch.float32)) / num_safe

    else:
        loss_safe = torch.Tensor([0.0])
        acc_safe = torch.Tensor([-1.0])

    return loss_dang, loss_safe, acc_dang, acc_safe


def loss_derivatives(states, actions, h, cbf):
    dsdt = dynamics(states, actions)
    states_next = states + dsdt * TIME_STEP

    h_next = cbf(states_next)
    deriv = h_next - h + TIME_STEP * ALPHA_CBF * h
    
    deriv_reshape = torch.reshape(deriv, (-1,))
    dang_mask = ttc_dangerous_mask(states)
    dang_mask_reshape = torch.reshape(dang_mask, (-1,))

    dang_deriv = torch.masked_select(deriv_reshape, dang_mask_reshape)
    safe_deriv = torch.masked_select(deriv_reshape, torch.logical_not(dang_mask_reshape))

    num_dang = dang_deriv.size(dim=0)
    num_safe = safe_deriv.size(dim=0)

    if num_dang:
        dang_deriv_max_val = torch.maximum(dang_deriv + GAMMA, torch.zeros_like(dang_deriv))
        loss_dang_deriv = torch.sum(dang_deriv_max_val) / num_dang
        acc_dang_deriv = torch.greater_equal(dang_deriv, 0)
        acc_dang_deriv = torch.sum(acc_dang_deriv.type(torch.float32)) / num_dang

    else:
        loss_dang_deriv = torch.Tensor([0.0])
        acc_dang_deriv = torch.Tensor([-1.0])

    if num_safe:
        safe_deriv_max_val = torch.maximum(-safe_deriv + GAMMA, torch.zeros_like(safe_deriv))
        loss_safe_deriv = torch.sum(safe_deriv_max_val) / num_safe
        acc_safe_deriv = torch.greater(safe_deriv, 0)
        acc_safe_deriv = torch.sum(acc_safe_deriv.type(torch.float32)) / num_safe

    else:
        loss_safe_deriv = torch.Tensor([0.0])
        acc_safe_deriv = torch.Tensor([-1.0])

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv


def loss_actions(states, goals, actions):
    state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
    state_gain = torch.from_numpy(state_gain).type_as(states)
    state_gain = state_gain.type(torch.float32)
    s_ref = torch.concat([states[:, :2] - goals, states[:, 2:]], dim=1)
    action_ref = torch.matmul(s_ref, torch.transpose(state_gain, 0, 1))
    action_ref_norm = torch.sum(torch.square(action_ref), dim=1)
    action_net_norm = torch.sum(torch.square(actions), dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)

    return loss


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
    ttc_dangerous = torch.logical_or(dist_dangerous, has_root_less_than_ttc)
    (ttc_dangerous, _) = torch.max(ttc_dangerous, dim=1)
    return ttc_dangerous


def ttc_dangerous_mask_np(s):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    x += np.expand_dims(np.eye(np.shape(s)[0]), 2)
    y += np.expand_dims(np.eye(np.shape(s)[0]), 2)
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - DIST_MIN_CHECK ** 2
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