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


def loss_barrier(h_trajectory, states_trajectory):
    """ Build the loss function for the control barrier functions.

    Args:
        h_trajectory (bs, n): The control barrier function of n agents for bs data-points.
        states_trajectory (bs, n, x): The x-dim states of n agents for bs data-points.
    """

    loss_dang = torch.Tensor([0.0])
    loss_safe = torch.Tensor([0.0])
    loss_safe_deriv = torch.Tensor([0.0])
    acc_dang = torch.Tensor([0.0])
    acc_safe = torch.Tensor([0.0])
    acc_safe_deriv = torch.Tensor([0.0])
    no_dang_sample = True
    no_safe_sample = True
    step_dang = 0
    step_safe = 0
    
    # all states for which derivative of h exists
    for i in range(len(states_trajectory)-1):
        states = states_trajectory[i]
        dang_mask = ttc_dangerous_mask(states)
        dang_mask_reshape = torch.reshape(dang_mask, (-1,))
        
        h = h_trajectory[i]
        h_next = h_trajectory[i + 1]
        h_deriv = (h_next - h) / TIME_STEP

        h_dang = torch.masked_select(h, dang_mask_reshape)
        h_safe = torch.masked_select(h, torch.logical_not(dang_mask_reshape))
        h_deriv_safe = torch.masked_select(h_deriv, torch.logical_not(dang_mask_reshape))

        num_dang = h_dang.size(dim=0)
        num_safe = h_safe.size(dim=0)
        
        if num_dang:
            if no_dang_sample:
                no_dang_sample = False
            
            step_dang += 1
            update_loss(h_dang, GAMMA, loss_dang, acc_dang, step_dang)

        if num_safe:
            if no_safe_sample:
                no_safe_sample = False
                
            step_safe += 1
            update_loss(-h_safe, GAMMA, loss_safe, acc_safe, step_safe)
            update_loss(-h_deriv_safe, GAMMA - ALPHA_CBF * h, loss_safe_deriv, acc_safe_deriv, step_safe)
            
    # last state
    states = states_trajectory[-1]
    dang_mask = ttc_dangerous_mask(states)
    dang_mask_reshape = torch.reshape(dang_mask, (-1,))
    
    h = h_trajectory[-1]

    h_dang = torch.masked_select(h, dang_mask_reshape)
    h_safe = torch.masked_select(h, torch.logical_not(dang_mask_reshape))

    num_dang = h_dang.size(dim=0)
    num_safe = h_safe.size(dim=0)
    
    if num_dang:
        if no_dang_sample:
            no_dang_sample = False
        
        step_dang += 1
        update_loss(h_dang, GAMMA, loss_dang, acc_dang, step_dang)

    if num_safe:
        if no_safe_sample:
            no_safe_sample = False
            
        step_safe += 1
        update_loss(-h_safe, GAMMA, loss_safe, acc_safe, step_safe)
            
    if no_dang_sample:
        acc_dang = torch.Tensor([-1.0])
        
    if no_safe_sample:
        acc_safe = torch.Tensor([-1.0])
        acc_safe_deriv = torch.Tensor([-1.0])

    return loss_dang, loss_safe, loss_safe_deriv, acc_dang, acc_safe, acc_safe_deriv


def update_loss(h, slack, loss, acc, step):
    max_val = torch.maximum(h + slack, torch.tensor(0))
    loss_curr = torch.mean(max_val)
    torch.add(loss, (loss_curr - loss) / step)  # online average of losses along trajectory
    acc_curr = torch.mean(torch.less(h, 0))
    torch.add(acc, (acc_curr - acc) / step)


def loss_derivatives(h_trajectory, states_trajectory):
    loss_dang_deriv = torch.Tensor([0.0])
    loss_safe_deriv = torch.Tensor([0.0])
    acc_dang_deriv = torch.Tensor([0.0])
    acc_safe_deriv = torch.Tensor([0.0])
    no_dang_sample = True
    no_safe_sample = True
    step_dang = 0
    step_safe = 0
    
    for i in range(len(states_trajectory)-1):
        states = states_trajectory[i]
        dang_mask = ttc_dangerous_mask(states)
        dang_mask_reshape = torch.reshape(dang_mask, (-1,))
        
        h = h_trajectory[i]
        h_next = h_trajectory[i + 1]
        h_deriv = (h_next - h) / TIME_STEP

        h_deriv_dang = torch.masked_select(h_deriv, dang_mask_reshape)
        h_deriv_safe = torch.masked_select(h_deriv, torch.logical_not(dang_mask_reshape))

        num_dang = h_deriv_dang.size(dim=0)
        num_safe = h_deriv_safe.size(dim=0)

        if num_dang:
            if no_dang_sample:
                no_dang_sample = False
                
            step_dang += 1
            update_loss(h_deriv_dang, GAMMA + ALPHA_CBF * h, loss_dang_deriv, acc_dang_deriv, step_dang)

        if num_safe:
            if no_safe_sample:
                no_safe_sample = False
                
            step_safe += 1
            update_loss(-h_deriv_safe, GAMMA - ALPHA_CBF * h, loss_safe_deriv, acc_safe_deriv, step_safe)
            
    if no_dang_sample:
        acc_dang_deriv = torch.Tensor([-1.0])
        
    if no_safe_sample:
        acc_safe_deriv = torch.Tensor([-1.0])

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv


def loss_actions(states_trajectory, goals_trajectory, actions_trajectory):
    loss_actions_traj = 0
    for i in range(len(states_trajectory)):
        states = states_trajectory[i]
        goals = goals_trajectory[i]
        actions = actions_trajectory[i]
        s_ref = torch.concat([states[:, :2] - goals, states[:, 2:]], dim=1)
        actions_ref = torch.matmul(s_ref, torch.transpose(FEEDBACK_GAIN, 0, 1))
        # actions_diff = actions - actions_ref
        # actions_diff_norm = torch.norm(actions_diff, dim = 1)
        # loss_actions_curr = torch.sum(actions_diff_norm)
        action_ref_norm = torch.sum(torch.square(actions_ref), dim=1)
        action_net_norm = torch.sum(torch.square(actions), dim=1)
        norm_diff = torch.abs(action_net_norm - action_ref_norm)
        loss_actions_curr = torch.mean(norm_diff)
        torch.add(loss_actions_traj, (loss_actions_curr - loss_actions_traj) / (i + 1))

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