import numpy as np
import torch
from config import *


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


def generate_data(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros(shape=(num_agents, 2), dtype=np.float32)
    goals = np.zeros(shape=(num_agents, 2), dtype=np.float32)

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(size=(2,)) * side_length
        dist_min = np.min(np.linalg.norm(states - candidate, axis=1))
        if dist_min <= dist_min_thres:
            continue
        states[i] = candidate
        i = i + 1

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i]
        dist_min = np.min(np.linalg.norm(goals - candidate, axis=1))
        if dist_min <= dist_min_thres:
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
    # dsdt = np.concatenate([states[:, 2:], actions], axis=1)
    dsdt = torch.cat([states[:, 2:], actions], dim=1)

    return dsdt


def loss_barrier(h, states):
    """ Build the loss function for the control barrier functions.

    Args:
        h (N, N, 1): The control barrier function.
        states (N, 4): The current state of N agents.
        r (float): The radius of the safe regions.
        ttc (float): The threshold of time to collision.
    """

    h_reshape = torch.reshape(h, (-1,))
    dang_mask = ttc_dangerous_mask(states)
    dang_mask_reshape = torch.reshape(dang_mask, (-1,))
    safe_mask_reshape = torch.logical_not(dang_mask_reshape)

    dang_h = torch.masked_select(h_reshape, dang_mask_reshape)
    safe_h = torch.masked_select(h_reshape, safe_mask_reshape)

    num_dang = dang_h.size(dim=0)
    num_safe = safe_h.size(dim=0)

    num_dang = torch.Tensor([num_dang])
    num_dang = num_dang.type(torch.float32)
    num_safe = torch.Tensor([num_safe])
    num_safe = num_safe.type(torch.float32)

    eps = [1e-3, 0]
    (dang_max_val, _) = torch.max(dang_h + eps[0], 0)
    loss_dang = torch.sum(dang_max_val) / (1e-5 + num_dang)
    (safe_max_val, _) = torch.max(-safe_h + eps[1], 0)
    loss_safe = torch.sum(safe_max_val) / (1e-5 + num_safe)

    acc_dang = torch.less_equal(dang_h, 0)
    acc_dang = torch.sum(acc_dang.type(torch.float32)) / (1e-5 + num_dang)
    acc_safe = torch.greater(safe_h, 0)
    acc_safe = torch.sum(safe_h.type(torch.float32)) / (1e-5 + num_safe)

    acc_dang = torch.where(
        torch.greater(num_dang, 0), acc_dang, -torch.Tensor([1.0]))
    acc_safe = torch.where(
        torch.greater(num_safe, 0), acc_safe, -torch.Tensor([1.0]))

    return loss_dang, loss_safe, acc_dang, acc_safe


def loss_derivatives(states, actions, h, cbf):
    dsdt = dynamics(states, actions)
    states_next = states + dsdt * TIME_STEP

    h_next = cbf.forward(states_next)
    deriv = h_next - h + TIME_STEP * ALPHA_CBF * h

    deriv_reshape = torch.reshape(deriv, (-1,))
    dang_mask = ttc_dangerous_mask(states)
    dang_mask_reshape = torch.reshape(dang_mask, (-1,))
    safe_mask_reshape = torch.logical_not(dang_mask_reshape)

    dang_deriv = torch.masked_select(deriv_reshape, dang_mask_reshape)
    safe_deriv = torch.masked_select(deriv_reshape, safe_mask_reshape)

    num_dang = dang_deriv.size(dim=0)
    num_safe = safe_deriv.size(dim=0)

    num_dang = torch.Tensor([num_dang])
    num_dang = num_dang.type(torch.float32)
    num_safe = torch.Tensor([num_safe])
    num_safe = num_safe.type(torch.float32)

    eps=[1e-3, 0]
    (dang_deriv_max_val, _) = torch.max(-dang_deriv + eps[0], 0)
    loss_dang_deriv = torch.sum(dang_deriv_max_val) / (1e-5 + num_dang)
    (safe_deriv_max_val, _) = torch.max(-safe_deriv + eps[1], 0)
    loss_safe_deriv = torch.sum(safe_deriv_max_val) / (1e-5 + num_safe)

    acc_dang_deriv = torch.greater_equal(dang_deriv, 0)
    acc_dang_deriv = torch.sum(acc_dang_deriv.type(torch.float32)) / (1e-5 + num_dang)
    acc_safe_deriv = torch.greater(safe_deriv, 0)
    acc_safe_deriv = torch.sum(acc_safe_deriv.type(torch.float32)) / (1e-5 + num_safe)

    acc_dang_deriv = torch.where(
        torch.greater(num_dang, 0), acc_dang_deriv, -torch.Tensor([1.0]))
    acc_safe_deriv = torch.where(
        torch.greater(num_safe, 0), acc_safe_deriv, -torch.Tensor([1.0]))

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv


def loss_actions(s, g, a):
    state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
    # print(state_gain)
    state_gain = torch.from_numpy(state_gain)
    state_gain = state_gain.type(torch.float32)
    s_ref = torch.concat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = torch.matmul(s_ref, torch.transpose(state_gain, 0, 1))
    action_ref_norm = torch.sum(torch.square(action_ref), dim=1)
    action_net_norm = torch.sum(torch.square(a), dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff).unsqueeze(0)

    return loss


def ttc_dangerous_mask(s):
    s_diff = torch.unsqueeze(s, dim=1) - torch.unsqueeze(s, dim=0)
    s_diff = torch.concat(
        [s_diff, torch.unsqueeze(torch.eye(s.size(dim=0)), dim=2)], dim=2)
    s_diff, _ = remove_distant_agents(s_diff, TOP_K)
    x, y, vx, vy, eye = torch.split(s_diff, 1, dim=2)
    x = x + eye
    y = y + eye
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - DIST_MIN_THRES ** 2
    dist_dangerous = torch.less(gamma, 0)

    has_two_positive_roots = torch.logical_and(
        torch.greater(beta ** 2 - 4 * alpha * gamma, 0),
        torch.logical_and(torch.greater(gamma, 0), torch.less(beta, 0)))
    root_less_than_ttc = torch.logical_or(
        torch.less(-beta - 2 * alpha * TIME_TO_COLLISION, 0),
        torch.less((beta + 2 * alpha * TIME_TO_COLLISION) ** 2, beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = torch.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerous = torch.logical_or(dist_dangerous, has_root_less_than_ttc)

    return ttc_dangerous


def ttc_dangerous_mask_np(s):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    x = x + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    y = y + np.expand_dims(np.eye(np.shape(s)[0]), 2)
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


def remove_distant_agents(x: torch.Tensor, k: int):
    n, _, c = x.shape
    if n <= k:
        return x
    d_norm = torch.sqrt(torch.sum(torch.square(x[:, :, :2]) + 1e-6, dim=2))
    topk = torch.topk(-d_norm, k)
    indices = topk[-1]
    row_indices = torch.unsqueeze(torch.arange(indices.size(dim=0)), dim=1) * torch.ones_like(indices)
    row_indices = torch.reshape(row_indices, (-1, 1))
    column_indices = torch.reshape(indices, (-1, 1))
    indices = torch.concat([row_indices, column_indices], dim=1)
    ind_1 = indices[:, 0]
    ind_2 = indices[:, 1]
    gathered = x[ind_1, ind_2]
    x = torch.reshape(gathered, (n, k, c))
        
    return x, indices
