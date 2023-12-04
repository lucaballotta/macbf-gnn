from copy import deepcopy
import torch.nn as nn
import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential
from torch import Tensor
from torch.optim import Adam
from torch.nn.utils.rnn import pack_sequence
from typing import Optional

from macbf_gnn.nn import MLP, CBFGNNLayer
from macbf_gnn.nn.predictor import PredictorMLP, PredictorLSTM
from macbf_gnn.controller import ControllerGNN
from macbf_gnn.env import MultiAgentEnv

from .base import Algorithm
from .buffer import Buffer


class CBFGNN(nn.Module):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int):
        super(CBFGNN, self).__init__()
        self.num_agents = num_agents
        self.feat_transformer = Sequential('x, edge_attr, edge_index', [
            (CBFGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=1024, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
        ])
        self.feat_2_CBF = MLP(in_channels=1024, out_channels=1, hidden_layers=(512, 128, 32), output_activation=nn.Tanh())


    def forward(self, data: Data) -> Tensor:
        """
        Get the CBF value for the input states.

        Parameters
        ----------
        data: Data
            batched data using Batch.from_data_list().

        Returns
        -------
        h: (bs x n,)
            CBF values for all agents
        """
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        h = self.feat_2_CBF(x)
        
        return h

    def attention(self, data: Data) -> Tensor:
        return self.feat_transformer.module_0.attention(data)


class MACBFGNN(Algorithm):

    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            batch_size: int = 500,
            params: Optional[dict] = None,
            use_all_data: bool = True
    ):
        super(MACBFGNN, self).__init__(
            env=env,
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )

        # models
        self.use_all_data = use_all_data
        if self.use_all_data:
            self.predictor = PredictorLSTM(
                input_dim=self.action_dim + self.edge_dim + 1,
                output_dim=self.edge_dim,
                hidden_size=256,
                dropout=0.1,
                device=self.device
            ).to(device)
        
        else:
            self.predictor = PredictorMLP(
                input_dim=self.action_dim + self.edge_dim + 1,
                output_dim=self.edge_dim,
                hidden_layers=(128, 128)
            ).to(device)
        
        self.cbf = CBFGNN(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            phi_dim=256
        ).to(device)
        self.controller = ControllerGNN(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            phi_dim=256,
            action_dim=self.action_dim,
        ).to(device)

        # optimizer
        self.optim_predictor = Adam(self.predictor.parameters(), lr=1e-3)
        self.optim_cbf = Adam(self.cbf.parameters(), lr=3e-4)
        self.optim_actor = Adam(self.controller.parameters(), lr=1e-3)

        # buffer to store data used in training
        self.buffer = Buffer()  # buffer for current episode
        self.memory = Buffer()  # replay buffer
        self.batch_size = batch_size

        # hyperparams
        if params is None:
            self.params = {  # default hyper-parameters
                'alpha': 1.0,
                'eps': 0.02,
                'inner_iter': 10,
                'loss_action_coef': 0.001,
                'loss_pred_coef': 1.,
                'loss_unsafe_coef': 1.,
                'loss_safe_coef': 1.,
                'loss_h_dot_coef': 0.1,
            }
        else:
            self.params = params


    @torch.no_grad()
    def act(self, data: Data) -> Tensor:
        if self._env.delay_aware:
            input_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=self.predictor(data.edge_attr),
                u_ref=data.u_ref
            )
            # h = self.cbf(input_data)
            # print('state', data.state_diff)
            # print('edge attr', data.edge_attr)
            # print('edge attr', data.edge_attr)
            # print('pred state', input_data.edge_attr)
            # true_state_diff = data.state_diff
            # true_state_diff_norm = torch.norm(true_state_diff, dim=1)
            # pred_err_norm = torch.norm(input_data.edge_attr - true_state_diff, dim=1)
            # print('loss pred', torch.mean(pred_err_norm / true_state_diff_norm).item())
            # print('cbf', h)
            # action = self.controller(input_data)
            # data_next = self._env.forward_graph(data, action)
            # input_cbf_next = Data(
            #     x=data_next.x,
            #     edge_index=data_next.edge_index,
            #     edge_attr=data_next.state_diff
            # )
            # h_next = self.cbf(input_cbf_next)
            # print('next cbf', h_next)
            # action = self.actor(input_data)
            # print('action', action)
            # print('action norm', torch.mean(torch.square(action).sum(dim=1)))
            # h_dot = (h_next - h) / self._env.dt
            # max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h + self.params['eps'])
            # print('loss_h_dot', torch.mean(max_val_h_dot).item())
            # print('actual safe', not torch.any(self._env.unsafe_mask(data)))
            # print('CBF-safe', torch.all(h>0).item())
            return self.controller(input_data)
        
        else:
            return self.controller(data)

    @torch.no_grad()
    def step(self, data: Data, prob: float) -> Tensor:
        if data.edge_index.numel():
            is_safe = True
            if torch.any(self._env.unsafe_mask(data)):
                is_safe = not is_safe
            
            if self._env.delay_aware and not self.use_all_data:
                data.edge_attr = torch.stack(
                    [edge_attr[-1, :] for edge_attr in data.edge_attr]
                )
            
            self.buffer.append(deepcopy(data), is_safe)
            action = self.act(data)
            if np.random.rand() < prob:
                action = torch.zeros_like(action, device=self.device)

        else:
            action = torch.zeros(self.num_agents, self.action_dim, device=self.device)
        
        return action


    def is_update(self, step: int) -> bool:
        return step % self.batch_size == 0


    def update(self, step: int, train_ctrl: bool, writer: SummaryWriter = None) -> dict:
        acc_safe = torch.zeros(1, dtype=torch.float)
        acc_unsafe = torch.zeros(1, dtype=torch.float)
        acc_h_dot = torch.zeros(1, dtype=torch.float)
        for i_inner in range(self.params['inner_iter']):
            
            # sample from the current buffer and the memory
            if self.memory.size == 0:
                graph_list = self.buffer.sample(self.batch_size // 5, True)
                
            else:
                curr_graphs = self.buffer.sample(self.batch_size // 10, True)
                prev_graphs = self.memory.sample(self.batch_size // 5 - self.batch_size // 10, True)
                graph_list = curr_graphs + prev_graphs
            
            # update models
            if graph_list:
                if self._env.delay_aware:
                    if self.use_all_data:
                        graphs = Batch.from_data_list(graph_list, exclude_keys=['edge_attr'])
                        batched_edge_attr = self.batch_edge_attr(graph_list)
                        batched_edge_attr.requires_grad = True
                        graphs.edge_attr = batched_edge_attr

                    else:
                        graphs = Batch.from_data_list(graph_list)
                    
                    # get current state difference predictions
                    pred_state_diff = self.predictor(graphs.edge_attr)
                    true_state_diff = graphs.state_diff
                    true_state_diff_norm = torch.norm(true_state_diff, dim=1)
                    pred_err_norm = torch.norm(pred_state_diff - true_state_diff, dim=1)
                    loss_pred = torch.mean(pred_err_norm / true_state_diff_norm)
                    
                    # get CBF values and the control inputs
                    input_data = Data(
                        x=graphs.x,
                        edge_index=graphs.edge_index,
                        edge_attr=pred_state_diff,
                        u_ref = graphs.u_ref
                    )
                    h = self.cbf(input_data)

                else:
                    graphs = Batch.from_data_list(graph_list)
                    graphs.edge_attr.requires_grad = True
                    h = self.cbf(graphs)

                # calculate loss
                # unsafe region h(x) < 0
                unsafe_mask = self._env.unsafe_mask(graphs)
                h_unsafe = h[unsafe_mask]
                if h_unsafe.numel():
                    max_val_unsafe = torch.relu(h_unsafe + self.params['eps'])
                    loss_unsafe = torch.mean(max_val_unsafe)
                    acc_unsafe = torch.mean(torch.less(h_unsafe, 0).type_as(h_unsafe))
                    
                else:
                    loss_unsafe = torch.tensor(0.0).type_as(h_unsafe)
                    acc_unsafe = torch.tensor(1.0).type_as(h_unsafe)
                
                # safe region h(x) > 0
                safe_mask = self._env.safe_mask(graphs)
                h_safe = h[safe_mask]
                if h_safe.numel():
                    max_val_safe = torch.relu(-h_safe + self.params['eps'])
                    loss_safe = torch.mean(max_val_safe)
                    acc_safe = torch.mean(torch.greater_equal(h_safe, 0).type_as(h_safe))
                    
                else:
                    loss_safe = torch.tensor(0.0).type_as(h_safe)
                    acc_safe = torch.tensor(1.0).type_as(h_safe)
                    
                # derivative loss h_dot + \alpha h > 0
                if self._env.delay_aware:
                    actions = self.controller(input_data)
                else:
                    actions = self.controller(graphs)

                graphs_next = self._env.forward_graph(graphs, actions, self.use_all_data)
                
                if self._env.delay_aware:
                    graphs_next.edge_attr.requires_grad = True
                    true_state_diff_next = graphs_next.state_diff
                    cbf_input_data_next = Data(
                        x=graphs_next.x,
                        edge_index=graphs_next.edge_index,
                        edge_attr=true_state_diff_next
                    )
                    h_next = self.cbf(cbf_input_data_next)

                    # update prediction loss
                    pred_state_diff_next = self.predictor(graphs_next.edge_attr)
                    true_state_diff_next_norm = torch.norm(true_state_diff_next, dim=1)
                    pred_err_next_norm = torch.norm(pred_state_diff_next - true_state_diff_next, dim=1)
                    loss_pred += self.params['loss_pred_next_ratio_coef'] * torch.mean(pred_err_next_norm / true_state_diff_next_norm)

                else:
                    h_next = self.cbf(graphs_next)
                
                h_dot = (h_next - h) / self._env.dt
                max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h + self.params['eps'])
                loss_h_dot = torch.mean(max_val_h_dot)
                acc_h_dot = torch.mean(torch.greater_equal((h_dot + self.params['alpha'] * h), 0).type_as(h_dot))
                
                # action loss
                loss_action = torch.mean(torch.square(actions).sum(dim=1))
                
                # total control loss
                loss_ctrl = self.params['loss_unsafe_coef'] * loss_unsafe + \
                        self.params['loss_safe_coef'] * loss_safe + \
                        self.params['loss_h_dot_coef'] * loss_h_dot + \
                        self.params['loss_action_coef'] * loss_action
                
                # backpropagation
                if not(train_ctrl) and self._env.delay_aware:

                    # update predictor
                    self.optim_predictor.zero_grad(set_to_none=True)
                    loss_pred.backward()
                    torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1e-3)
                    self.optim_predictor.step()
                
                else:

                    # update CBF and controller
                    self.optim_cbf.zero_grad(set_to_none=True)
                    self.optim_actor.zero_grad(set_to_none=True)
                    loss_ctrl.backward()
                    torch.nn.utils.clip_grad_norm_(self.cbf.parameters(), 1e-3)
                    torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1e-3)
                    self.optim_cbf.step()
                    self.optim_actor.step()

                # save loss
                writer.add_scalar('loss/unsafe', loss_unsafe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/safe', loss_safe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/derivative', loss_h_dot.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/action', loss_action.item(), step * self.params['inner_iter'] + i_inner)
                if self._env.delay_aware:
                    writer.add_scalar('loss/prediction', loss_pred.item(), step * self.params['inner_iter'] + i_inner)

                # save accuracy
                writer.add_scalar('acc/unsafe', acc_unsafe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('acc/safe', acc_safe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('acc/derivative', acc_h_dot.item(), step * self.params['inner_iter'] + i_inner)

        # merge the current buffer to the memory
        self.memory.merge(self.buffer)
        self.buffer.clear()

        train_info = {
            'loss/safe': loss_safe.item(),
            'loss/unsafe': loss_unsafe.item(),
            'loss/derivative': loss_h_dot.item(),
            'acc/safe': acc_safe.item(),
            'acc/unsafe': acc_unsafe.item(),
            'acc/derivative': acc_h_dot.item(),
        }
        if self._env.delay_aware:
            train_info['loss/prediction'] = loss_pred.item()

        return train_info
        
        
    def batch_edge_attr(self, graph_list):
        edge_attr = []
        for graph in graph_list:
            edge_attr.extend(graph.edge_attr)

        batched_edge_attr = pack_sequence(edge_attr, enforce_sorted=False)
        
        return batched_edge_attr

    
    def preprocess_edge_attr(self, edge_attr):
        no_delay_attr_idx = []
        no_delay_attr = []
        for edge_idx, edge in enumerate(edge_attr):
            if edge[-1][-1] == 0:
                no_delay_attr_idx.append(edge_idx)
                no_delay_attr.append(deepcopy(edge[-1][self._env.action_dim:-1]))

        delay_attr = [deepcopy(edge) for edge_idx, edge in enumerate(edge_attr) if edge_idx not in no_delay_attr_idx]
        if delay_attr:
            pred_attr = self.predictor(delay_attr)
            out_attr = torch.zeros([len(no_delay_attr) + len(delay_attr), self._env.edge_dim], device=self.device)
            out_attr[no_delay_attr_idx, :] = torch.stack(no_delay_attr)
            mask = torch.ones(len(no_delay_attr) + len(delay_attr), device=self.device)
            mask[no_delay_attr_idx] = 0
            out_attr[mask.bool(), :] = pred_attr

        else:
            out_attr = torch.stack(no_delay_attr)

        return out_attr
    
                
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.predictor.state_dict(), os.path.join(save_dir, 'predictor.pkl'))
        torch.save(self.cbf.state_dict(), os.path.join(save_dir, 'cbf.pkl'))
        torch.save(self.controller.state_dict(), os.path.join(save_dir, 'actor.pkl'))


    def load(self, load_dir: str):
        assert os.path.exists(load_dir)
        self.predictor.load_state_dict(torch.load(os.path.join(load_dir, 'predictor.pkl'), map_location=self.device))
        self.cbf.load_state_dict(torch.load(os.path.join(load_dir, 'cbf.pkl'), map_location=self.device))
        self.controller.load_state_dict(torch.load(os.path.join(load_dir, 'actor.pkl'), map_location=self.device))


    def apply(self, data: Data) -> Tensor:
        if data.edge_attr:
            input_data = Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=self.predictor(data.edge_attr),
                    u_ref=data.u_ref
                )
            h = self.cbf(input_data).detach()
            action = self.controller(input_data).detach()
            nominal = torch.zeros_like(action, device=self.device)
            data_next = self._env.forward_graph(data, nominal, self.use_all_data)
            data_next_pred = Data(
                        x=data_next.x,
                        edge_index=data_next.edge_index,
                        edge_attr=self.predictor(data_next.edge_attr)
                    )
            h_next = self.cbf(data_next_pred)
            h_dot = (h_next - h) / self._env.dt
            safe_nominal_mask = torch.sign(torch.relu(h_dot + self.params['alpha'] * h - self.params['eps'])).detach()
            return nominal * safe_nominal_mask + action * torch.logical_not(safe_nominal_mask)

        else:
            return torch.zeros(self.num_agents, self.action_dim, device=self.device)
