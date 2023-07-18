from copy import deepcopy
import torch.nn as nn
import os
import torch
import numpy as np
import cvxpy as cp
import torch_geometric

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential
from torch import Tensor
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
# from pytorch_forecasting.utils import unpack_sequence
from typing import Optional

from macbf_gnn.nn import MLP, CBFGNNLayer, Predictor
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

    def forward_explict(self, x: Tensor, edge_index: Tensor) -> Tensor:
        pass

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
            params: Optional[dict] = None
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
        self.predictor = Predictor(
            input_dim=self.action_dim + self.state_dim + 1,
            output_dim=self.state_dim
        ).to(device)
        self.cbf = CBFGNN(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            phi_dim=256
        ).to(device)
        self.actor = ControllerGNN(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            phi_dim=256,
            action_dim=self.action_dim
        ).to(device)

        # optimizer
        self.optim_predictor = Adam(self.predictor.parameters(), lr=1e-3)
        self.optim_cbf = Adam(self.cbf.parameters(), lr=3e-4)
        self.optim_actor = Adam(self.actor.parameters(), lr=1e-3)

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
        if data.edge_attr:
            data_pred = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=self.predictor(data.edge_attr),
                u_ref=data.u_ref
            )
            # print('')
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # print('actual safe', not torch.any(self._env.unsafe_mask(data)))
            # h = self.cbf(data_pred)
            # print('state', data.state_diff)
            # print('edge attr', data.edge_attr)
            # print('pred state', data_pred.edge_attr)
            # true_state_diff = data.state_diff
            # true_state_diff_norm = torch.norm(true_state_diff, dim=1)
            # pred_err_norm = torch.norm(data_pred.edge_attr - true_state_diff, dim=1)
            # print('loss pred', torch.mean(pred_err_norm / true_state_diff_norm))
            # print('cbf', h)
            # action = self.actor(data_pred)
            # data_next = self._env.forward_graph(data, action)
            # input_cbf_next = Data(
            #     x=data_next.x,
            #     edge_index=data_next.edge_index,
            #     edge_attr=data_next.state_diff
            # )
            # h_next = self.cbf(input_cbf_next)
            # print('next cbf', h_next)
            # action = self.actor(data_pred)
            # print('action', action)
            # print('action norm', torch.mean(torch.square(action).sum(dim=1)))
            # h_dot = (h_next - h) / self._env.dt
            # max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h + self.params['eps'])
            # print('loss_h_dot', torch.mean(max_val_h_dot).item())
            return self.actor(data_pred)
            
        else:
            return torch.zeros(self.num_agents, self.action_dim)

    @torch.no_grad()
    def step(self, data: Data, prob: float) -> Tensor:
        is_safe = True
        if torch.any(self._env.unsafe_mask(data)):
            is_safe = not is_safe
            
        self.buffer.append(deepcopy(data), is_safe)
        action = self.act(data)
        if np.random.rand() < prob:
            action = torch.zeros_like(action)
        
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
            
            graph_list = [graph for graph in graph_list if graph.edge_attr] # discard graphs with no data
            if graph_list:
                batched_edge_attr = self.batch_edge_attr(graph_list)
                batched_edge_attr.requires_grad = True
                graphs = Batch.from_data_list(graph_list, exclude_keys=['edge_attr'])
                graphs.edge_attr = batched_edge_attr
                
                # get current state difference predictions
                pred_state_diff = self.predictor(batched_edge_attr)
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
                # mask out the agents with no links at time t
                '''degree = torch_geometric.utils.degree(graphs.edge_index[0], graphs.num_nodes)
                agents_with_link = torch.nonzero(degree)
                degree_mask = torch.zeros_like(h)
                degree_mask = degree_mask.scatter_(0, agents_with_link, 1).bool()[:, 0]
                # h_dot_mask = degree_mask * safe_mask
                h_dot_mask = degree_mask

                h_next = self.cbf(graphs_next)[h_dot_mask]
                graphs_next_new_link = []
                for i_graph, this_graph in enumerate(graph_list):
                    this_graph_next = self._env.forward_graph(
                        this_graph, actions[i_graph * self.num_agents: (i_graph + 1) * self.num_agents])
                    graphs_next_new_link.append(self._env.add_communication_links(this_graph_next))
                graphs_next_new_link = Batch.from_data_list(graphs_next_new_link)
                h_next_new_link = self.cbf(graphs_next_new_link)[h_dot_mask]
                h_dot = (h_next - h[h_dot_mask]) / self._env.dt
                h_dot_new_link = (h_next_new_link - h[h_dot_mask]) / self._env.dt
                residue = (h_dot_new_link - h_dot).clone().detach()
                h_dot = residue + h_dot'''
                
                actions = self.actor(input_data)
                graphs_next = self._env.forward_graph(graphs, actions)
                graphs_next.edge_attr.requires_grad = True
                pred_state_diff_next = self.predictor(graphs_next.edge_attr)
                
                true_state_diff_next = graphs_next.state_diff
                true_state_diff_next_norm = torch.norm(true_state_diff_next, dim=1)
                pred_err_next_norm = torch.norm(pred_state_diff_next - true_state_diff_next, dim=1)
                loss_pred += self.params['loss_pred_next_ratio_coef'] * torch.mean(pred_err_next_norm / true_state_diff_next_norm)
                
                cbf_input_data_next = Data(
                    x=graphs_next.x,
                    edge_index=graphs_next.edge_index,
                    edge_attr=true_state_diff_next
                )
                # not_unsafe_mask = not(unsafe_mask)
                # h_not_unsafe = h[not_unsafe_mask]
                h_next = self.cbf(cbf_input_data_next)
                # h_dot = (h_next - h_not_unsafe) / self._env.dt
                h_dot = (h_next - h) / self._env.dt
                # max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h_not_unsafe + self.params['eps'])
                max_val_h_dot = torch.relu(-h_dot - self.params['alpha'] * h + self.params['eps'])
                loss_h_dot = torch.mean(max_val_h_dot)
                # acc_h_dot = torch.mean(torch.greater_equal((h_dot + self.params['alpha'] * h_not_unsafe), 0).type_as(h_dot))
                acc_h_dot = torch.mean(torch.greater_equal((h_dot + self.params['alpha'] * h), 0).type_as(h_dot))
                
                # action loss
                loss_action = torch.mean(torch.square(actions).sum(dim=1))
                
                # total control loss
                loss_ctrl = self.params['loss_unsafe_coef'] * loss_unsafe + \
                        self.params['loss_safe_coef'] * loss_safe + \
                        self.params['loss_h_dot_coef'] * loss_h_dot + \
                        self.params['loss_action_coef'] * loss_action

                # backpropagation
                if not train_ctrl:
                    self.optim_predictor.zero_grad(set_to_none=True)
                    loss_pred.backward()
                    torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1e-3)
                    self.optim_predictor.step()
                
                else:
                    #self.optim_predictor.zero_grad(set_to_none=True)
                    self.optim_cbf.zero_grad(set_to_none=True)
                    self.optim_actor.zero_grad(set_to_none=True)
                    loss_ctrl.backward()
                    #torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1e-3)
                    torch.nn.utils.clip_grad_norm_(self.cbf.parameters(), 1e-3)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1e-3)
                    #self.optim_predictor.step() 
                    self.optim_cbf.step()
                    self.optim_actor.step()

                # save loss
                writer.add_scalar('loss/unsafe', loss_unsafe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/safe', loss_safe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/derivative', loss_h_dot.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/action', loss_action.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('loss/prediction', loss_pred.item(), step * self.params['inner_iter'] + i_inner)

                # save accuracy
                writer.add_scalar('acc/unsafe', acc_unsafe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('acc/safe', acc_safe.item(), step * self.params['inner_iter'] + i_inner)
                writer.add_scalar('acc/derivative', acc_h_dot.item(), step * self.params['inner_iter'] + i_inner)

        # merge the current buffer to the memory
        self.memory.merge(self.buffer)
        self.buffer.clear()

        return {
            'h safe/avg': torch.mean(h_safe).item(),
            'h unsafe/avg': torch.mean(h_unsafe).item(),
            'loss/prediction': loss_pred.item(),
            'loss/safe': loss_safe.item(),
            'loss/unsafe': loss_unsafe.item(),
            'loss/derivative': loss_h_dot.item(),
            'acc/safe': acc_safe.item(),
            'acc/unsafe': acc_unsafe.item(),
            'acc/derivative': acc_h_dot.item(),
        }
        
        
    def batch_edge_attr(self, graph_list):
        edge_attrs = []
        for graph in graph_list:
            edge_attrs.extend(graph.edge_attr)
        
        batched_edge_attr = pack_sequence(edge_attrs, enforce_sorted=False)
        
        return batched_edge_attr
                
                
    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(self.predictor.state_dict(), os.path.join(save_dir, 'predictor.pkl'))
        torch.save(self.cbf.state_dict(), os.path.join(save_dir, 'cbf.pkl'))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pkl'))


    def load(self, load_dir: str):
        assert os.path.exists(load_dir)
        self.predictor.load_state_dict(torch.load(os.path.join(load_dir, 'predictor.pkl'), map_location=self.device))
        self.cbf.load_state_dict(torch.load(os.path.join(load_dir, 'cbf.pkl'), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'actor.pkl'), map_location=self.device))


    '''def apply(self, data: Data) -> Tensor:
        h = self.cbf(data).detach()
        action = self.actor(data).detach()
        nominal = torch.zeros_like(action)

        data_next = self._env.forward_graph(data, nominal)
        h_next = self.cbf(data_next)
        h_dot = (h_next - h) / self._env.dt
        max_val_h_dot = torch.relu((-h_dot - self.params['alpha'] * h))
        loss_h_dot = torch.mean(max_val_h_dot)
        if loss_h_dot <= 0:
            return nominal

        actions = list(torch.split(action, 1, dim=0))
        optim = []
        for i in range(self.num_agents):
            actions[i].requires_grad = True

        for i in range(self.num_agents):
            optim.append(Adam((actions[i],), lr=0.1))

        # consider the satisfaction of h_dot condition
        i_iter = 0
        max_iter = 30
        while True:
            action = torch.cat(actions, dim=0)
            data_next = self._env.forward_graph(data, action)
            h_next = self.cbf(data_next)
            h_dot = (h_next - h) / self._env.dt
            max_val_h_dot = torch.relu((-h_dot - self.params['alpha'] * h))
            loss_h_dot = torch.mean(max_val_h_dot)
            if loss_h_dot <= 0 or i_iter > max_iter:
                break
            else:
                val_agent = torch.nonzero(max_val_h_dot)[:, 0]
                for i in val_agent:
                    optim[i].zero_grad(set_to_none=True)
                loss_h_dot.backward()
                for i in val_agent:
                    optim[i].step()
                    actions[i].requires_grad = False
                    actions[i] -= 0.3 * torch.randn_like(actions[i].grad) * actions[i].grad
                    actions[i].requires_grad = True
                i_iter += 1

        return action'''