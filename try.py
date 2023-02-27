from macbf_gnn.env.drone import Drone
import torch
import matplotlib


matplotlib.use('TkAgg')
device = torch.device('cpu')
env = Drone(num_agents=10, device=device)
data = env.reset()
env.render()
